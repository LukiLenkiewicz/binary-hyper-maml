from collections import defaultdict
from copy import deepcopy
from time import time
import sys

import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

import backbone
from methods.hypernets.utils import get_param_dict, accuracy_from_scores
from methods.maml import MAML

from hypnettorch.hnets.chunked_mlp_hnet import HMLP, ChunkedHMLP


class GetSubnetFaster(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, zeros, ones, k_val):
        return torch.where(scores > k_val, zeros.to(scores.device), ones.to(scores.device))

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


class GetSubnetFasterSoft(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, zeros, ones, k_val):
        return torch.where(scores > k_val, zeros.to(scores.device), scores)

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


def percentile(scores, sparsity):
    k = 1 + round(.01 * float(sparsity) * (scores.numel() - 1))
    return scores.view(-1).kthvalue(k).values.item()


class HyperMAML(MAML):
    def __init__(self, model_func, n_way, n_support, n_query, params=None, approx=False):
        super(HyperMAML, self).__init__(model_func, n_way, n_support, n_query, params=params)
        self.loss_fn = nn.CrossEntropyLoss()

        self.hn_tn_hidden_size = params.hn_tn_hidden_size
        self.hn_tn_depth = params.hn_tn_depth
        self._init_classifier()

        self.enhance_embeddings = params.hm_enhance_embeddings

        self.n_task = 4
        self.task_update_num = 5
        self.train_lr = 0.01
        self.approx = approx  # first order approx.

        self.hn_sup_aggregation = params.hn_sup_aggregation
        self.hm_lambda = params.hm_lambda
        self.hm_save_delta_params = params.hm_save_delta_params
        self.hm_use_class_batch_input = params.hm_use_class_batch_input
        self.hn_adaptation_strategy = params.hn_adaptation_strategy
        self.hm_support_set_loss = params.hm_support_set_loss
        self.hm_maml_warmup = params.hm_maml_warmup
        self.hm_maml_warmup_epochs = params.hm_maml_warmup_epochs
        self.hm_maml_warmup_switch_epochs = params.hm_maml_warmup_switch_epochs
        self.hm_maml_update_feature_net = params.hm_maml_update_feature_net
        self.hm_update_operator = params.hm_update_operator
        self.hm_set_forward_with_adaptation = params.hm_set_forward_with_adaptation
        self.hn_val_lr = params.hn_val_lr
        self.hn_val_epochs = params.hn_val_epochs
        self.hn_val_optim = params.hn_val_optim

        self.alpha = 0
        self.hn_alpha_step = params.hn_alpha_step

        if self.hn_adaptation_strategy == 'increasing_alpha' and self.hn_alpha_step < 0:
            raise ValueError('hn_alpha_step is not positive!')

        self.single_test = False
        self.epoch = -1
        self.start_epoch = -1
        self.stop_epoch = -1

        self.bm_backbone_weights = params.bm_backbone_weights
        self.bm_detach_embedding = params.bm_detach_embedding
        self.bm_method = params.bm_method
        self.bm_activation = params.bm_activation
        self.bm_layer_size = params.bm_layer_size
        self.bm_num_layers = params.bm_num_layers
        self.bm_detach_feature_net = params.bm_detach_second_encoder
        self.bm_decrease_epochs = params.bm_decrease_epochs
        self.bm_gumbel_discretize = params.bm_gumbel_discretize
        self.bm_freeze_target_network = params.bm_freeze_target_network
        self.bm_fixed_size_mask = params.bm_fixed_size_mask
        self.bm_mask_size = params.bm_mask_size
        self.bm_chunk_emb_size = params.bm_chunk_emb_size
        self.bm_chunk_size = params.bm_chunk_size
        self.bm_adjust_eval = params.bm_adjust_eval

        #gumbel_softmax temp
        self.start_temp = 1.0
        self.end_temp = 0.2
        self.temp_diff = self.start_temp - self.end_temp

        hypernet_layers = [self.bm_layer_size for _ in range(self.bm_num_layers)]

        self.feature_query = model_func()

        self.calculate_embedding_size()

        shapes = []
        
        if self.bm_backbone_weights:
            backbone_shapes = [list(layer.shape) for layer in self.feature_query.parameters()]
            classifier_shapes = [list(layer.shape) for layer in self.classifier.parameters()]
            shapes =  backbone_shapes + classifier_shapes
        else:
            shapes = [list(layer.shape) for layer in self.classifier.parameters()]

        #ones percentage
        self.ones_ = 0
        self.all_units = sum(np.prod(shape) for shape in shapes)

        if self.bm_activation == "gumbel_softmax":
            shapes = [dim+[2] for dim in shapes]

        if self.bm_freeze_target_network:
            for param in self.feature_query.parameters():
                param.requires_grad = False

            for param in self.classifier.parameters():
                param.requires_grad = False

        self.hypernet = ChunkedHMLP(shapes, uncond_in_size=self.embedding_size, cond_in_size=0, chunk_emb_size=self.bm_chunk_emb_size,
                layers=hypernet_layers, chunk_size=self.bm_chunk_size, num_cond_embs=1)
        
    def _init_classifier(self):
        assert self.hn_tn_hidden_size % self.n_way == 0, f"hn_tn_hidden_size {self.hn_tn_hidden_size} should be the multiple of n_way {self.n_way}"
        layers = []

        for i in range(self.hn_tn_depth):
            in_dim = self.feat_dim if i == 0 else self.hn_tn_hidden_size
            out_dim = self.n_way if i == (self.hn_tn_depth - 1) else self.hn_tn_hidden_size

            linear = backbone.Linear_fw(in_dim, out_dim)
            linear.bias.data.fill_(0)

            layers.append(linear)

        self.classifier = nn.Sequential(*layers)

    def calculate_embedding_size(self):

        n_classes_in_embedding = 1 if self.hm_use_class_batch_input else self.n_way
        n_support_per_class = 1 if self.hn_sup_aggregation == 'mean' else self.n_support
        single_support_embedding_len = self.feat_dim + self.n_way + 1 if self.enhance_embeddings else self.feat_dim
        self.embedding_size = n_classes_in_embedding * n_support_per_class * single_support_embedding_len

    def apply_embeddings_strategy(self, embeddings):
        if self.hn_sup_aggregation == 'mean':
            new_embeddings = torch.zeros(self.n_way, *embeddings.shape[1:])

            for i in range(self.n_way):
                lower = i * self.n_support
                upper = (i + 1) * self.n_support
                new_embeddings[i] = embeddings[lower:upper, :].mean(dim=0)

            return new_embeddings.cuda()

        return embeddings

    def get_support_data_labels(self):
        return torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()  # labels for support data

    def get_hn_delta_params(self, support_embeddings):
        if self.bm_detach_embedding:
            support_embeddings = support_embeddings.detach()

        support_embeddings_resh = support_embeddings.reshape(1, -1)
        delta_params = self.hypernet(support_embeddings_resh)

        self.ones_ = 0

        if self.bm_activation == "gumbel_softmax":
            if self.epoch >= self.bm_decrease_epochs:
                temp = self.end_temp
            else:
                temp = self.start_temp - self.temp_diff*(self.epoch/self.bm_decrease_epochs)
            for i in range(len(delta_params)):
                delta_params[i] = F.gumbel_softmax(delta_params[i], tau=temp, hard=self.bm_gumbel_discretize, dim=-1)[...,-1]
                self.ones_ += float(torch.sum(delta_params[i]))
        elif self.bm_activation == "sigmoid":
            for i in range(len(delta_params)):
                delta_params[i] = torch.sigmoid(delta_params[i])
    
            if self.bm_fixed_size_mask:
                params_flat = [param.clone().detach().reshape(-1) for param in delta_params]
                concat = torch.cat(params_flat)

                k_val = torch.quantile(concat, self.bm_mask_size).item()

                for i in range(len(delta_params)):
                    delta_params[i] = GetSubnetFaster.apply(delta_params[i], torch.tensor([0.]), torch.tensor([1.]), k_val)
                    self.ones_ += torch.sum(delta_params[i]).item()
        elif self.bm_activation == "tanh":
            for i in range(len(delta_params)):
                delta_params[i] = torch.tanh(delta_params[i])
    
            if self.bm_fixed_size_mask:
                params_flat = [param.clone().detach().reshape(-1) for param in delta_params]
                concat = torch.cat(params_flat)

                k_val = torch.quantile(concat, self.bm_mask_size).item()

                for i in range(len(delta_params)):
                    delta_params[i] = GetSubnetFasterSoft.apply(delta_params[i], torch.tensor([0.]), torch.tensor([1.]), k_val)
                    self.ones_ += torch.sum(delta_params[i]).item()

        self.ones_list.append(self.ones_)

        return delta_params

    def _update_weight(self, weight, update_value, subnetwork=False):
        if self.hm_update_operator == 'multiply' or subnetwork:
            if weight.fast is None:
                weight.fast = weight * update_value
            else:
                weight.fast = weight.fast * update_value
        elif self.hm_update_operator == 'minus':
            if weight.fast is None:
                weight.fast = weight - update_value
            else:
                weight.fast = weight.fast - update_value
        elif self.hm_update_operator == 'plus':
            if weight.fast is None:
                weight.fast = weight + update_value
            else:
                weight.fast = weight.fast + update_value


    def _update_network_weights(self, delta_params_list, support_embeddings, support_data_labels, support_data):

        if self.bm_freeze_target_network:
            for param in self.feature_query.parameters():
                param.requires_grad = True

            for param in self.classifier.parameters():
                param.requires_grad = True

        # zeroing fast weights
        fast_parameters = []
        fet_fast_parameters = list(self.feature_query.parameters())
        for weight in self.feature_query.parameters():
            weight.fast = None
        self.feature_query.zero_grad()
        fast_parameters = fast_parameters + fet_fast_parameters

        clf_fast_parameters = list(self.classifier.parameters())
        for weight in self.classifier.parameters():
            weight.fast = None
        self.classifier.zero_grad()
        fast_parameters = fast_parameters + clf_fast_parameters

        classifier_offset = len(fet_fast_parameters)

        #prepare fast weights
        for k, weight in enumerate(self.feature_query.parameters()):
            update_value = delta_params_list[k]
            self._update_weight(weight, update_value, subnetwork=True)


        for k, weight in enumerate(self.classifier.parameters()):
            update_value = delta_params_list[classifier_offset+k]
            self._update_weight(weight, update_value, subnetwork=True)

        for task_step in range(self.task_update_num):
            support_embeddings_query = self.feature_query(support_data)
            scores = self.classifier(support_embeddings_query)

            set_loss = self.loss_fn(scores, support_data_labels)

            grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True, allow_unused=True)

            for k, weight in enumerate(self.feature_query.parameters()):
                update_value = self.train_lr * grad[k]
                self._update_weight(weight, update_value)

            for k, weight in enumerate(self.classifier.parameters()):
                update_value = (self.train_lr * grad[classifier_offset + k])
                self._update_weight(weight, update_value)

        if self.bm_freeze_target_network:
            for param in self.feature_query.parameters():
                param.requires_grad = False

            for param in self.classifier.parameters():
                param.requires_grad = False

    def _get_list_of_delta_params(self, maml_warmup_used, support_embeddings, support_data_labels):
        if not maml_warmup_used:

            if self.enhance_embeddings:
                with torch.no_grad():
                    logits = self.classifier.forward(support_embeddings).detach()
                    logits = F.softmax(logits, dim=1)

                labels = support_data_labels.view(support_embeddings.shape[0], -1)
                support_embeddings = torch.cat((support_embeddings, logits, labels), dim=1)

            for weight in self.parameters():
                weight.fast = None
            self.zero_grad()

            support_embeddings = self.apply_embeddings_strategy(support_embeddings)

            delta_params = self.get_hn_delta_params(support_embeddings)

            if self.hm_save_delta_params and len(self.delta_list) == 0:
                self.delta_list = [{'delta_params': delta_params}]

            return delta_params
        else:
            return [torch.zeros(*i).cuda() for (_, i) in self.target_net_param_shapes.items()]

    def forward(self, x):
        if self.bm_method=="two_encoders":
            out = self.feature_query.forward(x)
        else:
            out = self.feature.forward(x)

        if self.bm_detach_feature_net:
            out = out.detach()

        scores = self.classifier.forward(out)
        return scores

    def set_forward(self, x, is_feature=False, train_stage=False):
        """ 1. Get delta params from hypernetwork with support data.
        2. Update target- network weights.
        3. Forward with query data.
        4. Return scores"""

        assert is_feature == False, 'MAML do not support fixed feature'

        x = x.cuda()
        x_var = Variable(x)
        support_data = x_var[:, :self.n_support, :, :, :].contiguous().view(self.n_way * self.n_support,
                                                                            *x.size()[2:])  # support data
        query_data = x_var[:, self.n_support:, :, :, :].contiguous().view(self.n_way * self.n_query,
                                                                          *x.size()[2:])  # query data
        support_data_labels = self.get_support_data_labels()

        support_embeddings = self.feature(support_data)

        maml_warmup_used = (
                    (not self.single_test) and self.hm_maml_warmup and (self.epoch < self.hm_maml_warmup_epochs))

        delta_params_list = self._get_list_of_delta_params(maml_warmup_used, support_embeddings, support_data_labels)

        self._update_network_weights(delta_params_list, support_embeddings, support_data_labels, support_data)

        if self.hm_set_forward_with_adaptation and not train_stage:
            scores = self.forward(support_data)
            return scores, None
        else:
            if self.hm_support_set_loss and train_stage and not maml_warmup_used:
                query_data = torch.cat((support_data, query_data))

            # if self.bm_adjust_eval:
            #     if train_stage:
            #         scores = self.forward(query_data)
            #     else:
            #         self.eval()
            #         scores = self.forward(query_data)
            #         self.train()
            # else:
            #     scores = self.forward(query_data)
            scores = self.forward(query_data)
            
            # sum of delta params for regularization
            if self.hm_lambda != 0:
                total_delta_sum = sum([delta_params.pow(2.0).sum() for delta_params in delta_params_list])

                return scores, total_delta_sum
            else:
                return scores, None

    def set_forward_adaptation(self, x, is_feature=False):  # overwrite parrent function
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')

    def set_forward_loss(self, x):
        scores, total_delta_sum = self.set_forward(x, is_feature=False, train_stage=True)
        query_data_labels = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_query))).cuda()

        if self.hm_support_set_loss:
            support_data_labels = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
            query_data_labels = torch.cat((support_data_labels, query_data_labels))

        loss = self.loss_fn(scores, query_data_labels)

        if self.hm_lambda != 0:
            loss = loss + self.hm_lambda * total_delta_sum

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy().flatten()
        y_labels = query_data_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind == y_labels)
        task_accuracy = (top1_correct / len(query_data_labels)) * 100

        return loss, task_accuracy

    def set_forward_loss_with_adaptation(self, x):
        scores, _ = self.set_forward(x, is_feature=False, train_stage=False)
        support_data_labels = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_support))).cuda()

        loss = self.loss_fn(scores, support_data_labels)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy().flatten()
        y_labels = support_data_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind == y_labels)
        task_accuracy = (top1_correct / len(support_data_labels)) * 100

        return loss, task_accuracy

    def train_loop(self, epoch, train_loader, optimizer):  # overwrite parrent function
        print_freq = 10
        avg_loss = 0
        task_count = 0
        loss_all = []
        acc_all = []
        optimizer.zero_grad()

        self.delta_list = []

        self.ones_list = []
        
        # train
        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way == x.size(0), "MAML do not support way change"

            loss, task_accuracy = self.set_forward_loss(x)
            avg_loss = avg_loss + loss.item()  # .data[0]
            loss_all.append(loss)
            acc_all.append(task_accuracy)

            task_count += 1

            if task_count == self.n_task:  # MAML update several tasks at one time
                loss_q = torch.stack(loss_all).sum(0)
                loss_q.backward()

                optimizer.step()

                task_count = 0
                loss_all = []
            if i % print_freq == 0:
                print('Epoch {:d}/{:d} | Batch {:d}/{:d} | Loss {:f}'.format(self.epoch, self.stop_epoch, i,
                                                                             len(train_loader),
                                                                             avg_loss / float(i + 1)))

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)

        metrics = {"accuracy/train": acc_mean}

        if self.hn_adaptation_strategy == 'increasing_alpha':
            metrics['alpha'] = self.alpha

        if self.hm_save_delta_params and len(self.delta_list) > 0:
            delta_params = {"epoch": self.epoch, "delta_list": self.delta_list}
            metrics['delta_params'] = delta_params

        if self.alpha < 1:
            self.alpha += self.hn_alpha_step

        metrics["ones percentage"] = np.mean(self.ones_list)/self.all_units

        return metrics

    def test_loop(self, test_loader, return_std=False, return_time: bool = False):  # overwrite parrent function

        acc_all = []
        self.delta_list = []
        acc_at = defaultdict(list)

        iter_num = len(test_loader)

        eval_time = 0

        if self.hm_set_forward_with_adaptation:
            for i, (x, _) in enumerate(test_loader):
                self.n_query = x.size(1) - self.n_support
                assert self.n_way == x.size(0), "MAML do not support way change"
                s = time()
                acc_task, acc_at_metrics = self.set_forward_with_adaptation(x)
                t = time()
                for (k, v) in acc_at_metrics.items():
                    acc_at[k].append(v)
                acc_all.append(acc_task)
                eval_time += (t - s)

        else:
            for i, (x, _) in enumerate(test_loader):
                self.n_query = x.size(1) - self.n_support
                assert self.n_way == x.size(0), f"MAML do not support way change, {self.n_way=}, {x.size(0)=}"
                s = time()
                correct_this, count_this = self.correct(x)
                t = time()
                acc_all.append(correct_this / count_this * 100)
                eval_time += (t - s)

        metrics = {
            k: np.mean(v) if len(v) > 0 else 0
            for (k, v) in acc_at.items()
        }

        num_tasks = len(acc_all)
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        print("Num tasks", num_tasks)

        ret = [acc_mean]
        if return_std:
            ret.append(acc_std)
        if return_time:
            ret.append(eval_time)
        ret.append(metrics)

        return ret

    def set_forward_with_adaptation(self, x: torch.Tensor):
        self_copy = deepcopy(self)

        # deepcopy does not copy "fast" parameters so it should be done manually
        for param1, param2 in zip(self.parameters(), self_copy.parameters()):
            if hasattr(param1, 'fast'):
                if param1.fast is not None:
                    param2.fast = param1.fast.clone()
                else:
                    param2.fast = None

        metrics = {
            "accuracy/val@-0": self_copy.query_accuracy(x)
        }

        val_opt_type = torch.optim.Adam if self.hn_val_optim == "adam" else torch.optim.SGD
        val_opt = val_opt_type(self_copy.parameters(), lr=self.hn_val_lr)

        if self.hn_val_epochs > 0:
            for i in range(1, self.hn_val_epochs + 1):
                self_copy.train()
                val_opt.zero_grad()
                loss, val_support_acc = self_copy.set_forward_loss_with_adaptation(x)
                loss.backward()
                val_opt.step()
                self_copy.eval()
                metrics[f"accuracy/val_support_acc@-{i}"] = val_support_acc
                metrics[f"accuracy/val_loss@-{i}"] = loss.item()
                metrics[f"accuracy/val@-{i}"] = self_copy.query_accuracy(x)

        # free CUDA memory by deleting "fast" parameters
        for param in self_copy.parameters():
            param.fast = None

        return metrics[f"accuracy/val@-{self.hn_val_epochs}"], metrics

    def query_accuracy(self, x: torch.Tensor) -> float:
        scores, _ = self.set_forward(x, train_stage=True)
        return 100 * accuracy_from_scores(scores, n_way=self.n_way, n_query=self.n_query)

    def correct(self, x):
        scores, _ = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)

    def reshape_hypernetwork_output(self, weights, shapes):
        last_used_weight = 0
        reshaped_weights = []
        for shape in shapes:
            num_params = np.prod(shape)
            layer_weights = weights[last_used_weight:last_used_weight+num_params]
            reshaped_weights.append(layer_weights.reshape(shape))
            last_used_weight += num_params

        return reshaped_weights

    def _get_p_value(self):
        if self.epoch < self.hm_maml_warmup_epochs:
            return 1.0
        elif self.hm_maml_warmup_epochs <= self.epoch < self.hm_maml_warmup_epochs + self.hm_maml_warmup_switch_epochs:
            return (self.hm_maml_warmup_switch_epochs + self.hm_maml_warmup_epochs - self.epoch) / (self.hm_maml_warmup_switch_epochs + 1)
        return 0.0
