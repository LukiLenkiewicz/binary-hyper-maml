from collections import defaultdict
from copy import deepcopy
from time import time

import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

import backbone
from backbone import Linear_fw
from methods.hypernets.utils import accuracy_from_scores
from methods.vbh_meta_template import VBHMetaTemplate
from methods.hypernets.binary_maml_utils import Binarizer

from hypnettorch.hnets.chunked_mlp_hnet import ChunkedHMLP, HMLP

EMBEDDING_SIZE = 768

class VBHMAML(VBHMetaTemplate):
    def __init__(self, n_way, n_support, n_query, params=None, approx=False):
        super().__init__(n_way, n_support, change_way = False)
        self.loss_fn = nn.CrossEntropyLoss()

        self.hn_tn_hidden_size = params.hn_tn_hidden_size
        self.hn_tn_depth = params.hn_tn_depth

        self.enhance_embeddings = params.hm_enhance_embeddings

        self.n_task = 4
        self.task_update_num = 5
        self.train_lr = 0.01
        self.approx = approx  # first order approx.

        self.hn_sup_aggregation = params.hn_sup_aggregation
        self.hn_hidden_size = params.hn_hidden_size
        self.hm_save_delta_params = params.hm_save_delta_params
        self.hm_support_set_loss = params.hm_support_set_loss
        self.hm_update_operator = params.hm_update_operator
        self.hm_load_feature_net = params.hm_load_feature_net
        self.hm_feature_net_path = params.hm_feature_net_path
        self.hm_set_forward_with_adaptation = params.hm_set_forward_with_adaptation
        self.hn_val_lr = params.hn_val_lr
        self.hn_val_epochs = params.hn_val_epochs
        self.hn_val_optim = params.hn_val_optim
        self.hn_use_mask = params.hn_use_mask

        self.alpha = 0
        self.hn_alpha_step = params.hn_alpha_step


        self.single_test = False
        self.epoch = -1
        self.start_epoch = -1
        self.stop_epoch = -1

        self.bm_activation = params.bm_activation
        self.bm_layer_size = params.bm_layer_size
        self.bm_num_layers = params.bm_num_layers
        self.bm_mask_size = params.bm_mask_size


        self.calculate_embedding_size()

        self.classifier = Classifier_FW(EMBEDDING_SIZE, num_layers=self.hn_tn_depth, layer_size=self.hn_tn_hidden_size, num_classes=self.n_way)
        self.hypernet = self._init_hypernet()

    def _init_classifier(self):
        assert self.hn_tn_hidden_size % self.n_way == 0, f"hn_tn_hidden_size {self.hn_tn_hidden_size} should be the multiple of n_way {self.n_way}"
        layers = []

        for i in range(self.hn_tn_depth):
            in_dim = self.feat_dim if i == 0 else self.hn_tn_hidden_size
            out_dim = self.n_way if i == (self.hn_tn_depth - 1) else self.hn_tn_hidden_size

            linear = backbone.Linear_fw(in_dim, out_dim)
            linear.bias.data.fill_(0)

            layers.append(linear)

        classifier = nn.Sequential(*layers)
        return classifier

    def _init_hypernet(self):
        shapes = [list(layer.shape) for layer in self.classifier.parameters()]
        hypernet_layers = [self.bm_layer_size for _ in range(self.bm_num_layers)]

        hypernet = HMLP(shapes, uncond_in_size=3870, cond_in_size=0, layers=hypernet_layers, num_cond_embs=1)
        # hypernet = ChunkedHMLP(shapes, uncond_in_size=3870, cond_in_size=0, chunk_emb_size=8,
                # layers=hypernet_layers, chunk_size=325, num_cond_embs=1)

        return hypernet

    def calculate_embedding_size(self):
        n_classes_in_embedding = self.n_way
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
        support_embeddings_resh = support_embeddings.reshape(1, -1)
        delta_params = self.hypernet(support_embeddings_resh)

        params_flat = [param.clone().detach().reshape(-1) for param in delta_params]
        concat = torch.cat(params_flat)

        k_val = torch.quantile(concat, self.bm_mask_size).item()

        for i in range(len(delta_params)):
            delta_params[i] = Binarizer.apply(delta_params[i], k_val)

        return delta_params

    def _update_weight(self, weight, update_value):
        if self.hm_update_operator == 'minus':
            if weight.fast is None:
                weight.fast = weight - update_value
            else:
                weight.fast = weight.fast - update_value
        elif self.hm_update_operator == 'plus':
            if weight.fast is None:
                weight.fast = weight + update_value
            else:
                weight.fast = weight.fast + update_value
        elif self.hm_update_operator == 'multiply':
            if weight.fast is None:
                weight.fast = weight * update_value
            else:
                weight.fast = weight.fast * update_value

    def _update_network_weights(self, delta_params_list, support_embeddings, support_data_labels, train_stage=False):
        fast_parameters = list(self.classifier.parameters())
        for weight in self.classifier.parameters():
            weight.fast = None
        self.classifier.zero_grad()

        if self.hn_use_mask:
            for k, weight in enumerate(self.classifier.parameters()):
                update_value = delta_params_list[k]
                self._update_weight(weight, update_value)

        for task_step in range(self.task_update_num):
            scores = self.classifier(support_embeddings)

            set_loss = self.loss_fn(scores, support_data_labels)

            grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True,
                                        allow_unused=True)  # build full graph support gradient of gradient

            if self.approx:
                grad = [g.detach() for g in
                        grad]  # do not calculate gradient of gradient if using first order approximation

            for k, weight in enumerate(self.classifier.parameters()):
                update_value = (self.train_lr * grad[k])
                self._update_weight(weight, update_value)

    def _get_list_of_delta_params(self, support_embeddings, support_data_labels):
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

    def forward(self, x):
        scores = self.classifier.forward(x)
        return scores

    def set_forward(self, x, is_feature=False, train_stage=False):
        """ 1. Get delta params from hypernetwork with support data.
        2. Update target- network weights.
        3. Forward with query data.
        4. Return scores"""

        assert is_feature == False, 'MAML do not support fixed feature'

        x = x.cuda()
        x_var = Variable(x)
        support_embeddings = x_var[:, :self.n_support, 0, 0, :].contiguous().view(self.n_way * self.n_support,
                                                                            *x.size()[2:])  # support data
        support_embeddings = support_embeddings.reshape(self.n_way, -1)
        query_data = x_var[:, self.n_support:, :, :, :].contiguous().view(self.n_way * self.n_query,
                                                                          *x.size()[2:])  # query data
        query_data = query_data.reshape(query_data.shape[0], -1)
        support_data_labels = self.get_support_data_labels()


        delta_params_list = self._get_list_of_delta_params(support_embeddings, support_data_labels)

        self._update_network_weights(delta_params_list, support_embeddings, support_data_labels, train_stage)

        if self.hm_set_forward_with_adaptation and not train_stage:
            scores = self.forward(support_embeddings)
            return scores, None
        else:
            if self.hm_support_set_loss and train_stage:
                query_data = torch.cat((support_embeddings, query_data))

            scores = self.forward(query_data)

            # sum of delta params for regularization
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
            optimizer.zero_grad()
            if i % print_freq == 0:
                print('Epoch {:d}/{:d} | Batch {:d}/{:d} | Loss {:f}'.format(self.epoch, self.stop_epoch, i,
                                                                             len(train_loader),
                                                                             avg_loss / float(i + 1)))

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)

        metrics = {"accuracy/train": acc_mean}

        if self.hm_save_delta_params and len(self.delta_list) > 0:
            delta_params = {"epoch": self.epoch, "delta_list": self.delta_list}
            metrics['delta_params'] = delta_params

        if self.alpha < 1:
            self.alpha += self.hn_alpha_step

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

    def get_logits(self, x):
        self.n_query = x.size(1) - self.n_support
        logits, _ = self.set_forward(x)
        return logits

    def correct(self, x):
        scores, _ = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)


class Classifier_FW(nn.Module):
    def __init__(self, input_size=768, num_layers=0, layer_size=5, num_classes=100):
        super().__init__()
        layers = self._generate_layers(input_size, num_layers, layer_size, num_classes)
        self.net = nn.Sequential(*layers)

    def _generate_layers(self, input_size, num_hidden_layers, layer_size, num_classes):
        if num_hidden_layers == 0:
            return [Linear_fw(input_size, num_classes)]

        layers = [Linear_fw(input_size, layer_size), nn.ReLU()]
        for _ in range(num_hidden_layers-1):
            layers.append(Linear_fw(layer_size, layer_size))
            layers.append(nn.ReLU())

        layers.append(Linear_fw(layer_size, num_classes))
        return layers
    
    def forward(self, x):
        return self.net(x)
