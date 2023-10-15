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
from methods.hypernets.hypermaml import HyperMAML
from methods.hypernets.binary_maml_utils import Binarizer, SoftBinarizer

from hypnettorch.hnets.chunked_mlp_hnet import HMLP, ChunkedHMLP


class BinaryHyperMAML(HyperMAML):
    def __init__(self, model_func, n_way, n_support, n_query, params=None, approx=False):
        super(BinaryHyperMAML, self).__init__(model_func, n_way, n_support, n_query, params=params)
        self.bm_method = params.bm_method
        self.bm_activation = params.bm_activation
        self.bm_layer_size = params.bm_layer_size
        self.bm_num_layers = params.bm_num_layers
        self.bm_decrease_epochs = params.bm_decrease_epochs
        self.bm_gumbel_discretize = params.bm_gumbel_discretize
        self.bm_mask_size = params.bm_mask_size
        self.bm_chunk_emb_size = params.bm_chunk_emb_size
        self.bm_chunk_size = params.bm_chunk_size

        #gumbel_softmax temp
        self.start_temp = 1.0
        self.end_temp = 0.2
        self.temp_diff = self.start_temp - self.end_temp

        hypernet_layers = [self.bm_layer_size for _ in range(self.bm_num_layers)]

        if self.bm_method == "two_encoders":
            self.feature_query = model_func()
        elif self.bm_method == "one_encoder":
            self.feature_query = self.feature

        backbone_shapes = [list(layer.shape) for layer in self.feature_query.parameters()]
        classifier_shapes = [list(layer.shape) for layer in self.classifier.parameters()]
        shapes =  backbone_shapes + classifier_shapes

        if self.bm_activation == "gumbel_softmax":
            shapes = [dim+[2] for dim in shapes]

        self.hypernet = ChunkedHMLP(shapes, uncond_in_size=self.embedding_size, cond_in_size=0, chunk_emb_size=self.bm_chunk_emb_size,
                layers=hypernet_layers, chunk_size=self.bm_chunk_size, num_cond_embs=1)

    def get_hn_delta_params(self, support_embeddings):
        if self.bm_method == "one_encoder":
            support_embeddings = support_embeddings.detach()

        support_embeddings_resh = support_embeddings.reshape(1, -1)
        delta_params = self.hypernet(support_embeddings_resh)

        if self.bm_activation == "gumbel_softmax":
            if self.epoch >= self.bm_decrease_epochs:
                temp = self.end_temp
            else:
                temp = self.start_temp - self.temp_diff*(self.epoch/self.bm_decrease_epochs)
            for i in range(len(delta_params)):
                delta_params[i] = F.gumbel_softmax(delta_params[i], tau=temp, hard=self.bm_gumbel_discretize, dim=-1)[...,-1]
        elif self.bm_activation == "sigmoid":
            for i in range(len(delta_params)):
                delta_params[i] = torch.sigmoid(delta_params[i])
    
            params_flat = [param.clone().detach().reshape(-1) for param in delta_params]
            concat = torch.cat(params_flat)

            k_val = torch.quantile(concat, self.bm_mask_size).item()

            for i in range(len(delta_params)):
                delta_params[i] = Binarizer.apply(delta_params[i], k_val)

        elif self.bm_activation == "tanh":
            for i in range(len(delta_params)):
                delta_params[i] = torch.tanh(delta_params[i])
    
            params_flat = [param.clone().detach().reshape(-1) for param in delta_params]
            concat = torch.cat(params_flat)

            k_val = torch.quantile(concat, self.bm_mask_size).item()

            for i in range(len(delta_params)):
                delta_params[i] = SoftBinarizer.apply(delta_params[i], k_val)

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
        out = self.extract_query_features(x)

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
            scores = self.forward(query_data)
            
            # sum of delta params for regularization
            if self.hm_lambda != 0:
                total_delta_sum = sum([delta_params.pow(2.0).sum() for delta_params in delta_params_list])

                return scores, total_delta_sum
            else:
                return scores, None

    def extract_query_features(self, x):
        if self.bm_method == "two_encoders":
            return self.feature_query(x)
        return self.feature(x)