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
        self.bm_backbone_weights = params.bm_backbone_weights
        self.bm_method = params.bm_method
        self.bm_activation = params.bm_activation
        self.bm_layer_size = params.bm_layer_size
        self.bm_num_layers = params.bm_num_layers
        self.bm_decrease_epochs = params.bm_decrease_epochs
        self.bm_gumbel_discretize = params.bm_gumbel_discretize
        self.bm_freeze_target_network = params.bm_freeze_target_network
        self.bm_fixed_size_mask = params.bm_fixed_size_mask
        self.bm_mask_size = params.bm_mask_size
        self.bm_chunk_emb_size = params.bm_chunk_emb_size
        self.bm_chunk_size = params.bm_chunk_size

        #gumbel_softmax temp
        self.start_temp = 1.0
        self.end_temp = 0.2
        self.temp_diff = self.start_temp - self.end_temp

        hypernet_layers = [self.bm_layer_size for _ in range(self.bm_num_layers)]


        self.feature_query = model_func()

        shapes = []
        
        if self.bm_backbone_weights:
            backbone_shapes = [list(layer.shape) for layer in self.feature_query.parameters()]
            classifier_shapes = [list(layer.shape) for layer in self.classifier.parameters()]
            shapes =  backbone_shapes + classifier_shapes
        else:
            shapes = [list(layer.shape) for layer in self.classifier.parameters()]

        if self.bm_activation == "gumbel_softmax":
            shapes = [dim+[2] for dim in shapes]

        input_size = 70*5
        # input_size = 1606*5

        if self.bm_freeze_target_network:
            for param in self.feature_query.parameters():
                param.requires_grad = False

            for param in self.classifier.parameters():
                param.requires_grad = False

        self.hypernet = ChunkedHMLP(shapes, uncond_in_size=self.embedding_size, cond_in_size=0, chunk_emb_size=self.bm_chunk_emb_size,
                layers=hypernet_layers, chunk_size=self.bm_chunk_size, num_cond_embs=1)


    def calculate_embedding_size(self):

        n_classes_in_embedding = 1 if self.hm_use_class_batch_input else self.n_way
        n_support_per_class = 1 if self.hn_sup_aggregation == 'mean' else self.n_support
        single_support_embedding_len = self.feat_dim + self.n_way + 1 if self.enhance_embeddings else self.feat_dim
        self.embedding_size = n_classes_in_embedding * n_support_per_class * single_support_embedding_len

    def get_hn_delta_params(self, support_embeddings):
        if self.hm_detach_before_hyper_net:
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
    
            if self.bm_fixed_size_mask:
                params_flat = [param.clone().detach().reshape(-1) for param in delta_params]
                concat = torch.cat(params_flat)

                k_val = torch.quantile(concat, self.bm_mask_size).item()

                for i in range(len(delta_params)):
                    delta_params[i] = Binarizer.apply(delta_params[i], k_val)
        elif self.bm_activation == "tanh":
            for i in range(len(delta_params)):
                delta_params[i] = torch.tanh(delta_params[i])
    
            if self.bm_fixed_size_mask:
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

    def forward(self, x):
        if self.bm_method=="two_encoders":
            out = self.feature_query.forward(x)
        else:
            out = self.feature.forward(x)

        if self.hm_detach_feature_net:
            out = out.detach()

        scores = self.classifier.forward(out)
        return scores
