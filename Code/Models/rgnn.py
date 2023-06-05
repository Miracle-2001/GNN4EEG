import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mne
import copy
import json
import math
import os
from torch.autograd import Function
from torch_geometric.nn import global_add_pool, SGConv
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from .md_utils import *
from .trainer import Trainer


def get_dl_mat():
    # build dl_mat
    e=0.05
    g=0.03
    eps=1e-7
    dl_mat=[
        [1-3*e-2*g,g,g,e,e,e/3,e/3,e/3,eps],
        [g,1-3*e-2*g,g,e,e,e/3,e/3,e/3,eps],
        [g,g,1-3*e-2*g,e,e,e/3,e/3,e/3,eps],
        [e/3,e/3,e/3,1-3*e,e,eps,eps,eps,e],
        [e/3,e/3,e/3,e,1-4*e,e/3,e/3,e/3,e],
        [e/3,e/3,e/3,eps,e,1-3*e-2*g,g,g,e],
        [e/3,e/3,e/3,eps,e,g,1-3*e-2*g,g,e],
        [e/3,e/3,e/3,eps,e,g,g,1-3*e-2*g,e],
        [eps,eps,eps,e,e,e/3,e/3,e/3,1-3*e]
    ]
    return dl_mat

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class RGNN(torch.nn.Module):
    def __init__(self, device, num_nodes,  edge_weight, edge_index, num_features, num_hiddens, num_classes, num_layers, dropout=0.5, domain_adaptation=False):
        """
            num_nodes: number of nodes in the graph
            learn_edge_weight: if True, the edge_weight is learnable
            edge_weight: initial edge matrix
            num_features: feature dim for each node/channel
            num_hiddens: a tuple of hidden dimensions
            num_classes: number of emotion classes
            num_layers: number of layers
            dropout: dropout rate in final linear layer
            domain_adaptation: RevGrad
        """
        learn_edge_weight=True
        super(RGNN, self).__init__()
        self.device = device
        self.domain_adaptation = domain_adaptation
        self.num_nodes = num_nodes
        self.xs, self.ys = torch.tril_indices(
            self.num_nodes, self.num_nodes, offset=0)
        self.edge_index = torch.tensor(edge_index)
        edge_weight = edge_weight.reshape(self.num_nodes, self.num_nodes)[
            self.xs, self.ys]  # strict lower triangular values
        self.edge_weight = nn.Parameter(
            torch.Tensor(edge_weight).float(), requires_grad=learn_edge_weight)
        self.dropout = dropout
        self.conv1 = NewSGConv(num_features=num_features,
                               num_classes=num_hiddens, K=num_layers)
        self.fc = nn.Linear(num_hiddens, num_classes)

        # xavier init
        torch.nn.init.xavier_normal_(self.fc.weight)


        if self.domain_adaptation == True:
            self.domain_classifier = nn.Linear(num_hiddens, 2)
            torch.nn.init.xavier_normal_(self.domain_classifier.weight)

    def append(self, edge_index, batch_size):  # stretch and repeat and rename
        edge_index_all = torch.LongTensor(2, edge_index.shape[1] * batch_size)
        data_batch = torch.LongTensor(self.num_nodes * batch_size)
        for i in range((batch_size)):
            edge_index_all[:, i*edge_index.shape[1]                           :(i+1)*edge_index.shape[1]] = edge_index + i * self.num_nodes
            data_batch[i*self.num_nodes:(i+1)*self.num_nodes] = i
        return edge_index_all.to(self.device), data_batch.to(self.device)

    def forward(self, X, alpha=0,need_pred=True,need_dat=False):
        batch_size = len(X)
        # print("batch_size ",batch_size)
        x = X.reshape(-1, X.shape[-1])
        edge_index, data_batch = self.append(self.edge_index, batch_size)
        edge_weight = torch.zeros(
            (self.num_nodes, self.num_nodes), device=edge_index.device)
        edge_weight[self.xs.to(edge_weight.device), self.ys.to(
            edge_weight.device)] = self.edge_weight
        edge_weight = edge_weight + \
            edge_weight.transpose(1, 0) - torch.diag(edge_weight.diagonal())
        edge_weight = edge_weight.reshape(-1).repeat(batch_size)
        # edge_index: (2,self.num_nodes*self.num_nodes*batch_size)  edge_weight: (self.num_nodes*self.num_nodes*batch_size,)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        # domain classification
        domain_output = None
        if need_dat == True:
            reverse_x = ReverseLayerF.apply(x, alpha)
            domain_output = self.domain_classifier(reverse_x)
        if need_pred == True:
            x = global_add_pool(x, data_batch, size=batch_size)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.fc(x)

        # x.shape->(batch_size,num_classes)
        # domain_output.shape->(batch_size*num_nodes,2)

        # NO softmax!!!
        # x=torch.softmax(x,dim=-1)
        # if domain_output is not None:
        #     domain_output=torch.softmax(domain_output,dim=-1)
        if domain_output is not None:
            return x, domain_output
        else:
            return x


class RGNNTrainer(Trainer):
    def __init__(self, edge_index=None, edge_weight=None, num_classes=2, device=torch.device('cpu'),
                 domain_adaptation=False, distribution_learning=False, optimizer='Adam',
                 num_hiddens=400, num_layers=2, dropout=0.5,early_stop=20,
                 batch_size=8, lr=5e-3, l1_reg=0.0, l2_reg=0.0,num_epoch=80):  # ,model_save_path='./rgnn_weights2'):

        if edge_index is None or edge_weight is None:
            raise Exception("No edge_index and edge_weight")
        num_nodes = edge_weight.shape[0]
        dl_mat = get_dl_mat() # if distribution_learning else None
        # print("dl_mat", dl_mat)
        loss_module = nn.KLDivLoss(reduction='batchmean') if distribution_learning else nn.CrossEntropyLoss()
        
        super().__init__(RGNN, num_nodes, num_hiddens, num_classes, batch_size, num_epoch,
                            lr, l1_reg, l2_reg, dropout, early_stop, optimizer, device,
                            extension={'num_layers': num_layers, 
                                        'loss_module': loss_module,
                                        'edge_weight':edge_weight,
                                        'edge_index':edge_index,
                                        'domain_adaptation':domain_adaptation,
                                        'distribution_learning':distribution_learning,
                                        'dl_mat':dl_mat
                                        })
