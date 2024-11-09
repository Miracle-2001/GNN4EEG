import torch
import torch.nn as nn
import torch.nn.functional as F
from .md_utils import *
from .trainer import Trainer


def normalize_A(A: torch.Tensor, symmetry: bool=False) -> torch.Tensor:
    A = F.relu(A)
    if symmetry:
        A = A + torch.transpose(A, 0, 1)
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    else:
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    return L

class DGCNN_Model(torch.nn.Module):
    def __init__(self, device, num_nodes,  edge_weight, edge_index, num_features, num_hiddens, num_classes, num_layers,learn_edge_weight=True,dropout=0.5):
        """
            num_nodes: number of nodes in the graph
            learn_edge_weight: if True, the edge_weight is learnable
            edge_weight: initial edge matrix
            num_features: feature dim for each node/channel
            num_hiddens: a tuple of hidden dimensions
            num_classes: number of emotion classes
            num_layers: number of layers
            dropout: dropout rate in final linear layer
        """
        super(DGCNN_Model, self).__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.xs, self.ys = torch.tril_indices(
            self.num_nodes, self.num_nodes, offset=0)
        self.edge_index = torch.tensor(edge_index)
        edge_weight = edge_weight.reshape(self.num_nodes, self.num_nodes)[
            self.xs, self.ys]  # strict lower triangular values
        self.edge_weight = nn.Parameter(
            torch.Tensor(edge_weight).float(), requires_grad=True)
        self.dropout = dropout
        self.BN1 = nn.BatchNorm1d(num_features)
        self.conv1 = NewSGConv(num_features=num_features,
                               num_classes=num_hiddens, K=num_layers)
        # self.conv2 = torch.nn.Conv1d(self.num_nodes, 1, 1)
        self.fc1 = nn.Linear(num_nodes* num_hiddens , 64)
        self.fc2 = nn.Linear(64, num_classes)

    def append(self, edge_index, batch_size):  # stretch and repeat and rename
        edge_index_all = torch.LongTensor(2, edge_index.shape[1] * batch_size)
        data_batch = torch.LongTensor(self.num_nodes * batch_size)
        for i in range((batch_size)):
            edge_index_all[:, i*edge_index.shape[1]:(i+1)*edge_index.shape[1]] = edge_index + i * self.num_nodes
            data_batch[i*self.num_nodes:(i+1)*self.num_nodes] = i
        return edge_index_all.to(self.device), data_batch.to(self.device)

    def forward(self, X):
        batch_size = len(X)
        x = X #.reshape(-1, X.shape[-1])
        edge_index, data_batch = self.append(self.edge_index, batch_size)
        edge_weight = torch.zeros(
            (self.num_nodes, self.num_nodes), device=edge_index.device)
        edge_weight[self.xs.to(edge_weight.device), self.ys.to(
            edge_weight.device)] = self.edge_weight
        edge_weight = edge_weight + \
            edge_weight.transpose(1, 0) - torch.diag(edge_weight.diagonal())
        edge_weight = normalize_A(edge_weight)
        edge_weight = edge_weight.reshape(-1).repeat(batch_size)
        
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        x = x.reshape(-1, x.shape[-1])
        x = self.conv1(x, edge_index, edge_weight)
        x = x.view((batch_size, -1))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

