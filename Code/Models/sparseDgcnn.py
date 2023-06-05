import torch
import torch.nn as nn
import torch.nn.functional as F
from .md_utils import *
from .trainer import Trainer

class SparseDGCNN(torch.nn.Module):
    def __init__(self, device, num_nodes,  edge_weight, edge_index, num_features, num_hiddens, num_classes, num_layers, learn_edge_weight=True,dropout=0.5):
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
        super(SparseDGCNN, self).__init__()
        self.device = device
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
        self.conv2 = torch.nn.Conv1d(self.num_nodes, 1, 1)
        self.fc = nn.Linear(num_hiddens, num_classes)

    def append(self, edge_index, batch_size):  # stretch and repeat and rename
        edge_index_all = torch.LongTensor(2, edge_index.shape[1] * batch_size)
        data_batch = torch.LongTensor(self.num_nodes * batch_size)
        for i in range((batch_size)):
            edge_index_all[:, i*edge_index.shape[1]:(i+1)*edge_index.shape[1]] = edge_index + i * self.num_nodes
            data_batch[i*self.num_nodes:(i+1)*self.num_nodes] = i
        return edge_index_all.to(self.device), data_batch.to(self.device)

    def forward(self, X):
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
        x = self.conv1(x, edge_index, edge_weight)
        x = x.view((batch_size, self.num_nodes, -1))
        x = self.conv2(x)
        x = F.relu(x.squeeze(1))
        x = self.fc(x)
        # NO softmax!!!
        return x

class SparseDGCNNTrainer(Trainer):
    def __init__(self, edge_index=None, edge_weight=None, num_classes=2, device=torch.device('cpu'),
                 optimizer='Adam', num_hiddens=400, num_layers=2, dropout=0.5,early_stop=20,
                 batch_size=256, lr=5e-3, l1_reg=0.0,l2_reg=0.0, num_epoch=50): #l1_reg: i.e. λ in the paper   lr: i.e. τ in the paper

        if edge_index is None or edge_weight is None:
            raise Exception("No edge_index and edge_weight")
        num_nodes = edge_weight.shape[0]


        super().__init__(SparseDGCNN, num_nodes, num_hiddens, num_classes, batch_size, num_epoch,
                         lr, l1_reg,l2_reg, dropout, early_stop, optimizer, device,
                         extension={'num_layers': num_layers, 
                                    'loss_module': nn.CrossEntropyLoss(),
                                    'edge_weight':edge_weight,
                                    'edge_index':edge_index
                                    })
        