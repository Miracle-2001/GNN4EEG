import torch
import torch.nn as nn
from .md_utils import *
from .trainer import Trainer

class MLP(torch.nn.Module):
    def __init__(self,num_hiddens,num_nodes,num_features,num_layers,num_classes,device,dropout) :
        super(MLP,self).__init__()
        self.num_hiddens=num_hiddens
        self.num_nodes=num_nodes
        self.num_features=num_features
        self.num_layers=num_layers
        self.num_classes=num_classes
        self.dropout=dropout
        self.device=device
        
        net = nn.Sequential()
        net.add_module('flatten',nn.Flatten())
        for i in range(num_layers):
            if i==0:
                net.add_module(f'linear{i+1}',nn.Linear(num_nodes*num_features,num_hiddens))    
            else:
                net.add_module(f'linear{i+1}',nn.Linear(num_hiddens,num_hiddens))
            net.add_module(f'ReLu{i+1}',nn.ReLU())
            net.add_module(f'Dropout{i+1}',nn.Dropout(dropout))
        net.add_module(f'linear{num_layers+1}',nn.Linear(num_hiddens,num_classes))

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.01)

        net.apply(init_weights)
        self.net=net

    def forward(self,X):
        return self.net(X)

class MLPTrainer(Trainer):
    def __init__(self, num_nodes, num_hiddens=400, num_classes=2, num_layers=1, batch_size=256, num_epoch=50, lr=0.005, l1_reg=0, l2_reg=0, dropout=0.5,
                 optimizer='Adam', early_stop=20, device=torch.device('cpu')
                 ):
        super().__init__(MLP, num_nodes, num_hiddens, num_classes, batch_size, num_epoch,
                         lr, l1_reg, l2_reg, dropout, early_stop, optimizer, device,
                         extension={'num_layers': num_layers, 'loss_module': nn.CrossEntropyLoss()})