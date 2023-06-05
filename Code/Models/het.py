import torch
import torch.nn as nn
import torch.nn.functional as F
from .trainer import Trainer
from .md_utils import *

class permute(nn.Module):
    def __init__(self):
        super(permute, self).__init__()

    def forward(self, x):
        return x.permute(0, 2, 1)


class STDCN_with_GRU(nn.Module):
    def __init__(self, num_f, in_dim_with_c, out_dim_with_c, num_channel,device):
        super(STDCN_with_GRU, self).__init__()
        # Number of channels before entering GRU
        self.num_f = num_f
        self.in_dim = in_dim_with_c
        self.out_dim = out_dim_with_c
        self.device=device
        # Number of graph channels
        self.num_channels = num_channel
        self.per1 = permute()
        self.per2 = permute()
        self.gru = nn.GRU(input_size=self.in_dim, hidden_size=self.out_dim, num_layers=1,
                          batch_first=True, bidirectional=True)
        self.BN = torch.nn.BatchNorm1d(self.num_f)
        # GCN weight
        self.weight = nn.Parameter(torch.Tensor(1, 1))
        self.reset_parameters()

    def norm(self, H, add=True):
        # H shape（b,n,n）
        if add == False:
            H = H * ((torch.eye(H.shape[1]) ==
                     0).type(torch.FloatTensor)).unsqueeze(0)
        else:
            H = H * ((torch.eye(H.shape[1]) == 0).type(torch.FloatTensor)).unsqueeze(0).to(self.device) + torch.eye(
                H.shape[1]).type(torch.FloatTensor).unsqueeze(0).to(self.device)
        deg = torch.sum(H, dim=-1)
        # deg shape (b,n)
        deg_inv = deg.pow(-1 / 2)
        deg_inv = deg_inv.view((deg_inv.shape[0], deg_inv.shape[1], 1)) * torch.eye(H.shape[1]).type(
            torch.FloatTensor).unsqueeze(0).to(self.device)
        # deg_inv shape (b,n,n)
        H = torch.bmm(deg_inv, H)
        H = torch.bmm(H, deg_inv)
        H = torch.tensor(H, dtype=torch.float32).to(self.device)
        return H

    def reset_parameters(self):
        nn.init.constant_(self.weight, 10)

    def gcn_conv(self, X, H):
        # X shape (B,N,F)
        # H shape (B,N,N)

        for i in range(X.shape[-1]):  # 很迷的操作，为什么要一层一层来乘
            if i == 0:
                # X.unsqueeze(-2) shape:(B,N,1,F)
                out = torch.bmm(H, X.unsqueeze(-2)[:, :, :, i])
                # weight是一个1x1的tensor，所以相当于乘了一个系数……
                out = torch.matmul(out, self.weight)  # out size:(B,N,1)
            else:
                out = torch.cat((out, torch.matmul(torch.bmm(H, X.unsqueeze(-2)[:, :, :, i]), self.weight)),
                                dim=-1)
        return out  # (B,N,F)

    def forward(self, X, H):
        # Spatial
        # H shape (b,n,n)
        # x shape (b,n,f)
        X_ = F.leaky_relu(self.gcn_conv(X, self.norm(H)))
        # X_ shape (b,n,f)
        # temporal
        X_ = self.per1(X_)  # (b,f,n)
        X_, hn = self.gru(X_)  # (b,f,2*self.out_dim)
        X_ = self.BN(X_)
        X_ = self.per2(X_)  # (b,2*self.out_dim,f)
        # x_ shape (b,NEWn,out_channel)
        return X_


class Het(nn.Module):
    def __init__(self, device, num_nodes, num_time, num_freq, num_classes,num_hiddens=64, dropout=0.2):
        super(Het, self).__init__()
        # Load model configuration
        self.num_node = num_nodes
        self.num_band = num_freq
        self.final_out_node = num_nodes
        self.dropout = dropout
        self.sample_feature_num = num_time  # time step or number of band
        self.STDCN1_F = STDCN_with_GRU(
            self.num_band, self.num_node, self.final_out_node, 1, device)
        # Note that the feature number is multiplied by two after the bi-GRU comes out
        self.STDCN1_T = STDCN_with_GRU(
            self.sample_feature_num, self.num_node, self.final_out_node, 1, device)
        self.flatten = nn.Flatten()
        self.flatten = nn.Flatten()
        self.linF = nn.Linear(self.num_band, self.sample_feature_num // 2)
        self.linT = nn.Linear(self.sample_feature_num,
                              self.sample_feature_num // 2)
        self.all = self.sample_feature_num * self.final_out_node * 2
        self.Dropout = nn.Dropout(dropout)
        self.lin1 = nn.Linear(self.all, num_hiddens)
        self.act1 = nn.LeakyReLU()
        self.lin2 = nn.Linear(num_hiddens, num_classes)

    def forward(self, x_T,x_F,A):
        # x_F:(b,n,4) x_T:(b,n,sample_feature_num) A:(b,n,n)
        x_F = self.Dropout(x_F)
        x_T = self.Dropout(x_T)
        A = A.view((-1, self.num_node, self.num_node))
        x_F = x_F.view((-1, self.num_node, x_F.shape[-1]))
        x_T = x_T.view((-1, self.num_node, x_T.shape[-1]))
        # GCN and GRU
        out_F = self.STDCN1_F(x_F, A)  # (b,2*final_out_node,f)
        out_F = self.linF(out_F)  # (b,2*final_out_node,sample_feature_num/2)
        out_F = self.flatten(out_F)
        out_T = self.STDCN1_T(x_T, A)  # (b,2*final_out_node,t)
        out_T = self.linT(out_T)  # (b,2*final_out_node,sample_feature_num/2)
        out_T = self.flatten(out_T)
        # Fusion and classification
        out = torch.cat([out_F, out_T], dim=-1)  # (b,all)
        out = self.lin1(out)  # (b,64)
        out = self.act1(out)
        out = self.lin2(out)  # (b,2)
        return out

class HetTrainer(Trainer):
    def __init__(self, num_nodes, num_classes=2, device=torch.device('cpu'),
                 optimizer='Adam',
                 num_hiddens=64, dropout=0.2,early_stop=20,
                 batch_size=256, lr=5e-3, l1_reg=0.0, l2_reg=0.0, num_epoch=50):  # ,model_save_path='./rgnn_weights2'):
        
        super().__init__(Het, num_nodes, num_hiddens, num_classes, batch_size, num_epoch,
                         lr, l1_reg, l2_reg, dropout, early_stop, optimizer, device,
                         extension={'loss_module': nn.CrossEntropyLoss()
                                    })
        
