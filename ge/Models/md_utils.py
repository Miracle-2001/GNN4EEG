import numpy as np
import pandas as pd
import torch
import os
import math
from torch.autograd import Function
from torch_geometric.nn import global_add_pool, SGConv
from torch_scatter import scatter_add
import sklearn.metrics as metrics
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

class NormalDataset(Dataset):
    def __init__(self, data, label, device):
        super(NormalDataset, self).__init__()
        self.data = data
        self.label = label
        self.device = device

    def __getitem__(self, ind):
        X = np.array(self.data[ind])  # (seq_length, feat_dim) array
        Y = np.array(self.label[ind]) # (seq_length, feat_dim) arrays
        return torch.from_numpy(X).to(self.device, dtype=torch.float32), torch.from_numpy(Y).to(self.device, dtype=torch.int32)

    def __len__(self,):
        return self.data.shape[0]

class DATDataSet(Dataset):
    def __init__(self, data_source,data_target, label, device):
        super(DATDataSet, self).__init__()
        self.data_source = data_source
        self.data_target=data_target
        self.label = label
        self.device = device

    def __getitem__(self, ind):
        Xs = np.array(self.data_source[ind])  # (seq_length, feat_dim) array
        Xt= np.array(self.data_target[ind])  # (seq_length, feat_dim) array
        Y = np.array(self.label[ind]) # (seq_length, feat_dim) arrays
        return torch.from_numpy(Xs).to(self.device, dtype=torch.float32), torch.from_numpy(Xt).to(self.device, dtype=torch.float32),torch.from_numpy(Y).to(self.device, dtype=torch.int32)

    def __len__(self,):
        return self.data_source.shape[0]

class HetDataset(Dataset):
    def __init__(self, data_time, data_freq, data_mat, label, device):
        super(HetDataset, self).__init__()
        self.data_time = data_time
        self.data_freq = data_freq
        self.data_mat = data_mat
        self.label = label
        self.device = device

    def __getitem__(self, ind):
        X_time = np.array(self.data_time[ind])  # (num_nodes, num_time) array
        X_freq = np.array(self.data_freq[ind])  # (num_nodes, num_freq) array
        A = np.array(self.data_mat[ind])  # (num_nodes, num_nodes) array
        Y = np.array(self.label[ind])  # (seq_length, feat_dim) arrays
        return torch.from_numpy(X_time).to(self.device, dtype=torch.float32), torch.from_numpy(X_freq).to(self.device, dtype=torch.float32), torch.from_numpy(A).to(self.device, dtype=torch.float32), torch.from_numpy(Y).to(self.device, dtype=torch.int32)

    def __len__(self,):
        return self.data_time.shape[0]

def distribution_label(self, Y):
    distribution_labels = np.zeros((len(Y), self.num_classes))
    for i in range(len(Y)):
        distribution_labels[i, :] = self.dl_mat[Y[i]]
    return distribution_labels

def get_edge_weight(args):
    total_part = '''Fp1 Fp2 Fz F3 F4 F7 F8 FC1 FC2 FC5 FC6 Cz C3 C4 T7 T8 CP1 CP2 CP5 CP6 Pz P3 P4 P7 P8 PO3 PO4 Oz O1 O2'''.split()
    edge_pos_value = np.load('./Models/pos.npy')*100  # multiply 100
    edge_weight = np.zeros([len(total_part), len(total_part)])
    # edge_pos_value = [edge_pos[key] for key in total_part]
    delta = 2  # Choosing delta=2 makes the proportion of non_negligible connections exactly 20%
    edge_index = [[], []]
    if args.model=='DGCNN' or args.model=='SparseDGCNN':
        for i in range(len(total_part)):
            for j in range(len(total_part)):
                edge_index[0].append(i)
                edge_index[1].append(j)
                if i == j:
                    edge_weight[i][j] = 1
                else:
                    edge_weight[i][j] = np.sum(
                        [(edge_pos_value[i][k] - edge_pos_value[j][k])**2 for k in range(2)])
                    if delta/edge_weight[i][j] > 1:
                        edge_weight[i][j] = math.exp(-edge_weight[i][j]/2)
                    else:
                        edge_weight[i][j] = 0
    elif args.model=='RGNN':
        for i in range(len(total_part)):
            for j in range(len(total_part)):
                edge_index[0].append(i)
                edge_index[1].append(j)
                if i == j:
                    edge_weight[i][j] = 1
                else:
                    edge_weight[i][j] = np.sum([(edge_pos_value[i][k]
                                                - edge_pos_value[j][k])**2
                                                for k in range(2)])
                    edge_weight[i][j] = min(1, delta/edge_weight[i][j])
        global_connections = [
            ['Fp1', 'Fp2'],
            ['F3', 'F4'],
            ['FC5', 'FC6'],
            ['C3', 'C4'],
            ['CP5', 'CP6'],
            ['P3', 'P4'],
            ['PO3', 'PO4'],
            ['O1', 'O2']
        ]
        for item in global_connections:
            i = total_part.index(item[0])
            j = total_part.index(item[1])
            edge_weight[i][j] -= 1
            edge_weight[j][i] -= 1
    return edge_index, edge_weight

def get_edge_weight_from_electrode(model_name,edge_pos_value,global_connections=None):
    num_nodes=len(edge_pos_value)
    edge_weight = np.zeros([num_nodes, num_nodes])
    # edge_pos_value = [edge_pos[key] for key in total_part]
    delta = 2  # Choosing delta=2 makes the proportion of non_negligible connections exactly 20%
    edge_index = [[], []]
    if model_name=='DGCNN' or model_name=='SparseDGCNN':
        for i in range(num_nodes):
            for j in range(num_nodes):
                edge_index[0].append(i)
                edge_index[1].append(j)
                if i == j:
                    edge_weight[i][j] = 1
                else:
                    edge_weight[i][j] = np.sum(
                        [(edge_pos_value[i][k] - edge_pos_value[j][k])**2 for k in range(2)])
                    if delta/edge_weight[i][j] > 1:
                        edge_weight[i][j] = math.exp(-edge_weight[i][j]/2)
                    else:
                        edge_weight[i][j] = 0
    elif model_name=='RGNN':
        for i in range(num_nodes):
            for j in range(num_nodes):
                edge_index[0].append(i)
                edge_index[1].append(j)
                if i == j:
                    edge_weight[i][j] = 1
                else:
                    edge_weight[i][j] = np.sum([(edge_pos_value[i][k]
                                                - edge_pos_value[j][k])**2
                                                for k in range(2)])
                    edge_weight[i][j] = min(1, delta/edge_weight[i][j])
        if global_connections is not None:
            for item in global_connections:
                i = item[0]
                j = item[1]
                edge_weight[i][j] -= 1
                edge_weight[j][i] -= 1
    return edge_index, edge_weight


def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes


def add_remaining_self_loops(edge_index,
                             edge_weight=None,
                             fill_value=1,
                             num_nodes=None):
    # reposition the diagonal values to the end
    '''
    edge_weight.shape : (num_nodes*num_nodes*batch_size,)
    ()
    '''
    # actually return num_nodes
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    mask = row != col

    inv_mask = ~mask  # diagonal positions
    # print("inv_mask", inv_mask)

    loop_weight = torch.full(
        (num_nodes,),
        fill_value,
        dtype=None if edge_weight is None else edge_weight.dtype,
        device=edge_index.device)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        remaining_edge_weight = edge_weight[inv_mask]

        if remaining_edge_weight.numel() > 0:
            loop_weight[row[inv_mask]] = remaining_edge_weight

        edge_weight = torch.cat([edge_weight[mask], loop_weight], dim=0)
    loop_index = torch.arange(0, num_nodes, dtype=row.dtype, device=row.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)

    return edge_index, edge_weight


class NewSGConv(SGConv):
    def __init__(self, num_features, num_classes, K=1, cached=False,
                 bias=True):
        super(NewSGConv, self).__init__(num_features,
                                        num_classes, K=K, cached=cached, bias=bias)
        torch.nn.init.xavier_normal_(self.lin.weight)
    # allow negative edge weights
    @staticmethod
    # Note: here,num_nodes=self.num_nodes*batch_size
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        row, col = edge_index

        deg = scatter_add(torch.abs(edge_weight), row,
                          dim=0, dim_size=num_nodes)  # calculate degreematrix, i.e, D(stretched) in the paper.
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # calculate normalized adjacency matrix, i.e, S(stretched) in the paper.
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        if not self.cached or self.cached_result is None:
            edge_index, norm = NewSGConv.norm(
                edge_index, x.size(0), edge_weight, dtype=x.dtype,)

            for k in range(self.K):
                x = self.propagate(edge_index, x=x, norm=norm)
            self.cached_result = x

        return self.lin(self.cached_result)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

def get_het_adjacency_matrix(data, threshold=0.7, bins=10):
    '''
        Calculate mutual information among channels and get adjacent matrix
        :param sample: temporal domain data
        :param threshold: threshold
        :param bins:
        :return:
    '''
    # Number of channels
    num_of_v = data.shape[1]
    num_of_samples=data.shape[0]
    # Tensor to numpy array
    # ziyi
    # sample = sample.numpy()
    adj_matrix = np.zeros((num_of_samples,num_of_v, num_of_v))

    for t in range(num_of_samples):
        for i in range(num_of_v):
            for j in range(i, num_of_v):
                y = np.histogram2d(data[t][i], data[t][j], bins=bins)[0]
                adj_matrix[t][i][j] = metrics.mutual_info_score(None, None, contingency=y)
                adj_matrix[t][j][i] = adj_matrix[t][i][j]
    # Numpy array to tensor
    adj_matrix = torch.from_numpy(adj_matrix)
    # Replace inf and nan
    adj_matrix = torch.where(t.isinf(adj_matrix), t.full_like(adj_matrix, 0), adj_matrix)
    adj_matrix = torch.where(t.isnan(adj_matrix), t.full_like(adj_matrix, 0), adj_matrix)
    # Threshold
    adj_matrix[adj_matrix < threshold] = 0
    return adj_matrix

def l1_reg_loss(model,only=None,exclude=None): 
    """
    Returns the squared L1 norm of output layer of given model
    Note: 
        If para_name is not None, then this function will only calculate the squared L1 norm of that parameter.
        Otherwise, all the parameters will be caculated.
    """
    total_loss = 0
    if only is None and exclude is None:
        for name, param in model.named_parameters():
            total_loss += torch.sum(torch.abs(param))
    elif only is not None:
        for name, param in model.named_parameters():
            if name in only:
                total_loss += torch.sum(torch.abs(param))
    elif exclude is not None:
        for name, param in model.named_parameters():
            if name not in exclude:
                total_loss += torch.sum(torch.abs(param))
    return total_loss

def l2_reg_loss(model,only=None,exclude=None): 
    """
    Returns the squared L2 norm of output layer of given model
    Note: 
        If para_name is not None, then this function will only calculate the squared L2 norm of that parameter.
        Otherwise, all the parameters will be caculated.
    """
    total_loss = 0
    if only is None and exclude is None:
        for name, param in model.named_parameters():
            total_loss += torch.sum(torch.square(param))
    elif only is not None:
        for name, param in model.named_parameters():
            if name in only:
                total_loss += torch.sum(torch.square(param))
    elif exclude is not None:
        for name, param in model.named_parameters():
            if name not in exclude:
                total_loss += torch.sum(torch.square(param))
    return total_loss

def calc_accuracy(predict, label):
    if type(predict) is np.ndarray:
        num_of_sample = predict.shape[0]
    elif type(predict) is list:
        num_of_sample = len(predict)
        predict = np.array(predict)
        label = np.array(label)
    if num_of_sample == 0:
        return 0
    return np.sum(predict == label)/num_of_sample


def one_heat_code(label, K):
    one_heat_codes = np.zeros((len(label), K))
    for i, lb in enumerate(label):
        one_heat_codes[i][lb] = 1
    return one_heat_codes.astype(np.float32)

def draw(last_epoch, train_acc, eval_acc, train_loss, eval_loss, path, name):
    
    x = range(0, last_epoch+1)
    fig = plt.figure(figsize=(20, 8), dpi=80)
    ax1 = fig.add_subplot(1, 2, 1)
    plt.plot(x, train_acc, color='blue', label='train_acc')
    plt.plot(x, eval_acc, color='orange', label='eval_acc')
    plt.legend(loc="upper right")
    plt.title("accuracy", color='black')
    ax2 = fig.add_subplot(1, 2, 2)

    plt.plot(x, train_loss, color='blue', label='train_loss')
    plt.plot(x, eval_loss, color='orange', label='eval_loss')
    plt.legend(loc="upper right")
    plt.title("loss", color='black')
    plt.show()

    # save fig
    filepath = os.path.join(path, (name.split('.')[0])+'png')
    fig.savefig(filepath)