import torch
import torch.nn as nn
import torch.nn.functional as F
from .Models.md_utils import *
from .Models.trainer import *
from .Models.dgcnn import DGCNN
from .Models.rgnn import RGNN
from .Models.het import Het
from .Models.sparseDgcnn import SparseDGCNN

class DGCNNTrainer(Trainer):
    def __init__(self, edge_index=None, edge_weight=None, num_classes=2, device=torch.device('cpu'),
                 optimizer='Adam', num_hiddens=400, num_layers=2, dropout=0.5,early_stop=20,
                 batch_size=256, lr=5e-3, l1_reg=0.0, l2_reg=0.0, num_epoch=50):

        if edge_index is None or edge_weight is None:
            raise Exception("No edge_index and edge_weight")
        num_nodes = edge_weight.shape[0]

        super().__init__(DGCNN, num_nodes, num_hiddens, num_classes, batch_size, num_epoch,
                         lr, l1_reg, l2_reg, dropout, early_stop, optimizer, device,
                         extension={'num_layers': num_layers, 
                                    'loss_module': nn.CrossEntropyLoss(),
                                    'edge_weight':edge_weight,
                                    'edge_index':edge_index
                                    })

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
    
        
class HetTrainer(Trainer):
    def __init__(self, num_nodes, num_classes=2, device=torch.device('cpu'),
                 optimizer='Adam',
                 num_hiddens=64, dropout=0.2,early_stop=20,
                 batch_size=256, lr=5e-3, l1_reg=0.0, l2_reg=0.0, num_epoch=50):  # ,model_save_path='./rgnn_weights2'):
        
        super().__init__(Het, num_nodes, num_hiddens, num_classes, batch_size, num_epoch,
                         lr, l1_reg, l2_reg, dropout, early_stop, optimizer, device,
                         extension={'loss_module': nn.CrossEntropyLoss()
                                    })


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
        
