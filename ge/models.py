import torch
import torch.nn as nn
import torch.nn.functional as F
from .Models.md_utils import *
from .Models.trainer import *
from .Models.dgcnn import DGCNN_Model
from .Models.rgnn import RGNN_Model
from .Models.het import Het_Model
from .Models.sparseDgcnn import SparseDGCNN_Model
from .Models.md_utils import *

class DGCNNTrainer(Trainer):
    def __init__(self, edge_index=None, edge_weight=None, num_classes=2, device=torch.device('cpu'),
                 optimizer=torch.optim.Adam, num_hiddens=400, num_layers=2, dropout=0.5,early_stop=20,
                 batch_size=256, lr=5e-3, l1_reg=0.0, l2_reg=0.0, num_epoch=50):

        if edge_index is None or edge_weight is None:
            raise Exception("No edge_index and edge_weight")
        num_nodes = edge_weight.shape[0]

        super().__init__(DGCNN_Model, num_nodes, num_hiddens, num_classes, batch_size, num_epoch,
                         lr, l1_reg, l2_reg, dropout, early_stop, optimizer, device,
                         extension={'num_layers': num_layers, 
                                    'loss_module': nn.CrossEntropyLoss(),
                                    'edge_weight':edge_weight,
                                    'edge_index':edge_index
                                    })

class DGCNN(object):
    def __init__(self,num_nodes,num_hiddens,num_layers,electrode_position) :
        if(num_nodes!=len(electrode_position)):
            raise ValueError("The number of element in electrode_position should be equal to num_nodes.")
        self.num_nodes=num_nodes
        self.num_hiddens=num_hiddens
        self.num_layers=num_layers
        self.edge_index, self.edge_weight=get_edge_weight_from_electrode(model_name='DGCNN',edge_pos_value=electrode_position)
        self.trainer=None
        pass
    
    def train(self,data,labels, device=torch.device('cpu'),
                 optimizer=torch.optim.Adam, num_classes=2,num_hiddens=400, num_layers=2, dropout=0.5,
                 batch_size=256, lr=5e-3, l1_reg=0.0,l2_reg=0.0, num_epoch=50):
        self.trainer=DGCNNTrainer(edge_index=self.edge_index, edge_weight=self.edge_weight, num_classes=num_classes, device=device,
                               num_hiddens=num_hiddens, num_layers=num_layers, dropout=dropout,optimizer=optimizer,
                               batch_size=batch_size, lr=lr, l2_reg=l2_reg, num_epoch=num_epoch)
        self.trainer.train_only(data,labels)
        
    def predict(self,data): # here loader is 
        if self.runner is None:
            raise NotImplementedError("The model has not been trained yet.")
        return self.trainer.predict(data)
    


class RGNNTrainer(Trainer):
    def __init__(self, edge_index=None, edge_weight=None, num_classes=2, device=torch.device('cpu'),
                 domain_adaptation=False, distribution_learning=False, optimizer=torch.optim.Adam,
                 num_hiddens=400, num_layers=2, dropout=0.5,early_stop=20,
                 batch_size=8, lr=5e-3, l1_reg=0.0, l2_reg=0.0,num_epoch=80):  # ,model_save_path='./rgnn_weights2'):

        if edge_index is None or edge_weight is None:
            raise Exception("No edge_index and edge_weight")
        num_nodes = edge_weight.shape[0]
        dl_mat = get_dl_mat() # if distribution_learning else None
        # print("dl_mat", dl_mat)
        loss_module = nn.KLDivLoss(reduction='batchmean') if distribution_learning else nn.CrossEntropyLoss()
        
        super().__init__(RGNN_Model, num_nodes, num_hiddens, num_classes, batch_size, num_epoch,
                            lr, l1_reg, l2_reg, dropout, early_stop, optimizer, device,
                            extension={'num_layers': num_layers, 
                                        'loss_module': loss_module,
                                        'edge_weight':edge_weight,
                                        'edge_index':edge_index,
                                        'domain_adaptation':domain_adaptation,
                                        'distribution_learning':distribution_learning,
                                        'dl_mat':dl_mat
                                        })    
    
class RGNN(object):
    def __init__(self,num_nodes,num_hiddens,num_layers,electrode_position,global_connections=None) :
        if(num_nodes!=len(electrode_position)):
            raise ValueError("The number of element in electrode_position should be equal to num_nodes.")
        self.num_nodes=num_nodes
        self.num_hiddens=num_hiddens
        self.num_layers=num_layers
        self.edge_index, self.edge_weight=get_edge_weight_from_electrode(model_name='RGNN',edge_pos_value=electrode_position,global_connections=global_connections)
        self.trainer=None
        pass
    
    def train(self,train_data,train_labels,valid_data=None, device=torch.device('cpu'),
                 optimizer=torch.optim.Adam, num_classes=2,num_hiddens=400, num_layers=2, dropout=0.5,NodeDAT=False, EmotionDL=False,
                 batch_size=256, lr=5e-3, l1_reg=0.0,l2_reg=0.0, num_epoch=50):
        if NodeDAT is True and valid_data is None:
            raise ValueError("Validation data must be input in order to make the NodeDAT.")
        
        self.trainer=RGNNTrainer(edge_index=self.edge_index, edge_weight=self.edge_weight, num_classes=num_classes, device=device,
                               domain_adaptation=NodeDAT, distribution_learning=EmotionDL, optimizer=torch.optim.Adam,
                               num_hiddens=num_hiddens, num_layers=num_layers, dropout=dropout,
                               batch_size=batch_size, lr=lr, l2_reg=l2_reg, num_epoch=num_epoch)
        
        self.trainer.train_only(train_data,train_labels,valid_data)
        
    def predict(self,data): # here loader is 
        if self.runner is None:
            raise NotImplementedError("The model has not been trained yet.")
        return self.trainer.predict(data)


class SparseDGCNNTrainer(Trainer):
    def __init__(self, edge_index=None, edge_weight=None, num_classes=2, device=torch.device('cpu'),
                 optimizer=torch.optim.Adam, num_hiddens=400, num_layers=2, dropout=0.5,early_stop=20,
                 batch_size=256, lr=5e-3, l1_reg=0.0,l2_reg=0.0, num_epoch=50): #l1_reg: i.e. λ in the paper   lr: i.e. τ in the paper

        if edge_index is None or edge_weight is None:
            raise Exception("No edge_index and edge_weight")
        num_nodes = edge_weight.shape[0]


        super().__init__(SparseDGCNN_Model, num_nodes, num_hiddens, num_classes, batch_size, num_epoch,
                         lr, l1_reg,l2_reg, dropout, early_stop, optimizer, device,
                         extension={'num_layers': num_layers, 
                                    'loss_module': nn.CrossEntropyLoss(),
                                    'edge_weight':edge_weight,
                                    'edge_index':edge_index
                                    })


class SparseDGCNN(object):
    def __init__(self,num_nodes,num_hiddens,num_layers,electrode_position) :
        if(num_nodes!=len(electrode_position)):
            raise ValueError("The number of element in electrode_position should be equal to num_nodes.")
        self.num_nodes=num_nodes
        self.num_hiddens=num_hiddens
        self.num_layers=num_layers
        self.edge_index, self.edge_weight=get_edge_weight_from_electrode(model_name='SparseDGCNN',edge_pos_value=electrode_position)
        self.trainer=None
        pass
    
    def train(self,data,labels, device=torch.device('cpu'),
                 optimizer=torch.optim.Adam, num_classes=2,num_hiddens=400, num_layers=2, dropout=0.5,
                 batch_size=256, lr=5e-3, l1_reg=0.0,l2_reg=0.0, num_epoch=50):
        self.trainer=SparseDGCNNTrainer(edge_index=self.edge_index, edge_weight=self.edge_weight, num_classes=num_classes, device=device,
                               num_hiddens=num_hiddens, num_layers=num_layers, dropout=dropout,optimizer=optimizer,
                               batch_size=batch_size, lr=lr, l2_reg=l2_reg, num_epoch=num_epoch)
        self.trainer.train_only(data,labels)
        
    def predict(self,data): # here loader is 
        if self.runner is None:
            raise NotImplementedError("The model has not been trained yet.")
        return self.trainer.predict(data)
    
        
class HetTrainer(Trainer):
    def __init__(self, num_nodes, num_classes=2, device=torch.device('cpu'),
                 optimizer=torch.optim.Adam,
                 num_hiddens=64, dropout=0.2,early_stop=20,
                 batch_size=256, lr=5e-3, l1_reg=0.0, l2_reg=0.0, num_epoch=50):  # ,model_save_path='./rgnn_weights2'):
        
        super().__init__(Het_Model, num_nodes, num_hiddens, num_classes, batch_size, num_epoch,
                         lr, l1_reg, l2_reg, dropout, early_stop, optimizer, device,
                         extension={'loss_module': nn.CrossEntropyLoss()
                                    })
        
class HetEmotionNet(object):
    def __init__(self,num_nodes,num_hiddens,num_layers):
        self.num_nodes=num_nodes
        self.num_hiddens=num_hiddens
        self.num_layers=num_layers
        self.trainer=None
    
    def train(self,data,labels,num_freq, device=torch.device('cpu'),
                 optimizer=torch.optim.Adam, num_classes=2,num_hiddens=400, num_layers=2, dropout=0.5,
                 batch_size=256, lr=5e-3, l1_reg=0.0,l2_reg=0.0, num_epoch=50):
        self.trainer=HetTrainer(num_nodes=self.num_nodes, num_classes=num_classes, device=device,
                               num_hiddens=num_hiddens, num_layers=num_layers, dropout=dropout,optimizer=optimizer,
                               batch_size=batch_size, lr=lr, l2_reg=l2_reg, num_epoch=num_epoch)
        
        self.trainer.train_only(data,labels,mat_train=get_het_adjacency_matrix(data),num_freq=num_freq)
        
    def predict(self,data): # here loader is 
        if self.runner is None:
            raise NotImplementedError("The model has not been trained yet.")
        return self.trainer.predict(data,adj_mat=get_het_adjacency_matrix(data))



        

    
