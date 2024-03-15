import torch
import torch.nn as nn
import torch.nn.functional as F
from .Models.md_utils import *
from .Models.trainer import Trainer  # ,get_dl_mat
from .Models.dgcnn import DGCNN_Model
from .Models.rgnn import RGNN_Model
from .Models.het import Het_Model
from .Models.sparseDgcnn import SparseDGCNN_Model
from .Models.md_utils import *


class GNNModel(object):
    def __init__(self) -> None:
        ()


class DGCNNTrainer(Trainer):
    def __init__(self, edge_index=None, edge_weight=None, num_classes=2, device=torch.device('cpu'),
                 optimizer='Adam', num_hiddens=400, num_layers=2, dropout=0.5, early_stop=20,
                 batch_size=256, lr=5e-3, l1_reg=0.0, l2_reg=0.0, num_epoch=50):

        # if edge_index is None or edge_weight is None:
        #     raise Exception("No edge_index and edge_weight")
        num_nodes = 0
        if edge_weight is not None:
            num_nodes = edge_weight.shape[0]

        super().__init__(DGCNN_Model, num_nodes, num_hiddens, num_classes, batch_size, num_epoch,
                         lr, l1_reg, l2_reg, dropout, early_stop, optimizer, device,
                         extension={'num_layers': num_layers,
                                    'loss_module': nn.CrossEntropyLoss(),
                                    'edge_weight': edge_weight,
                                    'edge_index': edge_index
                                    })


class DGCNN(GNNModel):
    def __init__(self, num_nodes, num_hiddens, num_layers, electrode_position):
        '''
        Initialize a DGCNN model. Here the shape of electrode_position should be (num_nodes,2) or (num_nodes,3).
        '''

        if (num_nodes != len(electrode_position)):
            raise ValueError(
                "The number of element in electrode_position should be equal to num_nodes.")
        self.num_nodes = num_nodes
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.edge_index, self.edge_weight = get_edge_weight_from_electrode(
            model_name='DGCNN', edge_pos_value=electrode_position)
        self.trainer = None
        pass

    def train(self, data, labels, device=torch.device('cpu'),
              optimizer='Adam', num_classes=2, dropout=0.5,
              batch_size=256, lr=5e-3, l1_reg=0.0, l2_reg=0.0, num_epoch=50):
        '''
        This function is used to further define the training hyper-parameters and start the training progress. 

        Here, 'data' represents the data samples and 'labels' represents the corresponding labels.
        '''
        self.trainer = DGCNNTrainer(edge_index=self.edge_index, edge_weight=self.edge_weight, num_classes=num_classes, device=device,
                                    num_hiddens=self.num_hiddens, num_layers=self.num_layers, dropout=dropout, optimizer=optimizer,
                                    batch_size=batch_size, lr=lr, l2_reg=l2_reg, num_epoch=num_epoch)
        return self.trainer.train_only(data, labels)

    def train_and_eval(self, data_train, label_train, data_val, label_val, device=torch.device('cpu'),
                       optimizer='Adam', num_classes=2, dropout=0.5,
                       batch_size=256, lr=5e-3, l1_reg=0.0, l2_reg=0.0, num_epoch=50):
        self.trainer = DGCNNTrainer(edge_index=self.edge_index, edge_weight=self.edge_weight, num_classes=num_classes, device=device,
                                    num_hiddens=self.num_hiddens, num_layers=self.num_layers, dropout=dropout, optimizer=optimizer,
                                    batch_size=batch_size, lr=lr, l2_reg=l2_reg, num_epoch=num_epoch)
        self.trainer.train_and_eval(
            data_train, label_train, data_val, label_val)
        # print("after ",self.trainer.eval_acc_list)

    def predict(self, data):
        '''
        This function is used to give predictions of the input data. Note that before running this statement, the model should be already trained on certain dataset via 'DGCNN.train' function. Otherwise, an error will be raised.
        '''
        if self.trainer is None:
            raise NotImplementedError("The model has not been trained yet.")
        return self.trainer.predict(data)

    def save(self, path, name='best_model.dic.pkl'):
        if self.trainer is None:
            raise NotImplementedError("The model has not been trained yet.")
        if not os.path.exists(path):
            os.makedirs(path)
        self.trainer.save(path, name)

    def load(self, path, name='best_model.dic.pkl'):
        self.trainer = DGCNNTrainer()
        self.trainer.load(path, name)
        self.num_nodes = self.trainer.num_nodes
        self.num_hiddens = self.trainer.num_hiddens
        self.num_layers = self.trainer.num_layers
        self.edge_index = self.trainer.edge_index
        self.edge_weight = self.trainer.edge_weight


class RGNNTrainer(Trainer):
    def __init__(self, edge_index=None, edge_weight=None, num_classes=2, device=torch.device('cpu'),
                 domain_adaptation=False, distribution_learning=False, optimizer='Adam',
                 num_hiddens=400, num_layers=2, dropout=0.5, early_stop=20,
                 batch_size=8, lr=5e-3, l1_reg=0.0, l2_reg=0.0, num_epoch=80):  # ,model_save_path='./rgnn_weights2'):

        if edge_index is None or edge_weight is None:
            raise Exception("No edge_index and edge_weight")
        num_nodes = edge_weight.shape[0]
        # dl_mat = get_dl_mat() # if distribution_learning else None
        # print("dl_mat", dl_mat)
        loss_module = nn.KLDivLoss(
            reduction='batchmean') if distribution_learning else nn.CrossEntropyLoss()

        super().__init__(RGNN_Model, num_nodes, num_hiddens, num_classes, batch_size, num_epoch,
                         lr, l1_reg, l2_reg, dropout, early_stop, optimizer, device,
                         extension={'num_layers': num_layers,
                                    'loss_module': loss_module,
                                    'edge_weight': edge_weight,
                                    'edge_index': edge_index,
                                    'domain_adaptation': domain_adaptation,
                                    'distribution_learning': distribution_learning,
                                    # 'dl_mat':dl_mat
                                    })


class RGNN(GNNModel):
    def __init__(self, num_nodes, num_hiddens, num_layers, electrode_position, global_connections=None):
        '''
        Initialize a RGNN model. Here the shape of electrode_position should be (2,num_nodes) or (3,num_nodes). Global connections should be declared if the user want to introduce global inter-channel connections. (Detail intuition can be found in the original paper.)
        '''
        if (num_nodes != len(electrode_position)):
            raise ValueError(
                "The number of element in electrode_position should be equal to num_nodes.")
        self.num_nodes = num_nodes
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.edge_index, self.edge_weight = get_edge_weight_from_electrode(
            model_name='RGNN', edge_pos_value=electrode_position, global_connections=global_connections)
        self.trainer = None
        pass

    def train(self, train_data, train_labels, valid_data=None, device=torch.device('cpu'),
              optimizer='Adam', num_classes=2, dropout=0.5, NodeDAT=False,
              batch_size=256, lr=5e-3, l1_reg=0.0, l2_reg=0.0, num_epoch=50):
        '''
        This function is used to further define the training hyper-parameters and start the training progress. 

        Here, 'train_data' represents the data samples and 'train_labels' represents the corresponding labels. Besides, 'NodeDAT' is a boolean parameter which represents wether use Node-wise domain adaptation or not. If 'NodeDAT' is true, then the user should provide 'valid_data' as well. 'EmotionDL' represents the wether use emotional label distribution learning  or not.
        '''

        if NodeDAT is True and valid_data is None:
            raise ValueError(
                "Validation data must be input in order to make the NodeDAT.")
        EmotionDL = False
        self.trainer = RGNNTrainer(edge_index=self.edge_index, edge_weight=self.edge_weight, num_classes=num_classes, device=device,
                                   domain_adaptation=NodeDAT, distribution_learning=EmotionDL, optimizer=optimizer,
                                   num_hiddens=self.num_hiddens, num_layers=self.num_layers, dropout=dropout,
                                   batch_size=batch_size, lr=lr, l2_reg=l2_reg, num_epoch=num_epoch)

        return self.trainer.train_only(train_data, train_labels, valid_data)

    def train_and_eval(self, data_train, label_train, data_val, label_val, device=torch.device('cpu'),
                       optimizer='Adam', num_classes=2, dropout=0.5, NodeDAT=False, EmotionDL=False,
                       batch_size=256, lr=5e-3, l1_reg=0.0, l2_reg=0.0, num_epoch=50):

        self.trainer = RGNNTrainer(edge_index=self.edge_index, edge_weight=self.edge_weight, num_classes=num_classes, device=device,
                                   domain_adaptation=NodeDAT, distribution_learning=EmotionDL, optimizer=optimizer,
                                   num_hiddens=self.num_hiddens, num_layers=self.num_layers, dropout=dropout,
                                   batch_size=batch_size, lr=lr, l2_reg=l2_reg, num_epoch=num_epoch)

        self.trainer.train_and_eval(
            data_train, label_train, data_val, label_val)

    def predict(self, data):
        '''
        This function is used to give predictions of the input data. Note that before running this statement, the model should be already trained on certain dataset via 'RGNN.train' function. Otherwise, an error will be raised.

        '''

        if self.trainer is None:
            raise NotImplementedError("The model has not been trained yet.")
        return self.trainer.predict(data)

    def save(self, path, name='best_model.dic.pkl'):
        if self.trainer is None:
            raise NotImplementedError("The model has not been trained yet.")
        if not os.path.exists(path):
            os.makedirs(path)
        self.trainer.save(path, name)

    def load(self, path, name='best_model.dic.pkl'):
        self.trainer = DGCNNTrainer()
        self.trainer.load(path, name)
        self.num_nodes = self.trainer.num_nodes
        self.num_hiddens = self.trainer.num_hiddens
        self.num_layers = self.trainer.num_layers
        self.edge_index = self.trainer.edge_index
        self.edge_weight = self.trainer.edge_weight


class SparseDGCNNTrainer(Trainer):
    def __init__(self, edge_index=None, edge_weight=None, num_classes=2, device=torch.device('cpu'),
                 optimizer='Adam', num_hiddens=400, num_layers=2, dropout=0.5, early_stop=20,
                 batch_size=256, lr=5e-3, l1_reg=0.0, l2_reg=0.0, num_epoch=50):  # l1_reg: i.e. λ in the paper   lr: i.e. τ in the paper

        if edge_index is None or edge_weight is None:
            raise Exception("No edge_index and edge_weight")
        num_nodes = edge_weight.shape[0]

        super().__init__(SparseDGCNN_Model, num_nodes, num_hiddens, num_classes, batch_size, num_epoch,
                         lr, l1_reg, l2_reg, dropout, early_stop, optimizer, device,
                         extension={'num_layers': num_layers,
                                    'loss_module': nn.CrossEntropyLoss(),
                                    'edge_weight': edge_weight,
                                    'edge_index': edge_index
                                    })


class SparseDGCNN(GNNModel):
    def __init__(self, num_nodes, num_hiddens, num_layers, electrode_position):
        '''
        Initialize a SparseDGCNN model. Here the shape of 'electrode_position' should be (2,num_nodes) or (3,num_nodes). 
        '''
        if (num_nodes != len(electrode_position)):
            raise ValueError(
                "The number of element in electrode_position should be equal to num_nodes.")
        self.num_nodes = num_nodes
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.edge_index, self.edge_weight = get_edge_weight_from_electrode(
            model_name='SparseDGCNN', edge_pos_value=electrode_position)
        self.trainer = None
        pass

    def train(self, data, labels, device=torch.device('cpu'),
              optimizer='Adam', num_classes=2, dropout=0.5,
              batch_size=256, lr=5e-3, l1_reg=0.0, l2_reg=0.0, num_epoch=50):
        '''
        This function is used to further define the training hyper-parameters and start the training progress. 

        Here, 'data' represents the data samples and 'labels' represents the corresponding labels. 
        Note that because the sparse coefficient equals to L1 normalization coefficient, we use a single parameter 'l1_reg' to represent both. 
        '''

        self.trainer = SparseDGCNNTrainer(edge_index=self.edge_index, edge_weight=self.edge_weight, num_classes=num_classes, device=device,
                                          num_hiddens=self.num_hiddens, num_layers=self.num_layers, dropout=dropout, optimizer=optimizer,
                                          batch_size=batch_size, lr=lr, l2_reg=l2_reg, num_epoch=num_epoch)
        return self.trainer.train_only(data, labels)

    def train_and_eval(self, data_train, label_train, data_val, label_val, device=torch.device('cpu'),
                       optimizer='Adam', num_classes=2, dropout=0.5,
                       batch_size=256, lr=5e-3, l1_reg=0.0, l2_reg=0.0, num_epoch=50):
        self.trainer = SparseDGCNNTrainer(edge_index=self.edge_index, edge_weight=self.edge_weight, num_classes=num_classes, device=device,
                                          num_hiddens=self.num_hiddens, num_layers=self.num_layers, dropout=dropout, optimizer=optimizer,
                                          batch_size=batch_size, lr=lr, l2_reg=l2_reg, num_epoch=num_epoch)
        self.trainer.train_and_eval(
            data_train, label_train, data_val, label_val)

    def predict(self, data):  # here loader is
        '''
        This function is used to give predictions of the input data. Note that before running this statement, the model should be already trained on certain dataset via 'SparseDGCNN.train' function. Otherwise, an error will be raised.
        '''

        if self.trainer is None:
            raise NotImplementedError("The model has not been trained yet.")
        return self.trainer.predict(data)

    def save(self, path, name='best_model.dic.pkl'):
        if self.trainer is None:
            raise NotImplementedError("The model has not been trained yet.")
        if not os.path.exists(path):
            os.makedirs(path)
        self.trainer.save(path, name)

    def load(self, path, name='best_model.dic.pkl'):
        self.trainer = DGCNNTrainer()
        self.trainer.load(path, name)
        self.num_nodes = self.trainer.num_nodes
        self.num_hiddens = self.trainer.num_hiddens
        self.num_layers = self.trainer.num_layers
        self.edge_index = self.trainer.edge_index
        self.edge_weight = self.trainer.edge_weight


class HetTrainer(Trainer):
    def __init__(self, num_nodes, num_classes=2, device=torch.device('cpu'),
                 optimizer='Adam',
                 num_hiddens=64, dropout=0.2, early_stop=20,
                 batch_size=256, lr=5e-3, l1_reg=0.0, l2_reg=0.0, num_epoch=50):  # ,model_save_path='./rgnn_weights2'):

        super().__init__(Het_Model, num_nodes, num_hiddens, num_classes, batch_size, num_epoch,
                         lr, l1_reg, l2_reg, dropout, early_stop, optimizer, device,
                         extension={'loss_module': nn.CrossEntropyLoss()
                                    })


class HetEmotionNet(GNNModel):
    def __init__(self, num_nodes, num_hiddens):
        '''
        Initialize a HetEmotionNet model. Note that in HetEmotionNet the initial values in adjacency matrix is calculated according to mutual information, so the positions of electrodes is not a must.
        '''
        self.num_nodes = num_nodes
        self.num_hiddens = num_hiddens
        self.trainer = None

    def train(self, data_freq, data_time, labels, device=torch.device('cpu'),
              optimizer='Adam', num_classes=2, dropout=0.5,
              batch_size=256, lr=5e-3, l1_reg=0.0, l2_reg=0.0, num_epoch=50):
        '''
        This function is used to further define the training hyper-parameters and start the training progress. 

        Here, 'data_time' represents the data samples under time domain and 'data_freq' represents the data samples under frequency domain. 'labels' represents the corresponding labels.  
        '''
        self.num_freq = data_freq.shape[-1]
        if data_time.shape[0] != data_freq.shape[0]:
            raise ValueError(
                "The first dimension of data_time should be equal to the first dimension of data_freq.")
        data = np.concatenate((data_freq, data_time), axis=2)
        self.trainer = HetTrainer(num_nodes=self.num_nodes, num_classes=num_classes, device=device,
                                  num_hiddens=self.num_hiddens, dropout=dropout, optimizer=optimizer,
                                  batch_size=batch_size, lr=lr, l2_reg=l2_reg, num_epoch=num_epoch)

        return self.trainer.train_only(data, labels, mat_train=get_het_adjacency_matrix(data), num_freq=self.num_freq)

    def train_and_eval(self, data_train, label_train, data_val, label_val, num_freq, device=torch.device('cpu'),
                       optimizer='Adam', num_classes=2, dropout=0.5,
                       batch_size=256, lr=5e-3, l1_reg=0.0, l2_reg=0.0, num_epoch=50):
        self.num_freq = num_freq
        self.trainer = HetTrainer(num_nodes=self.num_nodes, num_classes=num_classes, device=device,
                                  num_hiddens=self.num_hiddens, dropout=dropout, optimizer=optimizer,
                                  batch_size=batch_size, lr=lr, l2_reg=l2_reg, num_epoch=num_epoch)

        self.trainer.train_and_eval(
            data_train, label_train, data_val, label_val, num_freq=num_freq)

    def predict(self, data_freq, data_time):  # here loader is
        '''
        This function is used to give predictions of the input data. Note that before running this statement, the model should be already trained on certain dataset via 'HetEmotionNet.train' function. Otherwise, an error will be raised.
        '''
        if self.trainer is None:
            raise NotImplementedError("The model has not been trained yet.")
        data = np.concatenate((data_freq, data_time), axis=2)
        return self.trainer.predict(data, adj_mat=get_het_adjacency_matrix(data))

    def save(self, path, name='best_model.dic.pkl'):
        if self.trainer is None:
            raise NotImplementedError("The model has not been trained yet.")
        if not os.path.exists(path):
            os.makedirs(path)
        self.trainer.save(path, name)

    def load(self, path, name='best_model.dic.pkl'):
        self.trainer = DGCNNTrainer()
        self.trainer.load(path, name)
        self.num_nodes = self.trainer.num_nodes
        self.num_hiddens = self.trainer.num_hiddens
