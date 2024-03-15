import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import os
import time
import inspect
from .md_utils import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, accuracy_score, ndcg_score, roc_auc_score, precision_score, f1_score
from imblearn.over_sampling import SMOTE


class Trainer(object):
    def __init__(self, model_f, num_nodes, num_hiddens=400, num_classes=2,
                 batch_size=256, num_epoch=50, lr=0.005, l1_reg=0, l2_reg=0, dropout=0.5, early_stop=20,
                 optimizer='Adam', device=torch.device('cpu'),
                 extension: dict = None):
        self.num_nodes = num_nodes
        self.num_hiddens = num_hiddens
        self.num_epoch = num_epoch
        self.num_classes = num_classes
        self.dropout = dropout
        self.lr = lr
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.device = device
        self.model_config_para_name = inspect.getfullargspec(
            model_f.__init__).args  # list the para of the model
        self.model_f = model_f
        self.model_name = model_f.__name__.split('_')[0]
        self.early_stop = early_stop
        if extension is not None:
            self.__dict__.update(extension)
        # save the training configs exclude model,num_features and model_config_para
        self.trainer_config_para = self.__dict__.copy()
        self.model = None
        self.num_features = None
        # save the model config para; keys are in model_config_para_name list
        self.model_config_para = dict()

    def do_epoch(self, data_loader, mode='eval', epoch_num=None, ret_predict=False):
        if mode == 'train':
            self.model = self.model.train()
        else:
            self.model = self.model.eval()
        epoch_loss = {'Entropy_loss': 0, 'L1_loss': 0, 'L2_loss': 0,
                      'NodeDAT_loss': 0}  # total loss of epoch
        total_samples = 0  # total samples in epoch
        epoch_metrics = {}

        num_correct_predict = 0
        num_correct_domain_predict = 0
        if self.num_epoch != 1:
            p = epoch_num/(self.num_epoch-1)
        else:
            p = 1
        beta = 2/(1+math.exp(-10*p))-1
        loss_model2 = nn.CrossEntropyLoss()

        total_class_predictions = []

        for i, batch in enumerate(data_loader):
            # print("data batch ",i)
            if self.model_name == 'Het':
                X_time, X_freq, A, Y = batch
                num_samples = X_time.shape[0]
                predictions = self.model(X_time, X_freq, A)
            elif self.model_name == 'RGNN':
                if self.domain_adaptation == True and mode == 'train':
                    Xs, Xt, Y = batch
                    num_samples = Xs.shape[0]
                    predictions, domain_output_Xs = self.model(
                        Xs, beta, need_dat=True)
                    middleX, domain_output_Xt = self.model(
                        Xt, beta, need_pred=False, need_dat=True)
                else:
                    X, Y = batch
                    num_samples = X.shape[0]
                    predictions = self.model(X, 0)
            else:
                X, Y = batch
                num_samples = X.shape[0]
                predictions = self.model(X)

            if self.model_name == 'RGNN' and self.distribution_learning == True:
                Entropy_loss = self.loss_module(
                    F.log_softmax(predictions, dim=-1), torch.Tensor(self.distribution_label(Y)).float().to(self.device))
            else:
                # Entropy_loss = self.loss_module(
                #     predictions, torch.Tensor(Y).long().to(self.device))
                # print(type(predictions))
                # print(predictions)
                Entropy_loss = self.loss_module(
                    predictions.float(),
                    torch.Tensor(Y.float()).long().to(self.device))

            class_predict = predictions.argmax(axis=-1)
            class_predict = class_predict.cpu().detach().numpy()
            if ret_predict == True:
                total_class_predictions += [
                    item for item in class_predict]

            Y = Y.cpu().detach().numpy()
            num_correct_predict += np.sum(class_predict == Y)
            L1_loss = self.l1_reg * l1_reg_loss(
                self.model, only=None if self.model_name == 'Het' else ['edge_weight'])  # if self.model_name == 'RGNN' else None) # Note: the L1_reg_loss of RGNN only contains matrix edge weight.
            L2_loss = self.l2_reg * l2_reg_loss(
                self.model, exclude=['edge_weight'])  # if  self.model_name=='SparseDGCNN' else None)

            if self.model_name == 'SparseDGCNN':
                loss = Entropy_loss+L2_loss
            else:
                loss = Entropy_loss + L1_loss + L2_loss

            if self.model_name == 'RGNN' and self.domain_adaptation == True and mode == 'train':
                domain_class = np.zeros(
                    (Xs.shape[0]+Xt.shape[0])*self.num_nodes)
                # [0,0,...,0,1,1,...,1]
                domain_class[Xs.shape[0]*self.num_nodes:] = 1

                NodeDAT_loss = loss_model2(torch.cat([domain_output_Xs, domain_output_Xt], dim=0), torch.Tensor(
                    domain_class).long().to(self.device))
                domain_class_predict = torch.cat(
                    [domain_output_Xs, domain_output_Xt], dim=0).argmax(-1).cpu().detach().numpy()
                num_correct_domain_predict += np.sum(
                    domain_class_predict == domain_class)
                loss += NodeDAT_loss  # add NodeDAT_loss to loss

            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=20)  # change to 20
                self.optimizer.step()

            if self.model_name == 'SparseDGCNN':
                with torch.no_grad():
                    for name, para in self.model.named_parameters():
                        if name in ['edge_weight']:
                            tmp = torch.sign(para.data)*torch.maximum(torch.zeros_like(
                                para.data), torch.abs(para.data)-0.5*self.lr*self.l1_reg)
                            para.copy_(tmp)

            if self.model_name == 'DGCNN' or self.model_name == 'SparseDGCNN':
                with torch.no_grad():
                    for name, para in self.model.named_parameters():
                        if name in ['edge_weight']:
                            tmp = F.relu(para.data)
                            para.copy_(tmp)

            with torch.no_grad():
                total_samples += num_samples
                epoch_loss['Entropy_loss'] += Entropy_loss.item()
                epoch_loss['L1_loss'] += L1_loss  # .item()
                epoch_loss['L2_loss'] += L2_loss  # .item()

        epoch_metrics['epoch'] = epoch_num
        epoch_metrics['loss'] = (
            epoch_loss['L2_loss']+epoch_loss['L1_loss']+epoch_loss['Entropy_loss']).item() / (total_samples/self.batch_size)
        epoch_metrics['num_correct'] = num_correct_predict
        epoch_metrics['acc'] = num_correct_predict/total_samples

        if ret_predict == False:
            return epoch_metrics
        else:
            return total_class_predictions

    def data_prepare_train_only(self, train_data, train_label, valid_data=None, mat_train=None, num_freq=None):
        # if use RGNN and NodeDAT, then valid_data must not be None

        label_class = set(train_label)
        assert (len(label_class) == self.num_classes)

        if self.model_name == 'Het':
            train_mat_list = mat_train  # get_het_adjacency_matrix(train_data)
            print("num_freq",num_freq)
            print('s1',train_data[:, :, num_freq:].shape)
            print('s2',train_data[:, :, :num_freq].shape)
            print('s3',train_mat_list.shape)
            train_dataset = HetDataset(
                train_data[:, :, num_freq:], train_data[:, :, :num_freq], train_mat_list, train_label, self.device)

        elif self.model_name == 'RGNN' and self.domain_adaptation == True:
            SMO = SMOTE(random_state=random.randint(0, 255))
            pre_label = np.zeros(valid_data.shape[0]+train_data.shape[0])
            pre_label[train_data.shape[0]:] = 1
            tmp_data, tmp_label = SMO.fit_resample(np.concatenate(
                (train_data, valid_data), axis=0).reshape(-1, self.num_nodes*self.num_features), pre_label)
            valid_oversampled = tmp_data[np.where(
                tmp_label == 1)].reshape(-1, self.num_nodes, self.num_features)
            # print(valid_oversampled.shape)
            train_dataset = DATDataSet(
                train_data, valid_oversampled, train_label, self.device)
        else:
            train_dataset = NormalDataset(train_data, train_label, self.device)

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

        return train_loader

    def data_prepare(self, train_data, train_label, valid_data=None, valid_label=None, mat_train=None, mat_val=None, num_freq=None):
        label_class = set(train_label)
        assert (len(label_class) == self.num_classes)

        if self.model_name == 'Het':
            train_mat_list = mat_train  # get_het_adjacency_matrix(train_data)
            valid_mat_list = mat_val   # get_het_adjacency_matrix(valid_data)
            train_dataset = HetDataset(
                train_data[:, :, num_freq:], train_data[:, :, :num_freq], train_mat_list, train_label, self.device)
            valid_dataset = HetDataset(
                valid_data[:, :, num_freq:], valid_data[:, :, :num_freq], valid_mat_list, valid_label, self.device)
        elif self.model_name == 'RGNN' and self.domain_adaptation == True:
            SMO = SMOTE(random_state=random.randint(0, 255))
            pre_label = np.zeros(valid_data.shape[0]+train_data.shape[0])
            pre_label[train_data.shape[0]:] = 1
            tmp_data, tmp_label = SMO.fit_resample(np.concatenate(
                (train_data, valid_data), axis=0).reshape(-1, self.num_nodes*self.num_features), pre_label)
            valid_oversampled = tmp_data[np.where(
                tmp_label == 1)].reshape(-1, self.num_nodes, self.num_features)
            # print(valid_oversampled.shape)
            train_dataset = DATDataSet(
                train_data, valid_oversampled, train_label, self.device)
            valid_dataset = NormalDataset(valid_data, valid_label, self.device)
        else:
            train_dataset = NormalDataset(train_data, train_label, self.device)
            valid_dataset = NormalDataset(valid_data, valid_label, self.device)

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(
            dataset=valid_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, valid_loader

    def train_and_eval(self, train_data, train_label, valid_data, valid_label,
                       num_freq=None, reload=True, nd_predict=True): 
        # print("train_and_eval !!!!")
        # for k, v in self.__dict__.items():
        #     print(k,v)
        if self.model_name == 'Het':
            self.num_freq = num_freq
            self.num_time = train_data.shape[-1]-num_freq
            self.trainer_config_para['num_freq'] = self.num_freq
            self.trainer_config_para['num_time'] = self.num_time
        else:
            self.num_features = train_data.shape[-1]
            # save num_features
            self.trainer_config_para['num_features'] = self.num_features
        # save model_config_para and save into trainer_config_para['model_config_para']
        for k, v in self.__dict__.items():
            if k in self.model_config_para_name:
                self.model_config_para.update({k: v})
        self.trainer_config_para['model_config_para'] = self.model_config_para

        self.model = self.model_f(**self.model_config_para)
        self.model.to(self.device)
        # print("using device: ", self.device)

        # print("gpu device ",self.device)

        # for name, param in self.model.named_parameters():
        #     print(name)
        #     print(param.data.shape)
            
        # print(self.optimizer)
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr)
        # print(self.optimizer)
        
        if self.model_name == 'Het':
            mat_train = get_het_adjacency_matrix(train_data)
            mat_val = get_het_adjacency_matrix(valid_data)
            train_loader, valid_loader = self.data_prepare(
                train_data, train_label, valid_data, valid_label, mat_train, mat_val, num_freq)
        else:
            train_loader, valid_loader = self.data_prepare(
                train_data, train_label, valid_data, valid_label)

        best_acc = 0
        early_stop = self.early_stop
        if early_stop is None:
            early_stop = self.num_epoch
        early_stop_num = 0
        best_epoch = -1
        eval_acc_list = []
        train_acc_list = []
        eval_loss_list = []
        train_num_correct_list = []
        eval_num_correct_list = []
        train_loss_list = []
        for i in range(self.num_epoch):
            time_start = time.time()
            train_metric = self.do_epoch(train_loader, 'train', i)
            eval_metric = self.do_epoch(valid_loader, 'eval', i)

            time_end = time.time()
            time_cost = time_end - time_start

            # if train_metric['epoch']%5==0:
            #     print('device', self.device, 'Epoch {:.0f} training_acc: {:.4f}  valid_acc: {:.4f}| train_loss: {:.4f}, valid_loss: {:.4f}, | time cost: {:.3f}'.format(
            #         train_metric['epoch'], train_metric['acc'], eval_metric['acc'], train_metric['loss'], eval_metric['loss'], time_cost))

            eval_acc_list.append(eval_metric['acc'])
            train_acc_list.append(train_metric['acc'])
            eval_loss_list.append(eval_metric['loss'])
            train_loss_list.append(train_metric['loss'])
            eval_num_correct_list.append(eval_metric['num_correct'])
            train_num_correct_list.append(train_metric['num_correct'])
            if eval_metric['acc'] > best_acc:
                early_stop_num = 0
                best_acc = eval_metric['acc']
                best_epoch = i
            else:
                early_stop_num += 1

        # print('best acc ', best_acc,
        #       ' best epoch ', best_epoch)

        self.eval_acc_list = eval_acc_list
        self.train_acc_list = train_acc_list
        

    def train_only(self, train_data, train_label, valid_data=None, mat_train=None,
                   num_freq=None):  # ,small=False,step=0.00001):
        # print("in train_only num_fe",num_freq)
        
        # self.num_epoch = 1
        if self.model_name == 'Het':
            self.num_freq = num_freq
            self.num_time = train_data.shape[-1]-num_freq
            self.trainer_config_para['num_freq'] = self.num_freq
            self.trainer_config_para['num_time'] = self.num_time
        else:
            self.num_features = train_data.shape[-1]
            # save num_features
            self.trainer_config_para['num_features'] = self.num_features
        # save model_config_para and save into trainer_config_para['model_config_para']
        for k, v in self.__dict__.items():
            if k in self.model_config_para_name:
                self.model_config_para.update({k: v})
        self.trainer_config_para['model_config_para'] = self.model_config_para

        self.model = self.model_f(**self.model_config_para)
        self.model.to(self.device)
        # print("gpu device ", self.device)

        # self.optimizer = torch.optim.Adam(
        #         self.model.parameters(), lr=self.lr)
        self.optimizer = self.optimizer(
            self.model.parameters(), lr=self.lr)

        
        if self.model_name == 'Het':
            train_loader = self.data_prepare_train_only(
                train_data, train_label, mat_train=mat_train, num_freq=num_freq)
            pass
        elif self.model_name == 'RGNN' and valid_data is not None:
            train_loader = self.data_prepare_train_only(
                train_data, train_label, valid_data)
        else:
            train_loader = self.data_prepare_train_only(
                train_data, train_label)

        train_acc_list = []
        train_num_correct_list = []
        train_loss_list = []
        # lr_list=[]
        for i in range(self.num_epoch):
            print("training epochs : ", i)
            train_metric = self.do_epoch(train_loader, 'train', i)
            # print('device',self.device,'fold', fold, 'Epoch {:.1f} training_acc: {:.4f}  valid_acc: {:.4f}| train_loss: {:.4f}, valid_loss: {:.4f}, | time cost: {:.3f}'.format(
            #     train_metric['epoch'], train_metric['acc'], eval_metric['acc'], train_metric['loss'], eval_metric['loss'], time_cost))
            train_acc_list.append(train_metric['acc'])
            train_loss_list.append(train_metric['loss'])
            train_num_correct_list.append(train_metric['num_correct'])

        # self.train_loss_list=train_loss_list
        # self.train_num_correct_list=train_num_correct_list
        self.train_acc_list = train_acc_list
        return train_acc_list

    def predict(self, data, adj_mat=None):  # inference
        if self.model is None:
            raise Exception(
                f"{self.model_name} model has not been trained yet.")

        data = torch.from_numpy(data).to(self.device, dtype=torch.float32)
        if self.model_name == 'Het':
            # adj_mat = torch.from_numpy(adj_mat).to(
            #     self.device, dtype=torch.float32)
            data_freq = data[:, :, :self.num_freq]
            data_time = data[:, :, self.num_freq:]
            mat_list = adj_mat
            self.model = self.model.eval()
            total_class_predictions = []
            with torch.no_grad():
                if data.shape[0] < 128:
                    predictions = self.model(data_time, data_freq, mat_list)
                    class_predict = predictions.argmax(axis=-1)
                    class_predict = class_predict.cpu().detach().numpy()
                    total_class_predictions += [item for item in class_predict]
                else:
                    for i in range(0, data.shape[0], 128):
                        if i+128 < data.shape[0]:
                            cur_data_time = data_time[i:i+128, :, :]
                            cur_data_freq = data_freq[i:i+128, :, :]
                            cur_mat_list = mat_list[i:i+128, :, :]
                        else:
                            cur_data_time = data_time[i:, :, :]
                            cur_data_freq = data_freq[i:, :, :]
                            cur_mat_list = mat_list[i:, :, :]
                        predictions = self.model(
                            cur_data_time, cur_data_freq, cur_mat_list)
                        class_predict = predictions.argmax(axis=-1)
                        class_predict = class_predict.cpu().detach().numpy()
                        total_class_predictions += [
                            item for item in class_predict]
        else:
            self.model = self.model.eval()
            total_class_predictions = []

            # predictions (before softmax)
            with torch.no_grad():
                if data.shape[0] < 128:
                    predictions = self.model(data)
                    class_predict = predictions.argmax(axis=-1)
                    class_predict = class_predict.cpu().detach().numpy()
                    total_class_predictions += [item for item in class_predict]
                else:
                    for i in range(0, data.shape[0], 128):
                        if i+128 < data.shape[0]:
                            cur_data = data[i:i+128, :, :]
                        else:
                            cur_data = data[i:, :, :]
                        predictions = self.model(cur_data)
                        class_predict = predictions.argmax(axis=-1)
                        class_predict = class_predict.cpu().detach().numpy()
                        total_class_predictions += [
                            item for item in class_predict]
        return np.array(total_class_predictions)

    def save(self, path, name='best_model.dic.pkl'):
        if self.model is None:
            raise Exception(
                f"{self.model_name} model has not been trained yet.")
        if not os.path.exists(path):
            os.makedirs(path)
        model_dict = {'state_dict': self.model.state_dict(),
                      'configs': self.trainer_config_para
                      }
        torch.save(model_dict, os.path.join(
            path, name))

    def load(self, path, name='best_model.dic.pkl'):
        self.model = None
        model_dic = torch.load(os.path.join(
            path, name), map_location='cpu')
        self.__dict__.update(model_dic['configs'])  # load trainer_config_para
        self.trainer_config_para = self.__dict__.copy()
        self.model = self.model_f(**self.model_config_para)
        self.model.load_state_dict(model_dic['state_dict'])
        self.model.to(self.device)
