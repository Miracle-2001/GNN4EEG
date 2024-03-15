import numpy as np
import pandas as pd
import torch
import os
import scipy.io as sio
import hdf5storage as hdf5
import random
import time
import joblib
import copy
import json
from .load_data import load_srt_de
from .models import *


class DataLoader(object):
    def __init__(self, protocol, data, labels, type, subject_id_list=None, num_freq=None):
        uni = np.unique(subject_id_list)
        if protocol == 'cross_subject':
            dict = {}
            for i in range(len(uni)):
                dict.update({uni[i]: i})
            sum = [0 for i in range(len(uni))]
            for i in range(len(subject_id_list)):
                sum[dict[subject_id_list[i]]] += 1
            for i in range(1, len(uni)):
                sum[i] += sum[i-1]
            data_new = np.zeros_like(data)
            labels_new = np.zeros_like(labels)
            subject_id_list_new = np.zeros_like(subject_id_list)
            pre_sum = copy.deepcopy(sum)
            self.pre_sum = pre_sum
            for i in range(data.shape[0]):
                pos = sum[dict[subject_id_list[i]]]-1
                sum[dict[subject_id_list[i]]] -= 1
                data_new[pos, :, :] = data[i, :, :]
                labels_new[pos] = labels[i]
                subject_id_list_new[pos] = subject_id_list[i]

        else:  # intra_subject. A random shuffle is used to reorder the data.
            lst = [i for i in range(data.shape[0])]
            random.shuffle(lst)
            data_new = np.zeros_like(data)
            labels_new = np.zeros_like(labels)
            subject_id_list_new = np.zeros_like(subject_id_list)
            for i in range(data.shape[0]):
                pos = lst[i]
                data_new[pos, :, :] = data[i, :, :]
                labels_new[pos] = labels[i]
                subject_id_list_new[pos] = subject_id_list[i]

        self.type = type
        self.protocol = protocol
        self.data = data_new
        self.labels = labels_new
        self.num_freq = num_freq
        self.subject_id_list = subject_id_list_new

    def getData(self, K, fold):
        if self.protocol == 'cross_subject':
            uni = np.unique(self.subject_id_list)
            subject_num = len(uni)
            if K > subject_num:
                raise ValueError(f"Under cross_subject protocol, there are {subject_num} subjects in the dataset, which is smaller than the fold number {K}. So, the dataset can not be divided into {K} sections.")

            step = int(subject_num/K)
            st = step*fold
            nd = step*(fold+1)  # [st,nd)
            if fold == K-1:
                nd = subject_num
            left = 0
            right = len(self.subject_id_list)
            if st != 0:
                left = self.pre_sum[st-1]
            if nd != subject_num:
                right = self.pre_sum[nd-1]

        else:
            tot_num = len(self.subject_id_list)
            if tot_num<K:
                raise ValueError(f"Under intra_subject protocol, there are {tot_num} samples in the dataset, which is smaller than the fold number {K}. So, the dataset can not be divided into {K} sections.")

            step = int(tot_num/K)
            st = step*fold
            nd = step*(fold+1)  # [st,nd)
            if fold == K-1:
                nd = tot_num
            left = st
            right = nd

        valid_list = np.arange(left, right)
        train_list = np.array(
            list(set(np.arange(0, len(self.subject_id_list))) - set(valid_list))).astype(int)
        data_train = self.data[train_list, :, :]
        label_train = self.labels[train_list]
        data_val = self.data[valid_list, :, :]
        label_val = self.labels[valid_list]
        train_subject_list = self.subject_id_list[train_list]
        return data_train, label_train, data_val, label_val, train_subject_list


def data_split(protocol, data, labels, subject_id_list=None, data_time=None):
    if protocol != "cross_subject" and protocol != "intra_subject":
        raise ValueError(
            "The protocol should be either 'cross_subject' or 'intra_subject'.")
    if protocol == "cross_subject" and subject_id_list is None:
        raise ValueError(
            "The subject_id_list should not be none when data splitting protocol is cross_subject")

    num_freq = data.shape[-1]
    if data_time is not None:
        data = np.concatenate((data, data_time), axis=2)

    dataloader = DataLoader(protocol, data, labels,
                            'user_defined', subject_id_list=subject_id_list,num_freq=num_freq)
    return dataloader


def data_FACED(protocol, categories, data_path):
    if protocol != "cross_subject" and protocol != "intra_subject":
        raise ValueError(
            "The protocol should be either 'cross_subject' or 'intra_subject'.")
    if categories != 2 and categories != 9:
        raise ValueError("The categories should be either 2 or 9.")
    if os.path.exists(data_path) == False:
        raise ValueError("The path of FACED dataset does not exist.")

    if categories != 2 and categories != 9:
        raise ValueError(
            "The label categories in FACED dataset should be either 2 or 9.")

    # Note: the signals are in dict['de_lds']
    data = hdf5.loadmat(data_path)['de_lds'] 
    # data shape: (123, 720 or 840, 120)  120=4*30  4 band and 30 nodes
    label_type = 'cls2' if categories == 2 else 'cls9'
    data, label_repeat, n_samples = load_srt_de(
        data, True, False, 1, label_type)
    # label_repeat shape: 720 or 840   720=24*30. 840=28*30   24/28 videos and 30 samples are generated by each video
   
    feature_shape = int(data.shape[-1]/30)
    labels=np.tile(label_repeat, data.shape[0])
    data=data.reshape(-1,feature_shape, 30).transpose([0, 2, 1])

    subject_id_list=[int(i/len(label_repeat)) for i in range(labels.shape[0])]
    dataloader = DataLoader(protocol, data, labels, 'FACED',subject_id_list=subject_id_list)
    return dataloader
    

def getGridPara(paras):
    if "dropout" in paras:
        dropout = paras["dropout"]
    else:
        dropout = 0.5
    if "batch_size" in paras:
        batch_size = paras["batch_size"]
    else:
        batch_size = 256
    if "lr" in paras:
        lr = paras["lr"]
    else:
        lr = 5e-3
    if "l1_reg" in paras:
        l1_reg = paras["l1_reg"]
    else:
        l1_reg = 0
    if "l2_reg" in paras:
        l2_reg = paras["l2_reg"]
    else:
        l2_reg = 0
    return lr, batch_size, dropout, l1_reg, l2_reg


def launch(curModel, data_train, label_train, data_val, label_val,
           device, optimizer, categories, dropout, batch_size, lr, l1_reg, l2_reg,
           mx_epoch, num_freq=None,
           NodeDAT=False, EmotionDL=False,
           ):
    if isinstance(curModel, RGNN):
        curModel.train_and_eval(
            data_train, label_train, data_val, label_val,
            device, optimizer, categories, dropout,
            batch_size=batch_size, lr=lr, l1_reg=l1_reg, l2_reg=l2_reg,
            NodeDAT=NodeDAT, EmotionDL=EmotionDL,
            num_epoch=mx_epoch
        )
    elif isinstance(curModel, HetEmotionNet):
        curModel.train_and_eval(
            data_train, label_train, data_val, label_val, num_freq,
            device, optimizer, categories, dropout,
            batch_size=batch_size, lr=lr, l1_reg=l1_reg, l2_reg=l2_reg,
            num_epoch=mx_epoch
        )
    else:
        curModel.train_and_eval(
            data_train, label_train, data_val, label_val,
            device, optimizer, categories, dropout,
            batch_size=batch_size, lr=lr, l1_reg=l1_reg, l2_reg=l2_reg,
            num_epoch=mx_epoch
        )


def evaluation(model:GNNModel, loader: DataLoader, protocol: str, grid: dict, categories, K, K_inner=None, device=torch.device('cpu'),
               optimizer="Adam", NodeDAT=False):
    EmotionDL=False
    # ,L1_reg=0,L2_reg=0,dropout=0,alpha=0,lr,epoch=100,batch_size,
    if protocol != "cv" and protocol != "ncv" and protocol != "fcv":
        raise ValueError(
            "The evaluation protocols must be 'cv', 'fcv' or 'ncv'.")
    model_paras = ["hiddens", "layers"]
    train_paras = ["lr", "epoch", "dropout", "batch_size", "l1_reg", "l2_reg"]
    grid_paras = []
    grid_epoch = [50]

    for k, v in grid.items():
        if k not in model_paras and k not in train_paras:
            raise ValueError(
                f"The parameter name {k} does not exist or can not be tuned.")
        if type(v) is not list and isinstance(v, (int, float)) == False:
            raise ValueError(
                f"The type of parameter value {v} must be list, int or float.")

        if k == "epoch":
            if type(v) is list:
                grid_epoch = v
            else:
                grid_epoch = [v]
            continue

        if len(grid_paras) == 0:
            if type(v) is list:
                for _ in v:
                    grid_paras.append({k: _})
            else:
                grid_paras.append({k: v})
        else:
            tmp = []
            if type(v) is list:
                for _ in v:
                    nt = {k: _}
                    for old in grid_paras:
                        ndict = copy.deepcopy(old)
                        ndict.update(nt)
                        tmp.append(ndict)
            else:
                nt = {k: v}
                for old in grid_paras:
                    ndict = copy.deepcopy(old)
                    ndict.update(nt)
                    tmp.append(ndict)
            grid_paras = tmp

    mx_epoch = max(grid_epoch)+1
    if protocol == "cv" or protocol == "fcv":
        result_list = []
        for paras in grid_paras:
            nModel = copy.deepcopy(model)
            if "hiddens" in paras:
                nModel.num_hiddens = paras["hiddens"]
            if "layers" in paras:
                nModel.num_layers = paras["layers"]

            acc_list = []
            lr, batch_size, dropout, l1_reg, l2_reg = getGridPara(paras)
            for fold in range(K):
                data_train, label_train, data_val, label_val, train_subject_list= loader.getData(
                    K, fold)
                curModel = copy.deepcopy(nModel)
                launch(curModel, data_train, label_train, data_val, label_val, device, optimizer, categories, dropout,
                       batch_size, lr, l1_reg, l2_reg, mx_epoch,
                       loader.num_freq, NodeDAT, EmotionDL)
                acc_list.append(
                    (curModel.trainer.eval_acc_list, data_val.shape[0]))

            # calc the acc result under the given paras

            acc_result = 0
            tot_samples = 0
            argmax_epoch = -1
            if protocol == 'cv':
                for fold in range(K):
                    max_acc = 0
                    for ep in grid_epoch:
                        max_acc = max(max_acc, acc_list[fold][0][ep])
                    acc_result += max_acc*acc_list[fold][1]
                    tot_samples += acc_list[fold][1]
                assert (tot_samples == loader.data.shape[0])
                acc_result /= tot_samples
            else:  # fcv
                acc_ep = []
                for ep in grid_epoch:
                    acc_ep.append(0)
                    tot_samples = 0
                    for fold in range(K):
                        acc_ep[-1] += acc_list[fold][0][ep]*acc_list[fold][1]
                        tot_samples += acc_list[fold][1]
                    assert (tot_samples == loader.data.shape[0])
                    acc_ep[-1] /= tot_samples
                acc_result = max(acc_ep)
                argmax_epoch = np.argmax(np.array(acc_ep))+1
            result_list.append({"paras":paras, "acc_mean":acc_result, "argmax_epoch":argmax_epoch})
        best_dict = result_list[0]
        for i in range(1, len(result_list)):
            if result_list[i]["acc_mean"] > best_dict["acc_mean"]:
                best_dict = result_list[i]
        return copy.deepcopy(best_dict), result_list

    else:  # ncv
        if K_inner is None:
            raise ValueError(
                "K_inner must not be None when the protocol is set as 'ncv'.")

        out_acc_list = []
        for out_fold in range(K):
            data_train_and_val, label_train_and_val, data_test, label_test, train_and_val_subject_list = loader.getData(
                K, out_fold)
            # print("out: ",out_fold," : ",label_train_and_val.shape,label_test.shape)
            result_list = []
            for paras in grid_paras:
                nModel = copy.deepcopy(model)
                if "hiddens" in paras:
                    nModel.num_hiddens = paras["hiddens"]
                if "layers" in paras:
                    nModel.num_layers = paras["layers"]

                acc_list = []
                lr, batch_size, dropout, l1_reg, l2_reg = getGridPara(paras)

                for in_fold in range(K_inner):
                    nLoader = DataLoader(loader.protocol, data_train_and_val, label_train_and_val,
                                         loader.type, subject_id_list=train_and_val_subject_list, num_freq=loader.num_freq)
                    data_train, label_train, data_val, label_val, train_subject_list = nLoader.getData(
                        K_inner, in_fold)
                    # print(in_fold," : ",label_train.shape,label_val.shape)
                    curModel = copy.deepcopy(nModel)

                    launch(curModel, data_train, label_train, data_val, label_val, device, optimizer, categories, dropout,
                           batch_size, lr, l1_reg, l2_reg, mx_epoch,
                           loader.num_freq, NodeDAT, EmotionDL)

                    acc_list.append(
                        (curModel.trainer.eval_acc_list, data_val.shape[0]))

                    # calc the acc result under the given paras

                acc_result = 0
                tot_samples = 0
                argmax_epoch = -1
                acc_ep = []
                for ep in grid_epoch:
                    acc_ep.append(0)
                    tot_samples = 0
                    for fold in range(K_inner):
                        acc_ep[-1] += acc_list[fold][0][ep] * \
                            acc_list[fold][1]
                        tot_samples += acc_list[fold][1]
                    # print(tot_samples," and ",data_train_and_val.shape[0])
                    assert (tot_samples == data_train_and_val.shape[0])
                    acc_ep[-1] /= tot_samples
                acc_result = max(acc_ep)
                argmax_epoch = np.argmax(np.array(acc_ep))+1

                result_list.append({"paras":paras, "acc_mean":acc_result, "argmax_epoch":argmax_epoch})
            best_dict = result_list[0]
            for i in range(1, len(result_list)):
                if result_list[i]["acc_mean"] > best_dict["acc_mean"]:
                    best_dict = result_list[i]

            nModel = copy.deepcopy(model)
            paras = best_dict["paras"]
            fcv_epoch = best_dict["argmax_epoch"]
            if "hiddens" in paras:
                nModel.num_hiddens = paras["hiddens"]
            if "layers" in paras:
                nModel.num_layers = paras["layers"]

            lr, batch_size, dropout, l1_reg, l2_reg = getGridPara(paras)

            launch(curModel, data_train_and_val, label_train_and_val, data_test, label_test, device, optimizer, categories, dropout,
                   batch_size, lr, l1_reg, l2_reg, fcv_epoch,
                   loader.num_freq, NodeDAT, EmotionDL)
            out_acc_list.append(
                {"fold":out_fold,
                 "best_paras":best_dict["paras"],
                 "train_acc_mean":best_dict["acc_mean"],
                 "test_acc_mean":curModel.trainer.eval_acc_list[fcv_epoch-1], 
                 "test_num_samples":data_test.shape[0]})
            # get best_tuple via inner_K fold cross-validation
        mean_acc = 0
        tot_samples = 0
        for _ in out_acc_list:
            mean_acc += _["test_acc_mean"]*_["test_num_samples"]
            tot_samples += _["test_num_samples"]
        # print(tot_samples," && ",loader.data.shape[0])
        assert (tot_samples == loader.data.shape[0])
        mean_acc /= tot_samples
        return mean_acc, out_acc_list
