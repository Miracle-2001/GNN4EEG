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
from load_data import load_srt_de
from Models.rgnn import RGNNTrainer
from Models.dgcnn import DGCNNTrainer
from Models.mlp import MLPTrainer
from Models.het import HetTrainer
from Models.sparseDgcnn import SparseDGCNNTrainer
from Models.md_utils import *
from sklearn.svm import LinearSVC
from load_data import load_srt_de


def data_prepare(args, fold):
    data_root_dir = args.data_root_dir
    fold_list = args.fold_list
    n_subs = args.n_subs
    n_per = args.n_per
    # band_used = args.band

    data_dir = os.path.join(data_root_dir, 'de_lds_fold%d.mat' % (fold))
    data = hdf5.loadmat(data_dir)['de_lds']
    data, label_repeat, n_samples = load_srt_de(
        data, True, False, 1, args.label_type)
    feature_shape = int(data.shape[-1]/30)
    # label shape: 720 or 840   720=24*30. 840=28*30   24/28 videos and 30 samples are generated by each video
    # data shape: (123, 720, 120 or 150 or 255*30)  720=24*30  120=4*30  30:channel num  4: band num

    val_sub = None
    val_list = None
    if args.subjects_type == 'inter':  # cross-subject / subject-independent
        if fold < args.n_folds - 1:
            val_sub = np.arange(n_per * fold, n_per * (fold + 1))
        else:
            val_sub = np.arange(n_per * fold, n_subs)
        train_sub = np.array(list(set(np.arange(n_subs)) - set(val_sub)))
        # here, we need to transpose the data making each sample (30,4)
        data_train = data[list(train_sub), :, :].reshape(-1,
                                                         feature_shape, 30).transpose([0, 2, 1])
        data_val = data[list(val_sub), :, :].reshape(-1,
                                                     feature_shape, 30).transpose([0, 2, 1])
        label_train = np.tile(label_repeat, len(train_sub))
        label_val = np.tile(label_repeat, len(val_sub))

    elif args.subjects_type == 'intra':  # subject-dependent
        val_seconds = 30 / args.n_folds
        train_seconds = 30 - val_seconds
        data_list = np.arange(0, len(label_repeat))
        # pick out the val sec
        val_list_start = np.arange(
            0, len(label_repeat), 30) + int(val_seconds * fold)
        val_list = val_list_start.copy()
        for sec in range(1, int(val_seconds)):
            val_list = np.concatenate(
                (val_list, val_list_start + sec)).astype(int)
        train_list = np.array(
            list(set(data_list) - set(val_list))).astype(int)
        # here, we need to transpose the data making each sample (30,4)
        data_train = data[:, list(train_list),
                          :].reshape(-1, feature_shape, 30).transpose([0, 2, 1])
        data_val = data[:, list(val_list), :].reshape(-1,
                                                      feature_shape, 30).transpose([0, 2, 1])

        # label_repeat repeated n_subs number of times
        label_train = np.tile(np.array(label_repeat)[train_list], n_subs)
        label_val = np.tile(np.array(label_repeat)[val_list], n_subs)
        # (123*648,30,4) and (123*648,)
        # (123*72,30,4) and (123*72,)
    return data_train, label_train, data_val, label_val, val_sub, val_list

def het_get_pre_mat(args, fold):
    data_root_dir = args.data_root_dir
    fold_list = args.fold_list
    n_subs = args.n_subs
    n_per = args.n_per
    # band_used = args.band
    data_dir = os.path.join(
        data_root_dir, 'het_mat_preprocessed%d.mat' % (fold))
    mat = hdf5.loadmat(data_dir)['adj_mat']
    print('adj_mat shape ', mat.shape)  # (123,720 or 840,30,30)
    val_sub = None
    val_list = None
    if args.subjects_type == 'inter':  # cross-subject / subject-independent
        if fold < args.n_folds - 1:
            val_sub = np.arange(n_per * fold, n_per * (fold + 1))
        else:
            val_sub = np.arange(n_per * fold, n_subs)
        train_sub = np.array(list(set(np.arange(n_subs)) - set(val_sub)))
        # here, we need to transpose the data making each sample (30,30)
        mat_train = mat[list(train_sub), :, :, :].reshape(-1,
                                                          30, 30)
        mat_val = mat[list(val_sub), :, :, :].reshape(-1,
                                                      30, 30)

    elif args.subjects_type == 'intra':  # subject-dependent
        val_seconds = 30 / args.n_folds
        train_seconds = 30 - val_seconds
        lens = 30*args.n_vids
        data_list = np.arange(0, lens)
        # pick out the val sec
        val_list_start = np.arange(
            0, lens, 30) + int(val_seconds * fold)
        val_list = val_list_start.copy()
        for sec in range(1, int(val_seconds)):
            val_list = np.concatenate(
                (val_list, val_list_start + sec)).astype(int)
        train_list = np.array(
            list(set(data_list) - set(val_list))).astype(int)
        # here, we need to transpose the data making each sample (30,30)
        mat_train = mat[:, list(train_list),
                        :, :].reshape(-1, 30, 30)
        mat_val = mat[:, list(val_list), :, :].reshape(-1,
                                                       30, 30)

    return mat_train, mat_val


def get_trainer(args):
    if args.cpu==True:
        device_using = torch.device('cpu')
    else:
        device_using = torch.device('cuda', args.device_index)
    if args.model == 'RGNN':
        # start initialize RGNN
        edge_index, edge_weight = get_edge_weight(args)
        trainer = RGNNTrainer(edge_index=edge_index, edge_weight=edge_weight, num_classes=args.num_classes, device=device_using,
                              domain_adaptation=args.NodeDAT, distribution_learning=args.EmotionDL,
                              num_hiddens=args.num_hiddens, num_layers=args.num_layers, dropout=args.dropout,
                              batch_size=args.batch_size, lr=args.lr, l1_reg=args.l1_reg, num_epoch=args.num_epoch)
    elif args.model == 'DGCNN':
        edge_index, edge_weight = get_edge_weight(args)
        trainer = DGCNNTrainer(edge_index=edge_index, edge_weight=edge_weight, num_classes=args.num_classes, device=device_using,
                               num_hiddens=args.num_hiddens, num_layers=args.num_layers, dropout=args.dropout,
                               batch_size=args.batch_size, lr=args.lr, l2_reg=args.l2_reg, num_epoch=args.num_epoch)
    elif args.model == 'SparseDGCNN':
        edge_index, edge_weight = get_edge_weight(args)
        trainer = SparseDGCNNTrainer(edge_index=edge_index, edge_weight=edge_weight, num_classes=args.num_classes, device=device_using,
                                     num_hiddens=args.num_hiddens, num_layers=args.num_layers, dropout=args.dropout,
                                     batch_size=args.batch_size, lr=args.lr, l1_reg=args.l1_reg, l2_reg=args.l2_reg, num_epoch=args.num_epoch)
    elif args.model == 'Het':
        trainer = HetTrainer(num_nodes=args.num_nodes, num_classes=args.num_classes, device=device_using, num_hiddens=args.num_hiddens,
                             dropout=args.dropout, batch_size=args.batch_size, lr=args.lr, l1_reg=args.l1_reg, num_epoch=args.num_epoch)
    elif args.model == 'SVM':
        # start initialize SVM
        trainer = LinearSVC(random_state=args.randSeed, C=args.C)
    elif args.model == 'MLP':
        trainer = MLPTrainer(num_nodes=args.num_nodes, num_hiddens=args.num_hiddens, num_classes=args.num_classes, num_layers=args.num_layers, batch_size=args.batch_size, num_epoch=args.num_epoch,
                             lr=args.lr, l1_reg=args.l1_reg, l2_reg=args.l2_reg, dropout=args.dropout, early_stop=args.early_stop, device=device_using)
    else:
        raise NotImplementedError(
            'Other models have not been implemented yet.')
    return trainer


def benchmark_old(args):
    data_root_dir = args.data_root_dir
    fold_list = args.fold_list
    n_subs = args.n_subs
    n_per = args.n_per
    for fold in fold_list:
        print('fold number: ', fold)

        data_train, label_train, data_val, label_val, val_sub, val_list = data_prepare(
            args, fold)

        start_time = time.time()
        # ----- train ----
        model_name = 'subject_%s_vids_%s_fold_%s_valid_%s.joblib' % (
            args.subjects_type, str(args.n_vids), str(fold), args.valid_method)

        trainer = get_trainer(args)
        if args.model == 'SVM':
            trainer.fit(data_train, label_train)
            joblib.dump(trainer, os.path.join(args.model_path, model_name))
            preds_train = trainer.predict(data_train)
            preds_val = trainer.predict(data_val)
        elif args.model == 'Het':
            mat_train, mat_val = het_get_pre_mat(args, fold)
            preds_train,preds_val= trainer.train(data_train, label_train, data_val,
                          label_val, fold, args.model_path, model_name,num_freq=args.band,mat_train=mat_train,mat_val=mat_val)
            # preds_train = trainer.predict(data_train, mat_train)
            # preds_val = trainer.predict(data_val, mat_val)
        else:
            preds_train,preds_val=trainer.train(data_train, label_train, data_val,
                          label_val, fold, args.model_path, model_name)
            # preds_train = trainer.predict(data_train)
            # preds_val = trainer.predict(data_val)

        end_time = time.time()
        # print("numofpreds_train ",preds_train.shape)
        train_acc = np.sum(preds_train == label_train) / len(label_train)
        val_acc = np.sum(preds_val == label_val) / len(label_val)

        print('thread id:', args.thread_id, ' fold:', fold, 'train acc:', train_acc,
              'val acc:', val_acc, 'time consumed:', end_time - start_time)

        subjects_results = preds_val
        if args.subjects_type == 'inter':
            subjects_results = subjects_results.reshape(
                val_sub.shape[0], -1)
            label_val = np.array(label_val).reshape(val_sub.shape[0], -1)
            val_result = [np.sum(subjects_results[i, :] == label_val[i, :]) / subjects_results.shape[1] for i in
                          range(0, val_sub.shape[0])]
            # result.subjects_score[val_sub] = val_result
            return (fold, val_acc, val_sub, val_result)
        elif args.subjects_type == 'intra':
            subjects_results = subjects_results.reshape(n_subs, -1)
            label_val = np.array(label_val).reshape(n_subs, -1)
            # result.subjects_results_[:, val_list] = subjects_results
            # result.label_val_[:, val_list] = label_val
            return (fold, val_acc, val_list, subjects_results, label_val)


def getRange(args):
    file = open('./gridSearchConfig.json', 'r')
    task = str(args.subjects_type)+str(args.num_classes)

    # rangeDict = json.load(file)[task][args.model]
    rangeDict={"lr":[args.lr],"num_hiddens":[args.num_hiddens],"l1_reg":[args.l1_reg],"l2_reg":[args.l2_reg]}
    return rangeDict


def train_val_split(args, fold, sub_fold, data_train_and_val, label_train_and_val, test_sub, test_list,het_train_and_val=None):
    if args.label_type == 'cls2':
        n_vids = 24
    elif args.label_type == 'cls9':
        n_vids = 28


    feature_shape = int(data_train_and_val.shape[-1])
    if args.subjects_type == 'intra':
        data_list = np.arange(0, n_vids*27)
        val_list_start = np.arange(
            0, n_vids*27, 27) + int(9 * sub_fold)
        val_list = val_list_start.copy()
        for sec in range(1, 9):  # 1~8
            val_list = np.concatenate(
                (val_list, val_list_start + sec)).astype(int)
        train_list = np.array(
            list(set(data_list) - set(val_list))).astype(int)
        # here, we need to transpose the data making each sample (30,4)
        data_train = data_train_and_val.reshape(123, -1, 30, feature_shape)[:, list(train_list),
                                                                        :, :].reshape(-1, 30, feature_shape)
        data_val = data_train_and_val.reshape(123, -1, 30, feature_shape)[:, list(val_list),
                                                                      :, :].reshape(-1, 30, feature_shape)
        if het_train_and_val is not None:
            data_train_het=het_train_and_val.reshape(123, -1, 30, 30)[:, list(train_list),
                                                                        :, :].reshape(-1, 30, 30)
            data_val_het=het_train_and_val.reshape(123, -1, 30, 30)[:, list(val_list),
                                                                      :, :].reshape(-1, 30, 30)

        label_train = label_train_and_val.reshape(
            123, -1)[:, list(train_list)].reshape(-1)
        label_val = label_train_and_val.reshape(
            123, -1)[:, list(val_list)].reshape(-1)

    elif args.subjects_type == 'inter':
        print(data_train_and_val.shape)
        print(het_train_and_val.shape)
        data_list = np.arange(0, data_train_and_val.shape[0]/(args.n_vids*30))
        start = sub_fold*36
        end = start+36
        if fold != 9 and sub_fold == 2:
            end = 111
        val_list = np.arange(start, end)
        train_list = np.array(
            list(set(data_list) - set(val_list))).astype(int)
        data_train = data_train_and_val.reshape(-1, args.n_vids*30, 30, feature_shape)[
            list(train_list), :, :, :].reshape(-1, 30, feature_shape)
        data_val = data_train_and_val.reshape(-1, args.n_vids*30, 30, feature_shape)[
            list(val_list), :, :, :].reshape(-1, 30, feature_shape)
        if het_train_and_val is not None:
            data_train_het=het_train_and_val.reshape(-1, args.n_vids*30, 30, 30)[
                                    list(train_list), :, :, :].reshape(-1, 30, 30)
            data_val_het=het_train_and_val.reshape(-1, args.n_vids*30, 30, 30)[
                                    list(val_list), :, :, :].reshape(-1, 30, 30)

        label_train = label_train_and_val.reshape(-1, args.n_vids*30)[
            list(train_list), :].reshape(-1)
        label_val = label_train_and_val.reshape(-1, args.n_vids*30)[
            list(val_list), :].reshape(-1)

    if het_train_and_val is not None:
        return data_train, label_train, data_val, label_val,data_train_het,data_val_het
    return data_train, label_train, data_val, label_val


def benchmark_new(args):
    data_root_dir = args.data_root_dir
    fold_list = args.fold_list
    n_subs = args.n_subs
    n_per = args.n_per
    band_used = args.band
    rangeDict = getRange(args)
    newargs = copy.deepcopy(args)
    for fold in fold_list:
        print('fold number: ', fold)
        now_fold_dir = os.path.join(args.model_path, 'subject_%s_vids_%s_fold_%s_valid_%s' % (
            args.subjects_type, str(args.n_vids), str(fold), args.valid_method))
        os.makedirs(now_fold_dir)

        data_train_and_val, label_train_and_val, data_test, label_test, test_sub, test_list = data_prepare(
            args, fold)
        
        if args.model=='Het':
            mat_train_and_val, mat_test = het_get_pre_mat(args, fold)
            print("mtrain ", mat_train_and_val.shape)
            print("mtest ",mat_test.shape)
            
        num_train_and_val = data_train_and_val.shape[0]
        num_test = data_test.shape[0]

        para_result_dict = {}
        best_para_dict = {}
        
        best_para_dict.update(
            {'lr': 0, 'num_hiddens': 0, 'l1_reg': 0, 'l2_reg': 0, 'num_epoch': 0})
        best_acc = {"val": 0, "train": 0}
        count = 0
        for lr, num_hiddens in zip(rangeDict["lr"], rangeDict["num_hiddens"]):
            for l1_reg, l2_reg in zip(rangeDict["l1_reg"], rangeDict["l2_reg"]):
                start_time0 = time.time()
                newargs.lr = lr
                newargs.num_hiddens = num_hiddens
                newargs.l1_reg = l1_reg
                newargs.l2_reg = l2_reg

                now_para_dir = os.path.join(
                    now_fold_dir, f'lr={lr}_num_hiddens={num_hiddens}_l1_reg={l1_reg}_l2_reg={l2_reg}')
                os.makedirs(now_para_dir)
                mean_acc_list = {'val': [0 for i in range(args.num_epoch)], 'train': [
                    0 for i in range(args.num_epoch)]}

                for sub_fold in range(3):
                    model_name = 'fold_%s_subfold_%s.joblib' % (
                        str(fold), str(sub_fold))

                    if args.model=="Het":
                        data_train, label_train, data_val, label_val ,mat_train,mat_val= train_val_split(
                            args, fold, sub_fold, data_train_and_val, label_train_and_val, test_sub, test_list,mat_train_and_val)
                    else:
                        data_train, label_train, data_val, label_val = train_val_split(
                            args, fold, sub_fold, data_train_and_val, label_train_and_val, test_sub, test_list)
                        
                    trainer = get_trainer(newargs)

                    start_time = time.time()

                    if args.model=="Het":
                        trainer.train(data_train, label_train, data_val,
                          label_val, sub_fold, now_para_dir, model_name,num_freq=args.band,mat_train=mat_train,mat_val=mat_val,reload=False,nd_predict=False)
                    else:
                        trainer.train(data_train, label_train, data_val,
                                        label_val, sub_fold, now_para_dir, model_name,reload=False,nd_predict=False)
                        
                    jfile = open(now_para_dir + "/" +
                                    model_name.split(".")[0] + '_acc_and_loss.json', 'r')
                    jdict = json.load(jfile)
                    eval_num_correct_list = jdict['eval_num_correct_list']
                    train_num_correct_list = jdict['train_num_correct_list']
                    for i in range(args.num_epoch):
                        mean_acc_list['val'][i] += eval_num_correct_list[i]
                        mean_acc_list['train'][i] += train_num_correct_list[i]
                    end_time = time.time()

                    # print("numofpreds_train ",preds_train.shape)
                    print('thread id:', args.thread_id, ' fold:', fold, 'sub_fold:', sub_fold, 'lr:', lr, 'num_hiddens:', num_hiddens, 'l1_reg:', l1_reg,
                            'l2_reg:', l2_reg, 'best_acc:', jdict['best_acc'], 'best_epoch:', jdict['best_epoch'], 'time consumed:', end_time - start_time)

                now_best_epoch = 0
                now_best_acc = {'val': 0, 'train': 0}
                for i in range(args.num_epoch):
                    mean_acc_list['val'][i] /= num_train_and_val
                    mean_acc_list['train'][i] /= 2*num_train_and_val
                    if mean_acc_list['val'][i] > now_best_acc['val']:
                        now_best_acc['val'] = mean_acc_list['val'][i]
                        now_best_acc['train'] = mean_acc_list['train'][i]
                        now_best_epoch = i

                para_result_dict.update({count: {"lr": lr, "num_hiddens": num_hiddens, "l1_reg": l1_reg, "l2_reg": l2_reg,
                                        "now_best_acc_train": now_best_acc['train'], "now_best_acc_val": now_best_acc['val'], "now_best_epoch": now_best_epoch}})
                count += 1
                json.dump({'fold': int(fold),
                            'now_best_train_acc': now_best_acc['train'],
                            'now_best_val_acc': now_best_acc['val'],
                            "now_best_epoch": now_best_epoch,
                            "lr": lr, 
                            "num_hiddens": num_hiddens, 
                            "l1_reg": l1_reg, 
                            "l2_reg": l2_reg,
                            'time consumed': end_time - start_time
                            }, open(now_para_dir +
                                    f'/fold_{fold}_mean_acc_and_loss.json', 'w'))
                if now_best_acc["val"] > best_acc["val"]:
                    best_acc["val"] = now_best_acc["val"]
                    best_acc["train"] = now_best_acc["train"]
                    best_para_dict.update({"lr": lr, 
                            "num_hiddens": num_hiddens, 
                            "l1_reg": l1_reg, 
                            "l2_reg": l2_reg,
                            "num_epoch":now_best_epoch
                            })

        print(f'fold {fold} chooses para: ', best_para_dict, ' best_acc val: ',
                best_acc['val'], ' best_acc train: ', best_acc['train'])
        
        newargs.lr=best_para_dict['lr']
        newargs.num_hiddens=best_para_dict['num_hiddens']
        newargs.l1_reg=best_para_dict['l1_reg']
        newargs.l2_reg=best_para_dict['l2_reg']
        newargs.num_epoch=best_para_dict['num_epoch']+1  #to get the number of total num_epoch should additionally add 1

        trainer = get_trainer(newargs)
        start_time = time.time()
        # all of the train_and_val
        model_name = 'fold_%s_model.joblib' % (
                        str(fold))
        

        if args.model=="Het":
            preds_train_and_val,preds_test = trainer.train(data_train_and_val, label_train_and_val, data_test,
                label_test, fold, now_fold_dir, model_name,num_freq=args.band,mat_train=mat_train_and_val,mat_val=mat_test,reload=False)
        else:
            preds_train_and_val,preds_test = trainer.train(data_train_and_val, label_train_and_val, data_test,
                                    label_test, fold, now_fold_dir, model_name,reload=False)
        
        
        # preds_train_and_val = trainer.predict(data_train_and_val)
        # preds_test = trainer.predict(data_test)
        end_time = time.time()
        train_and_val_acc = np.sum(
            preds_train_and_val == label_train_and_val) / len(label_train_and_val)
        test_acc = np.sum(preds_test == label_test) / len(label_test)
        print('--final test acc--  thread id:', args.thread_id, ' fold:', fold, 'train acc:', train_and_val_acc,
              'test acc:', test_acc, 'time consumed:', end_time - start_time)
        json.dump({'fold': int(fold),
                   'train_and_val_acc': train_and_val_acc,
                   'test_acc': test_acc,
                   'best_val_acc': best_acc['val'],
                   'best_para_dict': best_para_dict,
                   'para_result_dict': para_result_dict,
                   'time consumed': end_time - start_time
                   }, open(now_fold_dir +
                           f'/fold_{fold}_acc_and_loss.json', 'w'))

        subjects_results = preds_test
        if args.subjects_type == 'inter':
            subjects_results = subjects_results.reshape(
                test_sub.shape[0], -1)
            label_test = np.array(label_test).reshape(test_sub.shape[0], -1)
            test_result = [np.sum(subjects_results[i, :] == label_test[i, :]) / subjects_results.shape[1] for i in
                           range(0, test_sub.shape[0])]
            return (fold, test_acc, test_sub, test_result,para_result_dict)
        elif args.subjects_type == 'intra':
            subjects_results = subjects_results.reshape(n_subs, -1)
            label_test = np.array(label_test).reshape(n_subs, -1)
            return (fold, test_acc, test_list, subjects_results, label_test,para_result_dict)
    pass