import argparse
import numpy as np
import pandas as pd
import torch
import os
import scipy.io as sio
import random
import time
import joblib
import copy
import json
from benchmarks import *
from evaluate import draw_results
from multiprocessing.pool import Pool
from GNN4EEG.ge.load_data import load_srt_de
from manager_torch import GPUManager
# from pytorch_lightning import seed_everything


class Results(object):
    def __init__(self, args):
        self.val_acc_folds = np.zeros(args.n_folds)
        self.subjects_score = np.zeros(args.n_subs)
        self.acc_fold_list = [0]*10
        if args.subjects_type == 'intra':
            self.subjects_results_ = np.zeros(
                (args.n_subs, args.sec * args.n_vids))
            self.label_val_ = np.zeros((args.n_subs, args.sec * args.n_vids))


def init():
    parser = argparse.ArgumentParser(
        description='Finetune the pretrained model for EEG emotion recognition')
    # basic config
    parser.add_argument('--randSeed', default=7, type=int,
                        help='random seed')
    parser.add_argument('--training-fold', default='all', type=str,
                        help='the number of training fold, 0~9')
    parser.add_argument('--subjects-type', default='inter', type=str,
                        help='inter or intra subject', choices=['inter', 'intra'])
    parser.add_argument('--valid-method', default='10-folds',
                        type=str, help='the valid method, 10-folds or leave one out')
    parser.add_argument('--n-vids', default=24,
                        type=int, help='number of video')
    parser.add_argument('--multi-thread', default='True',
                        type=str, help='use MultiThread or not')
    parser.add_argument('--device-list', default='[0,3,5,6,7]',
                        type=str, help='available device list for multi thread training')
    parser.add_argument('--auto_device_count', default=5,
                        type=int, help='number of GPU auto find and use at the same time')
    parser.add_argument('--device-index', default=-1,
                        type=int, help='Device index.')
    parser.add_argument('--cpu', default='False',
                        type=str, help='Cpu use or not.')
    parser.add_argument('--model', default='RGNN', type=str, help='model type',
                        choices=['SVM', 'DGCNN', 'RGNN', 'Het', 'MLP', 'SparseDGCNN'])
    parser.add_argument('--early_stop', default=20,
                        type=int, help='early stop epochs')
    # parser.add_argument('--small', default='False',
    #                     type=str, help='whether use smaller dataset to find best lr')
    # parser.add_argument('--step',default=0.00001,
    #                     type=float,help='step size of lr between each epoch (only used when small==True)')
    # # data type and benchmark
    parser.add_argument('--per_person', default='False', type=str,
                        help='whether per person dependent.')
    parser.add_argument('--band', default=5, type=int,
                        help='number of bands used.')
    parser.add_argument('--benchmark', default='old', type=str,
                        help='benchmark type', choices=['old', 'new'])
    parser.add_argument("--num_nodes", default=30,
                        type=int, help='num of nodes')
    # GNN training
    parser.add_argument('--num_epoch', default=100, type=int,
                        help='number of epoches while training')
    parser.add_argument("--lr", default=0.001,
                        type=float, help='batch_size')
    parser.add_argument("--l1_reg", default=0, type=float, help='l1_reg')
    parser.add_argument("--l2_reg", default=0, type=float, help='l2_reg')
    parser.add_argument("--batch_size", default=256,
                        type=int, help='batch_size')
    parser.add_argument("--dropout", default=0.5,
                        type=float, help='dropout rate')
    parser.add_argument("--num_hiddens", default=800,
                        type=int, help='num of hiddens in W')
    # RGNN or DGCNN
    parser.add_argument('--NodeDAT', default='False',
                        type=str, help='use NodeDAT or not')
    parser.add_argument('--EmotionDL', default='False',
                        type=str, help='use EmotionDL or not')
    parser.add_argument("--num_layers", default=2,
                        type=int, help='exp num of S')
    # Het
    parser.add_argument("--num_time", default=250,
                        type=int, help='num of time steps')
    # SVM
    parser.add_argument('--C', default=0,
                        type=float, help='soft interval of SVM.')

    args = parser.parse_args()

    if args.n_vids == 28:
        args.label_type = 'cls9'
        args.num_classes = 9
    elif args.n_vids == 24:
        args.label_type = 'cls2'
        args.num_classes = 2

    args.EmotionDL = True if args.EmotionDL == 'True' and args.num_classes == 9 else False
    args.NodeDAT = True if args.NodeDAT == 'True' and args.subjects_type == 'inter' else False
    args.multi_thread = True if args.multi_thread == 'True' else False
    args.cpu = True if args.cpu == 'True' else False
    args.per_person = True if args.per_person == 'True' else False
    # args.small=True if args.small=='True' else False

    args.device_list = json.loads(args.device_list)
    if args.multi_thread:
        torch.multiprocessing.set_start_method('spawn')  # ,force=True)

    torch.manual_seed(args.randSeed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(args.randSeed)  # 为当前GPU设置随机种子
    # if you are using multi-GPU，为所有GPU设置随机种子
    torch.cuda.manual_seed_all(args.randSeed)
    np.random.seed(args.randSeed)  # Numpy module.
    random.seed(args.randSeed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # seed_everything(args.randSeed)

    if args.valid_method == '10-folds':
        args.n_folds = 10
    elif args.valid_method == 'loo':
        args.n_folds = args.n_subs

    args.n_subs = 123
    args.n_per = round(args.n_subs / args.n_folds)
    args.sec = 30

    # if args.benchmark=='new':
    if args.subjects_type=='intra' and args.n_vids==24:
        args.l1_reg=0.001
        args.l2_reg=0.001
    elif args.subjects_type=='inter' and args.n_vids==28:
        args.l1_reg=0.005
        args.l2_reg=0.005
    else:
        args.l1_reg=0.003
        args.l2_reg=0.003

    if args.model == 'Het':
        data_root_dir = './Data_freqAndTemp/'+str(args.band) + \
            'bands'+'/smooth_' + str(args.n_vids)
    else:
        data_root_dir = './Data_freqOnly/'+str(args.band) + \
            'bands'+'/smooth_' + str(args.n_vids)
    args.data_root_dir = data_root_dir

    now_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    args.now_time = now_time

    model_path = f'./result_benchmark_{args.benchmark}/{args.model}_analysis/{args.model}_result_{now_time}_{args.subjects_type}_{args.n_vids}'
    # code_path=f'./result/{args.model}_analysis'
    if os.path.exists(model_path) == False:
        os.makedirs(model_path)
    args.model_path = model_path
    # args.code_path=code_path

    json.dump(vars(args), open(model_path +
                               f'/args_{now_time}.json', 'w'))

    return args


def main(args):
    if args.benchmark == 'old':
        return benchmark_old(args)
    elif args.benchmark == 'new':
        return benchmark_new(args)


def print_error(value):
    print("multi_thread_error:", value)


def print_results(args, result):
    subjects_score = result.subjects_score
    if args.subjects_type == 'intra':
        subjects_results_ = result.subjects_results_
        label_val_ = result.label_val_

    print('acc mean: %.3f, std: %.3f' %
          (np.mean(result.acc_fold_list), np.std(result.acc_fold_list)))
    if args.subjects_type == 'intra':
        subjects_score = [np.sum(subjects_results_[i, :] == label_val_[
            i, :]) / subjects_results_.shape[1] for i in range(0, args.n_subs)]
        subjects_score = np.array(subjects_score).reshape(args.n_subs, -1)
    pd.DataFrame(subjects_score).to_csv(
        os.path.join(args.model_path,
                     'subject_%s_vids_%s_valid_%s.csv' % (args.subjects_type, str(args.n_vids), args.valid_method)))


if __name__ == '__main__':
    print("into main function")
    args = init()
    # It work only when valid method is 10-folds
    if args.training_fold == 'all':
        fold_list = np.arange(0, args.n_folds)
    else:
        # training_fold = 0~9
        fold_list = [int(args.training_fold)]

    result = Results(args)
    buc = []
    auto_choice_mode = 1

    

    if args.multi_thread:
        if args.auto_device_count > 0:
            args.device_list = [0 for i in range(args.auto_device_count)]
        k_max = int(len(fold_list) / len(args.device_list)) + 1
        for k in range(0, k_max):
            pool = Pool(len(args.device_list) + 1)
            item_list = []
            gm = GPUManager()
            for i in range(k * len(args.device_list), k * len(args.device_list) + len(args.device_list)):
                if i >= len(fold_list):
                    continue
                args_new = copy.deepcopy(args)
                if args.auto_device_count > 0:
                    args.device_list[(
                        i - k * len(args.device_list)) % len(args.device_list)] = gm.auto_choice(mode=auto_choice_mode)

                args_new.device_index = args.device_list[(
                    i - k * len(args.device_list)) % len(args.device_list)]

                args_new.thread_id = (
                    i - k * len(args.device_list)) % len(args.device_list)
                print("id ", args_new.thread_id,
                      "index ", args_new.device_index)
                args_new.fold_list = [fold_list[i]]

                item_list.append(pool.apply_async(main, args=(
                    args_new,), error_callback=print_error))
            pool.close()
            pool.join()
            for item in item_list:
                buc.append(item.get())
            time.sleep(10)
    else:
        gm = GPUManager()
        args.thread_id = 0
        # args.cuda = args.device_index
        # args.fold_list = fold_list
        # buc.append(main(args))
        if args.auto_device_count > 0:
            args.device_index = gm.auto_choice(mode=auto_choice_mode)
        for i in fold_list:
            args_new = copy.deepcopy(args)
            args_new.fold_list = [i]
            buc.append(main(args_new))

    para_mean_result_dict = {}
    if args.subjects_type == 'inter':
        for tup in buc:
            result.acc_fold_list[tup[0]] = tup[1]
            result.subjects_score[tup[2]] = tup[3]
    elif args.subjects_type == 'intra':
        for tup in buc:
            result.acc_fold_list[tup[0]] = tup[1]
            result.subjects_results_[:, tup[2]] = tup[3]
            result.label_val_[:, tup[2]] = tup[4]
    if args.benchmark == 'new':
        for tup in buc:
            if len(para_mean_result_dict) == 0:
                para_mean_result_dict = tup[-1]
            else:
                for k, v in tup[-1].items():
                    para_mean_result_dict[k]['now_best_acc_train'] += v['now_best_acc_train']
                    para_mean_result_dict[k]['now_best_acc_val'] += v['now_best_acc_val']
        for k in para_mean_result_dict.keys():
            para_mean_result_dict[k]['now_best_acc_train'] /= len(
                para_mean_result_dict)
            para_mean_result_dict[k]['now_best_acc_val'] /= len(
                para_mean_result_dict)
        json.dump({
            "para_mean_result_dict": para_mean_result_dict
        }, open(os.path.join(args.model_path, 'para_mean_result_dict.json'), 'w'))

    print_results(args, result)
    draw_results(args)
    # if args.small==True:
    #     find_best_lr(args)
