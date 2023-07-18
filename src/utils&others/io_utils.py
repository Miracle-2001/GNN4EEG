import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import shutil
import pickle
import random
import matplotlib.pyplot as plt

class DEDataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.FloatTensor(data) # n_samples * n_features
        self.label = torch.from_numpy(label)
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        one_seq = self.data[idx]
        one_label = self.label[idx]
        return one_seq, one_label


class EmotionDataset(Dataset):
    def __init__(self, data, label, timeLen, timeStep, n_segs, fs, transform=None):
        self.data = data.transpose() # n_samples * n_features
        self.timeLen = timeLen
        self.timeStep = timeStep
        self.n_segs = n_segs
        self.fs = fs
        self.transform = transform
        self.label = torch.from_numpy(label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # The sample should not be across different videos
        n_samples_remain_each = 30 - self.n_segs * self.timeStep
        one_seq = self.data[:, int((idx * self.timeStep + n_samples_remain_each * np.floor(idx / self.n_segs)) * self.fs):
                            int((idx * self.timeStep + self.timeLen + n_samples_remain_each * np.floor(idx / self.n_segs)) * self.fs)]
        one_label = self.label[idx]

        if self.transform:
            one_seq = self.transform(one_seq)

        one_seq = torch.FloatTensor(one_seq).unsqueeze(0)
        
        return one_seq, one_label


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.pkl'), 'w') as outfile:
            pickle.dump(args, outfile, default_flow_style=False)


class TrainSampler():
    def __init__(self, n_subs, n_times, batch_size, n_samples):
        self.n_per = int(np.sum(n_samples))
        self.n_subs = n_subs
        # Number of data points per session
        self.batch_size = batch_size
        self.n_samples_cum = np.concatenate((np.array([0]), np.cumsum(n_samples)))
        self.n_samples_per_trial = int(batch_size / len(n_samples))

        self.sub_pairs = []
        for i in range(self.n_subs):
            # j = i
            for j in range(i+1, self.n_subs):
                self.sub_pairs.append([i, j])
        random.shuffle(self.sub_pairs)
        self.n_times = n_times

    def __len__(self):
        return self.n_times * len(self.sub_pairs)

    def __iter__(self):
        for s in range(len(self.sub_pairs)):
            for t in range(self.n_times):
                [sub1, sub2] = self.sub_pairs[s]

                ind_abs = np.zeros(0)
                if self.batch_size < len(self.n_samples_cum)-1:
                    sel_vids = np.random.choice(np.arange(len(self.n_samples_cum)-1), self.batch_size)
                    for i in sel_vids:
                        ind_one = np.random.choice(np.arange(self.n_samples_cum[i], self.n_samples_cum[i+1]), 1, replace=False)
                        ind_abs = np.concatenate((ind_abs, ind_one))
                else:
                    for i in range(len(self.n_samples_cum)-2):
                        # np.random.seed(i)
                        ind_one = np.random.choice(np.arange(self.n_samples_cum[i], self.n_samples_cum[i+1]),
                                                   self.n_samples_per_trial, replace=False)
                        ind_abs = np.concatenate((ind_abs, ind_one))

                    i = len(self.n_samples_cum) - 2
                    # np.random.seed(i)
                    ind_one = np.random.choice(np.arange(self.n_samples_cum[i], self.n_samples_cum[i + 1]),
                                               int(self.batch_size - len(ind_abs)), replace=False)
                    ind_abs = np.concatenate((ind_abs, ind_one))
                    # print('ind abs length', len(ind_abs))

                assert len(ind_abs) == self.batch_size

                ind_this1 = ind_abs + self.n_per*sub1
                ind_this2 = ind_abs + self.n_per*sub2

                batch = torch.LongTensor(np.concatenate((ind_this1, ind_this2)))
                # print(batch)
                yield batch


class TrainSampler_sub():
    def __init__(self, n_subs_all, n_samples, batch_size=240, n_subs=10):
        self.n_subs = n_subs
        self.n_subs_all = n_subs_all
        # Number of data points per session
        self.batch_size = batch_size
        self.n_samples_sum = int(np.sum(n_samples))
        self.n_samples_per_sub = int(batch_size / n_subs)

        ind_all = np.zeros((n_subs_all, self.n_samples_sum))
        for i in range(n_subs_all):
            tmp = np.arange(self.n_samples_sum) + i * self.n_samples_sum
            np.random.shuffle(tmp)
            ind_all[i,:] = tmp
        np.random.shuffle(ind_all)
        self.ind_all = ind_all

        # self.n_times = int(n_subs_all * self.n_samples_sum // batch_size)
        self.n_times_sub = int(n_subs_all / n_subs)
        self.n_times_vid = int(self.n_samples_sum / self.n_samples_per_sub)


    def __len__(self):
        return self.n_times_sub * self.n_times_vid

    def __iter__(self):
        for i in range(self.n_times_vid):
            for j in range(self.n_times_sub):
                ind_sel = self.ind_all[j*self.n_subs: (j+1)*self.n_subs, self.n_samples_per_sub*i: self.n_samples_per_sub*(i+1)]
                ind_sel = ind_sel.reshape(-1)
                batch = torch.LongTensor(ind_sel)
                # print(batch)
                yield batch

class TrainSampler_video():
    def __init__(self, n_subs, n_times, batch_size, n_samples):
        self.n_per = int(np.sum(n_samples))
        self.n_subs = n_subs
        # Number of data points per session
        self.batch_size = batch_size
        self.n_samples_cum = np.concatenate((np.array([0]), np.cumsum(n_samples)))
        self.n_samples_per_trial = int(batch_size / len(n_samples))
        self.n_times = n_times
        self.subs = np.arange(self.n_subs)
        random.shuffle(self.subs)

    def __len__(self):
        return self.n_times * len(self.n_subs)

    def __iter__(self):
        for s in range(len(self.subs)):
            sub = self.subs[s]
            for t in range(self.n_times):
                ind_abs = np.zeros(0)

                for i in range(len(self.n_samples_cum)-2):
                    # np.random.seed(i)
                    ind_one = np.random.choice(np.arange(self.n_samples_cum[i], self.n_samples_cum[i+1]),
                                                2, replace=False)
                    ind_abs = np.concatenate((ind_abs, ind_one))

                i = len(self.n_samples_cum) - 2
                # np.random.seed(i)
                ind_one = np.random.choice(np.arange(self.n_samples_cum[i], self.n_samples_cum[i + 1]),
                                            2, replace=False)
                ind_abs = np.concatenate((ind_abs, ind_one))
                # print('ind abs length', len(ind_abs))

                assert len(ind_abs) == self.batch_size * 2

                ind_this = ind_abs + self.n_per*sub
                ind_this1 = ind_this[list(np.arange(0,len(ind_this),2))]
                ind_this2 = ind_this[list(np.arange(1,len(ind_this),2))]

                batch = torch.LongTensor(np.concatenate((ind_this1, ind_this2)))
                # print(batch)
                yield batch


def smooth_moving_average(data, filtLen):
    # data: [n_channs, n_points]
    if filtLen == 1:
        data_smoothed = data
    else:
        data_smoothed = np.zeros(data.shape)
        for i in range(data.shape[1]):
            if i < filtLen // 2:
                data_smoothed[:, i] = np.mean(data[:, :i + filtLen // 2], axis=1)
            elif i > data.shape[1] - filtLen // 2:
                data_smoothed[:, i] = np.mean(data[:, i - filtLen // 2:], axis=1)
            else:
                data_smoothed[:, i] = np.mean(data[:, i - filtLen // 2: i + filtLen // 2], axis=1)
    return data_smoothed
