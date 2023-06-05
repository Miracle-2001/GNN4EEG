import numpy as np
import scipy.io as sio
from io_utils import smooth_moving_average
import os
# import h5py

def load_srt_raw_newPre(data_dir, timeLen, timeStep, fs, channel_norm, time_norm, label_type):
    n_channs = 30
    n_points = 7500
    data_len = fs * timeLen
    n_segs = int((n_points/fs - timeLen) / timeStep + 1)
    print('n_segs:', n_segs)

    f = h5py.File(data_dir)
    for k, v in f.items():
        data = np.array(v)
    data = np.transpose(data, (3,2,1,0))
    data = data[np.concatenate((np.arange(36), np.arange(37,80))), :, :, :]
    n_subs = data.shape[0]
    # data = sio.loadmat(os.path.join(data_dir, 'data_all_prepared.mat'))['data_all_prepared']
    # data = np.load(os.path.join(data_dir, 'data.npy'))
    print('data loaded:', data.shape)

    # Only use positive and negative samples
    if label_type == 'cls2':
        vid_sel = list(range(12))
        vid_sel.extend(list(range(16,28)))
        data = data[:, vid_sel, :, :] # sub, vid, n_channs, n_points
        n_videos = 24
    else:
        n_videos = 28

    data = np.transpose(data, (0,1,3,2)).reshape(n_subs, -1, n_channs)

    if channel_norm:
        for i in range(data.shape[0]):
            data[i,:,:] = (data[i,:,:] - np.mean(data[i,:,:], axis=0)) / np.std(data[i,:,:], axis=0)

    if time_norm:
        data = (data - np.tile(np.expand_dims(np.mean(data, axis=2), 2), (1, 1, data.shape[2]))) / np.tile(
            np.expand_dims(np.std(data, axis=2), 2), (1, 1, data.shape[2])
        )

    n_samples = np.ones(n_videos)*n_segs

    if label_type == 'cls2':
        label = [0] * 12
        label.extend([1] * 12)
    elif label_type == 'cls9':
        label = [0] * 3
        for i in range(1,4):
            label.extend([i] * 3)
        label.extend([4] * 4)
        for i in range(5,9):
            label.extend([i] * 3)
        print(label)

    label_repeat = []
    for i in range(len(label)):
        label_repeat = label_repeat + [label[i]]*n_segs
    return data, label_repeat, n_samples, n_segs


def load_srt_de(data, channel_norm, isFilt, filtLen, label_type):
    # isFilt: False  filten:1   channel_norm: True
    n_subs = 123
    if label_type =='cls2':
        n_vids = 24
    elif label_type =='cls9':
        n_vids = 28
    n_samples = np.ones(n_vids).astype(np.int32) * 30   #(30,30,...,30)

    n_samples_cum = np.concatenate((np.array([0]), np.cumsum(n_samples))) #(0,30,60,...,810,840)
    if isFilt:
        data = data.transpose(0,2,1)
        # print(data.shape)
        # Smoothing the data
        for i in range(n_subs):
            for vid in range(len(n_samples)):
                data[i, :, int(n_samples_cum[vid]): int(n_samples_cum[vid+1])] = smooth_moving_average(data[
                                            i, :, int(n_samples_cum[vid]): int(n_samples_cum[vid+1])], filtLen)
        data = data.transpose(0,2,1)

    # Normalization for each sub
    if channel_norm:
        for i in range(data.shape[0]):
            data[i,:,:] = (data[i,:,:] - np.mean(data[i,:,:], axis=0)) / np.std(data[i,:,:], axis=0)

    if label_type == 'cls2':
        label = [0] * 12
        label.extend([1] * 12)
    elif label_type == 'cls9':
        label = [0] * 3
        for i in range(1,4):
            label.extend([i] * 3)
        label.extend([4] * 4)
        for i in range(5,9):
            label.extend([i] * 3)
        print(label)

    label_repeat = []
    for i in range(len(label)):
        label_repeat = label_repeat + [label[i]]*n_samples[i]
    return data, label_repeat, n_samples


def load_srt_pretrainFeat(datadir, channel_norm, timeLen, timeStep, isFilt, filtLen, label_type):
    if label_type == 'cls9':
        n_samples = np.ones(28).astype(np.int32) * 30
    elif label_type == 'cls2':
        n_samples = np.ones(24).astype(np.int32) * 30
    for i in range(len(n_samples)):
        n_samples[i] = int((n_samples[i] - timeLen) / timeStep + 1)

    if datadir[-4:] == '.npy':
        data = np.load(datadir)
        data[data < -10] = -5
    elif datadir[-4:] == '.mat':
        data = sio.loadmat(datadir)['de_lds']
        print('isnan total:', np.sum(np.isnan(data)))
        data[np.isnan(data)] = -8
        # data[data < -8] = -8
    
    # data_use = data[:, np.max(data, axis=0)>1e-6]
    # data = data.reshape(45, int(np.sum(n_samples)), 256)
    print(data.shape)
    print(np.min(data), np.median(data))

    n_samples_cum = np.concatenate((np.array([0]), np.cumsum(n_samples)))
    if isFilt:
        print('filtLen', filtLen)
        data = data.transpose(0,2,1)
        for i in range(data.shape[0]):
            for vid in range(len(n_samples)):
                data[i, :, int(n_samples_cum[vid]): int(n_samples_cum[vid+1])] = smooth_moving_average(data[
                                            i, :, int(n_samples_cum[vid]): int(n_samples_cum[vid+1])], filtLen)
        data = data.transpose(0,2,1)

    # Normalization for each sub
    if channel_norm:
        for i in range(data.shape[0]):
            data[i,:,:] = (data[i,:,:] - np.mean(data[i,:,:], axis=0)) / (np.std(data[i,:,:], axis=0) + 1e-3)

    if label_type == 'cls2':
        label = [0] * 12
        label.extend([1] * 12)
    elif label_type == 'cls9':
        label = [0] * 3
        for i in range(1,4):
            label.extend([i] * 3)
        label.extend([4] * 4)
        for i in range(5,9):
            label.extend([i] * 3)
        print(label)

    label_repeat = []
    for i in range(len(label)):
        label_repeat = label_repeat + [label[i]]*n_samples[i]
    return data, label_repeat, n_samples
