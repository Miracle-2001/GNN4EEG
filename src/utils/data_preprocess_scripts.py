import numpy as np
import scipy.io as sio
import pickle
import mne
import os
import scipy.io as sio
import numpy as np
from reorder_vids import video_order_load, reorder_vids, reorder_vids_back
import random
import argparse
import time


parser = argparse.ArgumentParser(description='Running norm the EEG data of baseline model')
parser.add_argument('--timeLen', default=5, type=int,
                    help='time length in seconds')
parser.add_argument('--use-data', default='de', type=str,
                    help='what data to use')
parser.add_argument('--normTrain', default='yes', type=str,
                    help='whether normTrain')
parser.add_argument('--save-dir', default=' ', type=str,
                    help='the save dir')
parser.add_argument('--dataset', default='both', type=str,
                    help='first_batch or second_batch')
parser = argparse.ArgumentParser(description='Smooth the EEG data of baseline model')

parser.add_argument('--use-data', default='de', type=str,
                    help='what data to use')
parser.add_argument('--n-vids', default=24, type=int,
                    help='use how many videos')
parser.add_argument('--randSeed', default=7, type=int,
                    help='random seed')
parser.add_argument('--smooth-length', default=30, type=int,
                    help='the length for lds smooth')
parser.add_argument('--dataset', default='both', type=str,
                    help='first_batch or second_batch')


args = parser.parse_args()

random.seed(args.randSeed)
np.random.seed(args.randSeed)



# Load the data
data_path = './Processed_data'
data_paths = os.listdir(data_path)
data_paths.sort()
n_vids = 28
chn = 30
fs = 250
sec = 30
data = np.zeros((len(data_paths), n_vids, chn, fs * sec))
for idx, path in enumerate(data_paths):
    f = open(os.path.join(data_path, path), 'rb')
    data_sub = pickle.load(f)
    data[idx, :, :, :] = data_sub[:, :-2, :]

# data shape :(sub, n_vids, chn, fs * sec)
print('data loaded:', data.shape)  # (123,28,30,7500)


n_subs = data.shape[0]

fs = 250
# freqs = [[4, 8], [8, 13], [13, 30], [30, 47]]
freqs = [[1,4],[4, 8], [8, 13], [13, 30], [30, 47]] # using 5 bands

de = np.zeros((n_subs, 30, 28*30, len(freqs)))
for i in range(len(freqs)):
    print('Current freq band: ', freqs[i])
    for sub in range(n_subs):
        for j in range(28):
            data_video = data[sub, j, :, :]
            print(data_video.shape)  # (30,30*250)
            low_freq = freqs[i][0]
            high_freq = freqs[i][1]
            data_video_filt = mne.filter.filter_data(
                data_video, fs, l_freq=low_freq, h_freq=high_freq)  # 使用了band pass滤波。调整后shape没有变化。
            data_video_filt = data_video_filt.reshape(
                30, -1, fs)  # (30,30,250)
            de_one = 0.5*np.log(2*np.pi*np.exp(1) *
                                (np.var(data_video_filt, 2)))  # (30,30)
            # n_subs, 30, 28*30, freqs

            de[sub, :, 30*j: 30*(j+1), i] = de_one


print(de.shape)  # (123,30,840,4) (subject,channel,n_vids*sec,n_band) 键值是这个被试的这个信道在观看这个视频的这一秒钟所有EEG信号(250个)的微分熵。
de = {'de': de}
sio.savemat('./de_features.mat', de)




displace = False
use_features = args.use_data
normTrain = args.normTrain
n_vids = args.n_vids
isCar = True
randomInit = False

root_dir = './'
save_dir = os.path.join(root_dir, 'running_norm_'+ str(n_vids))

bn_val = 1
# rn_momentum = 0.995
# print(rn_momentum)
# momentum = 0.9


n_total = 30*n_vids
n_counters = int(np.ceil(n_total / bn_val))

n_subs = 123
n_folds = 10
n_per = round(n_subs / n_folds)


for decay_rate in [0.990]:
    print(decay_rate)
    for fold in range(n_folds):
    # for fold in range(n_folds-1, n_folds):
        print(fold)
        if use_features == 'de':
            # data = sio.loadmat(os.path.join(save_dir, 'deFeature_all.mat'))['deFeature_all']
            data_name = 'de_features.mat'
            data = sio.loadmat(os.path.join(root_dir, data_name))['de']
            print(data.shape)
            data = data.transpose([0,2,3,1]).reshape(n_subs, 840, 30*5) 
            if n_vids == 24:
                data = np.concatenate((data[:, :12*30, :], data[:, 16*30:, :]), 1) #delete neutral-stimuli videos
        elif use_features == 'CoCA':
            if n_vids == 28:
                data = sio.loadmat(os.path.join(save_dir, 'de_CoCA_fold%d.mat' % fold))['de_all']
            elif n_vids == 24:
                data = sio.loadmat(os.path.join(save_dir, 'de_CoCA_fold%d.mat' % fold))['de_all']
        elif use_features == 'SA':
            if n_vids == 28:
                data = sio.loadmat(os.path.join(save_dir, 'de_%d.mat' % fold))['de_all']
            elif n_vids == 24:
                data = sio.loadmat(os.path.join(save_dir, 'de_%d.mat' % fold))['de_all']
        elif (use_features == 'pretrained') or (use_features == 'simseqclr'):
            if normTrain == 'yes':
                data = sio.loadmat(os.path.join(save_dir, str(fold), 'features1_de_1s_normTrain.mat'))['de']
            else:
                data = sio.loadmat(os.path.join(save_dir, str(fold), 'features1_de_1s.mat'))['de']
        print(data.shape)

        if fold < n_folds-1:
            val_sub = np.arange(n_per*fold, n_per*(fold+1))
        else:
            val_sub = np.arange(n_per*fold, n_per*(fold+1)-1)
        train_sub = list(set(np.arange(n_subs)) - set(val_sub))

        # reorder video. Because the videos are randomly sorted before each subject watch them
        # So, we must make the order 1,2,...,28
        vid_order = video_order_load(args.dataset, 28)

        data, vid_play_order_new = reorder_vids(data, vid_order)
        print(vid_play_order_new)

        data[np.isnan(data)] = -30
        # data[data<=-30] = -30
        
        data_mean = np.mean(np.mean(data[train_sub, :, :], axis=1), axis=0)
        data_var = np.mean(np.var(data[train_sub, :, :], axis=1), axis=0)
        
        data_norm = np.zeros_like(data)
        
        for sub in range(data.shape[0]):
            running_sum = np.zeros(data.shape[-1])
            running_square = np.zeros(data.shape[-1])
            decay_factor = 1.
            start_time = time.time()
            for counter in range(n_counters):
                data_one = data[sub, counter*bn_val: (counter+1)*bn_val, :]
                running_sum = running_sum + data_one
                running_mean = running_sum / (counter+1)
                # running_mean = counter / (counter+1) * running_mean + 1/(counter+1) * data_one
                running_square = running_square + data_one**2
                running_var = (running_square - 2 * running_mean * running_sum) / (counter+1) + running_mean**2

                curr_mean = decay_factor*data_mean + (1-decay_factor)*running_mean
                curr_var = decay_factor*data_var + (1-decay_factor)*running_var
                decay_factor = decay_factor*decay_rate

                # print(running_var[:3])
                # if counter >= 2:
                data_one = (data_one - curr_mean) / np.sqrt(curr_var + 1e-5)
                data_norm[sub, counter*bn_val: (counter+1)*bn_val, :] = data_one
            end_time = time.time()
            # print('time consumed: %.3f, counter: %d' % (end_time-start_time, counter+1))
        # data_norm.shape: (n_sub,n_vids*30,120)
        data_norm = reorder_vids_back(data_norm, vid_play_order_new)
        de = {'de': data_norm}
        print(data_norm.shape)
        if (use_features == 'de') or (use_features == 'CoCA'):
            if n_vids == 28:
                if isCar:
                    save_name = os.path.join(save_dir, 'normTrain_rnPreWeighted%.3f_newPre_%dvideo_car' % (decay_rate, n_vids))
                else:
                    save_name = os.path.join(save_dir, 'normTrain_rnPreWeighted%.3f_newPre_%dvideo' % (decay_rate, n_vids))
                if not os.path.exists(save_name):
                    os.makedirs(save_name)
                save_file = os.path.join(save_name, 'de_fold%d.mat' % fold)
            elif n_vids == 24:
                # if isCar:
                #     save_name = os.path.join(save_dir, 'normTrain_rnPreWeighted%.3f_newPre_car' % decay_rate)
                # else:
                #     save_name = os.path.join(save_dir, 'normTrain_rnPreWeighted%.3f_newPre' % decay_rate)
                if isCar:
                    save_name = os.path.join(save_dir, 'normTrain_rnPreWeighted%.3f_newPre_%dvideo_car' % (decay_rate, n_vids))
                else:
                    save_name = os.path.join(save_dir, 'normTrain_rnPreWeighted%.3f_newPre_%dvideo' % (decay_rate, n_vids))
                if not os.path.exists(save_name):
                    os.makedirs(save_name)
                save_file = os.path.join(save_name, 'de_fold%d.mat' % fold)
            print(save_file)
            sio.savemat(save_file, de)


def LDS(sequence):
    #print("shape ",sequence.shape) # (720, 120)

    ave = np.mean(sequence, axis=0)  # [120,]
    u0 = ave
    X = sequence.transpose((1, 0))  # [120, 720]

    V0 = 0.01
    A = 1
    T = 0.0001
    C = 1
    sigma = 1

    [m, n] = X.shape  # (120, 720)
    P = np.zeros((m, n))  # 
    u = np.zeros((m, n))  # 
    V = np.zeros((m, n))  # 
    K = np.zeros((m, n))  #

    K[:, 0] = (V0 * C / (C * V0 * C + sigma)) * np.ones((m,))
    u[:, 0] = u0 + K[:, 0] * (X[:, 0] - C * u0)
    V[:, 0] = (np.ones((m,)) - K[:, 0] * C) * V0

    for i in range(1, n):
        P[:, i - 1] = A * V[:, i - 1] * A + T
        K[:, i] = P[:, i - 1] * C / (C * P[:, i - 1] * C + sigma)
        u[:, i] = A * u[:, i - 1] + K[:, i] * (X[:, i] - C * A * u[:, i - 1])
        V[:, i] = (np.ones((m,)) - K[:, i] * C) * P[:, i - 1]

    X = u

    return X.transpose((1, 0))



n_vids = args.n_vids

root_dir = './running_norm_%s/normTrain_rnPreWeighted0.990_newPre_%svideo_car' % (n_vids, n_vids)
save_dir = './smooth_' + str(n_vids)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

n_subs = 123

n_folds = 10
n_per = round(n_subs/n_folds)
n_length = args.smooth_length

for fold in range(n_folds):
    data_dir = os.path.join(root_dir, 'de_fold' + str(fold) + '.mat')
    feature_de_norm = sio.loadmat(data_dir)['de']
    subs_feature_lds = np.ones_like(feature_de_norm)
    for sub in range(n_subs):
        subs_feature_lds[sub, : ,:] = LDS(feature_de_norm[sub,:,:])
    de_lds = {'de_lds': subs_feature_lds}
    save_file = os.path.join(save_dir, 'de_lds_fold' + str(fold) + '.mat')
    print(save_file)
    print(de_lds['de_lds'].shape) 
    sio.savemat(save_file, de_lds)
