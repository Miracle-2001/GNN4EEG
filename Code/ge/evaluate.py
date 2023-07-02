import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import os
import json


def draw_ratio(model_path,csvname,figname,cls):
    n_sub = 123
    path=os.path.join(model_path,csvname)
    pd_Data = pd.read_csv(path)
    acc_list = np.array(pd_Data[['0']]) * 100
    # print(acc_list)
    acc_mean = np.mean(acc_list)
    std = np.std(acc_list)

    
    print(figname + ' mean: %.1f' %acc_mean,' std:%.1f'%std)
    # plt.figure(figsize=(13, 10))
    plt.figure(figsize=(10, 10))
    title_name=figname+' mean: %.1f' %acc_mean+' std:%.1f'%std
    plt.title(title_name,fontsize=20,loc='center')
    x_haxis = [str(num) for num in range(1,n_sub+1+1)]
    y = np.vstack((acc_list,acc_mean)).flatten()
    y[:-1] = np.sort(y[:-1])
    x = np.arange(0,len(x_haxis))
    plt.ylim(0,100);
    # plt.xlabel('subjects',fontsize=20);
    plt.ylabel('Accuracy (%)',fontsize=30);
    plt.yticks(fontsize=25);plt.xticks(fontsize=25)
    plt.bar(x[:-1], y[:-1], facecolor='#D3D3D3', edgecolor='black', width=0.9,label='accuacy for each subject')
    plt.bar(x[-1]+5,y[-1],facecolor='#696969', edgecolor='black', width=2.5, label='averaged accuracy')
    plt.errorbar(x[-1]+5, y[-1], yerr=std, fmt='o', ecolor='black', color='#000000', elinewidth=1, capsize=2, capthick=1)
    # chance level
    y_ = np.ones((y.shape[0]+0)) * 1/int(cls) * 100
    x_ = np.arange(0,y_.shape[0])
    plt.plot(x_,y_,linestyle='dashed',color='#808080')
    # plt.legend(loc='upper right',fontsize=16,edgecolor='black')
    # plt.legend(fontsize=20)
    plt.savefig(os.path.join(model_path,figname + '.png'))
    plt.savefig(os.path.join(model_path,figname + '.eps'),format='eps')
    plt.clf()


def confusionMat(subs_data):
    # (n_sub, n_vids, secs)
    import seaborn as sns
    n_vids = subs_data.shape[1];n_subs = subs_data.shape[0]
    if n_vids == 28:
        cls = 9
    else:
        cls = 2
    confusion = np.zeros((cls,cls))
    for sub in range(0,subs_data.shape[0]):
        data = subs_data[sub,:,:]
        emotions = []
        for c in range(0,cls):
            emotions.append([])
            if cls == 2:
                emotions[c] = data[12*c:12*c+12,:].reshape(1,-1)
            if (cls == 9)&(c<4):
                emotions[c] = data[3*c:3*c+3,:].reshape(1,-1)
            elif (cls == 9)&(c==4):
                emotions[c] = data[12:16,:].reshape(1,-1)
            elif  (cls == 9)&(c>4):
                emotions[c] = data[16+(c-5)*3:16+(c-5)*3+3,:].reshape(1,-1)

        for real in range(0,cls):
            for predict in range(0,cls):
                confusion[real,predict] += (np.sum(emotions[real] == predict)/emotions[real].shape[1])/n_subs

    index = ['Anger','Disgust','Fear','Sadness','Neutral','Amusement','Inspiration','Joy','Tenderness']
    confusion = pd.DataFrame(confusion * 100,index=index,columns=index)
    annot_kws = {"fontsize": 13};
    plt.figure(figsize=(7, 6));
    # move the pic to the top
    plt.subplots_adjust(bottom=0.25,left=0.25)
    figure = sns.heatmap(confusion,annot=False,fmt=".1f", cmap='Blues',
                         xticklabels=index,annot_kws=annot_kws,square=True)
    plt.xlabel('Predicted label',fontsize=13);plt.ylabel('True label',fontsize=13)
    plt.xticks(rotation=60,fontsize=13)
    plt.yticks(rotation=0,fontsize=13)
    figure.get_figure().savefig('./clisa_confusion_mat.png')
    figure.get_figure().savefig('./clisa_confusion_mat.eps',format='eps')

def draw_acc_vs_lr(args, figname):
    path=args.model_path
    x = range(0, args.num_epoch*args.step, args.step)

    total_acc_list=[0 for i in range(args.num_epoch)]
    for fold in range(10):
        file=open(os.path.join(path,f'/fold_{fold}_acc_and_loss.json'),'r')
        dict=json.load(file)
        for i in range(args.num_epoch):
            total_acc_list[i]+=dict["eval_acc_list"][i]
    for i in range(args.num_epoch):
        total_acc_list[i]/=10
    
    max_lr=-1
    max_acc=0
    for i in range(args.num_epoch):
        if total_acc_list[i]>max_acc:
            max_acc=total_acc_list[i]
            max_lr=x[i]

    fig = plt.figure(figsize=(20, 8), dpi=80)
    plt.plot(x, total_acc_list, color='blue', label='mean accuracy')
    plt.legend(loc="upper right")
    plt.title(f"accuracy vs learning rate best acc:{max_acc},best lr:{max_lr}", color='black')
    plt.show()

    # save fig
    filepath = os.path.join(path, (figname.split('.')[0])+'png')
    fig.savefig(filepath)
    

def draw_results(args):
    csvname='subject_%s_vids_%s_valid_%s.csv' % (args.subjects_type, str(args.n_vids), args.valid_method)
    draw_ratio(args.model_path,csvname,'%s_acc_%s_%s_%s'%(args.model,args.subjects_type,str(args.n_vids),args.now_time),cls=args.num_classes)

def find_best_lr(args):
    draw_acc_vs_lr(args,"val_acc_vs_lr.png")
# # svm_inter_28_10_folds = './Svm_analysis/subject_inter_vids_28_valid_10-folds.csv'
# # svm_intra_28_10_folds = './Svm_analysis/subject_intra_vids_28_valid_10-folds.csv'
# # svm_inter_24_10_folds = './Svm_analysis/subject_inter_vids_24_valid_10-folds.csv'
# rgnn_intra_24_10_folds = 'subject_intra_vids_24_valid_10-folds.csv'
# # rgnn_intra_28_10_folds = './RGNN_analysis/rgnn_subject_intra_vids_28_valid_10-folds_20230209.csv'
# #rgnn_inter_28_10_folds = './RGNN_analysis/rgnn_subject_inter_vids_28_valid_10-folds.csv'
# # rgnn_inter_24_10_folds='./RGNN_analysis/rgnn_subject_inter_vids_24_valid_10-folds20230210.csv'
# # clisa_score = './Clisa_analysis/Clisa_score.csv'

# # draw_ratio(clisa_score,'clisa_acc',cls=9)
# # draw_ratio(svm_inter_28_10_folds,'svm_acc_inter_28',cls=9)
# # draw_ratio(svm_intra_28_10_folds,'svm_acc_intra_28',cls=9)
# # draw_ratio(svm_inter_24_10_folds,'svm_acc_inter_24',cls=2)
# draw_ratio(rgnn_intra_24_10_folds,'rgnn_acc_intra_24_20230301',cls=2)
# # draw_ratio(rgnn_intra_28_10_folds,'rgnn_acc_intra_28_20230209',cls=9)
# # draw_ratio(rgnn_inter_28_10_folds,'rgnn_acc_inter_28_202302091839',cls=9)
# # draw_ratio(rgnn_inter_24_10_folds,'rgnn_acc_inter_24_20230210',cls=2)
# # result_data = sio.loadmat('./Clisa_results.mat')['mlp']
# # confusionMat(result_data)

