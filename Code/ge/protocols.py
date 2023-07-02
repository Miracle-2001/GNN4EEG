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

def data_split(protocol,data,labels,subject_id_list):
    if protocol!="cross_subject" and protocol!="intra_subject":
        raise ValueError("The protocol should be either 'cross_subject' or 'intra_subject'.")
    
    
    pass

def data_FACED(protocol,categories,data_path):
    if protocol!="cross_subject" and protocol!="intra_subject":
        raise ValueError("The protocol should be either 'cross_subject' or 'intra_subject'.")
    if categories!=2 and categories!=9:
        raise ValueError("The categories should be either 2 or 9.")
    if os.path.exists(data_path)==False:
        raise ValueError("The path of FACED dataset is not exist.")
    pass

def evaluation(model,loader,protocol,K,K_inner,grid,categories,
               optimizer="Adam"):
    # ,L1_reg=0,L2_reg=0,dropout=0,alpha=0,lr,epoch=100,batch_size,
    
    pass