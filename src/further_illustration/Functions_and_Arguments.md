# Common Arguments
Here, we list some of the common arguments used in the functions. Other arguments which tightly related to specific functions will be introduced under those functions. 

| Args               | Default             | Description                                    |
|--------------------|---------------------|------------------------------------------------|
| num_nodes          |                     | The number of nodes in the adjacency matrix.   |
| num_hiddens        |                     | The number of dimensions of the hidden layer.  |
| num_layers         |                     | The number of layers in the GNN.               |
| electrode_position |                     | The 2D (or 3D) electrode position.                     |
| device             | torch.device('cpu') | The training device.                           |
| optimizer          | torch.optim.Adam    | The optimizer while training.                  |
| num_classes        | 2                   | The number of class to distinguish.            |
| dropout            | 0.5                 | The dropout rate.                              |
| batch_size         | 256                 | The size of a batch.                           |
| lr                 | 5.00E-03            | The learning rate.                             |
| l1_reg             | 0                   | The coefficient of the L1 normalization.       |
| l2_reg             | 0                   | The coefficient of the L2 normalization.       |
| num_epoch          | 50                  | The number of epochs in the training progress. |

Besides, many functions below contain 'data' (or 'train_data' in RGNN.train) and 'labels' parameters. The introductions of these two parameters is shown as followings.

'data' represent the samples and is recommanded preprocessed into **frequency domain**(Of course, you can still maintain it raw and under time domain). The type of 'data' must be numpy.ndarray.  The shape of 'data' is like '(num_samples,num_nodes,num_frequency_band).  
    
'labels' represent the corresponding class (integers,start from 0) of 'data' samples. The type of 'labels' must also be numpy.ndarray.  The shape of 'labels' is like (num_samples,). And 'data.shape[0]==labels.shape[0]' must be held. 
    
# Functions

Here, we introduce each function in GNN4EEG and list some of its particular parameters.  The functions can be roughly divided into 2 parts: 'Models' and 'Protocols'. 

For detailed usages, please refer to [examples](../../example.ipynb).

## Models
For 'Models' part, we present the initialization methods, training methods and predicting methods of the 4 models, i.e., DGCNN, RGNN, SparseDGCNN and HetEmotionNet. Each model is seperately equipped with initialization methods, training methods and predicting methods. All of these models inherit from GNNModel.

### DGCNN
- **models.DGCNN(num_nodes,num_hiddens,num_layers,electrode_position)**

    Initialize a DGCNN model. Here the shape of 'electrode_position' should be (2,num_nodes) or (3,num_nodes). 

- **models.DGCNN.train(self,data,labels, device=torch.device('cpu'),optimizer=torch.optim.Adam, num_classes=2,dropout=0.5,batch_size=256, lr=5e-3, l1_reg=0.0,l2_reg=0.0, num_epoch=50)**
    
    This function is used to further define the training hyper-parameters and start the training progress. 

    Here, 'data' represents the data samples and 'labels' represents the corresponding labels.
    
    
- **models.DGCNN.predict(self,data)**

    This function is used to give predictions of the input data. Note that before running this statement, the model should be already trained on certain dataset via 'DGCNN.train' function. Otherwise, an error will be raised.

### RGNN
Please note that because the EmotionDL regularization is complex and data oriented, we do not implement an API for that.

- **models.RGNN(num_nodes,num_hiddens,num_layers,electrode_position,global_connections=None)**

    Initialize a RGNN model. Here the shape of 'electrode_position' should be (2,num_nodes) or (3,num_nodes). Global connections should be declared if the user want to introduce global inter-channel connections. (Detail intuition can be found in the original paper.)

- **models.RGNN.train(self,train_data,train_labels,valid_data=None, device=torch.device('cpu'),optimizer=torch.optim.Adam, num_classes=2,dropout=0.5,NodeDAT=False, batch_size=256, lr=5e-3, l1_reg=0.0,l2_reg=0.0, num_epoch=50)**
    
    This function is used to further define the training hyper-parameters and start the training progress. 

    Here, 'train_data' represents the data samples and 'train_labels' represents the corresponding labels. Besides, 'NodeDAT' is a boolean parameter which represents wether use Node-wise domain adaptation or not. If 'NodeDAT' is true, then the user should provide 'valid_data' as well. 

- **models.RGNN.predict(self,data)**

    This function is used to give predictions of the input data. Note that before running this statement, the model should be already trained on certain dataset via 'RGNN.train' function. Otherwise, an error will be raised.


### SparseDGCNN
- **models.SparseDGCNN(num_nodes,num_hiddens,num_layers,electrode_position)**

    Initialize a SparseDGCNN model. Here the shape of 'electrode_position' should be (2,num_nodes) or (3,num_nodes). 

- **models.SparseDGCNN.train(self,data,labels, device=torch.device('cpu'),optimizer=torch.optim.Adam, num_classes=2, dropout=0.5,batch_size=256, lr=5e-3, l1_reg=0.0,l2_reg=0.0, num_epoch=50)**

    This function is used to further define the training hyper-parameters and start the training progress. 

    Here, 'data' represents the data samples and 'labels' represents the corresponding labels. 
    Note that because the sparse coefficient equals to L1 normalization coefficient, we use a single parameter 'l1_reg' to represent both. 

- **models.SparseDGCNN.predict(self,data)**

    This function is used to give predictions of the input data. Note that before running this statement, the model should be already trained on certain dataset via 'SparseDGCNN.train' function. Otherwise, an error will be raised.


### HetEmotionNet
- **models.HetEmotionNet(num_nodes,num_hiddens)**
    Initialize a HetEmotionNet model. Note that in HetEmotionNet the initial values in adjacency matrix is calculated according to mutual information, so the positions of electrodes is not a must.

- **models.HetEmotionNet.train(self,data_freq,data_time,labels, device=torch.device('cpu'),optimizer=torch.optim.Adam, num_classes=2, dropout=0.5,batch_size=256, lr=5e-3, l1_reg=0.0,l2_reg=0.0, num_epoch=50)**

    This function is used to further define the training hyper-parameters and start the training progress. 

    Here, 'data_time' represents the data samples under time domain and 'data_freq' represents the data samples under frequency domain. 'labels' represents the corresponding labels.  

- **models.HetEmotionNet.predict(self,data_freq,data_time)**

    This function is used to give predictions of the input data_freq (signals under frequency-domain) and data_time (signals under time-domain). Note that before using this function, the model should be already trained on certain dataset via 'HetEmotionNet.train' function. Otherwise, an error will be raised.


## Protocols
For 'Protocols' part, we present the data selecting and splitting method, including default dataset selection (FACED dataset) and user-defined data splitting interface. Afterwards, we also propose a 'evaluation' function to make grid search and cross-validation.

- **protocols.data_split(protocol, data, labels, subject_id_list=None, data_time=None)**
    
    This function is used to define data splitting protocols and store the dataset.
    
    Here, parameter 'protocol' must be either 'cross_subject' or 'intra_subject', the meaning of which can be found in the originial paper.

    As for 'subject_id_list', it is a list which record the subject id (integers,start from 0) of each data samples. So, naturally, 'data.shape[0]==len(subject_id_list)' must be held. The shape of 'subject_id_list' is like (num_samples,). Note that, when the 'protocol' is set to be 'intra_subject', parameter 'subject_id_list' is not a necessity (because we do not care which subject the signal belongs to under intra_subject protocol). 

    'data_time' is used when you are willing to use HetEmotionNet for evaluation. As described in the orginal paper, HetEmotionNet contains a two-stream structure, so both frequency-domain signals and original time-domain signals should be provided. Here, 'data_time' represent the original (or raw) time-domain signals. Of course, 'data.shape[0]==data_time.shape[0]' must be held. 


- **protocols.data_FACED(protocol, categories, data_path)**

    This function is used to load the samples in FACED dataset as well as define the data splitting protocol and classfication categories.
    
    To use this function, first, **users need to download the dataset, and preprocess it according to [this](./FACED_dataset_preparations.md)**. After running 'smooth_lds.py' scripts, dicts like *{'de_lds': subs_feature_lds}* (where the shape of subs_feature_lds is *(123, 720 or 840, 120)*) are saved using 'sio.savemat' and '.mat' files can be found. Afterwards, **use 'data_path' parameter** to input the path of your '.mat' file. Then, in protocols.data_FACED function, a statement 'data=hdf5.loadmat(data_path)['de_lds']' will be responsible for the loading of FACED dataset.

    For other parameters, 'protocol' must be either 'cross_subject' or 'intra_subject', the meaning of which can be found in the originial paper. 'categories' must be either 2 or 9.

- **protocols.evaluation(model:GNNModel, loader: DataLoader, protocol: str, grid: dict, categories, K, K_inner=None, device=torch.device('cpu'),optimizer="Adam", NodeDAT=False)**

    This function is used to start the evaluation. 
    
    'model' is a kind of GNNModel which means it should be one of DGCNN, RGNN, SparseDGCNN or HetEmotionNet. 'loader' is a kind of DataLoader, which can be initialized via protocols.data_split or protocols.data_FACED. 'protocol' is a string, representing the evaluation protocol, which is one of 'cv', 'fcv' or 'ncv'. 

    'grid' is a dict containing the tuning ranges of certain parameters. The key of it can include "hiddens", "layers","lr", "epoch", "dropout", "batch_size", "l1_reg", and "l2_reg". The type of the value of each key is must be list, int or float. Note that, if you have already defined the "num_hiddens" in the GNNModel, but still provide "hiddens" key in 'grid' dict, then the "num_hiddens" defined in GNNModel will be ignored.

    'categories' represent the classfication categories in the dataset, and each category label should appear in loader.labels at least once. 'K' represent the outer cross-validation fold number and 'K_inner' represent the inner cross-validation fold number (only 'ncv' protocols must contain 'K_inner'). 'NodeDAT' is a boolean parameter which represents wether use node-wise domain adaptation or not (only considered when RGNN model is used).

    This function will return two results. 
    
    For cv and fcv, it returns 'best_dict' and 'result_list'. The 'best_dict' stores the best parameters found during grid-search tuning, mean accuracy on validation set of all K folds and the fixed epoch number that generates this validation accuracy (when the protocol is 'cv', this epoch number is set to be -1). The 'result_list' stores dicts similar to 'best_dict' for each hyper-parameter combination.

    For ncv, it returns 'mean_acc' and 'out_acc_list'. 'mean_acc' is a float represents the mean accuracy on the test set of all K folds. 'out_acc_list' has K elements, recording the tuning result of the K outer folds. Each element stores the corresponding fold number, best parameters on that fold, mean accuracy on train set, mean accuracy on test set and the number of test set on that fold.

    For detailed usages, please refer to [examples](../../example.ipynb).
