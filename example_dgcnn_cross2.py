# A 2-class classification example for DGCNN using FACED dataset
from ge.protocols import *
from ge.models import *


if __name__ =='__main__':
    # load the FACED dataset
    data_path='./FACED_dataset_2_labels.mat'
    loader=data_FACED('cross_subject',2,data_path)
    print(loader.data.shape)

    # based hyper-parameters defination 
    num_nodes=30
    num_hiddens=80
    num_layers=2

    # load electrode_position 
    electrode_position=np.load('./src/utils&others/pos.npy')[:30]*100
    print(electrode_position)
    # model initialization
    model_DGCNN=DGCNN(num_nodes,num_hiddens,num_layers,electrode_position)

    subject_num=123
    section_size=loader.data.shape[0]/subject_num
    subject_id_list=np.array([int(i/section_size) for i in range(loader.data.shape[0])])
    print(subject_id_list.shape)
    
    K=10
    best_dict,out_acc_list=evaluation(model_DGCNN,loader,'cv',grid={"lr":0.001,"hiddens":80,'epoch':list(range(0,100)),'l1_reg':0.005,'l2_reg':0.005,
                                                                'batch_size':256,'dropout':0.5},
                    categories=2,K=K,device=torch.device('cuda:2'),optimizer='Adam')
    print(best_dict,out_acc_list)