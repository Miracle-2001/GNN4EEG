## Models

### DGCNN
- **models.DGCNN(num_nodes,num_hiddens,num_layers,electrode_position)**



### RGNN
- **models.RGNN(num_nodes,num_hiddens,num_layers,electrode_position,global_connections=None)**

### SparseDGCNN
- **models.SparseDGCNN(num_nodes,num_hiddens,num_layers,electrode_position)**

### HetEmotionNet
- **models.HetEmotionNet(num_nodes,num_hiddens,num_layers)**

## Protocols

- **protocols.data_split(protocol,data,labels,subject_id_list)**
- **protocols.data_FACED(protocol,categories,data_path)**
- **protocols.evaluation(model,loader,protocol,K,K_inner,grid,categories,optimizer="Adam")**