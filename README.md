
![logo](./src/pics/logo.png)
---
<!-- Electroencephalography (EEG) classification is a crucial task in neuroscience, neural engineering, and several commercial applications. Traditional EEG classification models, however, have often overlooked or inadequately leveraged the brain’s topological information. Recognizing this shortfall, there has been a burgeoning interest in recent years in harnessing the potential of Graph Neural Networks (GNN) to exploit the topological information by modeling features selected from each EEG channel in a graph structure. 

However, it remains challenging to evaluate the transferability of these models and implement GNN-based EEG classification models in practice due to the lack of easy-to-use toolkits and large-scale public benchmarks. To tackle this, we build GNN4EEG, a benchmark and toolkit for EEG classification with GNN. -->

GNN4EEG is a benchmark and toolkit focusing on Electroencephalography (EEG) classification tasks via Graph Neural Network (GNN), aiming to facilitate research in this direction. Researchers can arbitrarily choose their prefered GNN models, hyper-parameters and experimental protocols. Training and evaluating dataset can be flexibly chosen as the default *FACED dataset* (with detailed information listed in "Models and Dataset" chapter) or any *self-built datasets*.  The characteristics of our toolkit can be summarized as follows:

- **Large Benchmark**: We introduce a large
benchmark constructed with ***4*** EEG classification tasks based on
EEG data collected from the FACED dataset , consists of ***123*** subjects . 

- **Multiple SOTA Models**:  We implement  ***4*** state-of-the-art GNN-based EEG classification
models, i.e., DGCNN, RGNN, SparseDGCNN and HetEmotionNet.

- **Various Experimental Protocols**:  We provide comprehensive experimental settings and evaluation protocols, e.g., ***2*** data splitting protocols, and ***3*** cross-validation protocols.

- **Easy for Usage**: Our toolkit can proceed the whole process of training and tuning an available EEG classification model for real-time applications ***in just a few lines of code***. 

- **Flexible Framework**: Researchers can arbitrarily select their experimental settings and datasets.



<!-- Generally, GNN4EEG implements **4 EEG classification tasks** as the benchmark, **3 validation protocols** , and **4 GNN models** . -->

## Structure

Generally, GNN4EEG decomposes the whole training and evaluating progress into three modules:

- **Data Splitting**: First, it is necessary to choose the data splitting protocols, i.e., intra-subject or cross-subject. A list describing
the subject of each sample should be provided to guide the splitting.

- **Model Selection**: To initiate a specific model, parameters like
the number of classification categories, graph nodes, hidden layer dimension, and GNN layers should be included. Electrode positions, frequency values, and other options are also necessities for certain
GNN models.


- **Validation Protocols and Other Training Configurations**: The
final step is to declare the validation protocols and other configurations. As illustrated above, GNN4EEG provides three validation protocols, i.e., CV, FCV, and NCV. For detailed training configurations, the user can set the learning rate, dropout rate, number of
epochs, 𝐿1 and 𝐿2 regularization coefficient, batch size, optimizer,
and training device.

A data flow diagram is illustrated as following:

![Structure](./src/pics/structure.png)



## Getting Started

1. Install [Anaconda](https://docs.conda.io/en/latest/miniconda.html) with Python >= 3.5
2. Clone the repository

```bash
git clone https://github.com/Miracle-2001/GNN4EEG.git
```

3. Install requirements and step into the `src` folder

```bash
cd GNN4EEG
cd src
pip install -r requirements.txt
```

4. (optional) To download the FACED dataset, please refer to the [DOI link](https://doi.org/10.7303/syn50614194). Detailed steps will be discussed in [here](./src/further_illustration/FACED_dataset_preparations.md)
.

## Models and Dataset

**Models:**

We have implemented the following methods :

- [EEG Emotion Recognition Using Dynamical Graph Convolutional Neural Networks](https://ieeexplore.ieee.org/abstract/document/8320798) (DGCNN [IEEE Trans'20])

- [EEG-Based Emotion Recognition Using Regularized Graph Neural Networks](https://arxiv.org/pdf/1907.07835.pdf) (RGNN [IEEE Trans'20])

- [SparseDGCNN: Recognizing Emotion from Multichannel EEG Signals](https://ieeexplore.ieee.org/abstract/document/9321519) (SparseDGCNN [IEEE Trans'21])

- [HetEmotionNet: Two-Stream Heterogeneous Graph Recurrent Neural Network for Multi-modal Emotion Recognition](https://arxiv.org/pdf/2108.03354.pdf) (HetEmotionNet [ACM MM'21])

**Dataset:**

GNN4EEG built the large-scale benchmark with the Finer-grained Affective Computing EEG Dataset ([FACED](https://doi.org/10.7303/syn50614194)). As far
as we know, FACED is the largest affective computing dataset,
which is constructed by recording 32-channel EEG signals from a
large cohort of 123 subjects watching 28 emotion-elicitation video
clips.


**Experiments:**

We present the experimental setup and the evaluation results using the proposed GNN4EEG toolkit on FACED dataset. Analyses of
overall performances are elaborated here. The experiments are implemented on NVIDIA GeForce RTX 3090.

(Here, "intra-2" means binary intra-subject classification task and "cross-9" means 9 class cross-subject classification task. Others are similar.)

In the experiments, we set the fold number 𝐾 = 10 for all validation protocols and the
“inner” fold number 𝐾
′ = 3 for NCV. In intra-subject tasks, the 30
seconds EEG signals among all video clips and subjects are equally
split into 𝐾 folds. While in cross-subject tasks, the 123 subjects are
split into 𝐾 folds, with the last fold containing 15 subjects and the
former each containing 12 subjects.
We tune the number of hidden dimensions from {20, 40, 80} and
the learning rate from {0.0001, 0.001, 0.01} for all tasks and models.
Moreover, the dropout rate is 0.5, the number of GNN layers is 2,
the batch size is 256, and the maximum number of epochs is set
as 100. To address potential overfitting in different settings, we
have utilized different weights for the 𝐿1 and 𝐿2 norm in different
tasks. Specifically, both weights are set as 0.001 for intra-2, 0.005
for cross-9, and 0.003 for cross-2 and intra-9. 

![result](./src/pics/result.png)

## Functions and Arguments 

GNN4EEG implements 4 EEG classification tasks on FACED as the benchmark, 2 data splitting protocols, 3 validation
protocols, and 4 GNN models. Plenty optional parameters are provided for convience and flexibility.

Totally, GNN4EEG implements these functions:


- **protocols.data_split**
- **protocols.data_FACED**
- **protocols.evaluation**

- **models.DGCNN**
- **models.RGNN**
- **models.SparseDGCNN**
- **models.HetEmotionNet**

and each model is equipped with **train**, **predict**, **save** and **load** function.


Detailed arguments and usage will be further discussed in [here](./src/further_illustration/Functions_and_Arguments.md).

## Example Usage

Generally, to train and evaluate a model on a certain dataset, users can follow the steps below (according to the "Structure" chapter):

1. **Data Splitting**: Use protocols.data_split or protocols.data_FACED to load data and define the data splitting protocol.

2. **Model Selection**: Use models.* to select and set some basic hyper-parameters of your model.

3. **Validation Protocols and Other Training Configurations**: Use protocols.evaluation to define the validation protocols and put other training configurations into the parameter "grid". Then, the training and evaluating progress will be launched! (Hint: in this step, if you do not need a cross-validation to find proper hyper-parameters, then simply using *.train and *.predict is enough. Here, * represents your model declared in step 2)

Entire codes and other examples can be found in [here](example.ipynb)

## Citations

@article{zhang2023gnn4eeg,
  title={GNN4EEG: A Benchmark and Toolkit for Electroencephalography Classification with Graph Neural Network},
  author={Zhang, Kaiyuan and Ye, Ziyi and Ai, Qingyao and Xie, Xiaohui and Liu, Yiqun},
  journal={arXiv preprint arXiv:2309.15515},
  year={2023}
}

## Contact

Kaiyuan Zhang (<kaiyuanzhang2001@gmail.com>)

Ziyi Ye (<yeziyi1998@gmail.com>)
