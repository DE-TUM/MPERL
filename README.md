# MPERL for Entity Classification in Knowledge Graphs
This repository contains a Pytorch-based implementation of Markov Process and Evidential with Regularization Loss GCN (MPERL-GCN) for Entity Classification in Knowledge Graphs

## Setup
To use this package, you must install the following dependencies first: 
- Compatible with PyTorch 1.4.0 and Python 3.7.3.
- Dependencies can be installed using `requirements.txt`.

## Training
You can learn the representations on the datasets AIFB, MUTAG, BGS and AM via the following commands.


- AIFB: 
```shell
python run.py --data aifb --epochs 30 --bases 0 --hidden 16 --lr 0.01 --l2 5e-5 --lambda_p 0.2 --seed 0
```

- MUTAG: 
```shell
python run.py --data mutag --epochs 10 --bases 45 --hidden 16 --lr 0.005 --l2 1e-3 --drop 0.3 --lambda_p 0.2 --no_cuda --seed 0
```

- BGS: 
```shell
python run.py --data bgs --epochs 31 --bases 45 --hidden 16 --lr 0.005 --l2 5e-3 --drop 0.2 --no_cuda --lambda_p 0.2 --seed 0
```

- AM:
```shell
python run.py --data am --epochs 55 --bases 55 --hidden 10 --lr 0.005 --l2 5e-5 --lambda_p 0.2 --no_cuda --steplr 45 --seed 0
```
Note: Results depend on random seed and will vary between re-runs.
* `--bases` for RGCN basis decomposition
* `--data` denotes training datasets
* `--hidden` is the dimension of hidden GCN Layers
* `--lr` denotes learning rate
* `--l2` is the weight decay parameter of L2 regularization
* `--drop` is the dropout value for training GCN Layers
* `--lambda_p` the number of trials up to and including the first success for the geometric prior distribution. Expected number of Markov steps
* `--seed` the seed number to reproduce results




You can also run the following command to run MPERL-GCN on all datasets.
```shell
sh run.sh
```



## Benchmark Methods for Reproducibility
* [WL](https://github.com/BorgwardtLab/WWL): We used a tree depth of 2 and 3 iterations.
* [RDF2Vec](https://github.com/IBCNServices/pyRDF2Vec): For the embeddings, we used a learning rate of 0.025, dimension size of 500, a window size of 10, 10 SkipGram iterations and 25 negative samples. As Classifier, we used a linear SVM with l2 penalty, squared hinge loss and a regularization parameter of 1.0.
* [R-GCN](https://github.com/berlincho/RGCN-pytorch): We used a 2-layer model with 16 hidden units (10 for AM), a weight decay of 0.5e-3, a learning rate of 0.01 and apply Adam as optimizer over 50 epochs.
* [ConnectE](https://github.com/Adam1679/ConnectE): For the embeddings, we used a learning rate of 0.1, dimension size of 200, L2 norm with margin 2 and 1000 epochs. As Classifier, we used a linear SVM with l2 penalty, squared hinge loss and a regularization parameter of 1.0.
* [ASSET](https://github.com/dice-group/ASSET): We used the ConnectE Embeddings, a 2-layer NN as teacher and student model. 
* [E-R-GCN](https://github.com/TobiWeller/E-R-GCN): : We used a 2-layer model with 16 hidden units (10 for AM), a weight decay of 0.5e-3, a learning rate of 0.01 and apply Adam as optimizer over 50 epochs.
* [CompGCN](https://github.com/zhjwy9343/dgl/tree/master_compgcn_4_review/examples/pytorch/compgcn): For each dataset, we used the hyperparemets indicated in the provided DGL-CompGCN repository.

