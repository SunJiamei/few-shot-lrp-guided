few-shot-GNN-RN-lrp
================

This repo implements two few-shot learning models: graph neural net(GNN) and RelationNet.
The code is based on [CrossDomainFewShot](https://github.com/hytseng0509/CrossDomainFewShot).

We improve the two models using LRP explanations. The pre-trained models are available [here](https://drive.google.com/file/d/1tDuo5h0bf55NhuezmKcAEsHkZnYFSksj/view?usp=sharing).

## Prepare Dataset
The datasets used in this repo include cars, cub, miniImagenet, places, plantae. 

These datasets can be downloaded by running

`./filelists/process.py`

It will process each dataset into three `.json` files.

## Test pre-trained models
Please specify the params in `test.py` and test.

--name: `str` the path to the pre-trained model

--testset: `str` the testset name

--transductive: `bool` whether use transductive inference

--n_shot: `int` the number of support images per class

## To get LRP explanations

Please refer the  [lrpmodel.py](./methods/lrpmodel.py). There are two functions to generate the explanations of GNN and RN.

## To train the explanation-guided models

Please specify the `params.method` as `relationnetlrp` or `gnnnetlrp` to train the explanation-guided models.

For details of the models, please refer to `RelationNetLRP` in [./methods/relationnet.py](./methods/relationnet.py) and `GNNNetLRP` in [./methods/gnnnet.py](./methods/gnnnet.py).


### The same experiments using CAN models are [here](https://github.com/SunJiamei/few-shot-CANlrp)

### Acknowledgement
Many thanks to the developers of [innvestigate](https://github.com/albermax/innvestigate) and the important input from Phillip Seegerer on the LRP Pytorch implementation.
