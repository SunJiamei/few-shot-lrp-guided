few-shot-GNN-RN-lrp
================

This repo implements two few-shot learning models: graph neural net(GNN) and RelationNet.
The code is based on [CrossDomainFewShot](https://github.com/hytseng0509/CrossDomainFewShot).

We improve the two model using LRP explanations. The pre-trained models are availabel [here](https://drive.google.com/file/d/1pVLQz7z7njTaeBOxSr_gCyX7otSzEKdQ/view?usp=sharing).

## Prepare Dataset
The datasets used in this repo include: cars, cub, miniImagenet, places, plantae. 

These datasets can be downloaded by run

`./filelists/process.py`

This will process each dataset into three `.json` files.

## Test pre-trained models
Please specify the params in `test.py` and run.

--name: `str` the path to the pretrained model

--testset: `str` the testset name

--transductive: `bool` whether use transductive inference

--n_shot: `int` the number of support images per class

## To get LRP explanations

Pleas refer the  [lrpmodel.py](./methods/lrpmodel.py). There are two functions to generate the explanations of GNN and RN.

### The same experiments using CAN models are [here](https://github.com/SunJiamei/few-shot-CANlrp)