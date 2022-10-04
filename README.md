# PE-Net: Position Encoding based Convolutional Neural Networks for Machine Remaining Useful Life Prediction
This repository is implemented by [Ruibing Jin](https://ruibing-jin.github.io/).


**PE-Net is proposed to predict the machine remaining useful life (RUL) and achiveves state-of-the-art performaces on the C-MAPSS benchmark.**

## Abstract

Accurate remaining useful life (RUL) prediction is important in industrial systems. It prevents machines from working under failure conditions, and ensures that the industrial system works reliably and efficiently. Recently, many deep learning based methods have been proposed to predict RUL. Among these methods, recurrent neural network (RNN) based approaches show a strong capability of capturing sequential information. This allows RNN based methods to perform better than convolutional neural network (CNN) based approaches on the RUL prediction task. In this paper, we question this common paradigm and argue that existing CNN based approaches are not designed according to the classic principles of CNN, which reduces their performances. Additionally, the capacity of capturing sequential information is highly affected by the receptive field of CNN, which is neglected by existing CNN based methods. To solve these problems, we propose a series of new CNNs, which show competitive results to RNN based methods. Compared with RNN, CNN processes the input signals in parallel so that the temporal sequence is not easily determined. To alleviate this issue, a position encoding scheme is developed to enhance the sequential information encoded by a CNN. Hence, our proposed position encoding based CNN called PE-Net is further improved and even performs better than RNN based methods. Extensive experiments are conducted on the C-MAPSS dataset, where our PE-Net shows state-of-the-art performance.

<img src='pe_net.png' width='1280' height='520'>

## Disclaimer
This is an official PyTorch implementation of "[Position Encoding based Convolutional Neural Networks for Machine Remaining Useful Life Prediction](https://ieeexplore.ieee.org/document/9849459)"

If you have any question regarding the paper, please send a email to `jin_ruibing[at]i2r[dot]a-star[dot]edu[dot]sg`.

## Citing PE-Net
If you find PE-Net useful in your research, please consider citing:

    @InProceedings{jin2022position,
        title={Position Encoding Based Convolutional Neural Networks for Machine Remaining Useful Life Prediction},
        author={Jin, Ruibing and Wu, Min and Wu, Keyu and Gao, Kaizhou and Chen, Zhenghua and Li, Xiaoli},
        journal={IEEE/CAA Journal of Automatica Sinica},
        volume={9},
        number={8},
        pages={1427--1439},
        year={2022},
        publisher={IEEE}
    }

## Requirements
This code has been tested in the following environment:

- Pytorch: v1.7.1

- CUDA v11.0

- Ubuntu: 18.04

- cuDNN: 8.1.1

- Python: 3.8.12

You may try to run it on other enviroments, but the performances may be different.

## Installation

1. Install CUDA and cuDNN according to your system environment.
2. Install PyTroch according to the official website based on the Anaconda (select Conda Package).
3. The directory which you clone our reposity, is denoted as $(PENET). Install other packages in your corresponding conda envirement:
```
cd $(PENET)
pip -r install requirements.txt
```

## Dataset Preparation
 
Please organise the C-MAPSS dataset as follows:
```
./data
    ├── CMAPSSData
    │   ├──RUL_FD001.txt
    │   ├──RUL_FD002.txt
    │   ├──RUL_FD003.txt
    │   ├──RUL_FD004.txt
    │   ├──test_FD001.txt
    │   ├──test_FD002.txt
    │   ├──test_FD003.txt
    │   ├──test_FD004.txt
    │   ├──train_FD001.txt
    │   ├──train_FD002.txt
    │   ├──train_FD003.txt
    └───└──train_FD004.txt           
```

## Training
All your configurations are included in ./exps/*.yaml and the default value are defined in ./config.py. 

Please run:
```
python train.py --cfg <config-file>
```
For example, run the training experiment with our proposed CNN-C on the FD004:
```
python train.py --cfg exps/cnn_c_fd004.yaml
```
Run the training experiment with our proposed PE-Net on the FD004:
```
python train.py --cfg exps/pe_net_fd004.yaml
```
We have provided five config files in ./exp: cnn_a_fd004.yaml, cnn_b_fd004.yaml, cnn_c_fd004.yaml, cnn_d_fd004.yaml and pe_net_fd004.yaml. 

You can change the value in the .yaml file according to yourself.