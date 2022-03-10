# S2O
Code for CVPR 2022 Paper **'Enhancing Adversarial Training with Second-Order Statistics of Weights'**

# Requisite
Python 3.6+  
Pytorch 1.8.1+cu111

# How to use
AT+S2O for CIFAR10, ResNet18: run ./CIFAR10_AT_S2O/train_S2O.py  
We get best the performance between epoch 100-110  
clean: 83.65  PGD-20: 55.11  AA: 48.3

TRADES+AWP+S2O for CIFAR10, WRN34-10: run ./CIFAR10_AWP_S2O/train_S2O.py  
We get best the performance between epoch 100-200   
clean: 86.01  PGD-20: 61.12  AA: 55.9

MART+S2O for CIFAR10, WRN34-10: run ./CIFAR10_MART_S2O/train_S2O.py  
We get best the performance between epoch 100-200   
clean: 83.91  PGD-20: 59.29  AA: 54.1

# Evaluation
Following TRADES, we set epsilon=0.031, step_size=0.003 for PGD and CW evaluation. Auto attack evaluation is under standard version.


# Reference Code
[1] AT: https://github.com/locuslab/robust_overfitting  
[2] TRADES: https://github.com/yaodongyu/TRADES/  
[3] AutoAttack: https://github.com/fra31/auto-attack  
[4] MART: https://github.com/YisenWang/MART  
[5] AWP: https://github.com/csdongxian/AWP  
[6] AVMixup: https://github.com/hirokiadachi/Adversarial-vertex-mixup-pytorch
