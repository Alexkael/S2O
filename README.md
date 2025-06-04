# <p align="center">S2O</p>
<p align="center">Code for CVPR 2022/TPAMI 2025 Paper "S^2O: Enhancing Adversarial Training with Second-Order Statistics of Weights"</p> 

[![Arxiv](https://img.shields.io/badge/Arxiv-2303.10653-red.svg)](https://arxiv.org/abs/2203.06020)
[![Pub](https://img.shields.io/badge/Pub-CVPR'22-blue.svg)](https://arxiv.org/abs/2203.06020)
![License](https://img.shields.io/badge/license-MIT-yellow) 
 

# Requisite
Python 3.6+  
Pytorch 1.8.1+cu111

# How to use
AT+S2O for CIFAR10, ResNet18: run ./CIFAR10_AT_S2O/train_S2O.py  
We got the best performance between epoch 100-110  
clean: 83.65  PGD-20: 55.11  AA: 48.3

TRADES+AWP+S2O for CIFAR10, WRN34-10: run ./CIFAR10_AWP_S2O/train_S2O.py  
We got the best performance between epoch 100-200   
clean: 86.01  PGD-20: 61.12  AA: 55.9

MART+S2O for CIFAR10, WRN34-10: run ./CIFAR10_MART_S2O/train_S2O.py  
We got the best performance between epoch 100-200   
clean: 83.91  PGD-20: 59.29  AA: 54.1

TRADES+S2O for CIFAR10, WRN34-10: run ./CIFAR10_TRADES_S2O/train_S2O.py  
We got the best performance between epoch 100-110   
clean: 85.67  PGD-20: 58.34  AA: 54.1

For TRADES+AWP+S2O on CIFAR100,
please modify CIFAR10_AWP_S2O/train_S2O.py lines 207, 221: 0.8 -> 0.98; lines 209, 223: 0.2 -> 0.02

# Evaluation
Following TRADES, we set epsilon=0.031, step_size=0.003 for PGD and CW evaluation. Auto attack evaluation is under standard version.

# Citing this work
@article{jin2022enhancing,  
  title={Enhancing Adversarial Training with Second-Order Statistics of Weights},  
  author={Gaojie Jin and Xinping Yi and Wei Huang and Sven Schewe and Xiaowei Huang},  
  journal={CVPR},  
  year={2022}. 
}

@article{jin2025s2o,  
  title={S^2O: Enhancing Adversarial Training with Second-Order Statistics of Weights},  
  author={Gaojie Jin and Xinping Yi and Wei Huang and Sven Schewe and Xiaowei Huang},  
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},  
  year={2025}. 
}

# Reference Code
[1] AT: https://github.com/locuslab/robust_overfitting  
[2] TRADES: https://github.com/yaodongyu/TRADES/  
[3] AutoAttack: https://github.com/fra31/auto-attack  
[4] MART: https://github.com/YisenWang/MART  
[5] AWP: https://github.com/csdongxian/AWP  
[6] AVMixup: https://github.com/hirokiadachi/Adversarial-vertex-mixup-pytorch
