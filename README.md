# CIFAR-10 VGG Hyperparameter Sweep using Weights & Biases

This repository demonstrates how to perform **Bayesian hyperparameter optimization** on a VGG6 model trained on the CIFAR-10 dataset using **Weights & Biases Sweeps**.

---

## Project Structure
.  
├── train.py # Model training script  
├── sweep.yaml # W&B sweep configuration file  
├── best_model.pth # Model file saved with best configuration  
└── README.md # This guide  

---

## Prerequisites

Before running the sweep, make sure you have:

1. **Python 3.12**
2. **Weights & Biases (wandb) installed**
   ```bash
   pip install wandb
3. Logged in to W&B
   wandb login
   Use your W&B API key from https://wandb.ai/authorize

## Define Sweep Configuration

Sweep configuration is already defined in sweep.yaml


## Create the Sweep

wandb sweep sweep.yaml

This will output something like:
Created sweep with ID: neerajcher-indian-institute-of-technology-madras/cifar10-vgg-hparams/abc123


## Launch Agents

Each agent runs one experiment using parameters sampled by W&B.

Run the following command (replace <sweep_id> with your actual ID):
wandb agent neerajcher-indian-institute-of-technology-madras/cifar10-vgg-hparams/<sweep_id>


## The best Network Configuration:
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
VGG(  
  (features): Sequential(  
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
    (2): SELU(inplace=True)  
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
    (5): SELU(inplace=True)  
    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  
    (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
    (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
    (9): SELU(inplace=True)  
    (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
    (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
    (12): SELU(inplace=True)  
    (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  
  )  
  (classifier): Sequential(  
    (0): Linear(in_features=128, out_features=10, bias=True)  
  )  
)  
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
