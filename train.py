import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse

# this is the helper file I created
import nn_helper 

ap = argparse.ArgumentParser(description='train.py')
# Command Line ardguments

ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
ap.add_argument('--display_freq', type=int, dest="display_freq", action="store", default=20)

pa = ap.parse_args()
where = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
power = pa.gpu
epochs = pa.epochs
display_freq = pa.display_freq


trainloader, v_loader, testloader, train_data,validation_data,test_data  = nn_helper.load_and_tranform_data(where)


model, optimizer, criterion = nn_helper.build(structure,dropout,hidden_layer1,lr,power)

# train the neural network
nn_helper.train(model, optimizer, criterion, epochs, display_freq, trainloader, power)

#save the check point
nn_helper.save_checkpoint(model,train_data,path,structure,hidden_layer1,dropout,lr)


print("Training is completed !") 
