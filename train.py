# All file imports
import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
# from PIL import Image


def parse_args():
    '''process arguments from the command line'''
    parser = argparse.ArgumentParser(
        description='Build and Train your Neural Network')

    parser.add_argument('data_dir', type=str, required=True,
                        help='directory of the training data (required)')
    parser.add_argument('--save_dir', type=str,
                        help='directory where to save your neural network. By default it will save in current directory')
    parser.add_argument(
        '--arch', type=str, help='models to pretrain from (vgg13, vgg19, densenet)')
    parser.add_argument('--learning_rate', type=float,
                        help='learning rate of training')
    parser.add_argument('--hidden_units', type=int,
                        help='number of hidden units of the network')
    parser.add_argument('--epochs', type=int, help='number of training epochs')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='use GPU for training when available')

    # parse arguments
    args = parser.parse_args()
    return args


def get_datasets(data_dir):

    print(f'getting data from directory {data_dir}')

    # data_dir = 'data/flowers/'
    train_dir = data_dir + 'train'
    valid_dir = data_dir + 'valid'
    test_dir = data_dir + 'test'

    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([.485, .456, .406],
                                                               [.229, .224, .225])])

    test_tranforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([.485, .456, .406],
                                                             [.229, .224, .225])])

    # TODO: Load the datasets with ImageFolder
    image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_tranforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_tranforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(
        image_datasets, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)

    # train_batch = next(iter(trainloader))
    # img, lbls = train_batch
    # print(img.shape, lbls.shape)
    # print(lbls)

    return trainloader, testloader, validloader


def build_model(arch):
    pass


def main():
    trainloader, testloader, validloader = get_datasets(args.data_dir)


if __name__ == '__main__':
    main()
