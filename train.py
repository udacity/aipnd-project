# TODO 0: Add your information below for Programmer & Date Created.                                                                             
# PROGRAMMER: John Paul Hunter
# DATE CREATED: Monday, January 16, 2023                                 
# REVISED DATE: Friday, January 20, 2023
# PURPOSE: Training a network - 
#          train.py successfully trains a new network on a dataset of images
#          Training validation log -
#          The training loss, validation loss, and validation accuracy are printed out as a network trains
#          Model architecture -
#          The training script allows users to choose from at least two different architectures available from torchvision.models
#          Model hyperparameters -
#          The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs
#          Training with GPU - 
#          The training script allows users to choose training the model on a GPU
#
#   Use argparse Expected Call with <> indicating expected user input:
#   1. Train
#   Train a new network on a data set with train.py
#
#   Basic usage: python train.py assets/flower_data
#   Prints out training loss, validation loss, and validation accuracy as the network trains
#   Options:
#   Set directory to save checkpoints: python train.py assets/flower_data --save_dir ./
#   Choose architecture: python train.py assets/flower_data --arch "vgg13"
#   Set hyperparameters: python train.py assets/flower_data --learning_rate 0.01 --hidden_units 512 --epochs 20

# imports python modules
from time import time, sleep
from torchvision import datasets, transforms, models
import torch
from torch import nn
from torch import optim
from collections import OrderedDict


# import globals
import globals

# import functions created for this program
from get_input_args import get_train_input_args
from train_model import train_model


# main program function defined below
def main():
    # measures total program runtime by collecting start time
    start_time = time()
    # replace sleep(75) below with code you want to time
    sleep(1)
    
    # Define get_input_args function within the file get_input_args.py
    # This function retrieves Command Line Arugments from user as input from
    # the user running the program from a terminal window. This function returns
    # the collection of these command line arguments from the function call as
    # the variable in_arg
    in_arg = get_train_input_args()
    
    # retrieve and assign in_arg properties
    data_dir = in_arg.data_dir
    save_dir = in_arg.save_dir
    arch = in_arg.arch
    learning_rate = in_arg.learning_rate
    hidden_units = in_arg.hidden_units
    epochs = in_arg.epochs
    gpu = in_arg.gpu

    if in_arg is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")
    else:
        # prints command line agrs
        print("Command Line Arguments:",
                "\n   data_dir =", in_arg.data_dir,
                "\n   save_dir =", in_arg.save_dir,
                "\n   arch =", in_arg.arch,    
                "\n   learning_rate =", in_arg.learning_rate, 
                "\n   hidden_units =", in_arg.hidden_units,    
                "\n   epochs =", in_arg.epochs, 
                "\n   gpu =", in_arg.gpu)
    
    # check if user has asked for gpu and assign to device variable
    if gpu:
        device = "cuda"
    else:
        device = "cpu"
    
    # set the training and vailidation dirs the folder directory
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"
        
    # define the transforms for the training and validation sets
    train_transforms = transforms.Compose([transforms.RandomRotation(globals.ROTATION_AMOUNT),
                                       transforms.RandomResizedCrop(globals.CROP_AMOUNT),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(globals.COLOR_CHANNEL_1,
                                       globals.COLOR_CHANNEL_2)])

    valid_transforms = transforms.Compose([transforms.Resize(globals.RESIZE_AMOUNT),
                                      transforms.CenterCrop(globals.CROP_AMOUNT),
                                      transforms.ToTensor(),
                                      transforms.Normalize(globals.COLOR_CHANNEL_1,
                                      globals.COLOR_CHANNEL_2)])                                
    
    # load the datasets with ImageFoldeR
    # pass transforms in here, then run the next cell to see how the transforms look
    image_train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    image_valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
    # using the image datasets and the trainforms, define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(image_train_dataset, batch_size=globals.BATCH_SIZE, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(image_valid_dataset, batch_size=globals.BATCH_SIZE)
    
    # labels list json script
    import json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        cat_count = len(cat_to_name.keys())

    # choose pretrained model
    if arch =='vgg16':
        model = models.vgg16(pretrained = True)
        input_units = 25088
    elif arch =='vgg13':
        model = models.vgg13(pretrained = True)
        input_units = 25088
    elif arch =='alexnet':
        model = models.alexnet(pretrained = True)
        input_units = 9216
    else:
        print("Please input either \"vgg16\", \"vgg13\" or \"alexnet\"")
        
    for param in model.parameters():
        param.requires_grad = False
    
        classifier = nn.Sequential(
                nn.Linear(input_units, hidden_units),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(hidden_units, 256),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(256, cat_count),
                nn.LogSoftmax(dim = 1)
            )

        # replace current models classifier with this one
        model.classifier = classifier

        # specify optimizer (stochastic gradient descent) and use the default learning rate (0.001)
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        
    # freeze our feature paramaters - turn of gradients for our model
    for param in model.parameters():
        param.require_grad = False
    
    # specify loss function (categorical cross-entropy)
    criterion = nn.NLLLoss()

    # light 'em up
    print("Training starting!!!:  ") 
    print("device!:  "+ device) 
    
    # launch train_model func
    train_model(model, epochs, device, criterion, optimizer, train_dataloader, valid_dataloader)

    # define the checkpoint 
    save_checkpoint = {
        'arch': arch,
        'model': model,
        'classifier': classifier,
        'optimizer_state_dict': optimizer.state_dict,
        'class_to_idx': image_train_dataset.class_to_idx,
        'state_dict': model.state_dict()}

    # save the checkpoint    
    torch.save(save_checkpoint, save_dir + "/" + "checkpoint.pth")
        
    # display message to user
    print("Training complete, checkpoint file saved to:{}".format(save_dir+"/" + "checkpoint.pth"))
        
    # measure total program runtime by collecting end time
    end_time = time()
    
    # computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    
# call to main function to run the program
if __name__ == "__main__":
    main()
