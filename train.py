# TODO 0: Add your information below for Programmer & Date Created.                                                                             
# PROGRAMMER: John Paul Hunter
# DATE CREATED: Monday, January 16, 2023                                 
# REVISED DATE: 
# PURPOSE: Training a network - 
#          train.py successfully trains a new network on a dataset of images x
#          Training validation log -
#          The training loss, validation loss, and validation accuracy are printed out as a network trains x
#          Model architecture -
#          The training script allows users to choose from at least two different architectures available from torchvision.models x
#          Model hyperparameters -
#          The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs x
#          Training with GPU - 
#          The training script allows users to choose training the model on a GPU x
#
# Use argparse Expected Call with <> indicating expected user input:
#      python train.py --dir <directory with images> --arch <model>
#             --learning_rate <learning rate for the model>  --hidden_units <hidden units for the model> --epochs <epochsfor the model> --gpu <add it by itself for true>
#   Example call:
#    python train.py --data_dir assets/flower_data/ --arch vgg16 --learning_rate 0.0001 --hidden_units 2048 --epochs 1 --gpu 
##

# Imports python modules
from time import time, sleep
from torchvision import datasets, transforms, models
import torch
from torch import nn
from torch import optim
from collections import OrderedDict

# Imports functions created for this program
import globals
from get_input_args import get_train_input_args
from train_model import train_model

#load up globals
#globals()

# Main program function defined below
def main():
# Setup some constants


    # Measures total program runtime by collecting start time
    start_time = time()
    # Replace sleep(75) below with code you want to time
    sleep(1)
    
    # Define get_input_args function within the file get_input_args.py
    # This function retrieves Command Line Arugments from user as input from
    # the user running the program from a terminal window. This function returns
    # the collection of these command line arguments from the function call as
    # the variable in_arg
    in_arg = get_train_input_args()
    
    # Retrieve and assign in_arg properties
    data_dir = in_arg.data_dir
    arch = in_arg.arch
    device = in_arg.gpu
    save_dir = in_arg.save_dir
    epochs = in_arg.epochs
    learning_rate = in_arg.learning_rate
    hidden_units = in_arg.hidden_units
    
    # Set the training and vailidation dirs the folder directory
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"

    if in_arg is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")
    else:
        # prints command line agrs
        print("Command Line Arguments:",
                "\n   data_dir =", in_arg.data_dir,
                "\n   save_dir =", in_arg.save_dir,
                "\n   arch =", in_arg.arch,    
                "\n   learning_rate =", in_arg.learning_rate, 
                "\n   gpu =", in_arg.gpu)

        
    # Define the transforms for the training and validation sets
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
    
    # Load the datasets with ImageFoldeR
    # Pass transforms in here, then run the next cell to see how the transforms look
    image_train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    image_valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(image_train_dataset, batch_size=globals.BATCH_SIZE, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(image_valid_dataset, batch_size=globals.BATCH_SIZE)
    
    # labels list json script
    import json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        cat_count = len(cat_to_name.keys())

    # choose pretrained model
    # option #1
    if arch == 'resnet50':
        model = models.resnet50(pretrained=True)
        classifier = nn.Sequential(
            nn.Linear(in_features=hidden_units, out_features=cat_count),
            nn.LogSoftmax(dim=1))
        
        # replace current models classifier with this one
        model.fc = classifier

        # specify optimizer (stochastic gradient descent) and use the default learning rate (0.001)
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    
    # option #2
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088,4096)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(4096,hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(p=0.5)),
            ('fc3', nn.Linear(hidden_units, cat_count)),
            ('output', nn.LogSoftmax(dim=1))]))

        # replace current models classifier with this one
        model.classifier = classifier

        # specify optimizer (stochastic gradient descent) and use the default learning rate (0.001)
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    else:
        print("Please input either \"resnet50\" or \"vgg16\"")
        
    # Freeze our feature paramaters - turn of gradients for our model
    for param in model.parameters():
        param.require_grad = False
    
    # specify loss function (categorical cross-entropy)
    criterion = nn.NLLLoss()

    # light 'em up    
    print("Training starting!!!:  ") 
    
    # launch train_model func 
    train_model(model, epochs, device, criterion, optimizer, train_dataloader, valid_dataloader)

    # define the checkpoint 
    checkpoint = {
        'arch': model,
        'epochs': epochs,
        'dropout': 0.5,
        'classifier': classifier,
        'hidden_layers': hidden_units,
        'optimizer_state_dict': optimizer.state_dict,
        'class_to_idx': image_train_dataset.class_to_idx,
        'state_dict': model.state_dict()}

    # save the checkpoint    
    torch.save(checkpoint, save_dir + "/" + "checkpoint.pth")
        
    #display message to user
    print("Training complete, checkpoint file saved to:{}".format(save_dir+"/" + "checkpoint.pth"))
        
    # Measure total program runtime by collecting end time
    end_time = time()
    
    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    
# Call to main function to run the program
if __name__ == "__main__":
    main()
