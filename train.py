#Imports necessary tools
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from collections import OrderedDict
import json
import argparse
import time

#Imports cat_to_name.json file
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def get_arguemnts():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default="./flowers/", help='Determines which directory to pull information from')
	parser.add_argument('--save_dir', type=str, default='./checkpoint.pth', help='Enables user to choose directory for saving')
	parser.add_argument('--arch', type=str, default='vgg16', help='Determines which architecture you choose to utilize')
	parser.add_argument('--learning_rate', type=float, default=.001, help='Dictates the rate at which the model does its learning')
	parser.add_argument('--hidden_layer', type=int, default=1024, help='Dictates the hidden units for the hidden layer')
	parser.add_argument('--gpu', default='gpu', type=str, help='Determines where to run model: CPU vs. GPU')
	parser.add_argument('--epochs', type=int, default=3, help='Determines number of cycles to train the model')
	parser.add_argument('--dropout', type=float, default=0.5, help='Determines probability rate for dropouts')
	
	
	return parser.parse_args()
	

args = get_arguemnts()
#Pulls in and transforms data'
train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'
test_dir = args.data_dir + '/test'
# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])
}

# TODO: Load the datasets with ImageFolder
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
}

# TODO: Using the image datasets and the trainforms, define the dataloaders

trainloader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
validationloader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size =64,shuffle = True)
testloader = torch.utils.data.DataLoader(image_datasets['test'], batch_size = 64, shuffle = True)

dataloaders = {"train": trainloader, 
               "valid": validationloader,
               "test": testloader}

#Defines the model
def Classifier(arch='vgg16', dropout=0.5, hidden_layer=1024):
    if arch == 'vgg16':
        model = models.vgg19(pretrained=True)
        input_size = 25088
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = 1024
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, 1024)),
                          ('drop', nn.Dropout(dropout)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_layer, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    return model

#Establishs the model, criterion, and optimizer
model = Classifier(args.arch, args.dropout, args.hidden_layer)
for device in ['cpu', 'cuda']:

    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), args.learning_rate)

    model.to(device)

    for ii, (inputs, labels) in enumerate(trainloader):

        # Move input and label tensors to the GPU
        inputs, labels = inputs.to(device), labels.to(device)

        start = time.time()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ii==3:
            break
        
    print(f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds")
def train_model(model, optimizer, criterion):
    epochs = 5
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        test_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                model.train()
train_model(model, optimizer, criterion)          
# TODO: Save the checkpoint 
model.class_to_idx = image_datasets['train'].class_to_idx

checkpoint = {'input_size': 25088,
              'output_size': 102,
              'arch': 'arch',
              'learning_rate': 'learning_rate',
              'batch_size': 64,
              'epochs': 'epochs',
              'optimizer':optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}

torch.save(checkpoint, 'checkpoint.pth')
