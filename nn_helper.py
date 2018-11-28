import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

arch = {"vgg16":25088,
        "densenet121":1024,
        "alexnet":9216}


def load_data(where  = "./flowers" ):
    '''
    Arguments : the datas' path
    Returns : The loaders for the train, validation and test datasets

    This function receives the location of the image files, applies the necessery transformations (rotations,flips,normalizations and crops) and converts the images to tensor in order to be able to be fed into the neural network

    '''

    
    data_dir = where[0]
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Apply the required transfomations to the test dataset in order to maximize the efficiency of the learning
    #process


    train_transforms = transforms.Compose([transforms.RandomRotation(50),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # Crop and Resize the data and validation images in order to be able to be fed into the network

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    # The data loaders are going to use to load the data to the NN(no shit Sherlock)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    vloader = torch.utils.data.DataLoader(validation_data, batch_size =32,shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 20, shuffle = True)



    return trainloader , vloader, testloader


def nn_setup(structure='densenet121',dropout=0.5, hidden_layer1 = 120,lr = 0.001,power='gpu'):
    '''
    Arguments: The architecture for the network(alexnet,densenet121,vgg16), the hyperparameters for the network (hidden layer 1 nodes, dropout and learning rate) and whether to use gpu or not
    Returns: The set up model, along with the criterion and the optimizer fo the Training

    '''

    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("Im sorry but {} is not a valid model.Did you mean vgg16,densenet121,or alexnet?".format(structure))

    for param in model.parameters():
        param.requires_grad = False

        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
            ('dropout',nn.Dropout(dropout)),
            ('inputs', nn.Linear(25088, hidden_layer1)),
            ('relu1', nn.ReLU()),
            ('hidden_layer1', nn.Linear(hidden_layer1, 90)),
            ('relu2',nn.ReLU()),
            ('hidden_layer2',nn.Linear(90,80)),
            ('relu3',nn.ReLU()),
            ('hidden_layer3',nn.Linear(80,102)),
            ('output', nn.LogSoftmax(dim=1))
                          ]))


        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr )

        if torch.cuda.is_available() and power == 'gpu':
            model.cuda()

        return model, criterion, optimizer


def train_network(model, criterion, optimizer, epochs = 3, print_every=20, loader=None, power='gpu'):
    '''
    Arguments: The model, the criterion, the optimizer, the number of epochs, teh dataset, and whether to use a gpu or not
    Returns: Nothing

    This function trains the model over a certain number of epochs and displays the training,validation and accuracy every "print_every" step using cuda if specified. The training method is specified by the criterion and the optimizer which are NLLLoss and Adam respectively

    '''
    steps = 0
    running_loss = 0

    print("--------------Training is starting------------- ")
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(loader):
            steps += 1
            if torch.cuda.is_available() and power=='gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                vlost = 0
                accuracy=0


                for ii, (inputs2,labels2) in enumerate(loader):
                    optimizer.zero_grad()
                    if torch.cuda.is_available():
                        inputs2, labels2 = inputs2.to('cuda:0') , labels2.to('cuda:0')
                        model.to('cuda:0')

                    with torch.no_grad():
                        outputs = model.forward(inputs2)
                        vlost = criterion(outputs,labels2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                vlost = vlost / len(loader)
                accuracy = accuracy /len(loader)



                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Lost {:.4f}".format(vlost),
                       "Accuracy: {:.4f}".format(accuracy))


                running_loss = 0


    print("-------------- Finished training -----------------------")
    print("Dear User I the ulitmate NN machine trained your model. It required")
    print("----------Epochs: {}------------------------------------".format(epochs))
    print("----------Steps: {}-----------------------------".format(steps))
    print("That's a lot of steps")


def save_checkpoint(path='checkpoint.pth',structure ='densenet121', hidden_layer1=120,dropout=0.5,lr=0.001,epochs=12):
    '''
    Arguments: The saving path and the hyperparameters of the network
    Returns: Nothing

    This function saves the model at a specified by the user path

    '''
    model.class_to_idx = train_data.class_to_idx
    model.cpu
    torch.save({'structure' :structure,
                'hidden_layer1':hidden_layer1,
                'dropout':dropout,
                'lr':lr,
                'nb_of_epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)


def load_checkpoint(path='checkpoint.pth'):
    '''
    Arguments: The path of the checkpoint file
    Returns: The Neural Netowrk with all hyperparameters, weights and biases

    '''
    checkpoint = torch.load(path)
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    dropout = checkpoint['dropout']
    lr=checkpoint['lr']

    model,_,_ = nn_setup(structure , dropout,hidden_layer1,lr)

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])


def process_image(image_path):
    '''
    Arguments: The image's path
    Returns: The image as a tensor

    This function opens the image usign the PIL package, applies the  necessery transformations and returns the image as a tensor ready to be fed to the network

    '''

    for i in image_path:
        path = str(i)
    img = Image.open(i) # Here we open the image

    make_img_good = transforms.Compose([ # Here as we did with the traini ng data we will define a set of
        # transfomations that we will apply to the PIL image
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor_image = make_img_good(img)

    return tensor_image


def predict(image_path, model, topk=5,power='gpu'):
    '''
    Arguments: The path to the image, the model, the number of prefictions and whether cuda will be used or not
    Returns: The "topk" most probable choices that the network predicts

    '''

    if torch.cuda.is_available() and power=='gpu':
        model.to('cuda:0')

    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    if power == 'gpu':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img_torch)

    probability = F.softmax(output.data,dim=1)

    return probability.topk(topk)
