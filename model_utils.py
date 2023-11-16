import torch
from torch import nn, optim
from torchvision import models
import json

import data_loaders

import pathlib


def get_cat_to_name():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name


def train_model(arch, learning_rate, data_dir, hidden_units, epochs, save_dir=None, gpu=True):
    '''
    Trains a model given the provided hyperparameters
    :param arch:
    :return: trained model
    '''

    device = get_device(gpu)

    if save_dir is not None and not pathlib.Path(save_dir).exists():
        raise FileNotFoundError(f'Save directory {save_dir} not found.')

    if arch not in models.list_models():
        raise RuntimeError(f'No such model {arch}. Available models: {models.list_models()}')

    data = data_loaders.get_data_loaders(data_dir)

    dataloaders = data['data_loaders']
    datasets = data['datasets']


    model = models.get_model(arch, weights=models.get_model_weights(arch))

    for param in model.parameters():
        param.requires_grad = False

    classifier = get_classifier(hidden_units)

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)

    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders['test']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Test loss: {test_loss / len(dataloaders['test']):.3f}.. "
                      f"Test accuracy: {accuracy / len(dataloaders['test']):.3f}")
                running_loss = 0
                model.train()

    model.arch = arch
    model.hidden_units = hidden_units
    model.optimizer = optimizer
    model.epochs = epochs
    model.gpu = gpu
    model.class_to_idx = datasets['train'].class_to_idx

    if save_dir is not None:
        save_model(model, save_dir)

    return model


def get_classifier(hidden_units):
    cat_to_name = get_cat_to_name()
    classifier = nn.Sequential(nn.Linear(1024, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(hidden_units, len(cat_to_name)),
                               nn.LogSoftmax(dim=1))
    return classifier


def get_device(gpu):
    if gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        if gpu:
            print(f'Using cpu since cuda is not available')

        device = 'cpu'

    return device


def save_model(model, save_dir):
    if not pathlib.Path(save_dir).exists():
        raise FileNotFoundError(f'Save directory {save_dir} not found.')

    checkpoint = {
        'arch': model.arch,
        'hidden_units': model.hidden_units,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'epochs': model.epochs,
        'gpu': model.gpu,
        'class_to_idx': model.class_to_idx
    }

    torch.save(checkpoint, f'{save_dir}/checkpoint.pth')


def load_model(save_dir):
    if not pathlib.Path(save_dir).exists():
        raise FileNotFoundError(f'Save directory {save_dir} not found.')

    checkpoint = torch.load(f'{save_dir}/checkpoint.pth')

    device = get_device(checkpoint['gpu'])

    model = models.get_model(checkpoint['arch'], weights=models.get_model_weights(checkpoint['arch']))
    model.classifier = get_classifier(checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.arch = checkpoint['arch']
    model.hidden_units = checkpoint['hidden_units']
    model.epochs = checkpoint['epochs']
    model.gpu = checkpoint['gpu']
    model.class_to_idx = checkpoint['class_to_idx']

    model.to(device)
    model.eval()

    return model
