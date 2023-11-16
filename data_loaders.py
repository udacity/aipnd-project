from torchvision import datasets, transforms
import torch
import pathlib


def get_data_loaders(data_dir):
    if not pathlib.Path(data_dir).exists():
        raise FileNotFoundError(f'Data directory not found {data_dir}')

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=train_transforms),
        'test': datasets.ImageFolder(test_dir, transform=test_transforms),
        'validation': datasets.ImageFolder(valid_dir, transform=validation_transforms)
    }

    # Using the image datasets and the trainforms, define the dataloaders
    data_loaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64),
        'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64, shuffle=True)
    }

    return {
        'datasets': image_datasets,
        'data_loaders': data_loaders
    }
