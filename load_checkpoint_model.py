# PROGRAMMER: John Paul Hunter
# DATE CREATED: Wednesday, January 18, 2023                                 
# REVISED DATE: 
# PURPOSE: Function that loads the checkpoint model

import torch
from torchvision import models

def load_checkpoint_model(path):
    # load the checkpoint
    checkpoint = torch.load(path)
 
    #load up the checkpoint model
    checkpoint_model  = checkpoint['model']
  
    # freeze the parameters
    for param in checkpoint_model.parameters():
        param.requires_grad = False

    # load up the optimiser
    checkpoint_model.optimizer = checkpoint['optimizer_state_dict']

    # apply custom fc back to the model
    checkpoint_model.load_state_dict(checkpoint['state_dict'])
    checkpoint_model.class_to_idx = checkpoint['class_to_idx']

    # provide user feedback
    print(checkpoint['arch'] +  " checkpoint model Loaded")

    return checkpoint_model