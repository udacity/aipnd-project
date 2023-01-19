# PROGRAMMER: John Paul Hunter
# DATE CREATED: Wednesday, January 18, 2023                                 
# REVISED DATE: 
# PURPOSE: Function that performs the prediction  

from process_image import process_image

import torch

def predict_model(image_path, model, topk, this_device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    # run the modal in eval mode
    model.eval()

    # fix unexpected scalar type Double but found Float
    model.double()

    # model to gpu if this device suports
    model = model.to(this_device);
    
    # call our process_image func above
    image = process_image(image_path)
    
    # image to this device - unsqueeze the tensor
    image = image.to(this_device).unsqueeze(0)

    with torch.no_grad():
        
        # extract the top-k value-indices
        output = model.forward(image)
        topk ,topk_labels = torch.topk(output, topk)
        topk = topk.exp()

    # pupulate idx_to_class for labels
    idx_to_class = {model.class_to_idx[k]: k for k in model.class_to_idx}

    # create an empty array to store the class labels in
    classes = []

    # loop though predictions and populate classes
    for label in topk_labels.cpu().numpy()[0]:
        classes.append(idx_to_class[label])

    # retun topk and classes array
    return topk.cpu().numpy()[0], classes
