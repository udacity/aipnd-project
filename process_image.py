# PROGRAMMER: John Paul Hunter
# DATE CREATED: Wednesday, January 18, 2023                                 
# REVISED DATE: 
# PURPOSE: Function that processes an image

from PIL import Image
import torch
import numpy as np

import globals

def process_image(image):

    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # process a PIL image for use in a PyTorch model
    # resize where the sortest is 256, maintain aspect ratio
    size = globals.PIL_SIZE, globals.PIL_SIZE

    # open the image
    im  = Image.open(image)

    # use PIL thumbnail func to resize
    im.thumbnail(size)

    # retrieve image width and height
    im_width, im_height = im.size

    # set crop vars to crop out the centre
    crop_width, crop_height = globals.CROP_AMOUNT, globals.CROP_AMOUNT

    # work out crops
    left = (im_width - crop_width) / 2
    top = (im_height - crop_height) / 2
    right = (im_width + crop_width) / 2
    bottom = (im_height + crop_height) / 2

    # apply crops
    im =im.crop((left, top, right, bottom))
    
    # work our numpy colour channels
    np_image = np.array(im)
    np_image = np_image / 255

    # set up numpy colour channels as previous
    std = np.array(globals.COLOR_CHANNEL_1)
    mean = np.array(globals.COLOR_CHANNEL_2)

    # apply numpy colour channel calcs
    np_image = (np_image - mean) / std

    # apply to image
    image = np.transpose(np_image,(2,0,1))

    # return
    return torch.from_numpy(image)