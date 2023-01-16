# PROGRAMMER: John Paul Hunter
# DATE CREATED: Monday, January 16, 2023                                   
# REVISED DATE: 
# PURPOSE: define imput args
#
##
# Imports python modules
import argparse
import os

def get_input_args():
    # Creates parse 
    parser = argparse.ArgumentParser()

    # Creates command line arguments args.dir for path to images files, 
    # and checkpoint files
    parser.add_argument('--data_dir', type=str, default='assets/flower_data/', 
                        help='path to folder of images')
    parser.add_argument('--save_dir', type=str, default=os.getcwd(), 
                        help='path to checkpoint folder')
    # parse.add_argument statements BELOW to add type & help for:
    #          --arch - the CNN model architecture
    #          --learning_rate - set the learning rate for the model
    #          --epochs - amount of run through epochs
    #          --gpu - is this a gpu enabled architecture

    # args.arch which CNN model to use for classification
    parser.add_argument('--arch', type=str, default = 'vgg16', help='which CNN architecture would you like?' )
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='input the learning rate for the model')
    parser.add_argument('--hidden_units', nargs=2, type=int, default=2048, help='input hidden units')
    parser.add_argument('--epochs', type=int, default=1, help='input the amount of epochs to run through') 
    parser.add_argument('--gpu', action='store_true', default=False, help='is this a GPU enabled computer?')

    # Return parser.parse_args() parsed argument 
    # collection that you created with this function 
    return parser.parse_args()
