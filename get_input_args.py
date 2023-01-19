# PROGRAMMER: John Paul Hunter
# DATE CREATED: Monday, January 16, 2023                                   
# REVISED DATE: 
# PURPOSE: define imput args
#
##

# Imports python modules
import argparse
import os

def get_train_input_args():
    
    # setup parser
    parser = argparse.ArgumentParser()

    # Creates command line arguments args.dir for path to images files, 
    # and checkpoint files
    parser.add_argument("--data_dir", type=str, default="assets/flower_data/", 
                        help="path to folder of images")
    parser.add_argument("--save_dir", type=str, default=os.getcwd(), 
                        help="path to checkpoint folder")
    # parse.add_argument statements BELOW to add type & help for:
    #          --arch - the CNN model architecture
    #          --learning_rate - set the learning rate for the model
    #          --epochs - amount of run through epochs
    #          --gpu - is this a gpu enabled architecture leaving this off will result in false

    # args.arch which CNN model to use for classification
    parser.add_argument("--arch", type=str, default = "resnet50", help="which CNN architecture would you like?" )
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="input the learning rate for the model")
    parser.add_argument("--hidden_units", type=int, default=2048, help="input hidden units")
    parser.add_argument("--epochs", type=int, default=1, help="input the amount of epochs to run through") 
    parser.add_argument("--gpu", action="store_true", default=False, help="is this a GPU enabled computer?")

    # Return parser.parse_args() parsed argument 
    # collection that you created with this function 
    return parser.parse_args()


def get_predict_input_args():

    # setup parser
    parser = argparse.ArgumentParser()

    # Creates command line arguments args.dir for path to images files, 
    # and checkpoint files
    parser.add_argument("--image_location", type=str, default="assets/flower_data/valid/4/image_05638.jpg"
    ,help="path to image file for read in")
    parser.add_argument("--checkpoint_file", type=str, default=os.getcwd() + "/checkpoint.pth", 
                        help="path to checkpoint folder")
    # parse.add_argument statements BELOW to add type & help for:
    #          --gpu -is this a GPU enabled computer? (leaving this off will result in false)
    #          --topk -how many top-k predictions?
    #          --category_names - load a JSON file that maps the class values to other category names
    parser.add_argument("--gpu", action="store_true", default=False, help="is this a GPU enabled computer?")
    parser.add_argument("--topk", type=int, default=5, help="how many top-k predictions?")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json",
                        help="load a JSON file that maps the class values to other category names")
    

    # Return parser.parse_args() parsed argument 
    # collection that you created with this function 
    return parser.parse_args()
