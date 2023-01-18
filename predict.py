# PROGRAMMER: John Paul Hunter
# DATE CREATED: Wednesday, January 18, 2023                                 
# REVISED DATE: 
# PURPOSE: Predicting classes - 
#          The predict.py script successfully reads in an image and a checkpoint then 
#          prints the most likely image class and it's associated probability *
#          Top K classes -
#          The predict.py script allows users to print out the top K classes along with associated probabilities *
#          Displaying class names -
#          The predict.py script allows users to load a JSON file that maps the class values to other category names *
#          Predicting with GPU -
#          The predict.py script allows users to use the GPU to calculate the predictions *

# Use argparse Expected Call with <> indicating expected user input:
#      python predict.py --image_location <location of image to test> --checkpoint_file <path to checkpoint file>
#             --topk <how many top-k predictions>  --category_names <load a JSON file that maps the class values to other category names> 
#   Example call:
#    python predict.py --image_location assets/flower_data/valid/1/image_06739.jpg --checkpoint_file checkpoint.pth 
#                           --topk 5  --category_names cat_to_name.json --gpu
##

# Imports python modules
from time import time, sleep
import json

# Imports functions created for this program
from get_input_args import get_predict_input_args
from load_checkpoint_model import load_checkpoint_model
from predict_model import predict_model

# Main program function defined below
def main():
    # TODO 0: Measures total program runtime by collecting start time
    start_time = time()
    # Replace sleep(75) below with code you want to time
    sleep(1)
    
    # TODO 1: Define get_input_args function within the file get_input_args.py
    # This function retrieves 3 Command Line Arugments from user as input from
    # the user running the program from a terminal window. This function returns
    # the collection of these command line arguments from the function call as
    # the variable in_arg
    in_arg = get_predict_input_args()

    image_location = in_arg.image_location
    checkpoint_file = in_arg.checkpoint_file
    gpu = in_arg.gpu
    topk = in_arg.topk
    category_names = in_arg.category_names

    if in_arg is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")
    else:
        # prints command line agrs
        print("Command Line Arguments:",
                "\n   image_location =", in_arg.image_location,
                "\n   checkpoint_file =", in_arg.checkpoint_file,
                "\n   gpu =", in_arg.gpu,    
                "\n   topk =", in_arg.topk, 
                "\n   category_names =", in_arg.category_names)

    # grab device from input arg  
    device = "cpu"
    if in_arg.gpu:        
        device = "gpu"

    # load up the checkpoint model
    model = load_checkpoint_model(checkpoint_file)
    
    print("Beginning Prediction...")
    
    ps, tclass = predict_model(image_location, model, topk, device)

    with open(category_names, "r") as f:
        # get category names from json
        cat_to_name = json.load(f)
        
        print("Prediction:")
        print("Flower: {}".format(cat_to_name[tclass[0]]))
        print("Class: {}".format(tclass[0]))
        print("Probability: {:.2f}%".format(ps[0] * 100))
        
        print("Topk:")
        for idx, i in enumerate(tclass):
            print("Name: {flower_name}".format(flower_name=cat_to_name[tclass[idx]]))
            print("Class: {flower_class}".format(flower_class= tclass[idx]))
            print("Probability: {flower_ps:.2f}%".format(flower_ps=ps[idx] * 100))
    
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
