# PROGRAMMER: John Paul Hunter
# DATE CREATED: Wednesday, January 18, 2023                                 
# REVISED DATE: Friday, January 20, 2023
# PURPOSE: Predicting classes - 
#          The predict.py script successfully reads in an image and a checkpoint then 
#          prints the most likely image class and it's associated probability
#          Top K classes -
#          The predict.py script allows users to print out the top K classes along with associated probabilities
#          Displaying class names -
#          The predict.py script allows users to load a JSON file that maps the class values to other category names
#          Predicting with GPU -
#          The predict.py script allows users to use the GPU to calculate the predictions

# Use argparse Expected Call with <> indicating expected user input:
#   2. Predict
#   Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.
#
#   Basic usage: python predict.py assets/flower_data/valid/1/image_06739.jpg checkpoint.pth 
#   Options:
#   Return top K most likely classes: python predict.py assets/flower_data/valid/1/image_06739.jpg checkpoint.pth --top_k 3
#   Use a mapping of categories to real names: 
#               python predict.py assets/flower_data/valid/1/image_06739.jpg checkpoint.pth --category_names cat_to_name.json
#   Use GPU for inference: python predict.py assets/flower_data/valid/1/image_06739.jpg checkpoint.pth --gpu

# imports python modules
from time import time, sleep
import json

# imports functions created for this program
from get_input_args import get_predict_input_args
from load_checkpoint_model import load_checkpoint_model
from predict_model import predict_model

# main program function defined below
def main():
    
    # measures total program runtime by collecting start time
    start_time = time()
    
    # replace sleep(75) below with code you want to time
    sleep(1)
    
    # Define get_input_args function within the file get_input_args.py
    # This function retrieves 3 Command Line Arugments from user as input from
    # the user running the program from a terminal window. This function returns
    # the collection of these command line arguments from the function call as
    # the variable in_arg
    in_arg = get_predict_input_args()

    input = in_arg.input
    checkpoint = in_arg.checkpoint
    top_k = in_arg.top_k
    category_names = in_arg.category_names
    gpu = in_arg.gpu

    if in_arg is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")
    else:
        # prints command line agrs
        print("Command Line Arguments:",
                "\n   image_location =", in_arg.input,
                "\n   checkpoint_file =", in_arg.checkpoint,  
                "\n   top_k =", in_arg.top_k, 
                "\n   category_names =", in_arg.category_names,
                 "\n  gpu =", in_arg.gpu,  )

    # grab device from input arg  
    device = "cpu"
    if gpu:
        device = "cuda"

    # load up the checkpoint model
    model = load_checkpoint_model(checkpoint)
    
    print("Beginning Prediction...")
    print("device!:  "+ device) 
    ps, tclass = predict_model(input, model, top_k, device)

    with open(category_names, "r") as f:
        # get category names from json
        cat_to_name = json.load(f)
        
        print("Prediction:")
        print("Flower: {}".format(cat_to_name[tclass[0]]))
        print("Class: {}".format(tclass[0]))
        print("Probability: {:.2f}%".format(ps[0] * 100))
        
        print("Top_k:")
        for idx, i in enumerate(tclass):
            print("Name: {flower_name}".format(flower_name=cat_to_name[tclass[idx]]))
            print("Class: {flower_class}".format(flower_class= tclass[idx]))
            print("Probability: {flower_ps:.2f}%".format(flower_ps=ps[idx] * 100))
    
    # measure total program runtime by collecting end time
    end_time = time()
    
    # computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )

# call to main function to run the program
if __name__ == "__main__":
    main()
