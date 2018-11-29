# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.



Part 2 - Building the command line application

    a. Files created:
        i. nn_helper.py 
           This file is created to use modularity in programming for better reusablities. Almost all of the codes are taken from part 1.

    b. Files updated:
        i. train.py
        ii. predict.py

How to use this program:
    a. Pre-requistes : 
       i. Make sure GPU mode is enabled. Relevant code is
                
                "use_gpu = torch.cuda.is_available()\n",
                "if use_gpu:\n",
                "print(\"Using CUDA\")"
        ii. Use high RAM powered machine, otherwise you may get the below error:
                RuntimeError: $ Torch: not enough memory: you tried to allocate 6GB. Buy new RAM! at ..\aten\src\TH\THGeneral.cpp:204


    b. Fun part : how to use it?
        i. Clone the repository
        ii. Go to the source directory , and open command prompt

        USE TRAIN.PY

        iii. enter the below command to Prints out training loss, validation loss, and validation accuracy as the network trains
                python train.py flowers 
        iv. Set directory to save checkpoints: 
                python train.py flowers --arch "vgg16" 
        v. Explore other options as well. please note , this neural network only supports vgg16. Did not get enough time to explore other structure.

        Use PREDICT.PY
         
        vi. python predict.py 
            Change arguments and test it in different ways.

