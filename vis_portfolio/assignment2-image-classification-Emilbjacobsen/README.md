# Visual assignment 2

## Contribution
This code was made in collaboration with others in class. Though the result has been made by me. The code I used for the classifiers have mainly been provided by Ross during lessons.

## Assignment task

For this assignment, we'll be writing scripts which classify the ```Cifar10``` dataset.

You should write code which does the following:

- Load the Cifar10 dataset
- Preprocess the data (e.g. greyscale, reshape)
- Train a classifier on the data
- Save a classification report

You should write one script which does this for a logistic regression classifier **and** one which does it for a neural network classifier. In both cases, you should use the machine learning tools available via ```scikit-learn```.

## Methods

These two scripts load the cifar 10 datasets and use sickitlearn to train a logistic regression classifier and a neural network and saves reports of how well they each perform at classifying the images.

## Usage

### Hardware
I ran this script on a ucloud 16 cpu machine with 96 GB of memory.


### Data
Data can be found here: 
https://www.kaggle.com/c/cifar-10

### Running the script
If using ucloud, please make sure to have venv installed:
sudo apt-get update
sudo apt-get install python3-venv

Run "bash setup.sh" from terminal, it will make a virtual environment and install all nessescary packages in it.

For some reason the virtual environment won’t activate from my sh file, if you have the same problem, please just run this with assignment2-vis-Emilbjacobsen as working directory after running setup.sh, you can tell its activated if it’s in a parenthesis next to the coder in terminal:

source ./assignment2_viz_env/bin/activate

then go to src directory and paste:
"python3 script_lrc.py" and "python3 script_nnc.py" depending on whether you would like to train a logistic regression model or a neural network

The parameters of the models can be modified using argparse(see scripts for details), if no commands are specified it will go with the defaults




## Results

The code functions and produces two reports as expected. the models were trained on a dataset of images, the images were in these categories:
        airplane
        automobile
        bird
        cat
        deer
        dog
        frog
        horse 
        ship
        truck

 The two reports both show underwhelming performance on most stats, with a accuracy score of 31 for the logistic regression model and 35 for the neural network, so the neural network model is slightly better. Both models seem to especially struggle with animals, as for example cats have an f1 score of 16 and 17 in the models. All other animals also seen to have lower f1 scores than their vehicular counterparts, showing that both models struggle with the similar shapes of animals compared the other images who differ more in shape.
