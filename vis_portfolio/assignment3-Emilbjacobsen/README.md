# Assignment3

## Contribution
This code has been made in collaboration with others in class, though the final result has been made by me. The utils functions have been made by Ross

## Assignment task

In the previous assignments involving classification, we were performing a kind of simple feature extraction on images by making them greyscale and flattening them to a single vector. This vector of pixel values was then used as the input for some kind of classification model.

For this assignment, we're going to be working with an interesting kind of cultural phenomenon - fashion. On UCloud, you have access to a dataset of *Indo fashion* taken from this [Kaggle dataset](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset). There is a paper which goes along with it on *arXiv.org*, which you can read [here](https://arxiv.org/abs/2104.02830).

Your instructions for this assignment are short and simple:

- You should write code which trains a classifier on this dataset using a *pretrained CNN like VGG16*
- Save the training and validation history plots
- Save the classification report


## Methods

The dataset is prepared using generators to apply labels and prep the data. Then the script trains a pretrained vgg16 classifier with locked weights and some layers that can be trained to classify the dataset. The output of the model is a report, a loss curve and an accuracy curve.

## Usage

### Hardware
I ran this script on a ucloud 32 cpu machine with 192 GB of memory and it took about 20 hours.


### data
Data can be found here: 
https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset

place the zip file in the data folder and unzip it from terminal using:
unzip archive.zip 



### running the script
If using ucloud, please make sure to have venv installed:
sudo apt-get update
sudo apt-get install python3-venv

Set assignment3-Emilbjacobsen as working directory and run "bash setup.sh"

For some reason the virtual environment won’t activate from my sh file, if you have the same problem, please just run this with assignment3-vis-Emilbjacobsen as working directory after running setup.sh, you can tell its activated if it’s in a parenthesis next to the coder in terminal:

source ./assignment3_viz_env/bin/activate

Then go to the src dir and run this code:
python3 script.py

Some parameters are modifiable from the command line via argparse but are not needed for the code to run, see script for details.


## Results
The model achieves an overall accuracy of 82, with differing f1 scores for each class in the range from 0.61 to 0.95. specifically, gowns seem to be difficult for the model to pick up on.

                      precision    recall  f1-score   support

              blouse       0.93      0.96      0.95       500
         dhoti_pants       0.86      0.66      0.74       500
            dupattas       0.80      0.66      0.73       500
               gowns       0.75      0.51      0.61       500
           kurta_men       0.76      0.94      0.84       500
leggings_and_salwars       0.71      0.85      0.77       500
             lehenga       0.92      0.90      0.91       500
         mojaris_men       0.91      0.85      0.88       500
       mojaris_women       0.86      0.90      0.88       500
       nehru_jackets       0.92      0.90      0.91       500
            palazzos       0.94      0.76      0.84       500
          petticoats       0.91      0.86      0.88       500
               saree       0.76      0.95      0.85       500
           sherwanis       0.94      0.72      0.81       500
         women_kurta       0.59      0.90      0.71       500

            accuracy                           0.82      7500
           macro avg       0.84      0.82      0.82      7500
        weighted avg       0.84      0.82      0.82      7500

The accuracy and loss curves for the model seems to suggest that it starts overfitting at 6-7th epoch meaning that you could probably cut down on training time and still get similar results
