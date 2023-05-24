# Assignment 4 self-assigned

## Contribution
This code was made in collaboration with others in class. The not pretrained classifier came in large part from here, though i updated and edited it.

https://www.kaggle.com/code/pierrenicolaspiquin/classifying-art-pieces

The code for the vgg16 classifier came mostly from the above project and my assignment 3.



## Assignment description
This project trains 2 classifiers to tell the difference between different forms of art, on the same dataset. The dataset contains images from 5 different categories, drawings, paintings, engravings, sculptures and iconographies and contains around 9000 images. The aim of this project is to see whether if classifier trained specifically on the data performs can compete with pretrained general purpose models, in this case the vgg16 model. 

A classification report, a loss curve and an accuracy curve will be made for each model to see how they compare to each other, they will be saved to the output folder


## Methods
The dataset unzips to a validation, training split, but I redistributed the data into a train, vaidationl, test split to ensure that the classification report was based on unseen data.
The code for this project is split into three different scripts. The first transforms the data into test, val, train split and the other two scripts load the art dataset and train CNNs to classify the images into different categories. The models being trained is a normal CNN and a pretrained CNN called vgg16 with some extra trainable layers. They use the same method for loading the dataset and producing the outputs.





## Usage

### Hardware
I ran this script on a ucloud 32 cpu machine with 192 GB of memory and training the art classifier takes about 13 minutes and the pretrained takes around 25 minutes


### data
Data can be found here: 
https://www.kaggle.com/datasets/thedownhill/art-images-drawings-painting-sculpture-engraving

Unzip the zip file to the data folder. 
Please keep in mind that when unzipping the data there seemes to be a few corrupted images, the data_prepper.script deletes these images.


### running the script
Data can be found here: 
https://www.kaggle.com/datasets/thedownhill/art-images-drawings-painting-sculpture-engraving

Unzip the the zip file to the data folder. 
Please keep in mind that when unzipping the data there seems to be a few corrupted images, the data_prepper.script deletes these images

    running the script
If using ucloud, please make sure to have venv installed:
sudo apt-get update
sudo apt-get install python3-venv

set working directory to assignment4-Emilbjacobsen and run:
bash setup.sh

For some reason the virtual environment won’t activate from my sh file, if you have the same problem, please just run this with assignment4-vis-Emilbjacobsen as working directory after running setup.sh, you can tell its activated if it’s in a parenthesis next to the coder in terminal:

source ./assignment2_viz_env/bin/activate

Then unzip the zip file to the data folder and go to the src dir in terminal and paste this:

python3 data_prepper.py

this will convert the data to train, val, test split

Then just run the scripts from the src like so:

python3 script_train_class.py

and

python3 script_vgg16.py

Some of the model parameters are modifiable from the commandline for each model, if no specifications are put the models will run the default numbers, see script for further details.
--epochs 
--learning_rate
--batch_size (only for vgg16)





## Results
While both models have high performance, the pre trained vgg16 model does outperform the CNN classifier I trained myself on all categories, but not by much, there is only a 0,4 difference in accuracy. Both models show lower f1 score in categories 0 and 1, this is most likely because these also contains the least number of images, with 860 training images for the drawings category and 588 images for engravings, compared to 1612, 1350 and 1592 in the other categories. this suggest that both models would benefit from an expansion of the dataset.

Art trained CNN classifier report:

              precision    recall  f1-score   support

              precision    recall  f1-score   support

           0       0.62      0.50      0.56       185
           1       0.63      0.71      0.67       127
           2       0.84      0.95      0.89       346
           3       0.92      0.88      0.90       337
           4       0.85      0.81      0.83       290

    accuracy                           0.81      1285
   macro avg       0.77      0.77      0.77      1285
weighted avg       0.81      0.81      0.81      1285

vgg16 classifier report:

              precision    recall  f1-score   support

           0       0.64      0.67      0.66       185
           1       0.75      0.65      0.69       127
           2       0.95      0.98      0.96       346
           3       0.93      0.89      0.91       337
           4       0.89      0.93      0.91       290

    accuracy                           0.87      1285
   macro avg       0.83      0.82      0.83      1285
weighted avg       0.87      0.87      0.87      1285


The accuracy and loss curves suggest that the pretrained vgg16 model already starts overfitting around the 7th to 10th epoch, which was expected, due to its relatively fewer trainable parameters.

The art CNN on the other hand seems to keep improving even by the 27th epoch.



