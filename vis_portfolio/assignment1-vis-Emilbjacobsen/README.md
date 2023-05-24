# Visual assignment 1

## Contribution
This code has been made in collaboration with others in class, though the final result has been made by me

## Assignment task

For this assignment, you'll be using ```OpenCV``` to design a simple image search algorithm.

The dataset is a collection of over 1000 images of flowers, sampled from 17 different species. The dataset comes from the Visual Geometry Group at the University of Oxford, and full details of the data can be found [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/).

For this exercise, you should write some code which does the following:

- Define a particular image that you want to work with
- For that image
  - Extract the colour histogram using ```OpenCV```
- Extract colour histograms for all of the **other* images in the data
- Compare the histogram of our chosen image to all of the other histograms 
  - For this, use the ```cv2.compareHist()``` function with the ```cv2.HISTCMP_CHISQR``` metric
- Find the five images which are most simlar to the target image
  - Save a CSV file to the folder called ```out```, showing the five most similar images and the distance metric:

|Filename|Distance]
|---|---|
|target|0.0|
|filename1|---|
|filename2|---|

## Methods

This script takes a single image, makes a histogram for that image. It then iterates over each image and compares them, giving it a score that represents how alike the image is to the first image. The script produces 2 csv's, 1 with every file and its likeness to the original image and 1 with the top 5 most like the original image.

## Usage

### Hardware
I ran this script on a ucloud 8 cpu machine with 48 GB of memory.

### Data
The data can be found [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/).

### Running the script
If using ucloud, please make sure to have venv installed:
sudo apt-get update
sudo apt-get install python3-venv

Download the data from kaggle, place it in data directory and unpack the file using this command in terminal:

tar -xvzf 17flowers.tgz

and remove the 2 text files



Then make sure to have assignment1-vis-Emilbjacobsen as working dir and run the sh file in terminal with:

bash setup.sh

For some reason the virtual environment won’t activate from my sh file, if you have the same problem, please just run this with assignment1-vis-Emilbjacobsen as working directory after running setup.sh, you can tell its activated if it’s in a parenthesis next to the coder in terminal:

source ./assignment1_viz_env/bin/activate

Then run this code with src as working directory, remember to include path to the image you want to use fx:

python3 script.py ../data/jpg/image_0001.jpg

Keep in mind that the script moves the image you have used to the used_image folder, so remember to move it back if you want to rerun it.

also use the deactivate command in terminal when done.


## Results

When run, the program functions as it should and the csv files are produced and look as intended. When inspecting the 5 five flower images with most similar histograms, none of the flowers are the same species as the original, but some do share the yellow colour of the original image, others seem to not resemble in shape or in colour, so perhaps it is picking up on the background. Theese two examples seems to highlight this script to major weaknesses, it doesn't seem to account for shape, only colour similarities and the background and other objects in the foto can have a large impact on the results.