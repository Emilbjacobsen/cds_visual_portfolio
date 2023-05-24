#loading packages
import os
import sys
sys.path.append(os.path.join(".."))

# Importing packages
import cv2
import numpy as np
import sys
sys.path.append("..")
from utils.imutils import jimshow
from utils.imutils import jimshow_channel
import matplotlib.pyplot as plt
from zipfile import ZipFile
from PIL import Image
import shutil
import pandas as pd
import argparse
import shutil


# Create the argument parser
parser = argparse.ArgumentParser(description='Process images and compare histograms.')

# Add the argument for the first image
parser.add_argument('first_image', type=str, help='Path to the first image')

# Parse the command-line arguments
args = parser.parse_args()

# Get the path to the first image
first_image_path = args.first_image




# Function for getting histogram 
def histogram(input_path):

    # Reading in image
    flowder = cv2.imread(input_path)

    # Normalizing image
    hist1 = cv2.calcHist([flowder], [0,1,2], None, [256,256,256], [0,256, 0,256, 0,256])
    hist1 = cv2.normalize(hist1, hist1, 0, 1.0, cv2.NORM_MINMAX)# for some reason this doesnt work i think
    return hist1

#creating path for my image
flower_path = first_image_path

#path to flowers dir
path = os.path.join("..", "data", "jpg")

#function for getting a single image histogram then deleting that image
def get_image_his(flower_path):
    #getting my lovely flower
    my_lovely_flower = histogram(flower_path)

    # Creating a destination folder for the copied image
    destination_folder = os.path.join("..", "used_image")
    
    # Copying the image to the destination folder
    shutil.copy2(flower_path, destination_folder)

    #deleting that image so it does not get into the comparisons df
    os.remove(flower_path)

    return my_lovely_flower

#function for getting all histograms except for the previous image and saving a csv with those histograms
def get_all_csvs(flower_dir, my_lovely_flower):
    #creating empty list for paths to files
    all_paths = []
    #for loop creating every path to every file in rootdir to list using os walk
    for subdir, dirs, files in os.walk(flower_dir):
        for file in files:# second for loop feeding all paths to all_paths list using append and os join
            all_paths.append(os.path.join(subdir, file))
    
    #making list for all histograms
    output = []
    #for loop using my histogram function to feed all histograms into output
    for path in all_paths:

        #using histogram function to get histogram
        flower_pic = histogram(path)

        #comparing the images
        likeness = round(cv2.compareHist(my_lovely_flower, flower_pic, cv2.HISTCMP_CHISQR), 2)

        #putting the comparisons together
        final_data = ((path, likeness))

        #appending to output
        output.append(final_data)


    #creating dataframe
    df = pd.DataFrame(output)

    #renaming col
    col_names = ['Textfile', 'Distance']
    df.columns = col_names
    print(df)
    #creating path
    outpath1 = os.path.join("..", "out", "comparisons.csv")
    #saving csv
    df.to_csv(outpath1)
    # getting the 5 images most similar to the original
    smallest_rows = df.nsmallest(5, 'Distance')
    #saving csv
    outpath2 = os.path.join("..", "out", "comparisons5.csv")
    smallest_rows.to_csv(outpath2)




def main():



    #creating path for my image
    flower_path = os.path.join(first_image_path)

    #get single img
    my_loveley_flower = get_image_his(first_image_path)

    #get all histograms
    get_all_csvs(path, my_loveley_flower)

if __name__ == "__main__":
    main()