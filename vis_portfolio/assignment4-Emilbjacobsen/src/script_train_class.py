import os

# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)


# layers
from tensorflow.keras.layers import (Flatten,
                                     Dense,
                                     Dropout,
                                     BatchNormalization)

# generic model object
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

#scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# for plotting
import numpy as np
import matplotlib.pyplot as plt

# Image IO
import skimage.io
import skimage.transform

# for json files
import pandas as pd
import json

# utils plotting function
import sys
sys.path.append("..")
import utils.plot as pl

#argparse
import argparse

parser = argparse.ArgumentParser(description="CNN parameters")
parser.add_argument("--epochs", type=int, default=27, help="Number of epochs")
parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
args = parser.parse_args()


#defining shape of images
width = 128 # 368
height = 128 # 352
n_channels = 3

#getting categories
categories = ['drawings', 'engraving', 'iconography', 'painting', 'sculpture']
n_categories = len(categories)
category_embeddings = {
    'drawings': 0,
    'engraving': 1,
    'iconography': 2,
    'painting': 3,
    'sculpture': 4
}

#function for formatting images
def format_dataset(tuples_list, dataset_path):
    indexes = np.arange(len(tuples_list))
    np.random.shuffle(indexes)
    
    X = []
    y = []
    n_samples = len(indexes)
    cpt = 0
    for i in range(n_samples):
        t = tuples_list[indexes[i]]
        try:
            img = skimage.io.imread(os.path.join(dataset_path, t[0]))
            img = skimage.transform.resize(img, (width, height, n_channels), mode='reflect')
            X += [img]
            y_tmp = [0 for _ in range(n_categories)]
            y_tmp[category_embeddings[t[1]]] = 1
            y += [y_tmp]
        except OSError:
            pass
        
        cpt += 1
        
        if cpt % 1000 == 0:
            print("Processed {} images".format(cpt))

    X = np.array(X)
    y = np.array(y)
    
    return X, y

# functions for getting and prepping data
def get_data():

    # getting dir paths
    train_dir = os.path.join("..", "data", "train")
    val_dir = os.path.join("..", "data", "validation")
    test_dir = os.path.join("..", "data", "test")




    train_data = []
    for cat in categories:
        files = os.listdir(os.path.join(train_dir, cat))
        for file in files:
            train_data += [(os.path.join(cat, file), cat)]

    val_data = []
    for cat in categories:
        files = os.listdir(os.path.join(val_dir, cat))
        for file in files:
            val_data += [(os.path.join(cat, file), cat)]

    test_data = []
    for cat in categories:
        files = os.listdir(os.path.join(test_dir, cat))
        for file in files:
            test_data += [(os.path.join(cat, file), cat)]

    

    return train_data, test_data, val_data







def train_classifier(X_train, y_train, X_val, y_val):

    # creating datagenerator
    train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True)

    train_datagen.fit(X_train)

    # defining model layers
    # CNN model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=5, input_shape=(width, height, n_channels), activation='relu'))
    # model.add(Conv2D(8, kernel_size=5, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(48, kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(n_categories, activation='softmax'))

    # compile
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.learning_rate, decay_steps=10000, decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule)

    model.compile(optimizer = sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    # summarize model
    print(model.summary())
   # fit using a train and a validation generator
    # train_generator = DataGenerator(training_data, training_dataset_path)
    train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

    # test_generator = DataGenerator(test_data, test_dataset_path)
    history = model.fit_generator(generator=train_generator,
                                        validation_data=(X_val, y_val),
                                        epochs=args.epochs,
                                        verbose=1,
                                        steps_per_epoch=len(X_train) / 32)
    return history, model




# function for getting plots

def training_plot(history):
        # Training and validation history plots
    pl.plot_history(history, 27)
    output_path = os.path.join("..", "output", "myccn_train_and_val_plots.png")
    plt.savefig(output_path, dpi = 100)
    print("Plot is saved!")

#function for getting classification report
def get_report(test_dir, test_data, model):

    # making empty lists
    X_test = []
    y_test = []

    # for loop resizing images with skimage and adding path and label to lists
    for t in test_data:
        try:
            img = skimage.io.imread(os.path.join(test_dir, t[0]))
            img = skimage.transform.resize(img, (width, height, n_channels), mode='reflect')
            X_test += [img]
            y_test += [category_embeddings[t[1]]]
        except OSError:
            pass

    # Making them into numpy arrays
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # getting predictions for the test data
    pred = model.predict(X_test, verbose=1)
    y_pred = np.argmax(pred, axis=1)

    # getting report
    report = classification_report(y_test, y_pred)
    print(classification_report(y_test, y_pred))

    file_path = os.path.join("..", "output", "myccn_classification_report.txt")
    with open(file_path, "w") as f: #"writing" classifier report and saving it
        f.write(report)
    print("Classification report is saved!")


def main():
    # Load and prepare data
    train_data, test_data, val_data = get_data()

    train_dir = os.path.join("..", "data", "train")
    val_dir = os.path.join("..", "data", "validation")
    test_dir = os.path.join("..", "data", "test")

    # format data
    X_train, y_train = format_dataset(train_data, train_dir)
    X_val, y_val = format_dataset(val_data, val_dir)

    # train pretrained CNN
    history, model = train_classifier(X_train, y_train, X_val, y_val)

    # Training and validation history plots
    training_plot(history)

    # Predictions
    get_report(test_dir, test_data, model)



if __name__ == "__main__":
    main()