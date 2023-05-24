# path tools
import os
import cv2
# data loader
import numpy as np
from tensorflow.keras.datasets import cifar10

# machine learning tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# classificatio models
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# argparse
import argparse

# Create argument parser
parser = argparse.ArgumentParser(description="Logistic Regression Classifier")

# Add arguments
parser.add_argument("--penalty", type=str, default="none", choices=["l1", "l2", "elasticnet", "none"],
                    help="Type of regularization penalty")
parser.add_argument("--tol", type=float, default=0.1,
                    help="Tolerance for convergence")

parser.add_argument("--solver", type=str, default="saga", choices=["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                    help="Algorithm to use for optimization")
parser.add_argument("--multi_class", type=str, default="multinomial", choices=["ovr", "multinomial"],
                    help="Strategy for handling multiclass classification")

# Parse arguments
args = parser.parse_args()



def get_data():

    #loading dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    #making labels
    labels = ["airplane", 
            "automobile", 
            "bird", 
            "cat", 
            "deer", 
            "dog", 
            "frog", 
            "horse", 
            "ship", 
            "truck"]

    #converting to grayscale
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])

    #taking the pixel intensity to be a value between 0 and 1
    X_train_scaled = (X_train_grey)/255.0
    X_test_scaled = (X_test_grey)/255.0 

    #reshaping the data
    nsamples, nx, ny = X_train_scaled.shape
    X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny))

    nsamples, nx, ny = X_test_scaled.shape
    X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))

    return X_train_dataset, X_test_dataset, X_test, y_test, y_train, labels


def train_lrc(X_train_dataset, y_train):

    #defining logistic regression classifier
    clf = LogisticRegression(penalty=args.penalty,
                            tol=args.tol,
                            verbose=True,
                            solver=args.solver,
                            multi_class=args.multi_class).fit(X_train_dataset, y_train)
    
    return clf


def get_report(clf, X_test_dataset, y_test, labels):

    y_pred = clf.predict(X_test_dataset)


    report = classification_report(y_test, 
                                y_pred, 
                                target_names=labels)

    

    #saving report
    folder_path = os.path.join("..", "out")
    file_name = "lrc.txt"
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, "w") as f:
        f.write(report)
    print("reports saved")


def main():

    # Get data
    X_train_dataset, X_test_dataset, X_test, y_test, y_train, labels = get_data()
    
    # Train classifier
    clf = train_lrc(X_train_dataset, y_train)

    # Get report
    get_report(clf, X_test_dataset, y_test, labels)

if __name__ == "__main__":
    main()
