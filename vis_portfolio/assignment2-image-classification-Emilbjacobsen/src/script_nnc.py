import os
import cv2
import argparse
import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Image classification using MLP')
parser.add_argument('--hidden_layer_sizes', nargs='+', type=int, default=[64, 10], help='Hidden layer sizes')
parser.add_argument('--learning_rate', type=str, default='adaptive', help='Learning rate')
parser.add_argument('--max_iter', type=int, default=20, help='Maximum number of iterations')
args = parser.parse_args()

def get_data():
    # Load CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Convert images to grayscale
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])

    # Scale pixel intensities to range between 0 and 1
    X_train_scaled = (X_train_grey) / 255.0
    X_test_scaled = (X_test_grey) / 255.0

    # Reshape the data
    nsamples, nx, ny = X_train_scaled.shape
    X_train_dataset = X_train_scaled.reshape((nsamples, nx * ny))

    nsamples, nx, ny = X_test_scaled.shape
    X_test_dataset = X_test_scaled.reshape((nsamples, nx * ny))

    return X_train_dataset, X_test_dataset, y_train, y_test

def train_classifier(X_train_dataset, X_test_dataset, y_train):
    # Create and train the MLP classifier
    clf = MLPClassifier(random_state=42,
                        hidden_layer_sizes=tuple(args.hidden_layer_sizes),
                        learning_rate=args.learning_rate,
                        early_stopping=True,
                        verbose=True,
                        max_iter=args.max_iter).fit(X_train_dataset, y_train)
    return clf

def get_report(clf, X_test_dataset, y_test):
    # Make predictions on the test set
    y_pred = clf.predict(X_test_dataset)

    # Generate classification report
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    report = classification_report(y_test, y_pred, target_names=labels)

    # Save classification report to file
    folder_path = os.path.join("..", "out")
    file_name = "nnc.txt"
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, "w") as f:
        f.write(report)

    print("Report saved to:", file_path)

def main():

    # Getting and prepping data
    X_train_dataset, X_test_dataset, y_train, y_test = get_data()

    # Getting classifier
    clf = train_classifier(X_train_dataset, X_test_dataset, y_train)

    #getting classification report
    get_report(clf, X_test_dataset, y_test)


if __name__ == "__main__":
    main()