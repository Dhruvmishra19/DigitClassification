"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

###############################################################################
# Below we visualize the first 4 test samples and show their predicted
# digit value in the title.

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

###############################################################################
# :func:`~sklearn.metrics.classification_report` builds a text report showing
# the main classification metrics.

print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

###############################################################################
# If the results from evaluating a classifier are stored in the form of a
# :ref:`confusion matrix <confusion_matrix>` and not in terms of `y_true` and
# `y_pred`, one can still build a :func:`~sklearn.metrics.classification_report`
# as follows:


# The ground truth and predicted lists
y_true = []
y_pred = []
cm = disp.confusion_matrix

# For each cell in the confusion matrix, add the corresponding ground truths
# and predictions to the lists
for gt in range(len(cm)):
    for pred in range(len(cm)):
        y_true += [gt] * cm[gt][pred]
        y_pred += [pred] * cm[gt][pred]

print(
    "Classification report rebuilt from confusion matrix:\n"
    f"{metrics.classification_report(y_true, y_pred)}\n"
)
# plot_digits_classification.py
# This script trains two models (SVM and Decision Tree), compares their
# test accuracies, and saves the best-performing model to a file.

import os
from joblib import dump
from sklearn import svm, tree, metrics
from utils import load_data, preprocess_data, split_train_test

def main():
    """
    Main function to train, evaluate, and save the best model.
    """
    # 1. Load and prepare the data
    digits_data = load_data()
    X = preprocess_data(digits_data)
    y = digits_data.target

    # 2. Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=0.3, shuffle=True
    )

    # 3. Define the classifiers to be compared
    classifiers = {
        'SVM': svm.SVC(gamma=0.001),
        'DecisionTree': tree.DecisionTreeClassifier()
    }

    best_model = None
    best_accuracy = 0.0
    best_model_name = ""

    # 4. Loop through each classifier to train and evaluate it
    for name, clf in classifiers.items():
        print(f"--- Training {name} ---")
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, predicted)
        
        # Display the test accuracy
        print(f"Test accuracy of {name}: {accuracy:.4f}")

        # Check if this model is the best one so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = clf
            best_model_name = name

    print(f"\n--- Best model is {best_model_name} with an accuracy of {best_accuracy:.4f} ---")

    # 5. Save the best model artifact
    # Create a directory to store models if it doesn't already exist
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "best_model.joblib")
    
    dump(best_model, model_path)
    print(f"Best model has been saved to '{model_path}'")


if __name__ == "__main__":
    main()
