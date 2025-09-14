"""
================================
Recognizing hand-written digits
Using SVM and Decision Tree
================================
"""

# Standard imports
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm, tree
from sklearn.model_selection import train_test_split
from joblib import dump, load

###############################################################################
# Digits dataset
digits = datasets.load_digits()

# Visualize first 4 training images
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
plt.savefig("sample_training_digits.png")   # Save training samples plot

###############################################################################
# Flatten images into (n_samples, n_features)
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

###############################################################################
# Support Vector Classifier
svm_clf = svm.SVC(gamma=0.001)
svm_clf.fit(X_train, y_train)
predicted_svm = svm_clf.predict(X_test)

print(
    f"Classification report for SVM:\n"
    f"{metrics.classification_report(y_test, predicted_svm)}\n"
)

# Confusion Matrix - SVM
disp_svm = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted_svm)
disp_svm.figure_.suptitle("Confusion Matrix - SVM")
plt.savefig("svm_confusion_matrix.png")   # Save SVM confusion matrix
plt.close()

###############################################################################
# Decision Tree Classifier
dt_clf = tree.DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)
predicted_dt = dt_clf.predict(X_test)

print(
    f"Classification report for Decision Tree:\n"
    f"{metrics.classification_report(y_test, predicted_dt)}\n"
)

# Confusion Matrix - Decision Tree
disp_dt = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted_dt)
disp_dt.figure_.suptitle("Confusion Matrix - Decision Tree")
plt.savefig("decision_tree_confusion_matrix.png")   # Save Decision Tree confusion matrix
plt.close()

###############################################################################
# Save both models
dump(svm_clf, "svm_model.joblib")
dump(dt_clf, "dt_model.joblib")
print("✅ Models saved as svm_model.joblib and dt_model.joblib")

# Load both models
best_svm = load("svm_model.joblib")
best_dt = load("dt_model.joblib")
print("✅ Models loaded successfully!")
