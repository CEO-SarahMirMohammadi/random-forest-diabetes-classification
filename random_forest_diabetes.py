import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Load Scikit-learn diabetes dataset
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

print("Original dataset shape:")
print(X.shape)

print("\nOriginal target:")
print(y)


# Convert the continuous target into two classes
median_target = np.median(y)

y_class = (y > median_target).astype(int)

print("\nMedian target value:")
print(median_target)

print("\nClassification targets:")
print(y_class)

print("\nClass distribution:")
print(np.bincount(y_class))


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_class,
    test_size=0.5,
    random_state=0
)

print("\nTraining samples:")
print(X_train.shape[0])

print("Testing samples:")
print(X_test.shape[0])


# Create the Random Forest classifier
clf = RandomForestClassifier(
    random_state=0
)


# Train the model
clf.fit(X_train, y_train)

print("\nRandom Forest model trained successfully.")


# Make predictions
y_pred = clf.predict(X_test)


# Evaluate the model
N = y_test.shape[0]
C = (y_test == y_pred).sum()

print("\nEvaluation:")
print(
    "Total points: %d Correctly labeled points: %d"
    % (N, C)
)


# Calculate accuracy
accuracy = C / N

print("Accuracy: %.2f%%" % (accuracy * 100))