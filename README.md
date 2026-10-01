# Random Forest Diabetes Classification

A machine learning project demonstrating **Random Forest classification** using Scikit-learn's **Diabetes dataset**.

This project modifies Example 3.12, which originally performs Random Forest classification on the Iris dataset, and adapts it to Scikit-learn's Diabetes dataset.

Because the original Diabetes dataset contains a **continuous regression target**, the target is converted into two classes using its median value. This allows the dataset to be used with `RandomForestClassifier`.

## Overview

The original Example 3.12 uses:

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
```

This project replaces the Iris dataset with:

```python
from sklearn.datasets import load_diabetes
```

The Diabetes dataset is normally used for regression because its target is continuous.

To perform classification, the continuous target is transformed into two classes:

```text
Target ≤ median → Class 0
Target > median → Class 1
```

The resulting workflow is:

```text
Load Diabetes Dataset
        ↓
Extract Features and Continuous Target
        ↓
Calculate Target Median
        ↓
Convert Target to Binary Classes
        ↓
Train/Test Split
        ↓
Train Random Forest Classifier
        ↓
Predict Test Samples
        ↓
Compare Predictions with True Classes
        ↓
Calculate Accuracy
```

---

## Random Forest

### What is Random Forest?

**Random Forest** is an ensemble machine learning algorithm that combines multiple decision trees to make predictions.

Instead of relying on a single decision tree, Random Forest creates many trees and combines their predictions.

For classification, each tree produces a class prediction and the forest generally selects the class receiving the majority of votes.

Conceptually:

```text
                    Training Data
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Tree 1         Tree 2         Tree 3
          │              │              │
          ▼              ▼              ▼
       Class 0         Class 1         Class 0
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                   Majority Vote
                         │
                         ▼
                  Final Prediction
```

A Random Forest can generally model complex nonlinear relationships while reducing some of the instability associated with individual decision trees.

---

## Dataset

This project uses Scikit-learn's **Diabetes dataset** through:

```python
from sklearn.datasets import load_diabetes
```

The dataset contains:

* **442 samples**
* **10 numerical features**
* A continuous target variable

The ten features are standardized quantitative measurements related to diabetes progression.

The features are:

```text
age
sex
bmi
bp
s1
s2
s3
s4
s5
s6
```

The original target represents a quantitative measure of disease progression.

---

## Important Dataset Consideration

The Scikit-learn Diabetes dataset is fundamentally a **regression dataset**.

Its target is continuous rather than a class label.

Therefore, the following would normally be appropriate:

```python
RandomForestRegressor()
```

rather than:

```python
RandomForestClassifier()
```

However, this exercise specifically asks for **Random Forest classification**.

To satisfy the classification requirement, this project converts the continuous target into two classes based on the median target value.

---

## Target Transformation

First, the median target value is calculated:

```python
median_target = np.median(y)
```

Then each target value is converted into a binary class:

```python
y_class = (y > median_target).astype(int)
```

This produces:

```text
Class 0 → Target ≤ median
Class 1 → Target > median
```

Conceptually:

```text
                    Continuous Target
                           │
                           ▼
                    Calculate Median
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
          Target ≤ Median       Target > Median
                │                     │
                ▼                     ▼
             Class 0                Class 1
```

This creates a binary classification problem.

---

## Why Use the Median?

Using the median provides a simple and reproducible threshold for converting the continuous target into two groups.

It also produces approximately balanced classes because roughly half of the observations are below or equal to the median and half are above it.

This transformation is primarily intended for demonstrating the classification algorithm in this exercise.

It should not be interpreted as a clinical definition of diabetes severity or disease status.

---

## Project Structure

```text
random-forest-diabetes-classification/
│
├── random_forest_diabetes.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Requirements

| Requirement  | Purpose                                      |
| ------------ | -------------------------------------------- |
| Python 3.x   | Programming language                         |
| NumPy        | Median calculation and numerical operations  |
| Scikit-learn | Dataset, train/test split, and Random Forest |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/CEO-SarahMirMohammadi/random-forest-diabetes-classification.git
```

Move into the project directory:

```bash
cd random-forest-diabetes-classification
```

Create a virtual environment.

### Windows

```powershell
python -m venv .venv
```

Activate the virtual environment:

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

---

## Running the Program

Run:

```powershell
python random_forest_diabetes.py
```

The program will:

1. Load the Diabetes dataset.
2. Display the original dataset shape.
3. Display the original continuous target.
4. Calculate the target median.
5. Convert the continuous target into two classes.
6. Display the class distribution.
7. Split the dataset into training and testing sets.
8. Train a Random Forest classifier.
9. Predict the classes of the test samples.
10. Count correctly classified samples.
11. Calculate classification accuracy.

---

## Example Output

A typical execution will produce output similar to:

```text
Original dataset shape:
(442, 10)

Original target:
[...]

Median target value:
[...]

Classification targets:
[...]

Class distribution:
[...]

Training samples:
221

Testing samples:
221

Random Forest model trained successfully.

Evaluation:
Total points: 221 Correctly labeled points: XXX
Accuracy: XX.XX%
```

The exact predictions and accuracy depend on the trained Random Forest model.

---

## Train/Test Split

The dataset is divided using:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_class,
    test_size=0.5,
    random_state=0
)
```

The configuration uses:

```text
Training data → 50%
Testing data  → 50%
```

The `random_state=0` parameter makes the split reproducible.

---

## Model Training

The Random Forest classifier is created with:

```python
clf = RandomForestClassifier(
    random_state=0
)
```

The model is trained using:

```python
clf.fit(X_train, y_train)
```

During training, the Random Forest creates an ensemble of decision trees using the training data.

---

## Prediction

After training, predictions are generated using:

```python
y_pred = clf.predict(X_test)
```

Each test observation receives either:

```text
Class 0
```

or:

```text
Class 1
```

---

## Evaluation

The original exercise evaluates the model by counting correctly classified test observations.

The number of test observations is:

```python
N = y_test.shape[0]
```

The number of correct predictions is:

```python
C = (y_test == y_pred).sum()
```

The program displays:

```text
Total points: N
Correctly labeled points: C
```

---

## Accuracy

Accuracy is calculated as:

```python
accuracy = C / N
```

and displayed as a percentage:

```python
print("Accuracy: %.2f%%" % (accuracy * 100))
```

Accuracy represents the proportion of test observations that received the correct class prediction.

For example:

```text
100 test samples
90 correct predictions
```

would produce:

```text
90% accuracy
```

---

## Workflow

```text
                 Diabetes Dataset
                        │
                        ▼
                 442 Samples
                 10 Features
                        │
                        ▼
              Continuous Target
                        │
                        ▼
                Calculate Median
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
        Class 0              Class 1
      ≤ Median              > Median
              │                   │
              └─────────┬─────────┘
                        ▼
                 Train/Test Split
                    50% / 50%
                        │
                        ▼
              Random Forest Model
                        │
                        ▼
                    Training
                        │
                        ▼
                   Prediction
                        │
                        ▼
              Compare Predictions
                        │
                        ▼
                    Accuracy
```

---

## Key Concepts

### 1. Ensemble Learning

Random Forest is an ensemble method because it combines multiple decision trees into one predictive model.

### 2. Decision Trees

Each individual tree creates a sequence of feature-based decisions.

### 3. Random Forest

The forest combines the predictions of many trees to produce a final prediction.

### 4. Classification

After target transformation, this project performs binary classification:

```text
Class 0
Class 1
```

### 5. Continuous Target

The original Diabetes dataset contains a continuous target suitable for regression.

The project converts it into a binary target for this particular classification exercise.

### 6. Train/Test Split

The training data is used to fit the model while the test data is reserved for evaluation.

### 7. Accuracy

Accuracy measures how many test observations were classified correctly.

---

## Random Forest Parameters

The model currently uses Scikit-learn's default Random Forest configuration with a fixed random state:

```python
RandomForestClassifier(random_state=0)
```

Important Random Forest parameters include:

### `n_estimators`

Controls the number of decision trees:

```python
RandomForestClassifier(n_estimators=100)
```

More trees can provide more stable predictions, although they also require additional computation.

### `max_depth`

Controls the maximum depth of each tree:

```python
RandomForestClassifier(max_depth=5)
```

### `min_samples_split`

Controls the minimum number of samples required to split a node.

### `max_features`

Controls how many features are considered when looking for a split.

These parameters can be tuned to investigate model performance.

---

## Why Random Forest?

Random Forest has several useful characteristics:

* Combines multiple decision trees.
* Can model nonlinear relationships.
* Can capture feature interactions.
* Usually requires relatively little preprocessing.
* Provides feature importance estimates.
* Can be used for classification and regression.

However, Random Forest models can require more computational resources than a single decision tree, especially as the number of trees increases.

---

## Important Limitation

The classification target in this project is **created artificially from the continuous diabetes progression target**.

The original dataset was not designed as a binary classification dataset.

Therefore:

```text
Class 0 ≠ clinically diagnosed "no diabetes"
Class 1 ≠ clinically diagnosed "diabetes"
```

The two classes simply represent observations on opposite sides of the median target value.

This transformation is used solely to adapt the regression dataset to the classification exercise.

---

## Classification vs Regression

The same dataset can be used with a different Random Forest algorithm depending on the target.

### Classification

```python
RandomForestClassifier()
```

Used when the target consists of discrete classes.

### Regression

```python
RandomForestRegressor()
```

Used when the target is continuous.

For the original Scikit-learn Diabetes dataset, Random Forest regression would normally preserve the continuous target rather than converting it into classes.

---

## Future Improvements

Possible improvements include:

* Add a confusion matrix.
* Add a classification report.
* Calculate precision, recall, and F1-score.
* Tune the number of trees.
* Tune tree depth.
* Perform cross-validation.
* Analyze feature importance.
* Visualize feature importance.
* Compare Random Forest with a single Decision Tree.
* Compare Random Forest classification with Random Forest regression.
* Use `RandomForestRegressor` on the original continuous target.
* Experiment with different target thresholds.

---

## License

This project is licensed under the **MIT License**.

You are free to use, modify, distribute, and build upon this project under the terms of the MIT License.
