## Explaining why We use Cross Validation 
# Cross-validation is a technique used in machine learning to assess the performance of a model and to prevent overfitting. It involves splitting the dataset into multiple subsets, or "folds," and then training and evaluating the model on different combinations of these folds.
# The main reasons for using cross-validation are:
# 1. **Model Evaluation**: Cross-validation provides a more reliable estimate of a model's performance on unseen data compared to a single train-test split. By averaging the results across multiple folds, it gives a better indication of how the model will perform in real-world scenarios.
# 2. **Hyperparameter Tuning**: Cross-validation allows for the tuning of hyperparameters (the settings that govern the training process) by evaluating the model's performance on different folds.
# 3. **Data Utilization**: It maximizes the use of available data. In a single train-test split, a portion of the data is reserved for testing, which can lead to less training data. Cross-validation allows for more efficient use of the data by using different subsets for training and testing.
# 4. **Model Selection**: It helps in selecting the best model among different algorithms or configurations by comparing their performance across multiple folds.
# Overall, cross-validation is a crucial step in the machine learning workflow to ensure that the model
generalizes well to new, unseen data and to make informed decisions about model selection and hyperparameter tuning.
import numpy as np
from sklearn.model_selection import KFold
# Sample dataset
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([0, 1, 0, 1, 0])  # Binary labels
# Initialize KFold with 5 splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Perform cross-validation
for fold, (train_index, test_index) in enumerate(kf.split(X)):