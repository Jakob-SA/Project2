import os
import importlib_resources
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from scipy.io import loadmat
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from dtuimldmtools import confmatplot, rocplot
import pandas as pd



def logistic_regression_cv(X, y, K):

  results = []

  # K-fold cross-validation loop
  for fold in range(K):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/K, stratify=y)

    # Hyperparameter tuning for logistic regression (inner loop)
    lambda_interval = np.logspace(-6, 2, 50)
    best_lambda = None
    best_error_rate = float('inf')

    for lambda_value in lambda_interval:
      model = LogisticRegression(penalty='l2', C=1 / lambda_value)
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)
      error_rate = 1 - accuracy_score(y_test, y_pred)

      if error_rate < best_error_rate:
        best_lambda = lambda_value
        best_error_rate = error_rate

    # Evaluate the best model on the test set (outer loop)
    best_model = LogisticRegression(penalty='l2', C=1 / best_lambda)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    error_rate = 1 - accuracy_score(y_test, y_pred)



    results.append([fold + 1, best_lambda, error_rate])  # Adjust indexing for clarity
    df = pd.DataFrame(results, columns=['Fold', 'Lambda', 'Error Rate'])


  return df, y_pred

