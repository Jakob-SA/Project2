import os
import importlib_resources
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from dtuimldmtools import confmatplot, rocplot
import pandas as pd
from sklearn.metrics import accuracy_score


def logistic_regression_classifier(X_train, y_train, X_test, y_test):


     results = []

  # K-fold cross-validation loop
  for fold in range(K):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/K, stratify=y)

    # Hyperparameter tuning for logistic regression (inner loop)
    lambda_interval = np.logspace(-8, 2, 50)
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

  return results


    # Fit regularized logistic regression model to training data to predict
    # the type of wine
    lambda_interval = np.logspace(-8, 2, 50)
    train_error_rate = np.zeros(len(lambda_interval))
    test_error_rate = np.zeros(len(lambda_interval))
    coefficient_norm = np.zeros(len(lambda_interval))
    for k in range(0, len(lambda_interval)):
        mdl = LogisticRegression(penalty="l2", C=1 / lambda_interval[k])

        mdl.fit(X_train, y_train)

        y_train_est = mdl.predict(X_train).T
        y_test_est = mdl.predict(X_test).T

        train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
        test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)

        w_est = mdl.coef_[0]
        coefficient_norm[k] = np.sqrt(np.sum(w_est**2))

    min_error = np.min(test_error_rate)
    opt_lambda_idx = np.argmin(test_error_rate)
    opt_lambda = lambda_interval[opt_lambda_idx]

    return results


    # plt.figure(figsize=(8, 8))
    # # plt.plot(np.log10(lambda_interval), train_error_rate*100)
    # # plt.plot(np.log10(lambda_interval), test_error_rate*100)
    # # plt.plot(np.log10(opt_lambda), min_error*100, 'o')
    # plt.semilogx(lambda_interval, train_error_rate * 100)
    # plt.semilogx(lambda_interval, test_error_rate * 100)
    # plt.semilogx(opt_lambda, min_error * 100, "o")
    # plt.text(
    #     1e-8,
    #     3,
    #     "Minimum test error: "
    #     + str(np.round(min_error * 100, 2))
    #     + " % at 1e"
    #     + str(np.round(np.log10(opt_lambda), 2)),
    # )
    # plt.xlabel("Regularization strength, $\log_{10}(\lambda)$")
    # plt.ylabel("Error rate (%)")
    # plt.title("Classification error")
    # plt.legend(["Training error", "Test error", "Test minimum"], loc="upper right")
    # plt.ylim([0, 10])
    # plt.grid()
    # plt.show()

    # plt.figure(figsize=(8, 8))
    # plt.semilogx(lambda_interval, coefficient_norm, "k")
    # plt.ylabel("L2 Norm")
    # plt.xlabel("Regularization strength, $\log_{10}(\lambda)$")
    # plt.title("Parameter vector L2 norm")
    # plt.grid()
    # plt.show()

