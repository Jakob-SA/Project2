from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold
import numpy as np

def get_baseline_table(X, y, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    error_rates = []
    all_y_pred = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        dummy_clf = DummyClassifier(strategy="most_frequent")
        dummy_clf.fit(X_train, y_train)

        # Predict on the test data
        y_pred = dummy_clf.predict(X_test)

        # Calculate error rate
        accuracy = accuracy_score(y_test, y_pred)
        error_rate = 1 - accuracy
        error_rates.append(error_rate)

    return error_rates, y_pred