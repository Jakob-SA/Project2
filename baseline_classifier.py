
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier


def get_baseline_table(X_train, X_test, y_train, y_test):
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)

    # Predict on the test data
    y_pred = dummy_clf.predict(X_test)

    # Assuming you have y_test and y_pred as your true and predicted labels

    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1 - accuracy

    error_rates = []
    for i in range(10):
        error_rates.append(error_rate)
    return error_rates
