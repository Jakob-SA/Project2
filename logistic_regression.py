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


# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Construct the file path
filename = os.path.join(current_dir, 'optical_interconnection_network.csv')
# Load the optical_interconnection_network.csv data using the Pandas library
df = pd.read_csv(filename, delimiter=';')
# Delete the last five empty columns
df = df.iloc[:, :-5]
# Correct the indexing to extract class labels
classLabels = df.iloc[:, 2]
classNames = np.unique(classLabels)
classDict = dict(zip(classNames, range(len(classNames))))
# # Delete the column at index 2
# df.drop(df.columns[2], axis=1, inplace=True)

# One-hot encode the column at index 2 twice
column_to_encode = df.columns[2]
df_encoded = pd.get_dummies(df, columns=[column_to_encode], dtype=int)

# Map "client-server" to 1 and "async" to 0
df_encoded.iloc[:, 2] = df_encoded.iloc[:, 2].map({'Client-Server': '1', 'Asynchronous': '0'})


# Convert all strings from "0,922138" to floats in df_second_encoded
for i, col in enumerate(df_encoded.columns):
    if df_encoded[col].dtype == 'object':
        df_encoded[col] = df_encoded[col].str.replace(',', '.').astype(float)

attributeNames = np.asarray(df_encoded.columns)

encoded_filename = os.path.join(current_dir, 'optical_interconnection_network_encoded.csv')
df_encoded.to_csv(encoded_filename, index=False)

scaler = StandardScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns)

# Extract features and labels
label_column = 'Temporal Distribution'
X = df_normalized.drop(columns=[label_column]).values  # Replace 'class_label' with the actual label column name
y = df_normalized[label_column].values  # Replace 'class_label' with the actual label column name


K = 20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, stratify=y)

mu = np.mean(X_train, 0)
sigma = np.std(X_train, 0)

X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

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


plt.figure(figsize=(8, 8))
# plt.plot(np.log10(lambda_interval), train_error_rate*100)
# plt.plot(np.log10(lambda_interval), test_error_rate*100)
# plt.plot(np.log10(opt_lambda), min_error*100, 'o')
plt.semilogx(lambda_interval, train_error_rate * 100)
plt.semilogx(lambda_interval, test_error_rate * 100)
plt.semilogx(opt_lambda, min_error * 100, "o")
plt.text(
    1e-8,
    3,
    "Minimum test error: "
    + str(np.round(min_error * 100, 2))
    + " % at 1e"
    + str(np.round(np.log10(opt_lambda), 2)),
)
plt.xlabel("Regularization strength, $\log_{10}(\lambda)$")
plt.ylabel("Error rate (%)")
plt.title("Classification error")
plt.legend(["Training error", "Test error", "Test minimum"], loc="upper right")
plt.ylim([0, 10])
plt.grid()
plt.show()

plt.figure(figsize=(8, 8))
plt.semilogx(lambda_interval, coefficient_norm, "k")
plt.ylabel("L2 Norm")
plt.xlabel("Regularization strength, $\log_{10}(\lambda)$")
plt.title("Parameter vector L2 norm")
plt.grid()
plt.show()
