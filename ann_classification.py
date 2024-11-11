from cProfile import label
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
import torch
from sklearn import model_selection

from dtuimldmtools import (
    draw_neural_net,
    train_neural_net,
    visualize_decision_boundary,
)

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
print(attributeNames)

encoded_filename = os.path.join(current_dir, 'optical_interconnection_network_encoded.csv')
df_encoded.to_csv(encoded_filename, index=False)

scaler = StandardScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns)




# Extract features and labels
label_column = 'Temporal Distribution'
X = df_normalized.drop(columns=[label_column]).values  # Replace 'class_label' with the actual label column name
y = df_normalized[label_column].values  # Replace 'class_label' with the actual label column name
y = np.where(y == -1.0, 0.0, y)


N, M = X.shape
C = len(classNames)

# Parameters for neural network classifier
n_hidden_units = 8  # number of hidden units
n_replicates = 2  # number of networks trained in each k-fold
max_iter = 10000  # stop criterion 2 (max epochs in training)

# K-fold crossvalidation
K = 3  # only five folds to speed up this example
CV = model_selection.KFold(K, shuffle=True)
# Make figure for holding summaries (errors and learning curves)
summaries, summaries_axes = plt.subplots(1, 2, figsize=(10, 5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = [
    "tab:orange",
    "tab:green",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
    "tab:red",
    "tab:blue",
]

# Define the model, see also Exercise 8.2.2-script for more information.
model = lambda: torch.nn.Sequential(
    torch.nn.Linear(M, n_hidden_units),  # M features to H hiden units
    torch.nn.Tanh(),  # 1st transfer function,
    torch.nn.Linear(n_hidden_units, 1),  # H hidden units to 1 output neuron
    torch.nn.Sigmoid(),  # final tranfer function
)
loss_fn = torch.nn.BCELoss()

print("Training model of type:\n\n{}\n".format(str(model())))
errors = []  # make a list for storing generalizaition error in each loop
for k, (train_index, test_index) in enumerate(CV.split(X, y)):
    print("\nCrossvalidation fold: {0}/{1}".format(k + 1, K))

    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.Tensor(X[train_index, :])
    y_train = torch.Tensor(y[train_index]).view(-1, 1)
    X_test = torch.Tensor(X[test_index, :])
    y_test = torch.Tensor(y[test_index]).view(-1, 1)

    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(
        model,
        loss_fn,
        X=X_train,
        y=y_train,
        n_replicates=n_replicates,
        max_iter=max_iter,
    )

    print("\n\tBest loss: {}\n".format(final_loss))

    # Determine estimated class labels for test set
    y_sigmoid = net(X_test)
    y_test_est = (y_sigmoid > 0.5).type(dtype=torch.uint8)

    # Determine errors and errors
    y_test = y_test.type(dtype=torch.uint8)

    e = y_test_est != y_test
    error_rate = (sum(e).type(torch.float) / len(y_test)).data.numpy()
    errors.append(error_rate)  # store error rate for current CV fold

    # Display the learning curve for the best net in the current fold
    (h,) = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label("CV fold {0}".format(k + 1))
    summaries_axes[0].set_xlabel("Iterations")
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel("Loss")
    summaries_axes[0].set_title("Learning curves")

# Display the error rate across folds
summaries_axes[1].bar(
    np.arange(1, K + 1), np.squeeze(np.asarray(errors)), color=color_list
)
summaries_axes[1].set_xlabel("Fold")
summaries_axes[1].set_xticks(np.arange(1, K + 1))
summaries_axes[1].set_ylabel("Error rate")
summaries_axes[1].set_title("Test misclassification rates")

print("Diagram of best neural net in last fold:")
weights = [net[i].weight.data.numpy().T for i in [0, 2]]
biases = [net[i].bias.data.numpy() for i in [0, 2]]
tf = [str(net[i]) for i in [1, 3]]
# Convert attributeNames to a list
attributeNames = attributeNames.tolist()
draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

# Print the average classification error rate
print(
    "\nGeneralization error/average error rate: {0}%".format(
        round(100 * np.mean(errors), 4)
    )
)