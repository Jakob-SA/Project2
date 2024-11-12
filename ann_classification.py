from cProfile import label
import os
import importlib_resources
from matplotlib.cbook import index_of
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
def get_ann_table():
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
    y = np.where(y == -1.0, 0.0, y)


    N, M = X.shape
    C = len(classNames)

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


    # Initialize parameters
    n_replicates = 1
    max_iter = 10000
    K = 3  # Number of folds in cross-validation
    hidden_units_range = range(1, 3)  # Range of hidden units to try

    # Initialize lists to store results
    folds = []
    best_hidden_units = []
    error_rates = []

    CV = model_selection.KFold(K, shuffle=True)

    for k, (train_index, test_index) in enumerate(CV.split(X, y)):
        print("\nCrossvalidation fold: {0}/{1}".format(k + 1, K))
        hidden_errors = []

        for i in hidden_units_range:  # Trial from 1 to 3 hidden units
            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(M, out_features=i),  # M features to H hidden units
                torch.nn.Tanh(),  # 1st transfer function
                torch.nn.Linear(i, 1),  # H hidden units to 1 output neuron
                torch.nn.Sigmoid(),  # final transfer function
            )
            loss_fn = torch.nn.BCELoss()

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

            # Determine error rate on test set
            y_test_est = net(X_test)
            y_test_est = (y_test_est > 0.5).type(torch.uint8)
            error_rate = (y_test_est != y_test).sum().item() / y_test.shape[0]
            hidden_errors.append((i, error_rate))

            print("\n\tHidden units: {}, Best loss: {}, Error rate: {}\n".format(i, final_loss, error_rate))

        # Find the best number of hidden units for this fold
        best_hidden_units_fold, best_error_rate_fold = min(hidden_errors, key=lambda x: x[1])
        folds.append(k + 1)
        best_hidden_units.append(best_hidden_units_fold)
        error_rates.append(best_error_rate_fold)

    # Create a DataFrame to display the results
    results_df = pd.DataFrame({
        "Fold": folds,
        "Best Hidden Units": best_hidden_units,
        "Error Rate": error_rates
    })

    return results_df