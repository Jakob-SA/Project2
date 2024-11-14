import os
import numpy as np
import pandas as pd
from dtuimldmtools import similarity
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy import stats
from sklearn.discriminant_analysis import StandardScaler
from sklearn import model_selection
from dtuimldmtools import rlr_validate
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from matplotlib.pylab import (
    figure,
    grid,
    legend,
    loglog,
    semilogx,
    show,
    subplot,
    title,
    xlabel,
    ylabel,
)

def summary_stats(data):
    # Calculate summary statistics
    means = data.mean()
    stds = data.std()
    medians = data.median()
    ranges = data.max() - data.min()
    quantiles = data.quantile([0.25, 0.5, 0.75])
    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'Mean': means,
        'Standard Deviation': stds,
        'Median': medians,
        'Range': ranges,
        '25% Quantile': quantiles.loc[0.25],
        '50% Quantile': quantiles.loc[0.5],
        '75% Quantile': quantiles.loc[0.75]
    })
    return summary_df

def compare_models (y_true, y_pred_modelA,y_pred_modelB, alpha = 0.05):
    error_modelA = (y_true - y_pred_modelA) ** 2
    error_modelB = (y_true - y_pred_modelB) ** 2
    z_i = error_modelA - error_modelB
    n = len(z_i)
    
    mean_diff = np.mean(z_i)
    std_diff = np.sqrt(np.sum((z_i - mean_diff) ** 2) / (n * (n - 1)))
    dof = n - 1
    t_critical = stats.t.ppf(1 - alpha/2, dof)
    ci = (mean_diff - t_critical * std_diff, mean_diff + t_critical * std_diff)
    t_stat = mean_diff / std_diff
    p_value = 2 * stats.t.cdf(-abs(t_stat), dof)
    
    return mean_diff, ci, p_value


# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Construct the file path
filename = os.path.join(current_dir, 'optical_interconnection_network_encoded_1.csv')
# Load the optical_interconnection_network.csv data using the Pandas library
df = pd.read_csv(filename, delimiter=',')

df_subframe = df[(df['Node Number'] == 64) & (df['Thread Number'] == 4) ] 

df_subframe = df_subframe.drop(['Node Number', 'Thread Number','Temporal Distribution_Asynchronous','Temporal Distribution_Client-Server' ], axis=1)

print(df_subframe)



scaler = StandardScaler()

df_standardized = pd.DataFrame(scaler.fit_transform(df_subframe), columns = df_subframe.columns)




summary = summary_stats(df_standardized)

print(summary)



X = df_standardized.drop(columns=['Channel Waiting Time'])
y = df_standardized['Channel Waiting Time'].values

print(X)
print(y)

# Add offset attribute (bias term)
X = np.concatenate((np.ones((X.shape[0], 1)), X.values), axis=1)

attributeNames = ["Offset"] + list(df_subframe.drop(columns=['Channel Waiting Time']).columns)
N, M = X.shape  # Number of samples and features

# Cross-validation setup
K = 10

CV = model_selection.KFold(K, shuffle=True)
lambdas = np.power(10.0, range(-5, 9))

Error_train = np.empty((K, 1))
Error_test = np.empty((K, 1))
Error_train_rlr = np.empty((K, 1))
Error_test_rlr = np.empty((K, 1))
Error_train_nofeatures = np.empty((K, 1))
Error_test_nofeatures = np.empty((K, 1))
w_rlr = np.empty((M, K))
mu = np.empty((K, M - 1))
sigma = np.empty((K, M - 1))
w_noreg = np.empty((M, K))

k = 0

for train_index, test_index in CV.split(X, y):
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10

    (
        opt_val_err,
        opt_lambda,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
    ) = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)

    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]

    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = (
        np.square(y_train - y_train.mean()).sum(axis=0) / y_train.shape[0]
    )
    Error_test_nofeatures[k] = (
        np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0]
    )

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = (
        np.square(y_train - X_train @ w_rlr[:, k]).sum(axis=0) / y_train.shape[0]
    )
    Error_test_rlr[k] = (
        np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]
    )

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:, k] = np.linalg.solve(XtX, Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = (
        np.square(y_train - X_train @ w_noreg[:, k]).sum(axis=0) / y_train.shape[0]
    )
    Error_test[k] = (
        np.square(y_test - X_test @ w_noreg[:, k]).sum(axis=0) / y_test.shape[0]
    )
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    # m = lm.LinearRegression().fit(X_train, y_train)
    # Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    # Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Display the results for the last cross-validation fold
    if k == K - 1:
        figure(k, figsize=(12, 8))
        subplot(1, 2, 1)
        semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], ".-")  # Don't plot the bias term
        xlabel("Regularization factor")
        ylabel("Mean Coefficient Values")
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner
        # plot, since there are many attributes
        # legend(attributeNames[1:], loc='best')

        subplot(1, 2, 2)
        title("Optimal lambda: 1e{0}".format(np.log10(opt_lambda)))
        loglog(
            lambdas, train_err_vs_lambda.T, "b.-", lambdas, test_err_vs_lambda.T, "r.-"
        )
        xlabel("Regularization factor")
        ylabel("Squared error (crossvalidation)")
        legend(["Train error", "Validation error"])
        grid()

    # To inspect the used indices, use these print statements
    # print('Cross validation fold {0}/{1}:'.format(k+1,K))
    # print('Train indices: {0}'.format(train_index))
    # print('Test indices: {0}\n'.format(test_index))

    k += 1

show()
# Initialize parameters for ANN
hidden_layer_sizes_list = [(1,), (5,), (10,), (20,),(30,)]  # Example list of different hidden layer sizes to try
ann_errors_train = np.empty((K, len(hidden_layer_sizes_list)))
ann_errors_test = np.empty((K, len(hidden_layer_sizes_list)))

# Start cross-validation for ANN
k = 0
for train_index, test_index in CV.split(X, y):
    # Extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    # Standardize the data
    scaler_ann = StandardScaler()
    X_train = scaler_ann.fit_transform(X_train)
    X_test = scaler_ann.transform(X_test)

    # Loop over different ANN architectures
    for idx, hidden_layer_sizes in enumerate(hidden_layer_sizes_list):
        # Initialize and train the MLPRegressor
        ann_model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            max_iter=3000,
            random_state=42
        )
        ann_model.fit(X_train, y_train)

        # Compute the training and test errors
        y_train_pred = ann_model.predict(X_train)
        y_test_pred = ann_model.predict(X_test)
        ann_errors_train[k, idx] = np.mean((y_train - y_train_pred) ** 2)
        ann_errors_test[k, idx] = np.mean((y_test - y_test_pred) ** 2)

    k += 1

# Calculate mean training and test errors for ANN across all folds and configurations
mean_ann_errors_train = np.mean(ann_errors_train, axis=0)
mean_ann_errors_test = np.mean(ann_errors_test, axis=0)

# Display results for ANN
print("ANN Regression Results:")
for idx, hidden_layer_sizes in enumerate(hidden_layer_sizes_list):
    print(f"Hidden Layer Sizes: {hidden_layer_sizes}")
    print(f"- Training Error: {mean_ann_errors_train[idx]}")
    print(f"- Test Error: {mean_ann_errors_test[idx]}\n")




# Display results
print("Linear regression without feature selection:")
print("- Training error: {0}".format(Error_train.mean()))
print("- Test error:     {0}".format(Error_test.mean()))
print(
    "- R^2 train:     {0}".format(
        (Error_train_nofeatures.sum() - Error_train.sum())
        / Error_train_nofeatures.sum()
    )
)
print(
    "- R^2 test:     {0}\n".format(
        (Error_test_nofeatures.sum() - Error_test.sum()) / Error_test_nofeatures.sum()
    )
)
print("Regularized linear regression:")
print("- Training error: {0}".format(Error_train_rlr.mean()))
print("- Test error:     {0}".format(Error_test_rlr.mean()))
print(
    "- R^2 train:     {0}".format(
        (Error_train_nofeatures.sum() - Error_train_rlr.sum())
        / Error_train_nofeatures.sum()
    )
)
print(
    "- R^2 test:     {0}\n".format(
        (Error_test_nofeatures.sum() - Error_test_rlr.sum())
        / Error_test_nofeatures.sum()
    )
)

print("Weights in last fold:")
for m in range(M):
    print("{:>15} {:>15}".format(attributeNames[m], np.round(w_rlr[m, -1], 2)))


# Constructing the results table
folds = np.arange(1, K + 1)  # Outer fold indices

# Extracting optimal values and test errors
opt_hidden_units = [hidden_layer_sizes_list[np.argmin(ann_errors_test[k, :])] for k in range(K)]
ann_test_errors = [ann_errors_test[k, np.argmin(ann_errors_test[k, :])] for k in range(K)]
linear_reg_test_errors = Error_test_rlr.flatten()  # Test errors for regularized linear regression
baseline_test_errors = Error_test_nofeatures.reshape(-1)
opt_lambdas = [opt_lambda] * K  # The optimal lambda used across folds

# Constructing the DataFrame
results_df = pd.DataFrame({
    "Outer Fold (i)": folds,
    "h* (Optimal Hidden Units)": opt_hidden_units,
    "E_test (ANN)": ann_test_errors,
    "λ* (Optimal Lambda)": opt_lambdas,
    "E_test (Linear Regression)": linear_reg_test_errors,
    "E_test (Baseline)": baseline_test_errors
})

# Display the results table
print(results_df)


# Perform statistical testing between Regularized Linear Regression, ANN, and Baseline models

y_true = y  # Ground truth values
y_pred_lr = np.mean(Error_test_rlr)  # Mean test error for Linear Regression
y_pred_ann = np.mean(ann_test_errors)  # Mean test error for ANN
y_pred_baseline = np.mean(baseline_test_errors)  # Mean test error for Baseline

# ANN vs Baseline Comparison
mean_diff_ann_V_baseline, ci_ann_V_baseline, p_value_ann_V_baseline = compare_models(y_true, y_pred_ann, y_pred_baseline, alpha=0.05)

print("\nStatistical Comparison between ANN and Baseline:")
print(f"Mean Difference: {mean_diff_ann_V_baseline:.4f}")
print(f"95% Confidence Interval: ({ci_ann_V_baseline[0]:.4f}, {ci_ann_V_baseline[1]:.4f})")
print(f"P-value: {p_value_ann_V_baseline:.6f}")

# Interpret the p-value for ANN vs Baseline
if p_value_ann_V_baseline < 0.05:
    print("The difference between models is statistically significant.")
else:
    print("The difference between models is not statistically significant.")

# LR vs Baseline Comparison
mean_diff_lr_V_baseline, ci_lr_V_baseline, p_value_lr_V_baseline = compare_models(y_true, y_pred_lr, y_pred_baseline, alpha=0.05)

print("\nStatistical Comparison between Linear Regression and Baseline:")
print(f"Mean Difference: {mean_diff_lr_V_baseline:.4f}")
print(f"95% Confidence Interval: ({ci_lr_V_baseline[0]:.4f}, {ci_lr_V_baseline[1]:.4f})")
print(f"P-value: {p_value_lr_V_baseline:.6f}")

# Interpret the p-value for LR vs Baseline
if p_value_lr_V_baseline < 0.05:
    print("The difference between models is statistically significant.")
else:
    print("The difference between models is not statistically significant.")

# ANN vs LR Comparison
mean_diff_ann_V_lr, ci_ann_V_lr, p_value_ann_V_lr = compare_models(y_true, y_pred_ann, y_pred_lr, alpha=0.05)

print("\nStatistical Comparison between ANN and Linear Regression:")
print(f"Mean Difference: {mean_diff_ann_V_lr:.4f}")
print(f"95% Confidence Interval: ({ci_ann_V_lr[0]:.4f}, {ci_ann_V_lr[1]:.4f})")
print(f"P-value: {p_value_ann_V_lr:.6f}")

# Interpret the p-value for ANN vs LR
if p_value_ann_V_lr < 0.05:
    print("The difference between models is statistically significant.")
else:
    print("The difference between models is not statistically significant.")
