import os
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import ann_classification, baseline_classifier, logistic_regression_classifier


current_dir = os.path.dirname(__file__)

# Construct the file path
filename = os.path.join(current_dir, 'optical_interconnection_network.csv')
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

scaler = StandardScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns)

# Extract features and labels
label_column = 'Temporal Distribution'
X = df_normalized.drop(columns=[label_column]).values  # Replace 'class_label' with the actual label column name
y = df_normalized[label_column].values  # Replace 'class_label' with the actual label column name

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, stratify=y)

regression = logistic_regression_classifier.logistic_regression_cv(X, y, 10)

baseline = baseline_classifier.get_baseline_table(X_train, X_test,    y_train, y_test)

ann = ann_classification.get_ann_table()

combined_results = pd.DataFrame({
        "Fold": ann["Fold"],
        "Best Hidden Units": ann["Best Hidden Units"],
        "Error Rate":  ann["Error Rate"],
        "Lambda": regression["Lambda"],
        "Error Rate": regression["Error Rate"],
        "Baseline Error Rate": baseline

    })

print(combined_results)