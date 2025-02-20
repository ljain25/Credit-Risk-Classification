import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import warnings
import os


a1 = pd.read_excel("case_study1.xlsx")
a2 = pd.read_excel("case_study2.xlsx")


df1 = a1.copy()
df2 = a2.copy()


# REMOVING -99999 VALUES

# 1. In df1
df1["Age_Oldest_TL"].value_counts()           # -99999 appears only 45 times
df1 = df1[df1["Age_Oldest_TL"] != -99999]

# 2. In df2
df2["max_delinquency_level"].value_counts()       # number of times -99999 occurs = 35000

# If no. of -99999 in a column > 0.2*total records -> Remove the column. If <  0.2*total records -> remove the records containing it (if resultant no. of records doesn't go too low

columns_to_be_removed = []       # columns to be removed

for i in df2.columns:
    if df2[df2[i] == -99999].shape[0] > df2.shape[0]*0.2:
        columns_to_be_removed.append(i)

df2.drop(columns = columns_to_be_removed, inplace = True)

df3 = df2.copy()

for i in df2.columns:               # records to be removed
    df3 = df3[df3[i] != -99999]     # more than 42000 rows left after the treatment

if df3.shape[0] >= 0.2*df2.shape[0]:
    df2 = df3

df1.isnull().sum()
df2.isnull().sum()


# MERGING THE DATAFRAMES
df = df1.merge(df2, how = "inner", on = "PROSPECTID")

df.head()

df.info()       # has no null values

# DIVIDE THE FEATURES INTO CATEGORICAL AND NUMERICAL AND TREAT THEM SEPARATELY

# 1. CATEGORICAL COLUMNS
cat_cols = []
for i in df.columns:
    if df[i].dtype == "object":
        print(i)
        cat_cols.append(i)

df["MARITALSTATUS"].value_counts()
df["EDUCATION"].value_counts()
df["GENDER"].value_counts()
df["last_prod_enq2"].value_counts()
df["first_prod_enq2"].value_counts()
df["Approved_Flag"].value_counts()

# Comparing every column with Approved Flag column to see their association - chi square contingency table (two categorical columns)
# Keep only important columns
# Let alpha = 0.05
for i in cat_cols[:-1]:   # last column is Approved_Flag itself
    chi2, pval, _, _ = chi2_contingency(pd.crosstab(df[i], df["Approved_Flag"]))
    print(i, "---", pval)

# The p values for all these is less than alpha - we reject H0 - we can say these columns are associated with the Approved_flag column
# All columns are important


# 2. NUMERICAL COLUMNS
num_cols = []
for i in df.columns:
    if df[i].dtype != "object" and i not in ["PROSPECTID", "Approved_Flag"]:
        num_cols.append(i)

# Checking multicollinearity - calculating VIF (sequential)
vif_data = df[num_cols]
total_columns = vif_data.shape[1]
columns_to_be_kept = []
column_index = 0

for i in range(0, total_columns):

    vif_value = variance_inflation_factor(vif_data, column_index)
    print(column_index, "----", vif_value)

    if vif_value <= 6:
        columns_to_be_kept.append(num_cols[i])
        column_index = column_index + 1

    else:
        vif_data = vif_data.drop(columns = [num_cols[i]])

# Before VIF: 72 columns, After VIF: 39 columns

# Checking association of numerical columns with the Target variable and keep only the necessary
columns_to_be_kept_numerical = []

for i in columns_to_be_kept:
    a = list(df[i])
    b = list(df["Approved_Flag"])

    group_P1 = [value for (value, group) in zip(a,b) if group == "P1"]
    group_P2 = [value for (value, group) in zip(a,b) if group == "P2"]
    group_P3 = [value for (value, group) in zip(a,b) if group == "P3"]
    group_P4 = [value for (value, group) in zip(a,b) if group == "P4"]

    f_statistic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)

    if p_value <= 0.05:
        columns_to_be_kept_numerical.append(i)

# 37 numerical columns left

df = df[cat_cols+columns_to_be_kept_numerical]         # updating df with necessary categorical and numerical columns


# LABEL ENCODING FOR THE CATEGORICAL FEATURES
for i in cat_cols:
    print(df[i].unique())

# ordinal feature - Only EDUCATION - LABEL ENCODING

# SSC - 1
# 12TH - 2
# GRADUATE - 3
# UNDER GRADUATE - 3
# POST-GRADUATE - 4
# OTHERS - 1             --> THAT IS NOT YET VERIFIED BY THE BUSINESS END USER
# PROFESSIONAL - 3

df.loc[df["EDUCATION"] == "SSC", ["EDUCATION"]] = 1
df.loc[df["EDUCATION"] == "12TH", ["EDUCATION"]] = 2
df.loc[df["EDUCATION"] == "GRADUATE", ["EDUCATION"]] = 3
df.loc[df["EDUCATION"] == "UNDER GRADUATE", ["EDUCATION"]] = 3
df.loc[df["EDUCATION"] == "POST-GRADUATE", ["EDUCATION"]] = 4
df.loc[df["EDUCATION"] == "OTHERS", ["EDUCATION"]] = 1
df.loc[df["EDUCATION"] == "PROFESSIONAL", ["EDUCATION"]] = 3

df["EDUCATION"].value_counts()
df["EDUCATION"].dtype
df["EDUCATION"] = df["EDUCATION"].astype(int)
df.info()

df_encoded = pd.get_dummies(df, columns = ["MARITALSTATUS", "GENDER", "last_prod_enq2", "first_prod_enq2"], dtype = "int", drop_first = True)

df_encoded.info()


# MACHINE LEARNING MODEL FITTING

# 1. RANDOM FOREST

y = df_encoded["Approved_Flag"]
x = df_encoded.drop(columns = ["Approved_Flag"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

rf_classifier = RandomForestClassifier(n_estimators=200, random_state = 42)
rf_classifier.fit(x_train, y_train)
y_pred = rf_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

print()
print(f"Accuracy: {accuracy}")
print()

for i , v in enumerate(["p1", "p2", "p3", "p4"]):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 score: {f1_score[i]}")
    print()


# p3 is not being predicted well by random forest
# Accuracy - 76%


# 2. XGBOOST

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

xgb_classifier = xgb.XGBClassifier(objective = "multi:softmax", num_class = 4)

y = df_encoded["Approved_Flag"]
x = df_encoded.drop(columns = ["Approved_Flag"])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

x_train, x_test, y_train, y_test  =train_test_split(x, y_encoded, test_size = 0.2, random_state = 42)

xgb_classifier.fit(x_train, y_train)
y_pred = xgb_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

print()
print(f"Accuracy: {accuracy}")
print()

for i, v in enumerate(["p1", "p2", "p3", "p4"]):
    print(f"Class: {v}")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 score: {f1_score[i]}")
    print()

# Accuracy - 77%
# Better but not good prediction for p3


# 3. DECISION TREE
from sklearn.tree import DecisionTreeClassifier

y = df_encoded["Approved_Flag"]
x = df_encoded.drop(columns = ["Approved_Flag"])

x_train, x_test, y_train, y_test  =train_test_split(x, y_encoded, test_size = 0.2, random_state = 42)

dt_model = DecisionTreeClassifier(max_depth=20, min_samples_split=10)
dt_model.fit(x_train, y_train)
y_pred = dt_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

print()
print(f"Accuracy: {accuracy}")
print()

for i, v in enumerate(["p1", "p2", "p3", "p4"]):
    print(f"Class: {v}")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 score: {f1_score[i]}")
    print()

# Accuracy - 70% - significantly reduced


# XGBOOST comes out to be the best, move forward with it - do its hyper parameter tuning


# HYPER PARAMETER TUNING FOR XGBOOST

# Define hyperparameter grid
param_grid = {
    "colsample_bytree": [0.1,0.3,0.5,0.7,0.9],
    "learning_rate": [0.001, 0.01, 0.1, 1],
    "max_depth": [3, 5, 8, 10],
    "alpha": [1, 10, 100],
    "n_estimators": [10, 50, 100]
}

index = 0

answers_grid = {
    "combination": [],
    "train_accuracy": [],
    "test_accuracy": [],
    "colsample_bytree": [],
    "learning_rate": [],
    "max_depth": [],
    "alpha": [],
    "n_estimators": []
}

# loop through each combination of hyper parameters
for colsample_bytree in param_grid["colsample_bytree"]:
    for learning_rate in param_grid["learning_rate"]:
        for max_depth in param_grid["max_depth"]:
            for alpha in param_grid["alpha"]:
                for n_estimators in param_grid["n_estimators"]:

                    index += 1

                    model = xgb.XGBClassifier(objective = "multi_softmax",
                                           num_class = 4,
                                           colsample_bytree = colsample_bytree,
                                           learning_rate = learning_rate,
                                           max_depth = max_depth,
                                           alpha = alpha,
                                           n_estimators = n_estimators)

                    y = df_encoded["Approved_Flag"]
                    x = df_encoded.drop(["Approved_Flag"], axis = 1)

                    label_encoder = LabelEncoder()
                    y_encoded = label_encoder.fit_transform(y)

                    x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size = 0.2, random_state = 42)

                    model.fit(x_train, y_train)

                    # predict on training and testing data
                    y_pred_train = model.predict(x_train)
                    y_pred_test = model.predict(x_test)

                    # calculating accuracy for both training and testing data
                    accuracy_train = accuracy_score(y_train, y_pred_train)
                    accuracy_test = accuracy_score(y_test, y_pred_test)

                    # adding values to the lists
                    answers_grid["combination"].append(index)
                    answers_grid["train_accuracy"].append(accuracy_train)
                    answers_grid["test_accuracy"].append(accuracy_test)
                    answers_grid["colsample_bytree"].append(colsample_bytree)
                    answers_grid["learning_rate"].append(learning_rate)
                    answers_grid["max_depth"].append(max_depth)
                    answers_grid["alpha"].append(alpha)
                    answers_grid["n_estimators"].append(n_estimators)


pd.DataFrame({"combination": answers_grid["combination"],
              "train_accuracy": answers_grid["train_accuracy"],
              "test_accuracy": answers_grid["test_accuracy"],
              "colsample_bytree": answers_grid["colsample_bytree"],
              "learning_rate": answers_grid["learning_rate"],
              "max_depth": answers_grid["max_depth"],
              "alpha": alpha,
              "n_estimators": n_estimators}).to_excel("Accuracy Table.xlsx")

# The values for train_accuracy (0.99) and test_accuracy (0.76) is optimum for
# num_class = 4, colsample_bytree = 0.9, learning_rate = 1, max_depth = 10, alpha = 100, n_estimators = 100

# optimum model
model = xgb.XGBClassifier(objective = "multi:softmax",
                          num_class = 4,
                          colsample_bytree = 0.9,
                          learning_rate = 1,
                          max_depth = 10,
                          alpha = 100,
                          n_estimators = 100)

df_encoded.to_csv("training_data.csv", index = False)

# SAVE MODEL
import pickle
filename = "model.sav"
pickle.dump(model, open(filename, "wb"))
