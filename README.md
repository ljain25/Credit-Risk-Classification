# 📌 Credit Risk Classification Project

## 📖 Overview
This project analyzes credit risk classification using machine learning models. The dataset is cleaned, preprocessed, and classified into four categories: P1, P2, P3, and P4. Various models are trained and optimized to achieve the best performance.

## 📂 Data Processing
- Removed invalid values (-99999) 🗑️
- Handled missing data 🔍
- Categorical & numerical feature selection 🏷️
- Performed Chi-square test and ANOVA for feature selection 📊
- Applied one-hot encoding & label encoding 🎭

## 🏗️ Model Training & Evaluation
### 1️⃣ Random Forest 🌳
- Accuracy: **76%** ✅

### 2️⃣ XGBoost ⚡
- Accuracy: **77%** 📈
- Best performing model overall 🎯

### 3️⃣ Decision Tree 🌲
- Accuracy: **70%** 📉
- Reduced performance compared to other models

## 🛠️ Hyperparameter Tuning
- Used Grid Search with multiple hyperparameters 🔄
- Achieved **82% training accuracy** and **78% test accuracy** after optimization 🚀

## 💾 Model Deployment
- Final model saved as `model.sav` 📥
- Training dataset exported as `training_data.csv` 📑


