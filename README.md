# 📌 Credit Risk Classification Project

## 📖 Overview
Ever wondered how banks decide whether you're a safe bet or a financial daredevil? 💰🚀 This project takes on credit risk classification using machine learning, sorting customers into four risk levels: P1 (Saints) 😇, P2 (Cautious) 🤔, P3 (Risky) ⚠️, and P4 (Wildcards) 🎲. 

## 📂 Data Processing
Like a detective hunting for clues 🔍, I cleaned and prepped the dataset meticulously:
- Kicked out invalid values (-99999) like unwanted guests 🚫
- Filled in missing data (because guessing is better than ghosting) 👻
- Sorted categorical & numerical features like a neat freak 🏷️
- Ran Chi-square & ANOVA tests to pick the MVP features 🏆
- Applied Variance Inflation Factor (VIF) to remove overly clingy, redundant variables 🚮
- One-hot & label encoding because machines don’t speak human 🎭

## 🏗️ Model Training & Evaluation
I let three ML models duke it out in the arena:

### 1️⃣ Random Forest 🌳
- Accuracy: **76%** ✅

### 2️⃣ XGBoost ⚡
- Accuracy: **77%** 📈
- The MVP—best performer overall! 🏆

### 3️⃣ Decision Tree 🌲
- Accuracy: **70%** 📉
- Good effort, but lost to the competition 😓

## 🛠️ Hyperparameter Tuning
- Grid Search = The secret sauce for model optimization 🔄
- Final results: **82% training accuracy** & **78% test accuracy** 🚀

## 💾 Model Deployment
- Trained model saved as `model.sav` 📥
- Training dataset stored in `training_data.csv` 📑
- Built a **Streamlit app** (`app.py`) for fun, easy predictions 🎛️

## 🖥️ Streamlit Web App Features
- 📂 Upload a CSV file, let the magic happen
- 🤖 Automatically preprocesses & encodes data
- 📊 Predicts risk levels and displays results
- 📥 Downloadable CSV of risk classifications

