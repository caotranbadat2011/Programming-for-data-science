# Credit Card Fraud Detection using Logistic Regression (NumPy Only)
A complete machine learning pipeline implemented from scratch without scikit-learn.

## 📌 Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Method](#method)
4. [Installation & Setup](#installation--setup)
5. [Usage](#usage)
6. [Results](#results)
7. [Project Structure](#project-structure)
8. [Challenges & Solutions](#challenges--solutions)
9. [Future Improvements](#future-improvements)
10. [Contributors](#contributors)
11. [Contact](#contact)
12. [License](#license)

---

## 🚀 Introduction

### Motivation & Real-world Applications
- Prevent financial losses  
- Automatically flag abnormal transactions  
- Support banking and financial risk management  
- Provide an ML solution to a heavily imbalanced dataset  

### Project Objectives
- Perform operations using numpy without using loops
- Train and evaluate a Logistic Regression model implemented *entirely using NumPy*  
- Perform Exploratory Data Analysis (EDA)  
- Build a full preprocessing pipeline  
- Handle class imbalance using class weighting  
- Visualize results and metrics  
- Produce a fully reproducible ML workflow  

---

## 📊 Dataset

### Source
Dataset available on Kaggle:  
https://www.kaggle.com/mlg-ulb/creditcardfraud  
(The raw file is too large to include directly in this repository.)

### Characteristics
- 284,807 transactions  
- 492 fraud cases  
- Fraud ratio ≈ 0.17% (highly imbalanced)

### Feature Description
- **Time**: Seconds elapsed between a transaction and the first transaction  
- **Amount**: Transaction amount  
- **V1–V28**: PCA-transformed anonymized features  
- **Class**: Target label (0 = normal, 1 = fraud)

### Key Observations
- `Amount` contains extreme outliers  
- PCA features vary in scale  
- Severe class imbalance requires special treatment  

---

## ⚙️ Method

This project follows a complete machine learning workflow implemented from scratch using NumPy.  
The pipeline includes: **Data Exploration → Preprocessing → Model Building → Evaluation**.

---

### 1. Data Exploration (Notebook: `01_data_exploration.ipynb`)

The EDA step provides insights into the structure and behavior of the dataset before any modeling.  
Key analyses performed:

#### • Class Distribution
- Fraud accounts for only **0.17%** of all transactions.
- Confirmed the need for class imbalance handling.

#### • Feature Distributions
- `Amount` is highly skewed and contains extreme outliers.
- PCA components (`V1`–`V28`) follow transformed distributions that do not require outlier removal.

#### • Correlation Matrix
- PCA components show very low correlation due to PCA orthogonality.
- `Amount` and `Time` have minimal linear correlation with other features.

#### • Fraud vs Non-Fraud Comparison
- Several PCA components show clear distribution differences between fraud and non-fraud (e.g., V10, V12, V14).
- These insights confirm that Logistic Regression can learn meaningful separation.

The EDA step guided the design of the preprocessing pipeline.

---

### 2. Data Preprocessing (Notebook: `02_preprocessing.ipynb`)

#### (a) Missing Value Handling
- Replace missing values with column-wise medians.

#### (b) Outlier Processing (Amount Only)
`Amount` is clipped between the 1st and 99th percentiles to reduce extreme values: 

```python
x = min(max(x, P1), P99)
```python


This ensures stable gradient computation during training.

#### (c) Feature Standardization (Z-score Normalization)

\[
X_{\text{scaled}} = \frac{X - \mu}{\sigma}
\]

Ensures stable gradient descent and prevents sigmoid overflow.

#### (d) Train/Test Split
- 80% training  
- 20% testing  
- Random shuffle  

#### (e) Class Weighting
To handle imbalance:

\[
pos_weight = (#negative samples) / (#positive samples)
sample_weight = pos_weight if y == 1 else 1
\]

This forces the model to pay more attention to fraud samples.

---