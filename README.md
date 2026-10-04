# Task 1: Data Cleaning & Visualization Project

## Project Overview
This project processes, cleans, and visualizes raw passenger data from the Titanic dataset using Python, Pandas, Matplotlib, and Seaborn.

## Key Steps Performed
1. **Handling Missing Values:**
   - Imputed `Age` using the median value (28.0) to preserve distribution without skewing.
   - Replaced missing `Embarked` entries with the mode ('S').
   - Dropped the `Cabin` column due to >75% missing data.
2. **Outlier Treatment:**
   - Capped extreme fare outliers using the Interquartile Range (IQR) method.
3. **Data Type Casting:**
   - Converted categorical columns (`Survived`, `Pclass`, `Sex`, `Embarked`) to category data types.
4. **Visualizations:**
   - **Survival Rate by Gender:** Confirmed females had a significantly higher survival probability (~74% vs ~19%).
   - **Age Distribution:** Children under 10 had a higher likelihood of rescue.
   - **Class Analysis:** 1st class passengers showed higher survival rates compared to 3rd class passengers.
   - # Task 2: Predictive Modeling Using Machine Learning

## Objective
Build and evaluate supervised machine learning models to classify breast cancer tumors as Malignant or Benign using clinical features.

## Workflow
1. **Data Preprocessing:** Handled features from the Breast Cancer Wisconsin dataset; standardized inputs using `StandardScaler`.
2. **Model Training:** Implemented **Logistic Regression** and **Random Forest Classifier**.
3. **Evaluation:** Evaluated performance using Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.
4. **Results:**
   - Model: Random Forest Classifier
   - Accuracy: ~96%
   - ROC-AUC: ~0.99

## How to Run
```bash
pip install pandas matplotlib seaborn numpy
python task1_data_cleaning_visualization.py
