# Sampling and Model Evaluation
## By: Sachin Sushil Singh
## Roll no: 102103575

## Overview

This code demonstrates the application of various sampling methods to address class imbalance in a credit card fraud dataset. The resampled data is then used to train different classification models, and the resulting accuracies are recorded.

## Code Structure

- **Libraries Used:**
    - pandas
    - imbalanced-learn (imblearn)
    - scikit-learn

- **Models Used:**
    1. Logistic Regression
    2. Random Forest
    3. Support Vector Machine (SVM)
    4. XGBoost
    5. Multi-layer Perceptron (MLP)

- **Sampling Methods:**
    1. Random Over-sampling
    2. Random Under-sampling
    3. Tomek Links
    4. SMOTE (Synthetic Minority Over-sampling Technique)
    5. NearMiss

## Code Execution

1. **Read the Dataset:**
   - The code reads the credit card fraud dataset from a CSV file (`Creditcard_data.csv`) using pandas.

2. **Data Preprocessing:**
   - The dataset is split into features (x) and the target variable (y).
   - Train-test split is performed with a test size of 20%.

3. **Sampling and Model Training:**
   - Various sampling methods are applied to the training data before training the models.
   - Models are trained using different sampling techniques.

4. **Results:**
   - The accuracy scores for each model and sampling method combination are recorded.

5. **Saving Results:**
   - The results are saved in a CSV file (`sampling_results.csv`) for further analysis.

## Models and Sampling Strategies

- **Logistic Regression:**
  - Solver: newton-cg
  - Maximum Iterations: 1000

- **Random Forest:**
  - Default settings

- **SVM (Support Vector Machine):**
  - Default settings

- **XGBoost:**
  - Default settings

- **MLP (Multi-layer Perceptron):**
  - Maximum Iterations: 1000

## Results

The accuracy scores for each model and sampling method are recorded in the `sampling_results.csv` file.
It can be observed that Random Forest Algorithm and XGBost provide the most accuracy with Random Over Sampling.

| Model                | RandomOverSampler | RandomUnderSampler | TomekLinks | SMOTE | NearMiss |
|----------------------|-------------------|--------------------|------------|-------|----------|
| Logistic Regression  | 0.858             | 0.561              | 0.574      | 0.871 | 0.206    |
| Random Forest        | 0.994             | 0.600              | 0.710      | 0.994 | 0.394    |
| SVM                  | 0.697             | 0.748              | 0.626      | 0.690 | 0.348    |
| XGBoost              | 0.994             | 0.684              | 0.600      | 0.987 | 0.097    |
| MLP                  | 0.961             | 0.677              | 0.716      | 0.948 | 0.445    |

## How to Run

1. Ensure you have all the required libraries installed (`pandas`, `imbalanced-learn`, `scikit-learn`, `xgboost`).
2. Download the credit card fraud dataset and save it as `Creditcard_data.csv` in the same directory.
3. Run the provided Python script.

Feel free to customize the code for your specific needs or extend it for more advanced analyses.
