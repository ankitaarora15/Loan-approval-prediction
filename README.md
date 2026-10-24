# Loan Approval Prediction

## Project Overview

This project explores supervised machine learning models for predicting loan approval outcomes from applicant-related features. It compares multiple classification algorithms using a consistent train/test split and cross-validation.

## Dataset

- File: `loan_approval_data.csv`
- Original dataset size: 1,000 rows and 20 columns.
- After removing rows with missing target labels: 950 rows.
- Target variable: `Loan_Approved` (`Yes` or `No`).
- Missing values in input features are handled within the preprocessing pipeline.

**Dataset provenance:** The source of the dataset has not yet been established.

## Methods

### Data Preparation
- Inspected data types, missing values, duplicate records, numerical distributions, and categorical values.
- Removed rows with missing target labels rather than imputing unknown outcomes.
- Excluded `Applicant_ID` from the model features.
- Used a stratified 80/20 train/test split.
- Applied median imputation and standardization to numerical features.
- Applied most-frequent imputation and one-hot encoding to categorical features.
- Fitted preprocessing operations within scikit-learn pipelines to avoid fitting them on the held-out test set.

### Models Evaluated
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes

Grid search with cross-validation was used to tune the regularization parameter for Logistic Regression and the number of neighbors for KNN.

### Evaluation Metrics
Models were compared using accuracy, precision, recall, and F1-score for the approved-loan class (`Yes`). Five-fold stratified cross-validation was also performed on the training set.

## Results

Results on the held-out test set:

| Model | Accuracy | Precision (Yes) | Recall (Yes) | F1-score (Yes) |
|---|---:|---:|---:|---:|
| Logistic Regression (Original) | 0.884 | 0.852 | 0.767 | 0.807 |
| Logistic Regression (Tuned) | 0.879 | 0.836 | 0.767 | 0.800 |
| Gaussian Naive Bayes | 0.874 | 0.833 | 0.750 | 0.789 |
| KNN (Tuned) | 0.858 | 0.824 | 0.700 | 0.757 |
| KNN (Original) | 0.800 | 0.696 | 0.650 | 0.672 |

The original Logistic Regression achieved the highest test-set F1-score among the evaluated model versions. Tuning improved KNN's test-set F1-score, while tuning Logistic Regression did not improve its held-out test-set metrics.

## Limitations

- The dataset's provenance has not been verified.
- Results are based on one held-out test split and may vary on other data.
- Cross-validation results are estimates from the training data, not guarantees of future performance.
- This analysis does not establish causal relationships or evaluate fairness across applicant groups.
- These models are for educational analysis and should not be treated as validated systems for real-world lending decisions.

## Project Files

- `loan_approval_prediction.ipynb` — data exploration, preprocessing, model training, tuning, and evaluation.
- `loan_approval_data.csv` — dataset used in the analysis.
- `requirements.txt` — Python dependencies.

## How to Run

1. Install Python.
2. Download or clone this repository.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Launch Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open `loan_approval_prediction.ipynb` and run the cells in order.

Keep the notebook and CSV file in the same project folder so the relative dataset path works.

## Tools

Python, pandas, NumPy, Matplotlib, Seaborn, and scikit-learn.