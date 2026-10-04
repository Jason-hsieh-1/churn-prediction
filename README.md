# Bank Customer Churn Prediction

Predicting whether a bank's customers will leave (churn), using a binary classification approach in Python.

This project was built for a Kaggle classification challenge in the course **Machine Learning and Artificial Intelligence**. It compares four model families (logistic regression, random forest, support vector machine, and a neural network) and ranks them mainly by **ROC AUC**, the leaderboard metric. A **tuned random forest** came out on top, with a held-out AUC of **0.834** and a best Kaggle score of **0.85**.

All work is in a single notebook: [`Group_07_code.ipynb`](Group_07_code.ipynb).

---

## Table of contents

- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Key findings](#key-findings)
- [How to run](#how-to-run)
- [Repository structure](#repository-structure)
- [Known limitations and possible improvements](#known-limitations-and-possible-improvements)
- [License](#license)

---

## Dataset

The data comes from the course's Kaggle competition and is **not included in this repository**. It contains two files:

| File | Rows | Description |
|---|---|---|
| `train.csv` | 8,000 | Customer features and the `churn` label |
| `test.csv` | ~2,000 (IDs from 8001) | The same features without the label, used for the Kaggle submission |

**Features**

| Column | Type | Description |
|---|---|---|
| `ID` | int | Customer identifier (not used for modeling) |
| `credit_score` | int | Credit score (350–850) |
| `country` | categorical | France, Germany, or Spain |
| `gender` | categorical | Female or Male |
| `age` | int | Age in years (18–92) |
| `tenure` | int | Years as a customer (0–10) |
| `balance` | float | Account balance |
| `products_number` | int | Number of bank products held (1–4) |
| `credit_card` | binary | Has a credit card |
| `active_member` | binary | Is an active member |
| `estimated_salary` | float | Estimated annual salary |
| `churn` | binary | **Target**: 1 = left the bank, 0 = stayed |

Neither file has missing values. The target is imbalanced: about **20.6%** of training customers churned.

---

## Approach

```
train.csv ──► EDA ──► Preprocessing ──► 90/10 split ──► Train & tune 4 models ──► Compare AUC ──► Feature engineering ──► Final model ──► Kaggle submission
```

### 1. Exploratory data analysis
- Summary statistics, a pairplot, and histograms of the numeric features
- Churn rate broken down by **country** and **gender** (stacked bar charts)
- Correlation of every feature with churn, plus a full correlation heatmap

### 2. Preprocessing
- One-hot encoding of `country` and `gender`
- Min-max scaling of `credit_score`, `age`, `tenure`, `balance`, and `estimated_salary`
- A 90/10 train/validation split of `train.csv` (`random_state=42`), giving 7,200 rows for training and 800 for validation

### 3. Models
Each model was tuned with `GridSearchCV`, scored on `roc_auc`:

| Model | Search space | Best parameters | CV folds |
|---|---|---|---|
| Logistic regression | `C` ∈ {0.001, 0.01, 0.1, 1}, `max_iter` | `C=1`, `penalty='l2'` | 10 |
| Random forest | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` | not printed in notebook | 10 |
| SVM (polynomial kernel) | `C` ∈ {0.1, 1, 10}, `degree` ∈ {2, 3, 4}, `coef0` ∈ {0, 1, 2} | `C=1`, `degree=2`, `coef0=2` | 5 |
| Neural network (Keras) | 128 → 64 → 1 dense layers with BatchNorm and Dropout(0.1), Adam (lr=0.001), 20 epochs, standardized inputs | — | — |

### 4. Feature engineering
Three ratio features were added to see whether they would improve the random forest:

- `balance_per_product` = balance / products_number
- `balance_by_est_salary` = balance / estimated_salary
- `tenure_age_ratio` = tenure / age

---

## Results

### Validation set (800 held-out rows)

| Model | ROC AUC | Accuracy | Churn precision | Churn recall | Churn F1 |
|---|---|---|---|---|---|
| **Random forest (tuned)** | **0.834** | 0.85 | 0.80 | 0.43 | 0.56 |
| Random forest (tuned + engineered features) | 0.830 | **0.86** | **0.82** | 0.44 | **0.57** |
| Neural network | 0.827 | 0.85 | 0.73 | **0.48** | **0.58** |
| SVM (polynomial, tuned) | 0.813 | 0.84 | **0.87** | 0.34 | 0.49 |
| Logistic regression (tuned) | 0.722 | 0.81 | 0.68 | 0.21 | 0.32 |

### 10-fold cross-validation on the training split (default hyperparameters)

| Algorithm | ROC AUC mean | ROC AUC std | Accuracy mean | Accuracy std |
|---|---|---|---|---|
| Random forest | **85.69** | 1.31 | **86.12** | 1.19 |
| Kernel SVM | 82.99 | 2.16 | 85.12 | 1.29 |
| Logistic regression | 77.88 | 1.92 | 71.94 | 2.41 |

### Kaggle
- Tuned random forest: **0.84**
- Tuned random forest with engineered features: **0.85** (best leaderboard score)

### Model selection
The **tuned random forest** was picked as the final model because it had the highest held-out AUC. Adding the engineered features raised accuracy and churn precision slightly, and gave the best Kaggle score, but it did not raise validation AUC.

---

## Key findings

- **Age, balance, living in Germany, and being female** are positively correlated with churn.
- **Being an active member, living in France, and being male** are negatively correlated with churn.
- **Germany** has a noticeably higher churn rate than France or Spain, which are similar to each other.
- Every model is much better at spotting customers who stay than customers who leave. Churn recall peaks at roughly 0.48, which is typical for an imbalanced target with a 0.5 decision threshold.
- Tree ensembles fit this tabular data best. The linear model clearly lags behind, which suggests non-linear relationships between features and churn.

---

## How to run

The notebook was written for **Google Colab** and reads its data from Google Drive.

1. Get `train.csv` and `test.csv` from the course's Kaggle competition.
2. Upload them to `MyDrive/Colab Notebooks/` and create a `MyDrive/Colab Notebooks/submission/` folder for the output files.
3. Open `Group_07_code.ipynb` in Colab and run all cells.

**To run locally instead**, remove the `google.colab` drive-mount cell, change the file paths in the `pd.read_csv(...)` and `.to_csv(...)` calls, and install the dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost statsmodels tensorflow jupyter
```

> Note: the full grid searches (especially the random forest grid with 10-fold CV, and SVM with `probability=True`) can take a while on a CPU.

### Outputs
The notebook writes one Kaggle submission file (`ID`, `churn` probability) per model:

```
submission_logreg.csv
submission_rfc_tuning.csv
submission_svm_tuning.csv
submission_nn.csv
submission_rfc_tuning_new.csv
```

---

## Repository structure

```
churn-prediction/
├── Group_07_code.ipynb   # Full analysis: EDA, preprocessing, modeling, results, discussion
├── README.md
└── LICENSE               # GNU GPL v3
```

---

## Known limitations and possible improvements

These came up while reviewing the notebook and would be worth fixing in a future version:

- **Scaler fit on test data.** `MinMaxScaler` is fit again on the test set (`scaler.fit_transform(test3[...])`) instead of reusing the scaler fit on training data (`scaler.transform`). Train and test features can therefore end up on slightly different scales.
- **Neural network submission bug.** `submission_nn.csv` concatenates `pred_svm_tuning` instead of `pred_nn`, so it actually holds the SVM predictions. The test inputs for the network were also not passed through the `StandardScaler` used during training.
- **Redundant dummy columns.** Both `gender_Female` and `gender_Male` are kept. They are perfectly collinear, which matters for logistic regression.
- **Class imbalance is not addressed** in the tuned models. Options include `class_weight='balanced'`, resampling (for example SMOTE), or tuning the decision threshold to raise churn recall.
- **Single 800-row validation split.** Small AUC differences between models (for example 0.834 vs 0.830) are within noise; stratified k-fold estimates would be more reliable.
- **Unused imports.** XGBoost is imported but never trained. Gradient boosting (XGBoost, LightGBM, or CatBoost) is a natural next model to try on this kind of tabular data.
- **Hard-coded Colab paths** make the notebook harder to run outside Google Drive.

---

## License

This project is licensed under the **GNU General Public License v3.0**. See [`LICENSE`](LICENSE) for details.
