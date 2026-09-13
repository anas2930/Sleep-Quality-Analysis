# Sleep Efficiency Analysis

Exploratory data analysis and machine learning pipeline to predict **Sleep Efficiency** based on lifestyle and physiological factors, using the `Sleep_Efficiency.csv` dataset.

## Dataset

Features include:
- Age, Gender
- Bedtime, Wakeup time
- Sleep duration, Sleep efficiency (target)
- REM / Deep / Light sleep percentage
- Awakenings, Caffeine consumption, Alcohol consumption
- Smoking status, Exercise frequency

## Workflow

1. **Data Understanding** — inspect structure, dtypes, duplicates, and null values.
2. **Missing Value Imputation** — compared Mean, KNN, and RandomForest-based imputation (via GridSearchCV); final imputation uses **KNNImputer**.
3. **Feature Engineering** — parsed and dropped raw `Bedtime`/`Wakeup time` columns, cleaned column names, dropped `ID`.
4. **Outlier Handling** — distribution analysis (histograms, boxplots, skewness) with a conditional approach: 3-sigma bounds for near-normal features, IQR bounds for skewed ones.
5. **Correlation Analysis** — heatmap of feature correlations with sleep efficiency.
6. **Encoding** — one-hot encoding for `Gender` and `Smoking status`.
7. **Feature Selection** — Recursive Feature Elimination with Cross-Validation (RFECV) using a Random Forest estimator.
8. **Model Training & Tuning** — GridSearchCV/RandomizedSearchCV hyperparameter tuning for:
   - Linear Regression / Ridge
   - K-Nearest Neighbors
   - Decision Tree
   - Random Forest
   - Support Vector Regression (SVR)
   - XGBoost
9. **Model Evaluation** — compared models on MAE, MSE, R², and MAPE-based accuracy.

## Requirements
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
