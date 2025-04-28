# Titanic Survival Prediction

## Project Overview
This project predicts whether a passenger survived the Titanic disaster using Machine Learning models.

We apply data preprocessing, train multiple models, and select the best-performing one.

---

## Dataset
- File: `data/tested.csv`
- Features: Age, Gender, Ticket Class, Fare, Cabin, etc.
- Target: `Survived` (0 = No, 1 = Yes)

---

## Preprocessing
- **Handling Missing Values**: 
  - Numeric columns → Median Imputation
  - Categorical columns → Most Frequent Imputation
- **Encoding**: 
  - Categorical features → One-Hot Encoding
- **Scaling**: 
  - Numeric features → StandardScaler

---

## Models Used
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

---

## Performance Metrics

| Model                | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 80.1%    | 78.0%     | 74.0%  | 76.0%    |
| Random Forest        | 85.3%    | 84.1%     | 81.2%  | 82.6%    |
| Gradient Boosting    | 86.1%    | 85.3%     | 83.0%  | 84.1%    |

➡️ **Gradient Boosting** gave the best results.
