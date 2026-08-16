# Predictive Modeling for Workforce Retention

A machine learning project that predicts voluntary employee attrition and translates the model's findings into a business intervention strategy, built as a graduate capstone using the CRISP-DM framework.

## Overview

Employee turnover is costly and often preventable if at-risk employees can be identified early. This project builds a classification model on the IBM HR Analytics Employee Attrition dataset (1,470 employees, 35 variables) to predict which employees are likely to leave, then converts the model's output into an actionable retention strategy rather than stopping at a metric on a page.

## Objectives

- Identify the strongest predictors of voluntary employee attrition
- Build and compare classification models to flag at-risk employees
- Address the dataset's class imbalance (~84% stayed vs. ~16% left) without sacrificing recall
- Translate model results into a business-facing recommendation with estimated financial impact

## Methodology

Built end-to-end using the **CRISP-DM** framework:

1. **Business Understanding** — define attrition as the target problem and frame the cost of false negatives (missing an employee who leaves) as more expensive than false positives
2. **Data Understanding & Cleaning** — audit for missing values, duplicates, and zero-variance features
3. **Feature Engineering** — categorical encoding (label/one-hot), feature selection
4. **Class Balancing** — random oversampling of the minority (attrition) class to correct the imbalance
5. **Modeling** — Logistic Regression vs. Random Forest
6. **Evaluation** — confusion matrix, precision, recall, F1, ROC/AUROC — prioritizing recall over raw accuracy given the business cost of missed at-risk employees
7. **Deployment Recommendation** — Early Warning Dashboard concept + a 90-day "Stay Interview" intervention plan for flagged employees

## Results

- Final model: **Logistic Regression**, selected over Random Forest for its interpretability and stronger alignment with the business cost function
- **66% recall** in identifying employees who left
- Top predictive signals: **Overtime, Monthly Income, Age**
- Estimated **~$19.8M in preventable annual attrition cost** if the proposed early-intervention program were adopted

## Tech Stack

- **Language:** Python
- **Libraries:** pandas, scikit-learn, matplotlib
- **Techniques:** Logistic Regression, Random Forest, oversampling, one-hot/label encoding, ROC/AUROC analysis

## Repository Structure

```
├── data/               # Cleaned dataset used for modeling
├── notebooks/          # EDA, feature engineering, and modeling notebooks
├── report/             # Full written report (methodology, results, business case)
└── README.md
```

## Dataset

[IBM HR Analytics Employee Attrition & Performance dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) — 1,470 employee records, 35 features.

## Author

Mario Benjamin Garnett — Operations Research Analyst
