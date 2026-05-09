# comparison-of-five-machine-learning-models
A comparative analysis of CART, C4.5, Naive Bayes, Random Forest, and XGBoost for diabetes classification using the Pima Indians Diabetes Dataset.

# Diabetes Prediction Using Multiple Machine Learning Models

## Overview
This project implements and compares five different machine learning algorithms to predict diabetes based on diagnostic measurements. The models are trained and evaluated on the Pima Indians Diabetes Dataset, with performance metrics including accuracy, precision, recall, F1-score, and 5-fold cross-validation.

## Models Implemented
| Model | Description |
|-------|-------------|
| **CART** | Decision Tree using Gini impurity as splitting criterion |
| **C4.5 (Approximation)** | Decision Tree using Entropy (information gain) |
| **Naive Bayes** | Gaussian Naive Bayes classifier assuming feature independence |
| **Random Forest** | Ensemble of 100 decision trees with max depth 5 |
| **XGBoost** | Gradient boosting with extreme gradient optimization |

## Dataset
The dataset contains medical predictor variables and one target variable (`Outcome`):
- **Features**: Various diagnostic measurements (e.g., glucose level, BMI, age)
- **Target**: `0` = No Diabetes, `1` = Diabetes
- **Split**: 80% training / 20% testing

## Results

### Test Set Performance
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| CART | 79.22% | 74.47% | 63.64% | 68.63% |
| C4.5 | 78.57% | 70.37% | 69.09% | 69.72% |
| Naive Bayes | 76.62% | 66.10% | 70.91% | 68.42% |
| Random Forest | 76.62% | 68.63% | 63.64% | 66.04% |
| XGBoost | 70.78% | 58.06% | 65.45% | 61.54% |

### Cross-Validation (5-Fold)
| Model | Mean Accuracy | Std Dev |
|-------|---------------|---------|
| CART | 72.98% | ±4.81% |
| C4.5 | 72.81% | ±4.15% |
| Naive Bayes | 73.63% | ±3.57% |
| Random Forest | 76.71% | ±2.62% |
| XGBoost | 74.44% | ±2.88% |

## Visual Outputs
The code generates two decision tree visualizations:
- CART decision tree (Gini impurity)
- C4.5 decision tree (Entropy)

## Requirements
```bash
pip install pandas scikit-learn xgboost matplotlib
