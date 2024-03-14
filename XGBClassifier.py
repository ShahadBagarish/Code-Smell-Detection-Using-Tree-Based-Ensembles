import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score

# Load the dataset from CSV
data = pd.read_csv('Dataset/DataClass_Clean_GainRatio.csv')

# Assuming the last column is the target variable and the rest are features
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target variable

# Create XGBClassifier model
xgb_model = XGBClassifier()

# Perform 10-fold cross-validation for accuracy
cv_accuracy_scores = cross_val_score(xgb_model, X, y, cv=10, scoring='accuracy')

# Perform 10-fold cross-validation for AUC
cv_auc_scores = cross_val_score(xgb_model, X, y, cv=10, scoring='roc_auc')

# Print the cross-validation scores
print("Cross-Validation Accuracy Scores:", cv_accuracy_scores)
print("Mean CV Accuracy:", cv_accuracy_scores.mean())
print("Cross-Validation AUC Scores:", cv_auc_scores)
print("Mean CV AUC:", cv_auc_scores.mean())

# Plotting the cross-validation accuracy scores
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), cv_accuracy_scores, marker='o', linestyle='--', color='b', label='Accuracy')
plt.plot(range(1, 11), cv_auc_scores, marker='o', linestyle='--', color='r', label='AUC')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.title('Cross-Validation Accuracy and AUC Scores')
plt.legend()
plt.grid(True)
plt.show()
