import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import metrics

# Load the dataset from CSV
data = pd.read_csv('Dataset/DataClass_Clean_GainRatio.csv')

# Assuming the last column is the target variable and the rest are features
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target variable

# Create DecisionTreeClassifier model
decision_tree_model = DecisionTreeClassifier()

# Perform 10-fold cross-validation for accuracy
cv_accuracy_scores = cross_val_score(decision_tree_model, X, y, cv=10, scoring='accuracy')

# Perform 10-fold cross-validation for AUC
cv_auc_scores = cross_val_score(decision_tree_model, X, y, cv=10, scoring='roc_auc')

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
