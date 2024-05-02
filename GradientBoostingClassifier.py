import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score

# Load the dataset from CSV
# #1 
# data = pd.read_csv('Dataset/DataClass_Clean_GainRatio.csv')
# #2 
# data = pd.read_csv('Dataset/LongMethod_Clean_GainRatio.csv')
# #3 
# data = pd.read_csv('Dataset/FeatureEnvy_Clean_GainRatio.csv')
# #4 
# data = pd.read_csv('Dataset/LongParameterList_Clean_GainRatio.csv')
# #5 
# data = pd.read_csv('Dataset/GodClass_Clean_GainRatio.csv')
# #6 
data = pd.read_csv('Dataset/SwitchStatements_Clean_GainRatio.csv')

# Assuming the last column is the target variable and the rest are features
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target variable

# Create GradientBoostingClassifier model
gradient_boosting_model = GradientBoostingClassifier()

# Perform 10-fold cross-validation for accuracy
cv_accuracy_scores = cross_val_score(gradient_boosting_model, X, y, cv=10, scoring='accuracy')

# Perform 10-fold cross-validation for AUC
cv_auc_scores = cross_val_score(gradient_boosting_model, X, y, cv=10, scoring='roc_auc')

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

from scipy.stats import shapiro
# Perform 10-fold cross-validation and collect predictions
cv_predictions = cross_val_predict(gradient_boosting_model, X, y, cv=10)

# Perform Shapiro-Wilk test on the predictions
stat, p= shapiro(cv_predictions)

# Interpret the result
alpha = 0.05
is_normal = p > alpha
if p > alpha:
    print("normal")
else:
    print("Not normal")