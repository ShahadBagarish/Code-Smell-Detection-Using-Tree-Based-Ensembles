import pandas as pd
import matplotlib.pyplot as plt
from sklearn.experimental import enable_hist_gradient_boosting
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score

# Define dataset filenames
dataset_files = [
    'Dataset/DataClass_Clean_GainRatio.csv',
    'Dataset/LongMethod_Clean_GainRatio.csv',
    'Dataset/FeatureEnvy_Clean_GainRatio.csv',
    'Dataset/LongParameterList_Clean_GainRatio.csv',
    'Dataset/GodClass_Clean_GainRatio.csv',
    'Dataset/SwitchStatements_Clean_GainRatio.csv'
]

# Create an empty list to store accuracy scores for each dataset
cv_accuracy_scores_all = []

catboost_model = CatBoostClassifier()

# Iterate over each dataset
for file in dataset_files:
    # Load the dataset
    data = pd.read_csv(file)
    
    # Assuming the last column is the target variable and the rest are features
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target variable
    
    # Perform 10-fold cross-validation for accuracy
    cv_accuracy_scores = cross_val_score(catboost_model, X, y, cv=10, scoring='accuracy')
    
    # Append the accuracy scores to the list
    cv_accuracy_scores_all.append(cv_accuracy_scores)

# Define classifier names for each dataset
classifier_names = ['DataClass', 'LongMethod', 'FeatureEnvy', 'LongParameterList', 'GodClass', 'SwitchStatements']

# Define colors for the boxes
box_colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightskyblue', 'lightpink', 'lightgrey']

# Plotting the boxplot for cross-validation accuracy scores
plt.figure(figsize=(12, 8))
box = plt.boxplot(cv_accuracy_scores_all, labels=classifier_names, vert=False, patch_artist=True)

# Customize box colors
for patch, color in zip(box['boxes'], box_colors):
    patch.set_facecolor(color)

plt.xlabel('CAT Accuracy', fontsize=10)
plt.ylabel('Dataset', fontsize=10) 
plt.grid(True)

# Set y-axis tick parameters
plt.tick_params(axis='y', which='major', labelsize=8) 
plt.tick_params(axis='x', labelsize=8) 
plt.show()
