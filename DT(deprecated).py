from sklearn import tree
from sklearn.datasets import load_svmlight_file
import pandas as pd

 # Load a CSV dataset using Pandas
DataClass = pd.read_csv('Dataset/DataClass_Clean_GainRatio.csv', index_col="NOAM_type")
print (DataClass.head())

# #  # Convert the DataFrame to NumPy arrays 
# # X = df.drop('target_column', axis=1).values
# #  # Features 
# # y = df['target_column'].values 
# # # Target variable
                                                  
X, y = load_svmlight_file('Dataset/DataClass_Clean_GainRatio.csv',
    n_samples=2_000, n_features=10, n_classes=3, random_state=1
)
# X, y = make_gaussian_quantiles( n_samples=2_000, n_features=10, n_classes=3, random_state=1)


# We split the dataset into 2 sets: 70 percent of the samples are used for
# training and the remaining 30 percent for testing.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=42
)

# %%
# Training the `AdaBoostClassifier`

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

weak_learner = DecisionTreeClassifier(max_leaf_nodes=8)
n_estimators = 300

adaboost_clf = AdaBoostClassifier(
    estimator=weak_learner,
    n_estimators=n_estimators,
    algorithm="SAMME",
    random_state=42,
).fit(X_train, y_train)

# %%
# Analysis
# --------
# Convergence of the `AdaBoostClassifier`

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

dummy_clf = DummyClassifier()


def misclassification_error(y_true, y_pred):
    return 1 - accuracy_score(y_true, y_pred)


weak_learners_misclassification_error = misclassification_error(
    y_test, weak_learner.fit(X_train, y_train).predict(X_test)
)

dummy_classifiers_misclassification_error = misclassification_error(
    y_test, dummy_clf.fit(X_train, y_train).predict(X_test)
)

print(
    "DecisionTreeClassifier's misclassification_error: "
    f"{weak_learners_misclassification_error:.3f}"
)
print(
    "DummyClassifier's misclassification_error: "
    f"{dummy_classifiers_misclassification_error:.3f}"
)

# Now, we calculate the `misclassification_error`, i.e. `1 - accuracy`, of the
# additive model (:class:`~sklearn.tree.DecisionTreeClassifier`) at each
# boosting iteration on the test set to assess its performance.

import matplotlib.pyplot as plt
import pandas as pd

boosting_errors = pd.DataFrame(
    {
        "Number of trees": range(1, n_estimators + 1),
        "AdaBoost": [
            misclassification_error(y_test, y_pred)
            for y_pred in adaboost_clf.staged_predict(X_test)
        ],
    }
).set_index("Number of trees")
ax = boosting_errors.plot()
ax.set_ylabel("Misclassification error on test set")
ax.set_title("Convergence of AdaBoost algorithm")

plt.plot(
    [boosting_errors.index.min(), boosting_errors.index.max()],
    [weak_learners_misclassification_error, weak_learners_misclassification_error],
    color="tab:orange",
    linestyle="dashed",
)
plt.plot(
    [boosting_errors.index.min(), boosting_errors.index.max()],
    [
        dummy_classifiers_misclassification_error,
        dummy_classifiers_misclassification_error,
    ],
    color="c",
    linestyle="dotted",
)
plt.legend(["AdaBoost", "DecisionTreeClassifier", "DummyClassifier"], loc=1)
plt.show()

# %%
# The plot shows the missclassification error on the test set after each
# boosting iteration. We see that the error of the boosted trees converges to an
# error of around 0.3 after 50 iterations, indicating a significantly higher
# accuracy compared to a single tree, as illustrated by the dashed line in the
# plot.

weak_learners_info = pd.DataFrame(
    {
        "Number of trees": range(1, n_estimators + 1),
        "Errors": adaboost_clf.estimator_errors_,
        "Weights": adaboost_clf.estimator_weights_,
    }
).set_index("Number of trees")

axs = weak_learners_info.plot(
    subplots=True, layout=(1, 2), figsize=(10, 4), legend=False, color="tab:blue"
)
axs[0, 0].set_ylabel("Train error")
axs[0, 0].set_title("Weak learner's training error")
axs[0, 1].set_ylabel("Weight")
axs[0, 1].set_title("Weak learner's weight")
fig = axs[0, 0].get_figure()
fig.suptitle("Weak learner's errors and weights for the AdaBoostClassifier")
fig.tight_layout()



