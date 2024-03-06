import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.base import clone, BaseEstimator
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier

mnist = fetch_openml("mnist_784", version=1)
mnist.keys()

X, y = mnist["data"], mnist["target"]

X.shape
y.shape

some_digit = X.iloc[0]
some_digit_image = some_digit.values.reshape(28, 28)

plt.imshow(some_digit_image, cmap="binary")

y[0]
y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


y_train_5 = y_train == 5
y_test_5 = y_test == 5

sdg_clf = SGDClassifier(random_state=42)
sdg_clf.fit(X_train, y_train_5)

sdg_clf.predict([some_digit])

skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sdg_clf)
    X_train_folds = X_train.iloc[train_index]  # use iloc to index rows
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train.iloc[test_index]  # use iloc to index rows
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))

cross_val_score(sdg_clf, X_train, y_train_5, cv=3, scoring="accuracy")


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")

y_train_pred = cross_val_predict(sdg_clf, X_train, y_train_5, cv=3)

confusion_matrix(y_train_5, y_train_pred)

y_train_perfect_predictions = y_train_5
confusion_matrix(y_train_5, y_train_perfect_predictions)


precision_score(y_train_5, y_train_pred)

recall_score(y_train_5, y_train_pred)
f1_score(y_train_5, y_train_pred)

y_scores = sdg_clf.decision_function([some_digit])
threshold = 0

y_some_digit_pred = y_scores > threshold

threshold = 8000
y_some_digit_pred = y_scores > threshold

y_scores = cross_val_predict(
    sdg_clf, X_train, y_train_5, cv=3, method="decision_function"
)

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(
    precisions, recalls, thresholds, highlight_threshold=None
):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    # Add a vertical line at the specified threshold value
    if highlight_threshold is not None:
        plt.axvline(
            highlight_threshold,
            color="r",
            linestyle="--",
            label="Highlighted threshold",
        )

    # Add legend
    plt.legend(loc="lower left")

    # Add axis labels
    plt.xlabel("Threshold")
    plt.ylabel("Precision / Recall")

    # Add grid
    plt.grid(True)

    # Show the plot
    plt.show()


plot_precision_recall_vs_threshold(precisions, recalls, thresholds, 20000)

threshold_90_precision = thresholds[np.argmax(precisions >= 0.9)]


y_train_pred_90 = y_scores >= threshold_90_precision

precision_score(y_train_5, y_train_pred_90)
recall_score(y_train_5, y_train_pred_90)

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--")


plot_roc_curve(fpr, tpr)

roc_auc_score(y_train_5, y_scores)

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(
    forest_clf, X_train, y_train_5, cv=3, method="predict_proba"
)

y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")

roc_auc_score(y_train_5, y_scores_forest)
