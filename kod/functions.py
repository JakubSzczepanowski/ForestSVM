import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from typing import Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f_one = f1_score(y_true, y_pred, average="macro")

    print(f"Accuracy: {acc}\n Macro precision: {prec}\n Macro recall: {recall} \n Macro f1-score: {f_one}")

    cf_matrix = confusion_matrix(y_true, y_pred)

    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
      
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)

    plt.figure(figsize=(6,6))
    sns.heatmap(cf_matrix, annot=labels, fmt="")
    plt.title("Confusion matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()


def k_fold(X: pd.DataFrame, y: np.array, model: Any, k: int = 3, k_fold_func: callable = KFold, random_state: int = 42):

    kf = k_fold_func(n_splits = k, shuffle = True, random_state = random_state)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train) #Training the model
        y_pred = model.predict(X_test)

        print(f"Results for Fold {fold + 1}")
        metrics(y_pred=y_pred, y_true=y_test)
