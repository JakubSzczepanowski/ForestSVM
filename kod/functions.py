import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import metrics as mtr
from typing import Any, Iterable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from forestsvm import DecisionTreeID3
from sklearn.preprocessing import MinMaxScaler


def plot_confusion_matrix(y_true, y_pred) -> None:
    cf_matrix = confusion_matrix(y_true, y_pred)

    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
      
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)

    plt.figure(figsize=(5,5))
    sns.heatmap(cf_matrix, annot=labels, fmt="")
    plt.title("Confusion matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()


def plot_roc_curve(y_true, y_pred) -> None:
    fp_rate, tp_rate, _ = mtr.roc_curve(y_true, y_pred)
    auc = mtr.roc_auc_score(y_true, y_pred)

    #create ROC curve
    plt.figure(figsize=(5, 5))
    plt.plot(fp_rate, tp_rate, label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()


def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f_one = f1_score(y_true, y_pred, average="macro")

    return {"accuracy": acc, "precision": prec, "recall": recall, "f1-score": f_one}


def metrics(y_true, y_pred) -> dict:
    
    metrics = get_metrics(y_true, y_pred)

    print(f"Accuracy: {metrics['accuracy']}\n Macro precision: {metrics['precision']}\n Macro recall: {metrics['recall']} \n Macro f1-score: {metrics['f1-score']}")

    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_pred)

    return metrics

def get_scores(arr):
    mean = arr.mean()
    std = arr.std()
    min = arr.min()
    max = arr.max()

    return (mean, std, min, max)


def k_fold(X: pd.DataFrame, y: np.array, model: Any, k: int = 3, k_fold_func: callable = KFold, random_state: int = 42):
    kf = k_fold_func(n_splits = k, shuffle = True, random_state = random_state)

    fold_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []}

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, np.array(y_train)) #Training the model

        if isinstance(model, DecisionTreeID3):
            y_pred = model.predict_all(X_test)
        else:
            y_pred = model.predict(X_test)

        print(f"Results for Fold {fold + 1}")
        metrics_scores = metrics(y_true=y_test, y_pred=y_pred)

        fold_scores['accuracy'].append(metrics_scores['accuracy'])
        fold_scores['f1-score'].append(metrics_scores['f1-score'])
        fold_scores['precision'].append(metrics_scores['precision'])
        fold_scores['recall'].append(metrics_scores['recall'])

    scores_dict = {}

    for metric, values in fold_scores.items():
        scores = get_scores(np.array(values))
        scores_dict[metric] = scores
      
    results_df = pd.DataFrame(scores_dict)  
    results_df.insert(0, 'info', ['mean', 'std', 'min', 'max'])

    print(f"\n Results for all {k} folds:")  
    print(results_df)

def create_numerical_categories(df: pd.DataFrame, omit_columns: Iterable[str], ranges: int, normalize: bool = False):
    col = [c for c in df.columns if c not in omit_columns]
    if normalize:
        scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[col])
    
    interval = 1/ranges
    bins = [b*interval for b in range(ranges+1)]
    for c in col:
        df[c] = pd.cut(df[c], bins, labels=list(range(1, ranges+1)))
        df[c] = df[c].fillna(1)
