
from dataclasses import dataclass
from typing import Iterable, Literal
import pandas as pd
import math
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import sys
from sklearn.svm import SVC

@dataclass
class FeatureInfo:
    feature_name: str
    information_gain: float
    unique_values: list

class Node:

    def __init__(self, feature_name: str, value: any, depth_index: int = 0):
        self.feature_name = feature_name
        self.value = value
        self.depth_index = depth_index
        self.children: Iterable[Node] = []
        self.class_label = None

    def __str__(self) -> str:
        return '-'*self.depth_index + f'{self.feature_name}!{self.value}!{self.class_label}' + '\n'.join([str(child) for child in self.children])

    def add_level(self, data: Iterable):
        self.children = [Node(None, d, self.depth_index + 1) for d in data]
        return self.children

    def split_epoch(self, X: pd.DataFrame, y: np.array, y_old: np.array, rand_features:bool=False):
        if rand_features:
            X = X[np.random.choice(X.columns, size=math.floor(math.sqrt(len(X.columns))), replace=False)]
        unique, counts = np.unique(y, return_counts=True)
        if len(X.columns) == 0 or len(unique) == 1:
            class_index = DecisionTreeID3.get_max_counted_label_index(counts)
            self.class_label = unique[class_index]
            return
        elif len(X.index) == 0:
            unique_old, counts_old = np.unique(y_old, return_counts=True)
            class_index = DecisionTreeID3.get_max_counted_label_index(counts_old)
            self.class_label = unique_old[class_index]
            return

        best_feature = self.get_best_feature(X, y, counts)
        self.feature_name = best_feature.feature_name
        for child in self.add_level(best_feature.unique_values):
            new_indexes = X[self.feature_name] == child.value
            child.split_epoch(X.drop(self.feature_name, axis=1).loc[new_indexes], y[list(new_indexes)], y)

    def get_best_feature(self, X: pd.DataFrame, y: np.array, unique_label_counts: Iterable) -> FeatureInfo:
        best_feature = FeatureInfo('', -sys.maxsize-1, None)
        
        for feature_name in X.columns:
            feature_counts = {}
            for index, value in enumerate(X[feature_name]):
                class_name = y[index]
                if value not in feature_counts.keys():
                    feature_counts[value] = {}
                if class_name not in feature_counts[value].keys():
                    feature_counts[value][class_name] = 0
                feature_counts[value][class_name] += 1
            
            count = len(y)
            proportions = [] #proporcje wartości do całego zbioru
            entropies = []
            for counts_map in feature_counts.values():
                value_proportions = [] #proporcje klas dla konkretnej wartości
                value_count = np.sum(list(counts_map.values()))
                for value in counts_map.values():
                    value_proportions.append(value/value_count)
                entropies.append(DecisionTreeID3.entropy(value_proportions))
                proportions.append(value_count/count)
            whole_entropy = DecisionTreeID3.entropy([c/count for c in unique_label_counts])
            information_gain = DecisionTreeID3.calculate_information_gain(whole_entropy, entropies, proportions)
            if best_feature.information_gain < information_gain:
                best_feature.information_gain = information_gain
                best_feature.feature_name = feature_name
                best_feature.unique_values = list(feature_counts.keys())

        return best_feature


    def predict_cascade(self, row: pd.Series) -> any:
        if self.class_label is not None:
            return self.class_label
        value_closeness = {}
        for child in self.children:
            if row[self.feature_name] == child.value:
                return child.predict_cascade(row)
            if type(child.value) in (int, float, np.int64, np.float64) and type(row[self.feature_name]) in (int, float,np.int64, np.float64):
                value_closeness[child] = abs(row[self.feature_name]-child.value)
        if len(value_closeness) != 0:
            return max(value_closeness, key=value_closeness.get).predict_cascade(row)

class DecisionTreeID3:

    def __init__(self, rand_features:bool=False):
        self.root = Node(None, None)
        self.depth = 0
        self.rand_features = rand_features

    def __str__(self) -> str:
        return str(self.root)

    def fit(self, X: pd.DataFrame, y: np.array):
        self.root.split_epoch(X, y, y, self.rand_features)
        return self

    def predict(self, row: pd.Series) -> any:
        return self.root.predict_cascade(row)

    def predict_all(self, X: pd.DataFrame) -> np.array:
        predict_result = []
        for row in X.index:
            predict_result.append(self.root.predict_cascade(X.loc[row]))
        return np.array(predict_result)
         
    @staticmethod
    def entropy(proportions: list):
        return -np.sum([p*math.log2(p) for p in proportions if p != 0])

    @staticmethod
    def calculate_information_gain(whole_entropy: float, value_entropies: list, value_proportions: list):
        return whole_entropy-np.sum([ve*vp for ve, vp in zip(value_entropies, value_proportions)])

    @staticmethod
    def get_max_counted_label_index(counts: np.array) -> int:
        class_index = 0
        max_count = 0
        for index, c in enumerate(counts):
            if c > max_count:
                max_count = c
                class_index = index
        return class_index

class RandomForestClassifier:
    
    def __init__(self, n_estimators:int=100, rand_features:bool=True, combine_SVM:bool=True):
        self.n_estimators: int = n_estimators
        self.estimators: list[DecisionTreeID3] = []
        self.rand_features = rand_features
        self.combine_SVM = combine_SVM

    def fit(self, X: pd.DataFrame, y: np.array):
        n = len(X.index)
        switch = False
        for _ in range(self.n_estimators):
            bootstrap_samples = np.random.randint(n, size=n)
            if self.combine_SVM and switch:
                new_estimator = SVC().fit(X.iloc[bootstrap_samples], y[bootstrap_samples])
            else:
                new_estimator = DecisionTreeID3(self.rand_features).fit(X.iloc[bootstrap_samples], y[bootstrap_samples])
            self.estimators.append(new_estimator)
        return self

    def predict(self, X: pd.DataFrame) -> np.array:
        results = [None]*len(X.index)
        for index, row_index in enumerate(X.index):
            row = X.loc[row_index]
            counter = {}
            max_votes = ('', 0)
            for estimator in self.estimators:
                prediction = estimator.predict(row)
                if prediction in counter:
                    counter[prediction] += 1
                else:
                    counter[prediction] = 1
                if counter[prediction] > max_votes[1]:
                    max_votes = (prediction, counter[prediction])
            results[index] = max_votes[0]
        return np.array(results)