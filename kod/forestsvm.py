
from dataclasses import dataclass
from typing import Iterable, Literal
import pandas as pd
import math
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

class Node:

    def __init__(self, feature_name: str, value: any, depth_index: int = 0):
        self.feature_name = feature_name
        self.value = value
        self.depth_index = depth_index
        self.children: Iterable[Node] = []
        self.class_label = None

    def __str__(self) -> str:
        return '-'*self.depth_index + f'{self.feature_name}!{self.value}!{self.class_label}' + '\n'.join([str(child) for child in self.children])

    def parent_equal(self, other):
        if self is other:
            return self
        for child in self.children:
            child.parent_equal(other)

    def add_level(self, data: Iterable):
        self.children = [Node(None, d, self.depth_index + 1) for d in data]
        return self.children

    def split_epoch(self, X: pd.DataFrame, y: np.array):
        
        unique, counts = np.unique(y, return_counts=True)
        if len(X.columns) == 0 or len(unique) == 1:
            class_index = 0
            max_count = 0
            for index, c in enumerate(counts):
                if c > max_count:
                    max_count = c
                    class_index = index
            self.class_label = unique[class_index]
            return

        labels_length = len(y)
        positive_labels_prop = counts[1]/labels_length
        negative_labels_prop = counts[0]/labels_length
        whole_entropy = DecisionTreeID3.entropy(positive_labels_prop, negative_labels_prop)
        feature_unique_values: dict[str, np.array] = {}
        num_of_rows = len(X.index)
        best_feature = FeatureInfo('', 0)
        
        for feature_name in X.columns:
            positive_pairs: dict[str, int] = {}
            information_gain = whole_entropy
            for index, value in enumerate(X[feature_name]):
                if value not in positive_pairs.keys():
                    positive_pairs[value] = 0
                if y[index] == 1:
                    positive_pairs[value] += 1
            unique, counts = np.unique(X[feature_name], return_counts=True)
            for index, value in enumerate(unique):
                positive_prop = positive_pairs[value]/counts[index]
                entropy = DecisionTreeID3.entropy(positive_prop, 1-positive_prop)
                information_gain -= counts[index]/num_of_rows*entropy
            feature_unique_values[feature_name] = unique
            if best_feature.information_gain < information_gain:
                best_feature = FeatureInfo(feature_name, information_gain)
        self.feature_name = best_feature.feature_name
        for child in self.add_level(feature_unique_values[self.feature_name]):
            new_indexes = X[self.feature_name] == child.value
            child.split_epoch(X.drop(self.feature_name, axis=1).loc[new_indexes], y[new_indexes])

    def predict_cascade(self, row: pd.Series) -> any:
        if self.class_label is not None:
            return self.class_label
        for child in self.children:
            if row[self.feature_name] == child.value:
                return child.predict_cascade(row)

@dataclass
class FeatureInfo:
    feature_name: str
    information_gain: float

class DecisionTreeID3:

    def __init__(self):
        self.root = Node(None, None)
        self.depth = 0

    def __str__(self) -> str:
        return str(self.root)

    def fit(self, X: pd.DataFrame, y: np.array):
        self.root.split_epoch(X, y)

    def predict(self, X: pd.DataFrame) -> np.array:
        predict_result = []
        for row in range(len(X.index)):
            predict_result.append(self.root.predict_cascade(X.loc[row]))
        return np.array(predict_result)
         
    @staticmethod
    def entropy(positive_proportion: float, negative_proportion: float) -> float:
        return (-positive_proportion*math.log2(positive_proportion) if positive_proportion != 0 else 0) -(negative_proportion*math.log2(negative_proportion) if negative_proportion != 0 else 0)  

df = pd.DataFrame({'Opady': ['brak', 'mżawka', 'burza', 'burza', 'brak', 'brak'], 'Temperatura': ['ciepło', 'ciepło', 'ciepło', 'zimno', 'zimno', 'zimno'], 'Mgła': ['brak', 'lekka', 'brak', 'lekka', 'duża', 'brak'], 'Stan pogody': ['dobra', 'dobra', 'zła', 'zła', 'zła', 'dobra']})
X = df.drop('Stan pogody', axis=1)
y = df['Stan pogody'].copy()
enc = OrdinalEncoder(categories=[['zła', 'dobra']], dtype=np.int8)
y = enc.fit_transform(y.values.reshape(-1,1)).flatten()

tree = DecisionTreeID3()
tree.fit(X, y)
print(tree)

tree.predict(pd.DataFrame({'Opady': ['burza', 'brak', ], 'Temperatura': ['zimno', 'ciepło'], 'Mgła': ['duża', 'brak']}))