
from dataclasses import dataclass
from typing import Iterable, Literal
import pandas as pd
import math
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

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

    def split_epoch(self, X: pd.DataFrame, y: np.array, rand_features:bool=False):
        if rand_features:
            X = X[np.random.choice(X.columns, size=math.floor(math.sqrt(len(X.columns))), replace=False)]
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
        
        best_feature = self.get_best_feature(X, y, counts)
        self.feature_name = best_feature.feature_name
        for child in self.add_level(best_feature.unique_values):
            new_indexes = X[self.feature_name] == child.value
            child.split_epoch(X.drop(self.feature_name, axis=1).loc[new_indexes], y[list(new_indexes)])

    def get_best_feature(self, X: pd.DataFrame, y: np.array, unique_label_counts: Iterable) -> FeatureInfo:
        best_feature = FeatureInfo('', 0, None)
        
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
            if best_feature.information_gain <= information_gain:
                best_feature.information_gain = information_gain
                best_feature.feature_name = feature_name
                best_feature.unique_values = list(feature_counts.keys())
        
        return best_feature


    def predict_cascade(self, row: pd.Series) -> any:
        if self.class_label is not None:
            return self.class_label
        for child in self.children:
            if row[self.feature_name] == child.value:
                return child.predict_cascade(row)

class DecisionTreeID3:

    def __init__(self, rand_features:bool=False):
        self.root = Node(None, None)
        self.depth = 0
        self.rand_features = rand_features

    def __str__(self) -> str:
        return str(self.root)

    def fit(self, X: pd.DataFrame, y: np.array):
        self.root.split_epoch(X, y, self.rand_features)
        return self

    def predict(self, X: pd.DataFrame) -> np.array:
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

class RandomForestClassifier:
    
    def __init__(self, n_estimators:int=100, rand_features:bool=True):
        self.n_estimators: int = n_estimators
        self.estimators: list[DecisionTreeID3] = []
        self.rand_features = rand_features

    def fit(self, X: pd.DataFrame, y: np.array):
        n = len(X.index)
        for _ in range(self.n_estimators):
            bootstrap_samples = np.random.randint(n, size=n)
            new_estimator = DecisionTreeID3(self.rand_features).fit(X.iloc[bootstrap_samples], y[bootstrap_samples])
            self.estimators.append(new_estimator)
        return self

    def predict(self, X: pd.DataFrame) -> np.array:
        counters = [None]*len(X.index)
        for tree in self.estimators:
            prediction = tree.predict(X)
            for index, label in enumerate(prediction):
                if counters[index] is None:
                    counters[index] = {}
                if label in counters[index]:
                    counters[index][label] += 1
                else:
                    counters[index][label] = 1

        return np.array(list(map(lambda elem: max(elem, key=elem.get), counters)))

# df = pd.read_csv('PlayTennis.csv')
# X = df.drop('Play Tennis', axis=1)
# y = df['Play Tennis'].copy()
# tree_classifier = DecisionTreeID3()
# tree_classifier.fit(X,np.array(y))

# df = pd.DataFrame({'Opady': ['brak', 'mżawka', 'burza', 'burza', 'brak', 'brak'], 'Temperatura': ['ciepło', 'ciepło', 'ciepło', 'zimno', 'zimno', 'zimno'], 'Mgła': ['brak', 'lekka', 'brak', 'lekka', 'duża', 'brak'], 'Stan pogody': ['dobra', 'dobra', 'zła', 'zła', 'zła', 'dobra']})
# X = df.drop('Stan pogody', axis=1)
# y = df['Stan pogody'].copy()
# enc = OrdinalEncoder(categories=[['zła', 'dobra']], dtype=np.int8)
# y = enc.fit_transform(y.values.reshape(-1,1)).flatten()

# forest = RandomForestClassifier()
# forest.fit(X, y)

# forest.predict(pd.DataFrame({'Opady': ['burza', 'brak', ], 'Temperatura': ['zimno', 'ciepło'], 'Mgła': ['duża', 'brak']}))