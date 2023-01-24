
from dataclasses import dataclass
from typing import Iterable
import pandas as pd
import math
import numpy as np
import sys
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin

@dataclass
class FeatureInfo:
    feature_name: str
    information_gain: float
    unique_values: list

class Node:

    def __init__(self, feature_name: str, value: any, depth_index: int = 0):
        self.feature_name = feature_name    #nazwa atrybutu w węźle
        self.value = value                  #wartość atrybutu w węźle
        self.depth_index = depth_index      #indeks głębokości
        self.children: Iterable[Node] = []  #węzły potomne
        self.class_label = None             #etykieta klasy (jak różne od None -> liść)

    #prosty opis struktury drzewa
    def __str__(self) -> str:
        return '-'*self.depth_index + f'{self.feature_name}!{self.value}!{self.class_label}' + '\n'.join([str(child) for child in self.children])

    #dodawanie węzłów potomnych
    def add_level(self, data: Iterable):
        self.children = [Node(None, d, self.depth_index + 1) for d in data]
        return self.children

    #metoda do wywołań rekurencyjnych w procesie uczenia (jedno wywołanie na podział według wartości atrybutu)
    def split_epoch(self, X: pd.DataFrame, y: np.array, y_old: np.array, rand_features: bool = False):

        #losowanie pierwiastek z n atrybutów
        if rand_features:
            X = X[np.random.choice(X.columns, size=math.floor(math.sqrt(len(X.columns))), replace=False)]

        #unikalne etykiety i ich ilość
        unique, counts = np.unique(y, return_counts=True)

        #warunki stopu
        if len(X.columns) == 0 or len(unique) == 1:
            class_index = DecisionTreeID3.get_max_counted_label_index(counts)
            self.class_label = unique[class_index]
            return
        elif len(X.index) == 0:
            unique_old, counts_old = np.unique(y_old, return_counts=True)
            class_index = DecisionTreeID3.get_max_counted_label_index(counts_old)
            self.class_label = unique_old[class_index]
            return

        #wybór najlepszego atrybutu
        best_feature = self.get_best_feature(X, y, counts)
        self.feature_name = best_feature.feature_name
        for child in self.add_level(best_feature.unique_values):
            new_indexes = X[self.feature_name] == child.value
            #wywołanie rekurencyjne z wycięciem wybranego atrybutu oraz wierszy według wartości
            child.split_epoch(X.drop(self.feature_name, axis=1).loc[new_indexes], y[list(new_indexes)], y)

    def get_best_feature(self, X: pd.DataFrame, y: np.array, unique_label_counts: Iterable) -> FeatureInfo:
        best_feature = FeatureInfo('', -sys.maxsize-1, None)
        
        #liczenie klas dla każdej wartości dla każdego atrybutu
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
            #proporcje wartości dla całego zbioru
            proportions = []
            entropies = []
            for counts_map in feature_counts.values():
                #proporcje klas dla konkretnej wartości
                value_proportions = []
                #ilość danej wartości
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

    #metoda do wywołań rekurencyjnych w procesie predykcji
    def predict_cascade(self, row: pd.Series) -> any:
        if self.class_label is not None:
            return self.class_label
        value_closeness = {}
        for child in self.children:
            if row[self.feature_name] == child.value:
                return child.predict_cascade(row)
            if type(child.value) in (int, float, np.int64, np.float64) and type(row[self.feature_name]) in (int, float, np.int64, np.float64):
                value_closeness[child] = abs(row[self.feature_name]-child.value)
        if len(value_closeness) != 0:
            return max(value_closeness, key=value_closeness.get).predict_cascade(row)

class DecisionTreeID3:

    def __init__(self, rand_features:bool=False):
        self.root = Node(None, None)
        self.rand_features = rand_features

    def __str__(self) -> str:
        return str(self.root)

    def fit(self, X: pd.DataFrame, y: np.array):
        self.root.split_epoch(X, y, y, self.rand_features)
        return self

    #predykcja dla pojedynczego wiersza
    def predict(self, row: pd.Series) -> any:
        return self.root.predict_cascade(row)

    #predykcja dla całego zbioru
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

class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, n_estimators: int = 100, rand_features: bool = True, combine_SVM_prop: float = 0):
        self.n_estimators: int = n_estimators
        self.estimators: list[DecisionTreeID3] = []
        self.rand_features: bool = rand_features
        self.combine_SVM_prop: int = math.floor(n_estimators*combine_SVM_prop)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            if parameter == 'combine_SVM_prop':
                value = math.floor(self.n_estimators*value)
            setattr(self, parameter, value)
        return self

    def fit(self, X: pd.DataFrame, y: np.array):
        n = len(X.index)
        #przełącznik jeżeli używamy klasyfikatora SVM
        switch = self.combine_SVM_prop
        for _ in range(self.n_estimators):
            #bootstrapowe losowanie ze zwracaniem
            bootstrap_samples = np.random.randint(n, size=n)
            if switch:
                new_estimator = SVC().fit(X.iloc[bootstrap_samples], y[bootstrap_samples])
                switch -= 1
            else:
                new_estimator = DecisionTreeID3(self.rand_features).fit(X.iloc[bootstrap_samples], y[bootstrap_samples])
            self.estimators.append(new_estimator)
        return self

    def predict(self, X: pd.DataFrame) -> np.array:
        results = [None]*len(X.index)
        #iteracja po każdym wierszu
        for index, row_index in enumerate(X.index):
            row = X.loc[row_index]
            counter = {}
            max_votes = ('', 0)
            #predykcja dla każdego estymatora i głosowanie
            for estimator in self.estimators:
                if isinstance(estimator, SVC):
                    row_array = np.array(row).reshape(1,-1)
                    prediction = estimator.predict(row_array)
                    prediction = prediction[0]
                else:
                    prediction = estimator.predict(row)
                if prediction in counter:
                    counter[prediction] += 1
                else:
                    counter[prediction] = 1
                if counter[prediction] > max_votes[1]:
                    max_votes = (prediction, counter[prediction])
            results[index] = max_votes[0]
        return np.array(results)