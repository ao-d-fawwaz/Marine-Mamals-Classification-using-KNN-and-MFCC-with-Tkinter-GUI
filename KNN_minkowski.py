from re import X
import numpy as np
from collections import Counter

import scipy.spatial.distance
from scipy.spatial.distance import cityblock
from scipy.spatial import distance
from sklearn.metrics.pairwise import paired_manhattan_distances


def Minkowski_distance(x1,x2, p=4):

    return distance.minkowski(x1,x2,p)

class KNN_minkowski:

    def __init__(self,k=3):
        self.k=k
    def fit(self, X,y):
        self.X_train=X
        self.y_train=y
    def predict(self, X):
        predictions=[self._predict(x) for x in X]
        return predictions
    def _predict(self, x):
        distances=[Minkowski_distance(x,x_train) for x_train in self.X_train]
        k_indices=np.argsort(distances)[:self.k]
        print(k_indices)
        k_nearest_labels=[self.y_train[i] for i in k_indices]
        print(k_nearest_labels)
        most_common=Counter(k_nearest_labels).most_common()
        return most_common[0][0]
