import numpy as np
from collections import Counter
def euclidean_distance(x1,x2):
    #euclidean distance
    # Get the square of the difference of the 2 vectors
    square = np.square(x1 - x2)
    # Get the sum of the square
    sum_square = np.sum(square)
    distance = np.sqrt(sum_square)
    return distance
class KNN:
    def __init__(self,k=3):
        self.k=k
    def fit(self, X,y):
        self.X_train=X
        self.y_train=y
    def predict(self, X):
        predictions=[self._predict(x) for x in X]
        # print([self._predict(x)for x in X])
        # print("hallo")
        return predictions
    def _predict(self, x):
        # converter = LabelEncoder()
        distances=[euclidean_distance(x,x_train) for x_train in self.X_train]
        # print(distances)
        k_indices=np.argsort(distances)[:self.k]
        print(k_indices)
        k_nearest_labels=[self.y_train[i] for i in k_indices]
        print(k_nearest_labels)
        most_common=Counter(k_nearest_labels).most_common()
        return most_common[0][0]
