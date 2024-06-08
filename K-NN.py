from math import sqrt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def euclidean_distance(p1, p2):
    sum = 0
    for i in range(len(p1)):
        sum += (p1[i] - p2[i]) ** 2
    return sqrt(sum)

def knn_custom(x_train, x_test, y_train, y_test, k=5):
    distances = []
    y_pred = []
    for xtest in x_test:
        # Calculate the distance between the test instance and all training instances
        for i in range(len(x_train)):
            distances.append((i, euclidean_distance(xtest, x_train[i])))
        # Sort by distance
        distances.sort(key=lambda tup: tup[1])
        # Get the k nearest neighbors
        neighbours = [y_train[distances[i][0]] for i in range(k)]
        # Predict the class with the most votes
        y_pred.append(max(set(neighbours), key=neighbours.count))
        # Clear distances for the next test instance
        distances = []
    # Calculate the accuracy
    c = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            c += 1
    print('accuracy = ' + str(c * 100 / len(y_test)))

if __name__ == '__main__':
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.20)
    knn_custom(x_train, x_test, y_train, y_test)
