import matplotlib.pyplot as plt
import numpy as np
import data_processing


class KNN:
    def __init__(self, k):
        self.k = k
        self.data = None
        self.labels = None

    def fit(self, data, labels):
        self.data = data
        self.labels = list(map(lambda x: x[0], labels))

    def predict(self, v):
        norms = []
        for n, vi in enumerate(self.data):
            norm = sum((vi[i] - v[i])**2 for i in range(len(vi)))
            norms.append([self.labels[n], norm])

        norms = sorted(norms, key=lambda x: x[1])[:self.k]
        labels = list(map(lambda x: x[0], norms))
        return max(set(labels), key=labels.count)


if __name__ == '__main__':
    data = data_processing.data_generator(10, 3, 2, centers=((0, 0), (5, 5), (-5, 5)))
    data = list(map(lambda x: x.tolist(), data))
    knn_classifier = KNN(k=6)
    knn_classifier.fit(*data)
    print(knn_classifier.predict((0, 0)))
