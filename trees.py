import data_processing
import numpy as np
import math


class Test:
    def __init__(self, label, label_index):
        self.label = label
        self.label_index = label_index
        self.decisions = None

    @staticmethod
    def entropy(values, labels):
        vectors = list(zip(values, labels))

        def ent(o, total):
            z = total - o
            return 0 if z*o == 0 else (-z*math.log2(z/total) - o*math.log2(o/total))/total

        set_sizes = {value: len(list(filter(lambda x: x[1] == value, vectors))) for value in set(values)}
        no_ones = {value: sum(list(zip(*filter(lambda x: x[1] == value, vectors)))[1]) for value in set(values)}
        return sum([ent(no_ones[value], set_sizes[value]) * set_sizes[value]/len(vectors) for value in set(values)])

    def fit(self, values, labels):
        vectors = list(zip(values, labels))
        set_sizes = {value: len(list(filter(lambda x: x[1] == value, vectors))) for value in set(values)}
        no_ones = {value: sum(list(zip(*filter(lambda x: x[1] == value, vectors)))[1]) for value in set(values)}
        self.decisions = {value: 2*no_ones[value]>set_sizes[value] for value in set(values)}

    def predict(self, vector):
        return self.decisions[vector[self.label_index]]


class IdentificationTree:
    def __init__(self):
        pass
