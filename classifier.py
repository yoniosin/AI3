from hw3_utils import *
from collections import Counter, namedtuple
import numpy as np
import heapq
import random
import pickle
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from functools import reduce
import csv


def euclidain_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def split_crosscheck_groups(data_set, num_folds):
    X = data_set[0]
    y = np.array(data_set[1])
    true_samples = list(np.where(y)[0])
    true_samples_num = len(true_samples)

    false_samples = list(np.where(~y)[0])
    false_samples_num = len(false_samples)

    true_per_fold = int(true_samples_num / num_folds)
    false_per_fold = int(false_samples_num / num_folds)

    indexes_per_fold = [[] for _ in range(num_folds)]

    for i in range(num_folds):
        for t_i in range(true_per_fold):
            indexes_per_fold[i].append(true_samples.pop(random.randrange(len(true_samples))))

        for t_i in range(false_per_fold):
            indexes_per_fold[i].append(false_samples.pop(random.randrange(len(false_samples))))

    for i in range(num_folds):
        indexes = indexes_per_fold[i]
        fold = (X[indexes], list(y[indexes]))

        with open('ecg_fold' + str(i) + '.data', 'wb') as f:
            pickle.dump(fold, f)


def load_k_fold_data(i):
    with open('ecg_fold' + str(i) + '.data', 'rb') as f:
        return pickle.load(f)


Sample = namedtuple('Sample', ['features', 'label'])


class knn_classifier(abstract_classifier):
    def __init__(self, k, data, labels):
        self.k = k
        self.data = data
        self.labels = labels

    def classify(self, features):
        dist_heap = [Sample(euclidain_distance(sample, features), label) for sample, label in
                     zip(self.data, self.labels)]
        heapq.heapify(dist_heap)
        nearest = [heapq.heappop(dist_heap).label for _ in range(self.k)]
        label_count = Counter(nearest)
        return max(label_count, key=lambda m: label_count[m])


class knn_factory(abstract_classifier_factory):
    def __init__(self, k):
        self.k = k

    def train(self, data, labels):
        return knn_classifier(self.k, data, labels)


def apply_PCA(data_set):
    Y = data_set[1]
    X = data_set[0]
    X_scale = (X - X.mean(0)) / (np.var(X, axis=0) ** 0.5)
    data_scale = sklearn.preprocessing.scale(data_set[0])

    pca = PCA(n_components=3)
    pca.fit(data_scale)
    X = pca.transform(data_scale)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y)


def evaluate(classifier_factory: abstract_classifier_factory, k):
    train_idx = list(range(k))
    test_idx = train_idx.pop(np.random.choice(train_idx))
    train_data = np.empty([0])
    train_labels = []

    train_folds = [load_k_fold_data(idx) for idx in train_idx]
    train_data, train_labels = reduce(lambda a, b: (np.concatenate(a[0], b[0]), np.concatenate(a[1], b[1])),
                                      train_folds)

    classifier: abstract_classifier = classifier_factory.train(train_data, train_labels)

    test_data, test_labels = load_k_fold_data(test_idx)
    N = len(test_labels)
    accuracy = 0
    error = 0
    for sample, label in zip(test_data, test_labels):
        if classifier.classify(sample) == label:
            accuracy += 1
        else:
            error += 1

    return accuracy / N, error / N


def compare_k(k_vec):
    _res = {}
    for k_val in k_vec:
        _res[k_val] = evaluate(knn_factory(k_val), 2)

    return _res


def print_csv(res_dict):
    csv = open('experiment6.csv', "w")
    for key in res_dict.keys():
        row = str(key) + "," + str(res_dict[key][0]) + "," + str(res_dict[key][1]) + '\n'

        csv.write(row)


if __name__ == '__main__':
    # data_set = load_data()

    # split_crosscheck_groups(data_set, 2)
    # load_k_fold_data(1)

    res = compare_k([1, 3, 5, 7, 13])
    print_csv(res)

    print('a')
