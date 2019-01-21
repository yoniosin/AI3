from hw3_utils import *
from collections import Counter, namedtuple
import numpy as np
import heapq
import random
import pickle
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from mpl_toolkits.mplot3d import Axes3D
from functools import reduce
import csv


def euclidain_distance(vec1, vec2, weights=None):
    if weights is None:
        return np.linalg.norm(vec1 - vec2)

    return ((vec1 - vec2) ** 2) @ weights


def split_crosscheck_groups(data_set, num_folds, suffix=''):
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

        with open('ecg_fold' + str(i) + suffix + '.data', 'wb') as f:
            pickle.dump(fold, f)


def load_k_fold_data(i, suffix):
    with open('ecg_fold' + str(i) + suffix + '.data', 'rb') as f:
        return pickle.load(f)


Sample = namedtuple('Sample', ['features', 'label'])


def normalize_data(data):
    var = np.var(data, axis=0)
    return (data - np.mean(data, axis=0)) / var ** 0.5


class knn_classifier(abstract_classifier):
    def __init__(self, k, data, labels, weights):
        self.k = k
        self.normalizer = Normalizer(data)
        self.data = self.normalizer.transform(data)
        self.labels = labels
        self.weights = weights

    def classify(self, features):
        norm_features = self.normalizer.transform(features)
        dist_heap = [Sample(euclidain_distance(sample, norm_features, self.weights), label) for sample, label in
                     zip(self.data, self.labels)]
        heapq.heapify(dist_heap)
        nearest = [heapq.heappop(dist_heap).label for _ in range(self.k)]
        label_count = Counter(nearest)
        return max(label_count, key=lambda m: label_count[m])


class knn_factory(abstract_classifier_factory):
    def __init__(self, k):
        self.k = k

    def train(self, data, labels, weights=None):
        return knn_classifier(self.k, data, labels, weights)


class tree_factory(abstract_classifier_factory):
    def __init__(self, criterion):
        self.criterion = criterion

    def train(self, data, labels):
        return tree_classifier(self.criterion, data, labels)


class tree_classifier(abstract_classifier):
    def __init__(self, criterion, data, labels):
        self.tree = DecisionTreeClassifier(criterion=criterion)
        self.tree.fit(X=data, y=labels)

    def classify(self, features):
        return self.tree.predict(X=features)


class perceptron_factory(abstract_classifier_factory):
    def train(self, data, labels):
        return perceptron_classifier(data, labels)


class perceptron_classifier(abstract_classifier):
    def __init__(self, data, labels):
        self.normalizer = Normalizer(data)
        self.perceptron = Perceptron()
        self.perceptron.fit(X=self.normalizer.transform(data), y=labels)

    def classify(self, features):
        return self.perceptron.predict(self.normalizer.transform(features))


def apply_PCA(data_set, n_components):
    X = data_set[0]
    Y = data_set[1]

    normalizer = Normalizer(X)
    X_scale = normalizer.transform(X)

    pca = PCA(n_components=n_components)
    pca.fit(X_scale)
    projected_X = pca.transform(X_scale)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y)

    return projected_X, Y


def evaluate(classifier_factory: abstract_classifier_factory, k, suffix=''):
    test_folds = [load_k_fold_data(idx, suffix) for idx in range(k)]
    N = 0
    accuracy = 0
    error = 0

    for fold in range(k):
        train_data, train_labels = reduce(lambda a, b: (np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]])),
                                          test_folds[1:])
        classifier: abstract_classifier = classifier_factory.train(train_data, train_labels)
        val_data, val_labels = test_folds[0]
        evaluate_tree(classifier.tree, val_data[0].reshape(-1, 1))
        N += len(val_labels)
        right_decision_path = []
        wrong_decision_path = []
        for sample, label in zip(val_data, val_labels):
            if classifier.classify(sample.reshape(1, -1)) == label:
                decision_path = right_decision_path
                accuracy += 1
            else:
                decision_path = wrong_decision_path
                error += 1
            decision_path.append(classifier.tree.decision_path(sample.reshape(1, -1)).indices)

        test_folds.append(test_folds.pop(0))

    return accuracy / N, error / N


def compare_k(k_vec, factory):
    _res = {}
    for k_val in k_vec:
        _res[k_val] = (evaluate(factory(k_val), 2))

    return _res


def print_csv(res_dict, idx):
    csv = open('experiment' + idx + '.csv', "w")
    for key in res_dict.keys():
        row = str(key) + "," + str(res_dict[key][0]) + "," + str(res_dict[key][1]) + '\n'

        csv.write(row)


def plot_results(res):
    k_list = res.keys()
    accuracy, error = list(zip(*res.values()))

    plt.figure()
    plt.plot(k_list, accuracy, label='accuracy')
    max_value = max(accuracy)
    max_i = [k for i, k in enumerate(k_list) if accuracy[i] == max_value]
    plt.axhline(max_value,
                label='max value of accuracy = ' + "{0:.4f}".format(max_value) + "\n attained by k=" + str(max_i[0]),
                color='r')
    plt.axvline(x=max_i, linestyle='--', color='r')
    plt.legend()
    plt.title('Accuracy of classifier')
    plt.ylabel('accuracy score')
    plt.xlabel('k - number of folds')
    plt.show()


def section3():
    data_set = load_data()
    split_crosscheck_groups(data_set, 2)


def section6():
    data_set = load_data()
    k_list = [1, 3, 5, 7, 13]
    res = compare_k(k_list, knn_factory)
    print_csv(res, '6')
    plot_results(res)


def section7():
    res = {}

    tree_classifier = tree_factory('entropy')
    res[1] = (evaluate(tree_classifier, 2))

    perceptron_classifier = perceptron_factory()
    res[2] = evaluate(perceptron_classifier, 2)

    print_csv(res, '12')


def part_c():
    tree_classifier = tree_factory('entropy')
    res = evaluate(tree_classifier, 2)
    print('a')


if __name__ == '__main__':
    # section3()
    # section6()
    # section7()
    part_c()
    print('a')
