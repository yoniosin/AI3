from hw3_utils import *
from collections import Counter, namedtuple
import numpy as np
import heapq
import random
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from functools import reduce
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm


class Normalizer:
    def __init__(self, data):
        self.mean = np.mean(data, axis=0)
        self.var = np.var(data, axis=0)

    def transform(self, data):
        return (data - self.mean) / self.var ** 0.5


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


class SvmFactory(abstract_classifier_factory):
    def __init__(self, kernel):
        self.kernel = kernel

    def train(self, data, labels, weights=None):
        return SvmClassifier(self.kernel, data, labels, weights)


class SvmClassifier(abstract_classifier):
    def __init__(self, kernel, data, labels, weights):
        self.normalizer = Normalizer(data)
        self.svm = svm.SVC(kernel=kernel, gamma='scale')
        self.svm.fit(self.normalizer.transform(data), labels)

    def classify(self, features):
        norm_features = self.normalizer.transform(features)
        return self.svm.predict(norm_features)[0]


class TreeKnnClassifier(abstract_classifier):
    def __init__(self, criterion, depth, k, samples, labels):
        self.tree = DecisionTreeClassifier(criterion=criterion, max_depth=depth)
        self.tree.fit(X=samples, y=labels)

        factory = knn_factory(k)
        train_sets = self.getLeafIDs(samples, labels)
        self.knn_classifiers = {leaf: factory.train(train_sets[leaf]['samples'], train_sets[leaf]['labels'],
                                                    train_sets[leaf]['weights'])
                                for leaf in train_sets.keys()}

    def getLeafIDs(self, data, labels):
        train_sets = {}
        for sample, label in zip(data, labels):
            leaf = self.getLeaf(sample)
            try:
                train_sets[leaf]['samples'].append(sample)
                train_sets[leaf]['labels'].append(label)

            except KeyError:
                train_sets[leaf] = {'samples': [sample], 'labels': [label], 'weights': self.generateWeights(sample)}

        return train_sets

    def getLeaf(self, sample):
        return self.tree.decision_path(sample.reshape(1, -1)).indices[-1]

    def generateWeights(self, sample):
        weights = np.ones(sample.shape)
        for node in self.tree.decision_path(sample.reshape(1, -1)).indices[:-1]:
            weights[self.tree.tree_.feature[node]] = 0

        return weights

    def classify(self, sample):
        """
        Finds relevant knn classifier, and returns it's result on the given sample.
        :param sample:
        :return:
        """
        return self.knn_classifiers[self.getLeaf(sample)].classify(sample)


class TreeKnnFactory(abstract_classifier_factory):
    def __init__(self, criteria, depth, k):
        self.criteria = criteria
        self.depth = depth
        self.k = k

    def train(self, samples, labels):
        return TreeKnnClassifier(self.criteria, self.depth, self.k, samples, labels)


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
        return self.tree.predict(X=features)[0]

class forest_factory(abstract_classifier_factory):
    def train(self, data, labels):
        return forest_clasiffier(data, labels)

class forest_clasiffier(abstract_classifier):
    def __init__(self, data, labels):
        knn_fact_5 = knn_factory(5)
        knn_fact_1 = knn_factory(1)
        svm_factory = SvmFactory('rbf')
        self.classifiers_list = [
            knn_fact_1.train(data[0], labels[0]),
            svm_factory.train(data[1], labels[1]),
            knn_fact_5.train(data[2], labels[2])
        ]

    def classify(self, features):
        res = [classifier.classify(features) for classifier in self.classifiers_list]
        return max(set(res), key=res.count)

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


def evaluate(classifier_factory: abstract_classifier_factory, k, suffix='', component=None):
    need_to_load_fourier = suffix == 'forest'
    if not need_to_load_fourier:
        test_folds = [load_k_fold_data(idx, suffix) for idx in range(k)]
    else:
        test_folds = [load_k_fold_data(idx, 'time') for idx in range(k)]
        test_folds_fourier = [load_k_fold_data(idx, 'fourier') for idx in range(k)]
        test_folds_scale = [load_k_fold_data(idx, 'scale') for idx in range(k)]

    N = 0
    accuracy = 0
    error = 0
    precision_vec = [0, 0]
    recall_vec = [0, 0]

    for fold in range(k):
        train_data, train_labels = reduce(lambda a, b: (np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]])),
                                          test_folds[1:])
        val_data, val_labels = test_folds[0]

        if need_to_load_fourier:
            train_data_fourier, train_labels_fourier = reduce(lambda a, b: (np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]])),
                                              test_folds_fourier[1:])
            val_data_fourier, val_labels_fourier = test_folds_fourier[0]
            train_data_scale, train_labels_scale = reduce(
                lambda a, b: (np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]])),
                test_folds_scale[1:])
            val_data_scale, val_labels_scale= test_folds_scale[0]

        # if it is fourier data, apply LPF / BPF (window)
        if (suffix in ['fourier', 'pca'] or need_to_load_fourier) and component is not None:
            train_data = train_data[:, component[0]:component[1]]
            val_data = val_data[:, component[0]:component[1]]
            if need_to_load_fourier:
                train_data_fourier = train_data_fourier[:, component[0]:component[1]]
                val_data_fourier = val_data_fourier[:, component[0]:component[1]]

                train_data_scale= train_data_scale[:, component[0]:component[1]]
                val_data_scale = val_data_scale[:, component[0]:component[1]]

                train_data = [train_data, train_data_fourier, train_data_scale]
                train_labels = [train_labels, train_labels_fourier, train_labels_scale]

        classifier: abstract_classifier = classifier_factory.train(train_data, train_labels)
        N += len(val_labels)
        error_dict = {True: 0, False: 0}
        accuracy_dict = {True: 0, False: 0}
        for sample, label in zip(val_data, val_labels):
            class_res = classifier.classify(sample.reshape(1, -1))
            if class_res == label:
                accuracy_dict[label] += 1
                accuracy += 1
                if class_res:
                    precision_vec[0] += 1
                    recall_vec[0] += 1
            else:
                error += 1
                error_dict[label] += 1

            if class_res:
                precision_vec[1] += 1
            if label:
                recall_vec[1] += 1

        test_folds.append(test_folds.pop(0))
        if need_to_load_fourier:
            test_folds_fourier.append(test_folds_fourier.pop(0))
            test_folds_scale.append(test_folds_scale.pop(0))

    precision = precision_vec[0] / precision_vec[1]
    recall = recall_vec[0] / recall_vec[1]
    f1 = 2 * (precision * recall) / (precision + recall)
    return accuracy / N, error / N


def compare_k(k_vec, factory, suffix='', n_component=None):
    _res = {}
    for k_val in k_vec:
        print('evaluate' + str(k_val))
        _res[k_val] = (evaluate(factory(k_val), 2, suffix, n_component))

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


if __name__ == '__main__':
    section3()
    section6()
    section7()
    print('a')
