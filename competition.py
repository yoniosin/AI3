from hw3_utils import *
from classifier import knn_factory, evaluate, apply_PCA, split_crosscheck_groups
from sklearn.tree import DecisionTreeClassifier


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


if __name__ == '__main__':
    # k_list = [1, 3, 5, 7, 13]
    # res = compare_k(k_list, TreeKnnFactory)
    # data = load_data()
    # x_PCA = apply_PCA(data, n_components=100)
    # split_crosscheck_groups(x_PCA, 2, 'competition')
    tree_knn_factory = TreeKnnFactory('entropy', 1, 5)
    res = evaluate(tree_knn_factory, 2, 'competition')
    print('a')
