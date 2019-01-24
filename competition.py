from hw3_utils import *
from classifier import *
# import neurokit as nk
from imblearn.over_sampling import SMOTE, ADASYN
import scipy
from sklearn.decomposition import PCA


def remove_corrupted_data(data):
    for i in [319, 519, 550, 908]:
        data[0] = np.delete(data[0], (i), axis=0)
        del data[1][i]

    return data


def create_time(data, fold_num, sample_type):
    # split_crosscheck_groups(data, fold_num, 'time' + sample_type)
    sample_len = data[0][0].shape[0]
    t = np.arange(sample_len)
    plot_random_samples(data, t)


def create_fourier(data, fold_num, sample_type):
    X = data[0]
    # X = (X.T - X.mean(axis=1)).T
    X_fft = np.asarray([np.fft.fft(x) for x in X])
    X_fft_shift = np.asarray([np.fft.fftshift(x) for x in X_fft])
    data_fourier = [np.abs(X_fft), data[1]]
    data_shift = [np.abs(X_fft_shift), data[1]]

    sample_len = data[0][0].shape[0]
    f = np.linspace(-np.pi, np.pi, sample_len)
    plot_random_samples(data_shift, f)
    # split_crosscheck_groups(data_fourier, fold_num, 'fourier' + sample_type)


def create_scale(data, fold_num, sample_type):
    sample_len = data[0][0].shape[0]

    # scale train data
    X = data[0]
    X_scale = []
    for i, x in enumerate(X):
        x_crop = np.asarray(list(filter(lambda v: abs(v) > 0.05, x))).reshape(-1, 1)
        x_scale = scipy.misc.imresize(x_crop, (sample_len, 1)).reshape(-1) / 255
        X_scale.append(x_scale)
    X_scale = np.asarray(X_scale)


    # scale test data
    test = data[2]
    test_scale = []
    for i, x in enumerate(test):
        test_crop = np.asarray(list(filter(lambda v: abs(v) > 0.05, x))).reshape(-1, 1)
        scale = scipy.misc.imresize(test_crop, (sample_len, 1)).reshape(-1) / 255
        test_scale.append(scale)
    test_scale = np.asarray(test_scale)


    data_scale = [X_scale, data[1], test_scale]
    t = np.arange(sample_len)
    plot_random_samples(data_scale, t)
    # split_crosscheck_groups(data_scale, 2, 'scale' + sample_type)

    return data_scale

def prepare_data(fold_num):
    org_data = list(load_data())
    X_resampled, y_resampled = SMOTE().fit_resample(org_data[0], org_data[1])
    print(sorted(Counter(y_resampled).items()))
    data_resampled = [X_resampled, y_resampled]

    for data, sample_type in zip([org_data], ['']):
        # remove corrupted data
        data = remove_corrupted_data(data)

        # time
        create_time(data, fold_num, sample_type)

        # fourier
        create_fourier(data, fold_num, sample_type)

        # scale
        create_scale(data, fold_num, sample_type)

        # pca
        create_PCA(data, fold_num, sample_type)

def create_PCA(data_set, fold_num, sample_type):

    X = data_set[0]
    Y = data_set[1]

    normalizer = Normalizer(X)
    X_scale = normalizer.transform(X)

    pca = PCA()
    pca.fit(X_scale)
    projected_X = pca.transform(X_scale)
    pca_data = [projected_X, Y]
    # split_crosscheck_groups(pca_data, fold_num, 'pca' + sample_type)

def plot_random_samples(_data, x):
    cdict = {True: (0.2, 0.5, 0.8), False: (0.8, 0.5, 0.2)}
    fig, ax = plt.subplots()
    for g in np.unique(_data[1]):
        ix = np.where(_data[1] == g)
        [ax.plot(x, (_data[0][tot_i]).T, c=cdict[g], label=(g if i == 0 else "")) for i, tot_i in
         enumerate(ix[0][10:15])]
    ax.legend()
    ax.set_title('Random Samples of the Data')
    plt.show()


# classifiers_list = ['svm']
classifiers_list = ['knn', 'tree', 'svm', 'perceptron', 'knn_tree']
k_list = [1, 3, 5, 7, 13]

factory = {'tree': tree_factory('entropy'),
           'perceptron': perceptron_factory(),
           'svm': SvmFactory('rbf'),
           'forest': forest_factory(),
           'knn_tree': TreeKnnFactory('entropy', 2, 1)
           }

if __name__ == '__main__':

    # train the competition classifier
    competition_factory = knn_factory(5)
    data = list(load_data())
    filtered_data = remove_corrupted_data(data)
    scaled_data = create_scale(filtered_data, 1, 'Competition')
    competition_classifier = competition_factory.train(scaled_data[0], scaled_data[1])

    results = [competition_classifier.classify(feature) for feature in scaled_data[2]]
    write_prediction(results)
    '''
    fold_num = 2
    # prepare_data(fold_num)
    dict_res = {}

    # forest
    fourier_component = [0, 30]
    res_forest = evaluate(factory['forest'], fold_num, 'forest', fourier_component)
    dict_res['forest'] = res_forest

    # fourier analyze
    method = 'fourier'
    dict_res[method] = {}
    fourier_component_list = [(0, 10), (0, 30), (0, 50), (1, 11), (1, 31), (1, 51)]
    for classifier in classifiers_list:
        dict_res[method][classifier] = {}
        for component in fourier_component_list:
            if classifier == 'knn':
                res_knn = compare_k(k_list, knn_factory, method, component)
                dict_res[method]['knn'][component] = res_knn
            else:
                res = evaluate(factory[classifier], fold_num, method, component)
                dict_res[method][classifier][component] = res

    # pca analyze
    method = 'pca'
    dict_res[method] = {}
    pca_component_list = [(0, 187), (0, 150), (0, 100), (0, 70), (0, 50), (0, 30), (0, 15)]
    for classifier in classifiers_list:
        dict_res[method][classifier] = {}
        for component in pca_component_list:
            if classifier == 'knn':
                res_knn = compare_k(k_list, knn_factory, method, component)
                dict_res[method]['knn'][component] = res_knn
            else:
                res = evaluate(factory[classifier], fold_num, method, component)
                dict_res[method][classifier][component] = res

    for method in ['scale', 'time']:
        dict_res[method] = {}
        for classifier in classifiers_list:
            dict_res[method][classifier] = {}
            if classifier == 'knn':
                res_knn = compare_k(k_list, knn_factory, method)
                dict_res[method]['knn'] = res_knn
            else:
                res = evaluate(factory[classifier], fold_num, method)
                dict_res[method][classifier] = res
    '''

    print('done')
