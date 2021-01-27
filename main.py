import time
import matplotlib.pyplot as plt
import numpy as np
import mnist
# save numpy array as csv file
from numpy import asarray
from numpy import save
from numpy import load
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import svm_2.kernel as svm2

from pszt import svm as svm_pszt

from svo import SVM as svo 

from svm_3 import svm_3

X_train, y_train, X_test, y_test = mnist.load()
limit = 1000
X_train = X_train.astype(np.float64)[0:limit, :]
X_test = X_test.astype(np.float64)[0:limit, :]
y_train = y_train[0:limit]
y_test = y_test[0:limit]


def ovr_create_label_array(arr, digit):
    """
    Zwraca array w typie [1 -1 -1 -1 ... ] na podstawie arraya labeli
    :param arr: [ndarray] y_test lub y_train
    :param digit: [int] cyfra po jakiej chcemy przefiltrować
    :return: [ndarray] gotowy ndarray
    """
    new = np.ndarray(len(arr), dtype=np.int8)
    for idx, field in enumerate(arr):
        if field == np.int8(digit):
            new[idx] = 1
        else:
            new[idx] = -1
    return new


def ovo_create_one_digit_array(arr, arr_label, digit1, digit2):
    """
    Zwraca ndarray złożony z samych obrazków konkretnych cyfr
    :param arr:
    :param arr_label:
    :param digit: [int] cyfra po jakiej chcemy przefiltrować
    :return: [ndarray] gotowy ndarray
    """
    # new = np.ndarray(0, dtype=np.float64)
    new = np.ndarray(14000, dtype=np.int8)
    new_label = np.ndarray(14000, dtype=np.int8)
    idx = 1
    for ide,field in enumerate(arr_label):
        if field == np.int8(digit1):
            new[idx] = ide
            new_label[idx] = 1
            idx += 1
        elif field == np.int8(digit2):
            new[idx] = ide
            new_label[idx] = -1
            idx += 1

    # print(idx,np.sum(new[idx-1]), np.sum(new[idx]), np.sum(new[idx+1]))
    return new[1:idx], new_label[1:idx]



if __name__ == '__main__':
    ovr_train_y = {}
    ovr_test_y = {}
    ovo_test_x = {}
    ovo_train_x = {}
    ovo_test_y = {}
    ovo_train_y = {}

    # zapis
    # for i in range(0, 10):
    #     ovr_train_y[i] = ovr_create_label_array(y_train, i)
    #     ovr_test_y[i] = ovr_create_label_array(y_test, i)
    #     # ovo_test_x[i] = ovo_create_one_digit_array(X_train, y_train, i)
    #     # ovo_train_x[i] = ovo_create_one_digit_array(X_test, y_test, i)
    #     for j in range(0, 10):
    #         if i == j:
    #             continue
    #         ovo_test_x[(i, j)], ovo_test_y[(i, j)] = ovo_create_one_digit_array(X_test, y_test, i, j)
    #         ovo_train_x[(i, j)], ovo_train_y[(i, j)] = ovo_create_one_digit_array(X_train, y_train, i, j)
    #         save(f'ovo_test_x({i},{j}).npy', ovo_test_x[(i, j)])
    #         save(f'ovo_test_y({i},{j}).npy', ovo_test_y[(i, j)])
    #         save(f'ovo_train_x({i},{j}).npy', ovo_train_x[(i, j)])
    #         save(f'ovo_train_y({i},{j}).npy', ovo_train_y[(i, j)])
    # print(len(ovo_train_y.keys()), ovo_train_y.keys())

    # odczyt
    # for i in range(0, 10):
    #     for j in range(0, 10):
    #         if i == j:
    #             continue
    #         ovo_test_x[(i, j)] = load(f'prepared_datasets/60000/ovo_test_x({i},{j}).npy')
    #         ovo_test_y[(i, j)] = load(f'prepared_datasets/60000/ovo_test_y({i},{j}).npy')
    #         ovo_train_x[(i, j)] = load(f'prepared_datasets/60000/ovo_train_x({i},{j}).npy')
    #         ovo_train_y[(i, j)] = load(f'prepared_datasets/60000/ovo_train_y({i},{j}).npy')
    # print(len(ovo_train_y.keys()), ovo_train_y.keys())
    # print(ovo_train_x[(1, 2)][1:10])
    # # print( X_train[x] for x in ovo_train_x[(1, 2)][1:10])
    # print([y_train[x] for x in ovo_train_x[(1, 2)][1:10]])
    # print(X_train[ovo_train_x[(1, 2)][1:10]])
    # print(y_train[ovo_train_y[(1, 2)]][1:10])
    # X = X_train[ovo_train_x[(0, 1)]]/16
    # y = ovo_train_y[(0, 1)]
    

    # pca = PCA(n_components=28)
    # pca.fit_transform(X_train)
    # pca.transform(X_test)

    X = X_train[(y_train == 0) | (y_train == 3)] / 256
    y = y_train[(y_train == 0) | (y_train == 3)].astype(np.double)

    positive_indices = (y == 0)
    negative_indices = (y == 3)

    y[positive_indices] = np.ones(sum(positive_indices)).reshape(-1)
    y[negative_indices] = -np.ones(sum(negative_indices)).reshape(-1)

    X_t = X_test[(y_test == 0) | (y_test == 3)] / 256
    y_t = y_test[(y_test == 0) | (y_test == 3)].astype(np.double)

    positive_indices = (y_t == 0)
    negative_indices = (y_t == 3)

    y_t[positive_indices] = np.ones(sum(positive_indices)).reshape(-1)
    y_t[negative_indices] = -np.ones(sum(negative_indices)).reshape(-1)


    # # predicted = clf.predict(X_test)
    # test_X = X_test[ovo_test_x[(0, 1)]]/16
    # test_y = ovo_test_y[(0, 1)]
    # print(y_test[ovo_test_x[(0, 1)]][:10])
    # print(test_y[:10])
    # clasifier = svm_pszt.SVM_NonLinear().fit(X, y)
    #
    # results = clasifier.predict(X_t)
    #
    # classifier = svm.SVC(gamma='scale')
    # classifier.fit(X, y)
    # results = classifier.predict(X_t)

    # KernelSVM
    # classifier = svm_3.SVM()
    classifier = svm_pszt.SVM_NonLinear()
    print('odpalam fit X,y o długościach', len(y), len(X))
    dupa = classifier.fit(X, y, gamma=0)
    # classifier.train(X, y, {})
    print('długości danych trainowych: ', len(y), len(X))
    print('długości danych testowych: ', len(y_t),len(X_t))
    print('predict wektorem testowym X_t o długości: ', len(X_t))
    xtestowe = X_t
    ytestowe = y_t
    results = dupa.predict(xtestowe)
    print('wynik z predicta X_t: ', results.shape)

    # comparison = y == results
    # print(np.sum(np.int8(comparison))/len(y))
    # print(np.ones(y.shape)[(y == results)])
    print(np.sum(np.ones(ytestowe.shape)[(ytestowe == results)]) / len(ytestowe))


