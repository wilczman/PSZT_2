# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
import mnist
# save numpy array as csv file
from numpy import asarray
from numpy import save
from numpy import load
# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

# import svm_2.kernel as svm

from pszt import svm as svm_pszt

X_train, y_train, X_test, y_test = mnist.load()

X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)
y_train = y_train
y_test = y_test


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
    for i in range(0, 10):
        for j in range(0, 10):
            if i == j:
                continue
            ovo_test_x[(i, j)] = load(f'prepared_datasets/ovo_test_x({i},{j}).npy')
            ovo_test_y[(i, j)] = load(f'prepared_datasets/ovo_test_y({i},{j}).npy')
            ovo_train_x[(i, j)] = load(f'prepared_datasets/ovo_train_x({i},{j}).npy')
            ovo_train_y[(i, j)] = load(f'prepared_datasets/ovo_train_y({i},{j}).npy')
    # print(len(ovo_train_y.keys()), ovo_train_y.keys())
    # print(ovo_train_x[(1, 2)][1:10])
    # # print( X_train[x] for x in ovo_train_x[(1, 2)][1:10])
    # print([y_train[x] for x in ovo_train_x[(1, 2)][1:10]])
    # print(X_train[ovo_train_x[(1, 2)][1:10]])
    # print(y_train[ovo_train_y[(1, 2)]][1:10])
    X = X_train[ovo_train_x[(0, 1)]]/16
    y = ovo_train_y[(0, 1)]

    # predicted = clf.predict(X_test)
    test_X = X_test[ovo_test_x[(0, 1)]]/16
    test_y = ovo_test_y[(0, 1)]
    print(y_test[ovo_test_x[(0, 1)]][:10])
    print(test_y[:10])
    # clasifier = svm_pszt.SVM_NonLinear().fit(X/256, y)

    # results = clasifier.predict(test_X)

    classifier = svm.SVC(gamma='scale')
    classifier.fit(X, y)
    results = classifier.predict(test_X)

    sum = 0
    for i, res in enumerate(results):
        if res == test_y[i]:
            sum += 1

    print(sum / len(results))


