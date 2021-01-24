# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy
import mnist
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

X_train, y_train, X_test, y_test = mnist.load()

X_train = X_train.astype(numpy.float64)
X_test = X_test.astype(numpy.float64)
y_train = y_train
y_test = y_test

def ovr_create_label_array(arr, digit):
    """
    Zwraca array w typie [1 -1 -1 -1 ... ] na podstawie arraya labeli
    :param arr: [ndarray] y_test lub y_train
    :param digit: [int] cyfra po jakiej chcemy przefiltrować
    :return: [ndarray] gotowy ndarray
    """
    new = numpy.ndarray(len(arr), dtype=numpy.int8)
    for idx, field in enumerate(arr):
        if field == numpy.int8(digit):
            new[idx] = 1
        else:
            new[idx] = -1
    return new


def ovo_create_digit_array(arr, arr_label, digit):
    """
    Zwraca ndarray złożony z samych obrazków konkretnych cyfr
    :param arr:
    :param arr_label:
    :param digit: [int] cyfra po jakiej chcemy przefiltrować
    :return: [ndarray] gotowy ndarray
    """
    # new = numpy.ndarray(0, dtype=numpy.float64)
    new = numpy.ndarray((7000, 784), dtype=numpy.float64)
    idx = 1
    for field in arr_label:
        if field == numpy.int8(digit):
            new[idx] = arr[idx]
            # numpy.append(new, arr[idx])
            idx += 1
    # print(idx,numpy.sum(new[idx-1]), numpy.sum(new[idx]), numpy.sum(new[idx+1]))
    return new[1:idx]


if __name__ == '__main__':
    ovr_train_y = {}
    ovr_test_y = {}
    ovo_test_x = {}
    ovo_train_x = {}

    for i in range(0, 10):
        ovr_train_y[i] = ovr_create_label_array(y_train, i)
        ovr_test_y[i] = ovr_create_label_array(y_test, i)
        ovo_test_x[i] = ovo_create_digit_array(X_train, y_train, i)
        ovo_train_x[i] = ovo_create_digit_array(X_test, y_test, i)





