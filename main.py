import time
import matplotlib.pyplot as plt
import numpy as np
import mnist
from pszt import ensembles
# save numpy array as csv file
# from numpy import asarray
# from numpy import save
# from numpy import load
from sklearn import datasets, metrics, svm
# from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
# import svm_2.kernel as svm2
from pszt import svm as svm_pszt
# from svo import SVM as svo
# from svm_3 import svm_3

X_train, y_train, X_test, y_test = mnist.load()
limit_train = 5000
limit_test = 2000
X_train = X_train.astype(np.float64)[0:limit_train, :]
X_test = X_test.astype(np.float64)[0:limit_test, :]
y_train = y_train[0:limit_train]
y_test = y_test[0:limit_test]


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
    classifier = svm_pszt.SVM
    clf = ensembles.OVOEnsemble(classifier, rbf_kernel)

    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test).astype(np.int8)

    _, axes = plt.subplots(nrows=1, ncols=12, figsize=(15, 5))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(28, 28)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title(f'Prediction: {prediction}')

    # print(f"Classification report for classifier {clf}:\n"
    #       f"{metrics.classification_report(y_test, predicted)}\n")

    # disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
    # disp.figure_.suptitle("Confusion Matrix")
    # print(f"Confusion matrix:\n{disp.confusion_matrix}")
    plt.show()

    confusiomMatrix = metrics.confusion_matrix(y_test, predicted)
    metrics.ConfusionMatrixDisplay(confusiomMatrix, display_labels=np.unique(y_test))
    
    print(np.sum(np.ones(y_test.shape)[(y_test == predicted)]) / len(y_test))
    
