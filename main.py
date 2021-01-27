import time
import matplotlib.pyplot as plt
import numpy as np
import mnist
from sklearn import datasets, metrics, svm
from sklearn.metrics.pairwise import rbf_kernel
from pszt import svm as svm_pszt
from pszt import ensembles

X_train, y_train, X_test, y_test = mnist.load()
limit_train = 10000
limit_test = 2000
X_train = X_train.astype(np.float64)[0:limit_train, :]
X_test = X_test.astype(np.float64)[0:limit_test, :]
y_train = y_train[0:limit_train]
y_test = y_test[0:limit_test]

if __name__ == '__main__':

    print("Testowanie liniowego klasyfikatora SVM w wersji OVO")
    classifier = svm_pszt.SVM
    clf = ensembles.OVOEnsemble(classifier)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test).astype(np.int8)
    confusiomMatrix = metrics.confusion_matrix(y_test, predicted)
    disp = metrics.ConfusionMatrixDisplay(confusiomMatrix, display_labels=np.unique(y_test))
    disp.plot()
    correct_rate = np.sum(np.ones(y_test.shape)[(y_test == predicted)]) / len(y_test)
    print(f"Procent porawnych klasyfikacji lin SVM OVO: {correct_rate * 100}%")


    print("Testowanie liniowego klasyfikatora SVM w wersji OVR")
    classifier = svm_pszt.SVM
    clf = ensembles.OVREnsemble(classifier)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test).astype(np.int8)
    confusiomMatrix = metrics.confusion_matrix(y_test, predicted)
    disp = metrics.ConfusionMatrixDisplay(confusiomMatrix, display_labels=np.unique(y_test))
    disp.plot()
    correct_rate = np.sum(np.ones(y_test.shape)[(y_test == predicted)]) / len(y_test)
    print(f"Procent porawnych klasyfikacji lin SVM OVR: {correct_rate * 100}%")


    print("Testowanie klasyfikatora SVM z wykorzystaniem kernela RBF w wersji OVO")
    classifier = svm_pszt.SVM
    clf = ensembles.OVOEnsemble(classifier, rbf_kernel)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test).astype(np.int8)
    confusiomMatrix = metrics.confusion_matrix(y_test, predicted)
    disp = metrics.ConfusionMatrixDisplay(confusiomMatrix, display_labels=np.unique(y_test))
    disp.plot()
    correct_rate = np.sum(np.ones(y_test.shape)[(y_test == predicted)]) / len(y_test)
    print(f"Procent porawnych klasyfikacji RBF SVM OVO: {correct_rate * 100}%")


    print("Testowanie klasyfikatora SVM z wykorzystaniem kernela RBF w wersji OVR")
    classifier = svm_pszt.SVM
    clf = ensembles.OVREnsemble(classifier, rbf_kernel)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test).astype(np.int8)
    confusiomMatrix = metrics.confusion_matrix(y_test, predicted)
    disp = metrics.ConfusionMatrixDisplay(confusiomMatrix, display_labels=np.unique(y_test))
    disp.plot()
    correct_rate = np.sum(np.ones(y_test.shape)[(y_test == predicted)]) / len(y_test)
    print(f"Procent porawnych klasyfikacji RBF SVM OVR: {correct_rate * 100}%")
    
