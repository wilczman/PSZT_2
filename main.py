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
    classifier = svm_pszt.SVM
    clf = ensembles.OVREnsemble(classifier, rbf_kernel)

    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test).astype(np.int8)

    _, axes = plt.subplots(nrows=1, ncols=12, figsize=(15, 5))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(28, 28)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title(f'Prediction: {prediction}')


    plt.show()

    confusiomMatrix = metrics.confusion_matrix(y_test, predicted)
    disp = metrics.ConfusionMatrixDisplay(confusiomMatrix, display_labels=np.unique(y_test))
    disp.plot()
    print(np.sum(np.ones(y_test.shape)[(y_test == predicted)]) / len(y_test))
    
