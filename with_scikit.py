# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy
import mnist
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
limit = 4000

digits = datasets.load_digits()
# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, digits.images, digits.target):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     ax.set_title('Training: %i' % label)

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)
# print(X_train2[1])
# print(y_train2[1])
# print(X_test2[1])
# print(y_test2[1])

X_train, y_train, X_test, y_test = mnist.load()
X_train = (X_train[1:limit, :]/16.0).astype(numpy.uint8).astype(numpy.float64)
X_test = (X_test[1:limit, :]/16.0).astype(numpy.uint8).astype(numpy.float64)
y_train = y_train[1:limit]
y_test = y_test[1:limit]

# print(X_train[2])
# print(y_train[2])
# print(X_test[1])
# print(y_test)

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma='scale')

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(28, 28)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Prediction: {prediction}')

# print(f"Classification report for classifier {clf}:\n"
#       f"{metrics.classification_report(y_test, predicted)}\n")

disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
# print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

print(numpy.sum(numpy.ones(y_test.shape)[(y_test == predicted)]) / len(y_test))