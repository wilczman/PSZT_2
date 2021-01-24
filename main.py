import mnist
import numpy
from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt
import time
# mnist.init()
t = time.time()
x_train, t_train, x_test, t_test = mnist.load()
# X_train, X_test, y_train, y_test = mnist.load()
# X_train = X_train[1:5000, :]
# X_test = X_test[1:5000, :]
# y_train = y_train[1:5000]
# y_test = y_test[1:5000]
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
print(x_train)
for i, x in enumerate(x_train[1:10]):
    x_train_2.append(x.reshape(28, 28))
# for x in x_train[1:10]:
#     print(x)

for ax, image in zip(axes, x_test):
    ax.set_axis_off()
    image = image.reshape(28, 28)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Prediction: none')

plt.show()

# clf = svm.SVC(gamma=0.001)
# clf.fit(x_train[1:100,:], t_train[1:100])
# predicted = clf.predict(x_test[1:100,:])
# print(predicted)
# print(time.time()-t)

# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, digits.images, digits.target):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     ax.set_title('Training: %i' % label)


# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, prediction in zip(axes, x_test, predicted):
#     ax.set_axis_off()
#     image = image.reshape(8, 8)
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     ax.set_title(f'Prediction: {prediction}')







# import matplotlib.pyplot as plt
#
# img = x_train[344,:].reshape(28,28) # First image in the training set.
# plt.imshow(img,cmap='gray')
# plt.show() # Show the image

