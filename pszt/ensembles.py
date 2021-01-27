import numpy as np
import itertools

class OVOEnsemble(object):

    def __init__(self, classifier):
        self._classifier = classifier
        

    def fit(self, X, y):
        self._labels = np.unique(y)
        self._scaler = np.max(X)
        self._X_train = X / self._scaler
        self._y_train = y

        self._models = dict()

        for pClass, nClass in itertools.combinations(self._labels, 2):
            X = self._X_train[(self._y_train == pClass) | (self._y_train == nClass)]
            y = self._y_train[(self._y_train == pClass) | (self._y_train == nClass)].astype(np.double)

            positive_indices = (y == pClass)
            negative_indices = (y == nClass)
            y[positive_indices] = np.ones(sum(positive_indices)).reshape(-1)
            y[negative_indices] = -np.ones(sum(negative_indices)).reshape(-1)

            self._models[(pClass, nClass)] = self._classifier()
            self._models[(pClass, nClass)].fit(X, y)

    def predict(self, X_in):
        X = X_in / self._scaler
        votes = np.zeros((X.shape[0], self._labels.shape[0]))
        sorter = np.argsort(self._labels)

        for labels, model in self._models.items():

            result = model.predict(X)

            positive_indices = (result == 1)
            negative_indices = (result == -1)

            result_labels = np.zeros(result.shape)
            result_labels[positive_indices] = np.repeat(labels[0], sum(positive_indices))
            result_labels[negative_indices] = np.repeat(labels[1], sum(negative_indices))

            labels_indices = sorter[np.searchsorted(self._labels, result_labels, sorter = sorter)]

            for idx_sample, idx_label in enumerate(labels_indices):
                votes[idx_sample, idx_label] += 1

        final_idx = np.argmax(votes, axis=1)

        prediction = np.zeros(X.shape[0])
        for i, label_idx in enumerate(final_idx):
            prediction[i] = self._labels[label_idx]
            
        return prediction


class OVREnsemble(object):

    def __init__(self, classifier):
        self._classifier = classifier
        

    def fit(self, X, y):
        self._labels = np.unique(y)
        self._scaler = np.max(X)
        self._X_train = X / self._scaler
        self._y_train = y

        self._models = [None] * self._labels.shape[0]

        for i, pClass in enumerate(self._labels):
            X = self._X_train
            y = self._y_train.astype(np.double)

            positive_indices = (y == pClass)
            negative_indices = (y != pClass)
            y[positive_indices] = np.ones(sum(positive_indices)).reshape(-1)
            y[negative_indices] = -np.ones(sum(negative_indices)).reshape(-1)

            self._models[i] = self._classifier()
            self._models[i].fit(X, y)

    def predict(self, X_in):
        X = X_in / self._scaler
        votes = np.zeros((X.shape[0], self._labels.shape[0]))

        for label_idx, model in enumerate(self._models):

            result = model.predict(X)

            positive_indices = (result == 1)
            negative_indices = (result == -1)

            result_labels = np.zeros(result.shape)
            result_labels[positive_indices] = np.repeat(self._labels[label_idx], sum(positive_indices))
            result_labels[negative_indices] = np.repeat(self._labels.shape[0], sum(negative_indices))

            for idx_sample, result in enumerate(result_labels):
                idx_label = np.where(result == self._labels)
                if idx_label == self._labels.shape[0]:
                    continue
                votes[idx_sample, idx_label] += 1

        final_idx = np.argmax(votes, axis=1)

        prediction = np.zeros(X.shape[0])
        for i, label_idx in enumerate(final_idx):
            prediction[i] = self._labels[label_idx]
            
        return prediction


            

            