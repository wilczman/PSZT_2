import numpy as np 
import cvxopt
from numpy.linalg import norm
from sklearn.metrics.pairwise import rbf_kernel

SUPPORT_VECTOR_MULTIPLIER_THRESHOLD = 1e-5

class Kernel(object):
    '''
    Klasa implementująca wybrane funkcje jądra
    '''
    @staticmethod
    def rbf(sigma = 1):
        return lambda x_1, x_2: np.exp(-norm(np.subtract(x_1, x_2), 2) / (2*sigma*sigma))


class SVM_NonLinear(object):
    '''
    Klasa której podstawowym zadaniem jest zbudowanie klasyfikatora opartego
    na maszynie wektorów nośnych
    '''

    def __init__(self, kernel = Kernel.rbf(), C = 1.0):
        self.kernel = kernel
        self.C = C
        self.vectors = None
        self.labels = None
        self.weights = None
        self.bias = None

    def fit(self, X, y, gamma=None):
        '''
        Metoda budująca klasyfikator na podstawie danych X i odpowiadającym im wartościom y
        :param X: macierz n x m, gdzie n to ilość próbek w zbiorze, a m to liczba atrybutów pojedynczej próbki
        :param y: wektor n elementowy, o wartościach y_i = {-1, 1} 
        '''
        self.gamma = 1/X.shape[1] if gamma == None else gamma

        lagrange_multipliers = self.get_lagrange_multipliers(X, y)

        support_vectors_indices = ((self.C > lagrange_multipliers) & (lagrange_multipliers > SUPPORT_VECTOR_MULTIPLIER_THRESHOLD))
        
        # svm_multipliers = np.zeros(lagrange_multipliers.shape)
        # svm_vectors = np.zeros(X.shape)
        # svm_labels = np.zeros(y.shape)

        svm_multipliers = lagrange_multipliers[support_vectors_indices]
        svm_vectors = X[support_vectors_indices]
        svm_labels = y[support_vectors_indices]
        self.weights = svm_multipliers
        self.vectors = svm_vectors
        self.labels = svm_labels

        bias = np.mean(
            svm_labels - SVM_NonLinear_Classifier(
                weights=svm_multipliers,
                vectors=svm_vectors,
                labels=svm_labels,
                bias=0.0,
                kernel=self.kernel
            ).predict(svm_vectors)
        )
        self.bias = bias

        return SVM_NonLinear_Classifier(
                weights=svm_multipliers,
                vectors=svm_vectors,
                labels=svm_labels,
                bias=bias,
                kernel=self.kernel
            )


    def get_lagrange_multipliers(self, X, y):
        n_samples, _ = X.shape
        K = np.zeros((n_samples, n_samples))

        K = rbf_kernel(X)

        P = cvxopt.matrix(np.outer(y, y)*K)
        q = cvxopt.matrix(-np.ones((n_samples,1)))
        G = cvxopt.matrix(np.concatenate((np.eye(n_samples), -np.eye(n_samples))))
        h = cvxopt.matrix(np.concatenate((self.C * np.ones((n_samples, 1)), np.zeros((n_samples, 1)))))
        b = cvxopt.matrix(0.0)
        A = cvxopt.matrix(y.reshape(1, -1).astype(np.double))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)['x']

        return np.ravel(solution)

    def predict(self, X):
        ker = rbf_kernel(self.vectors, X)  # kernel nieliniowy
        # ker = np.matmul(self.vectors, X.T)  # kernel liniowy
        alfa_y = np.multiply(self.labels.reshape(-1, 1), self.weights.reshape(-1, 1))
        result1 = np.matmul(ker.T, alfa_y).T[0] + self.bias

        # wersja iteracyjna również działa
        # result1 = np.zeros(X.shape[0])
        # for i, x in enumerate(X):
        #     y = 0
        #     for j, X_v in enumerate(self.vectors):
        #         y += self.weights[j] * self.labels[j] * rbf_kernel(np.array([X_v]), np.array([x]))
        #     result1[i] = y+self.bias
        return np.sign(result1)


class SVM_NonLinear_Classifier(object):

    def __init__(self, weights, vectors, labels, bias, kernel):
        self.kernel = kernel
        self.weights = weights
        self.vectors = vectors
        self.labels = labels
        self.bias = bias

    def predict(self, X):
        ker = rbf_kernel(self.vectors, X)
        alfa_y = np.multiply(self.labels.reshape(-1, 1), self.weights.reshape(-1, 1))
        result1 = np.matmul(ker.T, alfa_y).T[0] + self.bias

        # wersja iteracyjna również działa
        # result1 = np.zeros(X.shape[0])
        # for i, x in enumerate(X):
        #     y = 0
        #     for j, X_v in enumerate(self.vectors):
        #         y += self.weights[j] * self.labels[j] * rbf_kernel(np.array([X_v]), np.array([x]))
        #     result1[i] = y+self.bias
        return np.sign(result1)


class SVM_Linear():

    def __init__(self):
        pass

    def fit(self, X, y):
        pass


class SVM_Linear_Classifier():

    def __init__(self, sepplane):
        pass

    def fit(self, X, y):
        pass