import numpy as np 
import cvxopt
from numpy.linalg import norm

SUPPORT_VECTOR_MULTIPLIER_THRESHOLD = 1e-5

class Kernel(object):
    '''
    Klasa implementująca wybrane funkcje jądra
    '''
    @staticmethod
    def rbf(sigma = 1):
        return lambda x_1, x_2: np.exp(-norm(np.subtract(x_1, x_2)) * (2*sigma*sigma))


class SVM_NonLinear(object):
    '''
    Klasa której podstawowym zadaniem jest zbudowanie klasyfikatora opartego
    na maszynie wektorów nośnych
    '''

    def __init__(self, kernel = Kernel.rbf(), C = 1.0):
        self.kernel = kernel
        self.C = C

    def fit(self, X, y):
        '''
        Metoda budująca klasyfikator na podstawie danych X i odpowiadającym im wartościom y
        :param X: macierz n x m, gdzie n to ilość próbek w zbiorze, a m to liczba atrybutów pojedynczej próbki
        :param y: wektor n elementowy, o wartościach y_i = {-1, 1} 
        '''
        X = X/256
        lagrange_multipliers = self.get_lagrange_multipliers(X, y)

        support_vectors_indices = ((self.C > lagrange_multipliers) & (lagrange_multipliers > SUPPORT_VECTOR_MULTIPLIER_THRESHOLD))
        
        svm_multipliers = lagrange_multipliers[support_vectors_indices]
        svm_vectors = X[support_vectors_indices]
        svm_labels = y[support_vectors_indices]

        bias = np.mean([
            y_k - SVMClassifier(
                weights=svm_multipliers,
                vectors=svm_vectors,
                labels=svm_labels,
                bias=0.0,
                kernel=self.kernel
            ).predict(x_k)
            for (y_k, x_k) in zip(svm_labels, svm_vectors)
        ])

        print(svm_multipliers.shape)
        print(svm_vectors.shape)
        print(svm_labels.shape)

        return SVMClassifier(
                weights=svm_multipliers,
                vectors=svm_vectors,
                labels=svm_labels,
                bias=bias,
                kernel=self.kernel
            )

        

    def get_lagrange_multipliers(self, X, y):
        n_samples, _ = X.shape
        K = np.zeros((n_samples, n_samples))

        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self.kernel(x_i, x_j)
    
        q = cvxopt.matrix(-np.ones((n_samples,1)))
        P = cvxopt.matrix(np.outer(y, y)*K)
        A = cvxopt.matrix(y.reshape(1, n_samples).astype(np.double))
        b = cvxopt.matrix(0.0)
        G = cvxopt.matrix(np.concatenate((np.eye(n_samples), -np.eye(n_samples))))
        h = cvxopt.matrix(np.concatenate((self.C * np.ones((n_samples, 1)), np.zeros((n_samples, 1)))))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)['x']

        return np.ravel(solution)


class SVMClassifier(object):

    def __init__(self, weights, vectors, labels, bias, kernel):
        self.kernel = kernel
        self.weights = weights
        self.vectors = vectors
        self.labels = labels
        self.bias = bias

    def predict(self, x):
        result = self.bias
        for a_i, y_i, x_i in zip(self.weights, self.labels, self.vectors):
            result += a_i * y_i * self.kernel(x_i, x)
        return np.sign(result).item()
