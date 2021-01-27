import numpy as np 
import cvxopt
from numpy.linalg import norm
from sklearn.metrics.pairwise import rbf_kernel as _rbf_kernel

class SVM(object):
    '''
    Klasa której podstawowym zadaniem jest zbudowanie klasyfikatora opartego
    na maszynie wektorów nośnych
    '''

    def __init__(self, kernel=None, C = 1.0, threshold=1e-5):
        self.kernel = kernel if kernel is not None else self.linear_kernel()
        self.C = C
        self.vectors = None
        self.labels = None
        self.weights = None
        self.bias = None
        self.threshold = threshold
        self.bias = 0

    def fit(self, X, y, gamma=None):
        '''
        Metoda budująca klasyfikator na podstawie danych X i odpowiadającym im wartościom y
        :param X: macierz n x m, gdzie n to ilość próbek w zbiorze, a m to liczba atrybutów pojedynczej próbki
        :param y: wektor n elementowy, o wartościach y_i = {-1, 1} 
        '''
        self.gamma = 1/X.shape[1] if gamma == None else gamma

        lagrange_multipliers = self.get_lagrange_multipliers(X, y)

        support_vectors_indices = ((self.C > lagrange_multipliers) & (lagrange_multipliers > self.threshold))
        
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
            svm_labels - self.predict(svm_vectors)
        )
        self.bias = bias

    def get_lagrange_multipliers(self, X, y):
        n_samples, _ = X.shape
        K = np.zeros((n_samples, n_samples))

        K = self.kernel(X)

        P = cvxopt.matrix(np.outer(y, y)*K)
        q = cvxopt.matrix(-np.ones((n_samples,1)))
        G = cvxopt.matrix(np.concatenate((np.eye(n_samples), -np.eye(n_samples))))
        h = cvxopt.matrix(np.concatenate((self.C * np.ones((n_samples, 1)), np.zeros((n_samples, 1)))))
        b = cvxopt.matrix(0.0)
        A = cvxopt.matrix(y.reshape(1, -1).astype(np.double))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)['x']

        return np.ravel(solution)

    def predict(self, X):
        ker = self.kernel(self.vectors, X)
        alfa_y = np.multiply(self.labels.reshape(-1, 1), self.weights.reshape(-1, 1))
        result = np.matmul(ker.T, alfa_y).T[0] + self.bias
        return np.sign(result)

    
    def linear_kernel(self):
        def ker(X, Y=None):
            if Y is None:
                return np.matmul(X, X.T)
            else:
                return np.matmul(X, Y.T)
        return ker
