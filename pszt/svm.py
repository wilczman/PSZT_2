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
        return lambda x_1, x_2: np.exp(-norm(np.subtract(x_1, x_2)) / (2*sigma*sigma))


class SVM_NonLinear(object):
    '''
    Klasa której podstawowym zadaniem jest zbudowanie klasyfikatora opartego
    na maszynie wektorów nośnych
    '''

    def __init__(self, kernel = Kernel.rbf(), C = 1.0):
        self.kernel = kernel
        self.C = C

    def fit(self, X, y, gamma=None):
        '''
        Metoda budująca klasyfikator na podstawie danych X i odpowiadającym im wartościom y
        :param X: macierz n x m, gdzie n to ilość próbek w zbiorze, a m to liczba atrybutów pojedynczej próbki
        :param y: wektor n elementowy, o wartościach y_i = {-1, 1} 
        '''
        self.gamma = 1/X.shape[1] if gamma == None else gamma

        lagrange_multipliers = self.get_lagrange_multipliers(X, y)

        support_vectors_indices = ((self.C > lagrange_multipliers) & (lagrange_multipliers > SUPPORT_VECTOR_MULTIPLIER_THRESHOLD))
        
        svm_multipliers = np.zeros(lagrange_multipliers.shape)
        svm_vectors = np.zeros(X.shape)
        svm_labels = np.zeros(y.shape)

        svm_multipliers[support_vectors_indices] = lagrange_multipliers[support_vectors_indices]
        svm_vectors[support_vectors_indices] = X[support_vectors_indices]
        svm_labels[support_vectors_indices] = y[support_vectors_indices]


        bias = np.mean(
            svm_labels - SVM_NonLinear_Classifier(
                weights=svm_multipliers,
                vectors=svm_vectors,
                labels=svm_labels,
                bias=0.0,
                kernel=self.kernel
            ).predict(svm_vectors)
        )

        # print(svm_multipliers.shape)
        # print(svm_vectors.shape)
        # print(svm_labels.shape)

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

        # print("K start")
        # X_norm = np.linalg.norm(X, axis=-1)
        K = rbf_kernel(X)
        # for i, x_i in enumerate(X):
        #     for j, x_j in enumerate(X):
        #         K[i, j] = self.kernel(x_i, x_j)
        #     print(K[i, j])
        # print("K end")
        # print(K.shape)
        # print(np.mean(K), np.min(K), np.max(K))

        # P = cvxopt.matrix(np.outer(y, y)*K)
        # q = cvxopt.matrix(-np.ones((n_samples,1)))
        # G = cvxopt.matrix(np.concatenate((np.eye(n_samples), -np.eye(n_samples))))
        # h = cvxopt.matrix(np.concatenate((self.C * np.ones((n_samples, 1)), np.zeros((n_samples, 1)))))
        # b = cvxopt.matrix(0.0)
        # A = cvxopt.matrix(y.reshape(1, n_samples).astype(np.double))

        P = cvxopt.matrix(np.outer(y, y)*K)
        q = cvxopt.matrix(-np.ones((n_samples,1)))
        G = cvxopt.matrix(np.concatenate((np.eye(n_samples), -np.eye(n_samples))))
        h = cvxopt.matrix(np.concatenate((self.C * np.ones((n_samples, 1)), np.zeros((n_samples, 1)))))
        b = cvxopt.matrix(0.0)
        A = cvxopt.matrix(y.reshape(1, -1).astype(np.double))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)['x']

        return np.ravel(solution)


class SVM_NonLinear_Classifier(object):

    def __init__(self, weights, vectors, labels, bias, kernel):
        self.kernel = kernel
        self.weights = weights
        self.vectors = vectors
        self.labels = labels
        self.bias = bias

    def predict(self, X):
        print('predict fun przed obliczeniami, długość przekazanego X: ', X.shape)
        print('DłUGOŚC SELF.weights: ', self.weights.shape)
        print('DłUGOŚC SELF.LABELS: ', self.labels.shape)
        a = rbf_kernel( self.vectors, X)
        # print(a)
        b = np.matmul(a, self.weights)


        dupa = np.matmul(self.weights.reshape(-1, 1), self.labels.reshape(-1, 1).T)
        # print('DłUGOŚC a: ', a.shape, a)
        # print('DłUGOŚC dupa = weights x labels: ', dupa.shape)
        c = np.matmul(b.reshape(-1, 1), self.labels.reshape(1, -1))
        # c = np.matmul(dupa, a)

        result = np.sum(c, axis=0) + self.bias
        # print('parametry predict: a,b,c: ', a.shape, b.shape, c.shape)
        print('result długość: ', result.shape)
        print('predict fun po obliczeniach, długość przekazanego X: ', X.shape)
        return np.sign(result)


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