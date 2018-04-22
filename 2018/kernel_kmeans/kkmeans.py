# -*- coding: utf-8 -*-
import numpy as np

class KernelKMeans(object):

    def __init__(self, gram_mat, n_clusters=2, max_iter=500):

        self.gram_mat = gram_mat
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.y = np.random.randint(low=0, high=self.n_clusters, size=self.gram_mat.shape[0])

        for _ in range(self.max_iter):

            err = np.zeros((self.n_clusters, self.gram_mat.shape[0]))
            numc = np.bincount(self.y)

            for c in range(self.n_clusters):
                err[c,:] -= 2*np.sum(self.gram_mat[:, self.y == c], axis=1) / numc[c]
                err[c,:] += np.sum(self.gram_mat[self.y == c][:,self.y == c]) / (numc[c] ** 2)

            self.y = np.argmin(err, axis=0)

    def __call__(self):
        return self.y



class CovMat(object):

    def __init__(self, size):
        self.c = 0
        self.cov_mat = np.zeros((size,size))

    def add(self, vec):
        gram_mat = np.matrix(vec).T * np.matrix(vec)
        self.cov_mat += gram_mat
        self.c += 1

    def __call__(self):
        return self.cov_mat / float(c)


        



import matplotlib.pyplot as plt


def make_dataset(N):
    X = X = np.zeros((N, 2))
    X[: N / 2, 0] = 10 * np.cos(np.linspace(0.2 * np.pi, N / 2, num=N / 2))
    X[N / 2:, 0] = np.random.randn(N / 2)
    X[: N / 2, 1] = 10 * np.sin(np.linspace(0.2 * np.pi, N / 2, num=N / 2))
    X[N / 2:, 1] = np.random.randn(N / 2)
    return X


if __name__ == '__main__':

    X = make_dataset(100)

    gram_mat = np.zeros((100,100))

    for i in range(100):
        #gram_mat[i] = np.sum(X[i]*X, axis=1)
        gram_mat[i] = np.exp(-0.1*np.linalg.norm(X[i]-X, axis=1))
        #gram_mat[i] = np.random.rand(100)


    kk = KernelKMeans(gram_mat, n_clusters=2, max_iter=2000)
    y_linear = kk()


    plt.subplot(111)
    plt.scatter(X[y_linear == 0][:, 0], X[y_linear == 0][:, 1], c="blue")
    plt.scatter(X[y_linear == 1][:, 0], X[y_linear == 1][:, 1], c="red")
    plt.axis("scaled")

    plt.show()


