import numpy as np
import matplotlib.pyplot as plt


# @data: 1*dim array
# @mean: 1*dim array
# @cov: dim*dim array
# @return: a real number of probability of the data given mean and cov.
def Gaussian(data, mean, cov):
    dim = data.size
    cov_det = np.linalg.det(cov)
    cov_inv = np.linalg.pinv(cov)
    k = np.sqrt(np.power(2*np.pi, dim)*abs(cov_det))
    z = -0.5*np.dot(np.dot((data-mean), cov_inv), (data-mean))
    return (1/k)*np.exp(z)


dim = 2
N = 100
data = np.random.rand(N, dim)*50
K = 5
P = np.array([1/K for i in range(K)])
mean = [[] for k in range(K)]
for k in range(K):
    # the random number is importent!
    # the iteration won't go on if the initial mean points are all the same!
    mean[k] = np.mean(data, axis=0)+np.random.rand()*5
cov = [0]*K
for k in range(K):
    # init the cov by the covariance of all the data
    cov[k] = np.cov(data.T)
threshold = 0.001
likelyhood = 0
likelyhood_old = 1
# gamma is the probability that the nth data belongs to the kth Gaussian.
gamma = np.array([np.zeros(K) for i in range(N)])
while (np.abs(likelyhood-likelyhood_old) > threshold):
    likelyhood_old = likelyhood

    # E step
    for i in range(N):
        P_i = [P[k]*Gaussian(data[i], mean[k], cov[k]) for k in range(K)]
        for k in range(K):
            gamma[i][k] = P_i[k]/np.sum(P_i)

    # M step
    for k in range(K):
        Pk = np.sum([gamma[i][k] for i in range(N)])
        P[k] = Pk/N
        mean[k] = (1.0/Pk)*np.sum([gamma[i][k]*data[i]
                                   for i in range(N)], axis=0)
        data_diff = data-mean[k]
        cov[k] = (1.0/Pk)*np.sum([gamma[i][k]*data_diff[i].reshape(dim, 1)*data_diff[i]
                                  for i in range(N)], axis=0)

    likelyhood = np.sum([np.log(np.sum(
        [P[k] * Gaussian(data[n], mean[k], cov[k]) for k in range(K)])) for n in range(N)])

    # print(mean)
    print(likelyhood)
