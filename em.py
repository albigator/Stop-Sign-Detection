import numpy as np
import os
from scipy.stats import multivariate_normal as mvn

def EM(xtrain,K):
    numEl,dim = xtrain.shape 
    log_likelihoods = []
    u = xtrain[np.random.choice(numEl, K), :]
    cov = [60*np.eye(dim)] * K
    
    for i in range(K):
        cov[i]=np.multiply(cov[i],np.random.rand(dim,dim))
    pi = [1./K] * K
    z = np.zeros((numEl, K))    

    while 1:
        for k in range(K):
            tmp = pi[k] * mvn.pdf(xtrain, u[k], cov[k])
            z[:,k]=tmp.reshape((numEl,))

        log_likelihood = np.sum(np.log(np.sum(z, axis = 1)))

        print(log_likelihood)
        if log_likelihood>-60000: break

        log_likelihoods.append(log_likelihood)

        z = (z.T / np.sum(z, axis = 1)).T

        N_ks = np.sum(z, axis = 0)

        for k in range(K):
            u[k] = 1. / N_ks[k] * np.sum(z[:, k] * xtrain.T, axis = 1).T
            diff = np.matrix(xtrain - u[k])
            cov[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(diff.T,  z[:, k]), diff))
            pi[k] = 1. / numEl * N_ks[k]

        if len(log_likelihoods) < 2 : continue

    np.save('weights',pi)
    np.save('sigma',cov)
    np.save('mean',u)

xtrain = np.load('train_set.npy')
EM(xtrain,7)
