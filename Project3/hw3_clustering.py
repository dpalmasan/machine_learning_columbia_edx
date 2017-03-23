##############################################################################################################
# Author: Diego Palma S.
#
# Machine Learning (ColumbiaX, edX) Project 3
#
# Extracted from project description:
# WHAT YOUR PROGRAM OUTPUTS
#
# You should write your K-means and EM-GMM codes to learn 5 clusters. Run both algorithms for 10 iterations. 
# You can initialize your algorithms arbitrarily. We recommend that you initialize the K-means centroids by 
# randomly selecting 5 data points. For the EM-GMM, we also recommend you initialize the mean vectors in the 
# same way, and initialize pi to be the uniform distribution and each Sigma_k to be the identity matrix.  
##############################################################################################################

import sys
import numpy as np


def load_data(input_file):
    """
    Loads the dataset. It assumes a *.csv file without header, and the output variable
    in the last column 
    """
    data = np.genfromtxt(input_file, delimiter=',', skip_header=0, names=None)
    return data

def KMeans(X, K=5, maxit=10, saveLog=True):
    """
    Apply KMeans for clustering a dataset given as input, and the number of clusters (K).
    Input: x1, ..., xn where x in R^d, and K
    Output: Vector c of cluster assignments, and K mean vectors mu
    """
    
    # Sample Size
    N = X.shape[0]

    # Initialize output variables
    c = np.zeros(N)
    mu = X[np.random.choice(N, K, replace=False), :]
    
    for i in xrange(N):
        kmin = 1
        minDist = float('Inf')
        for k in xrange(K):
            dist = np.linalg.norm(X[i, :] - mu[k, :])
            if dist < minDist:
                minDist = dist
                kmin = k

        c[i] = kmin + 1

    
    cNew = np.zeros(N)

    it = 1
    while it <= maxit and not all(c == cNew):
        # Write to output file if required (Project Requirement)
        if saveLog:
            with open('centroids-' + str(it) + '.csv', 'w') as f:
                for mu_i in mu:
                    for j in xrange(len(mu_i) - 1):
                        f.write(str(mu_i[j]) + ',')
                    f.write(str(mu_i[len(mu_i) - 1]) + '\n')

        c = np.copy(cNew)
        for i in xrange(N):
            kmin = 1
            minDist = float('Inf')
            for k in xrange(K):
                dist = np.linalg.norm(X[i, :] - mu[k, :])
                if dist < minDist:
                    minDist = dist
                    kmin = k

            cNew[i] = kmin + 1
        for k in xrange(1, K + 1):
            Xk = X[cNew == k, :]
            mu[k - 1] =  np.sum(Xk, axis=0) / Xk.shape[0]
        it += 1

    return (c, mu, it)


def gauss(mu, cov, x):
    """
    Computes gaussian parametrized by mu and cov, given x. Make sure
    x dimensions are of correct size
    """
    d = len(x)
    den = np.sqrt(np.linalg.det(cov))*(2*np.pi)**(0.5*d)
    num = np.exp(-0.5 * np.dot(x - mu, np.linalg.solve(cov, np.transpose(x - mu))))
    return num/den

# TODO: Define a criteria for convergence (stopping criteria)
def EM_GMM(X, K=5, maxit=10, saveLog=True):
    """
    Algorithm: Maximum Likelihood EM for the Gaussian Mixture Model
    Input: x1, ..., xn, x in R^d
    Output: pi, mu, cov
    """

    N = X.shape[0]
    D = X.shape[1]

    # Initialization
    mu = X[np.random.choice(N, K, replace=False), :]
    pi = [1.0/K for k in xrange(K)]
    sigma = [np.identity(D) for k in xrange(K)]
    
    for it in xrange(maxit):
        # E-Step  
        phi = []
        for i in xrange(N):
            normalization_factor = sum([pi[j]*gauss(mu[j, :], sigma[j], X[i, :]) for j in xrange(K)])
            phi.append(np.array([pi[k]*gauss(mu[k, :], sigma[k], X[i, :])/normalization_factor for k in xrange(K)]))
            

        phi = np.array(phi)

        n = np.sum(phi, axis=0)

        # M-Step
        for k in xrange(K):
            pi[k] = n[k]/N

            mu[k] = np.zeros([1, D])
            for i in xrange(N):
                mu[k] = mu[k] + phi[i, k]*X[i, :]
            mu[k] = mu[k]/n[k]

            prod_sigma = np.zeros([D, D])
            for i in xrange(N):
                xmu = (X[i, :] - mu[k])[np.newaxis]
                prod_sigma = prod_sigma + phi[i, k]*xmu.T.dot(xmu)
            sigma[k] = prod_sigma / n[k]

        sigma = np.array(sigma)

        if saveLog:
            with open('pi-' + str(it + 1) + '.csv', 'w') as f:
                for k in xrange(K - 1):
                    f.write(str(pi[k]) + '\n')
                f.write(str(pi[K - 1]))

            with open('mu-' + str(it + 1) + '.csv', 'w') as f:
                for m in mu:
                    for k in xrange(D-1):
                        f.write(str(m[k]) + ',')
                    f.write(str(m[D-1]) + '\n')

            for k in xrange(K):
                cov_k = sigma[k]
                with open('Sigma-' + str(k + 1) + '-' + str(it + 1) + '.csv', 'w') as f:
                    for i in xrange(cov_k.shape[0]):
                        for j in xrange(cov_k.shape[1] - 1):
                            f.write(str(cov_k[i, j]) + ',')
                        f.write(str(cov_k[i, cov_k.shape[1] - 1]) + '\n')
    return (pi, mu, sigma)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: python hw3_clustering.py <X.csv>'
        sys.exit(0)

    # Load data
    X = load_data(sys.argv[1])

    # Apply clustering techniques
    (c, mu, it) = KMeans(X)
    (pi, mu, sigma) = EM_GMM(X)
