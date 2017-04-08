import sys
import numpy as np
import matplotlib.pyplot as plt

def load_data(input_file):
    """
    Loads the dataset. It assumes a *.csv file without header, and the output variable
    in the last column 
    """
    data = np.genfromtxt(input_file, delimiter=',', skip_header=0, names=None)
    return data

def getUsersProductsN(data):
    """
    Returns the number of users and products
    """
    return (len(np.unique(data[:, 0])), len(np.unique(data[:, 1])))


def PMF(M, Omega, N1, N2, d=5, sigma2=0.1, lam=2, maxiter=50, plot=False):
    """
    Implements probabilistic matrix factorization
    Input: An incomplete ratings matrix M, as indexed by the set Sigma. Rank d, sigma2, lambda
    and number of iterations.

    Ouptut: N1 user locations, ui in R^d, and N2 object locations vj in R^d
    """

    # Used to store objective function values through iterations
    Ls = []

    # Initialization (each vj is N(0, 1/lam*I)
    vjs = []
    for j in xrange(N2):
        vj = np.random.normal(loc=0.0, scale=1.0/lam, size=d)
        vjs.append(vj)

    I = np.eye(d)
    vjs = np.array(vjs)

    open('objective.csv', 'w').close()
    for it in xrange(maxiter):
        uis = []        
        for i in xrange(N1):
            Omega_ui = Omega[Omega[:, 0] == i, 1]
            V = vjs[Omega_ui]
            Mij = M[i, Omega_ui]    
         
            # Computing ui, we are with the convention that ui in R^d is [u1, ..., u_d]
            ui = np.linalg.solve((lam*sigma2*I + V.T.dot(V)), V.T.dot(Mij)[np.newaxis].T)[:, 0]
            uis.append(ui)
        uis = np.array(uis)
        
        k = 0
        for j in xrange(N2):
            Omega_vj = Omega[Omega[:, 1] == j, 0]
            U = uis[Omega_vj]
            Mij = M[Omega_vj, j]
            vj = np.linalg.solve((lam*sigma2*I + U.T.dot(U)), U.T.dot(Mij)[np.newaxis].T)[:, 0]
            vjs[k] = vj
            k = k + 1
        
        t1 = 0.0
        for i, j in Omega:
             t1 += (M[i, j] - uis[i, :].dot(vjs[j, :]))**2

        t1 = t1/(2*sigma2)

        t2 = 0.0
        for i in xrange(N1):
            t2 += np.linalg.norm(uis[i, :])
        t2 = lam/2.0 * t2
        
        t3 = 0.0
        for j in xrange(N2):
            t3 += np.linalg.norm(vjs[j, :])
        t3 = lam/2.0 * t3
        L = -t1 - t2 - t3

        # Save objective function at iter
        Ls.append(L)
        
        with open('objective.csv', 'a') as f:
            f.write(str(L) + '\n')

        if it == 9 or it == 24 or it == 49:
            with open('U-' + str(it + 1) + '.csv', 'w') as f:
                for u in uis:
                    for i in xrange(len(u) - 1):
                        f.write(str(u[i]) + ',')
                    f.write(str(u[i]) + "\n")

            with open('V-' + str(it + 1) + '.csv', 'w') as f:
                for v in vjs:
                    for j in xrange(len(v) - 1):
                        f.write(str(v[j]) + ',')
                    f.write(str(v[j]) + "\n")

    if plot:
        plt.plot(xrange(maxiter), Ls)
        plt.ylabel('Objective')
        plt.xlabel('iterations')
        plt.show()
        
        
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: python hw4_PMF.py <ratings.csv>'
        sys.exit(0)

    # Loading data
    data = load_data(sys.argv[1])

    # Python indexing is from 0 to N - 1, recall this for the implementation
    Omega = data[:, (0, 1)].astype(int) - 1
    (Nu, Nv) = getUsersProductsN(data)
    M = np.zeros((Nu, Nv))
    M[Omega[:, 0], Omega[:, 1]] = data[:, 2]
    PMF(M, Omega, Nu, Nv, plot=True)
    
