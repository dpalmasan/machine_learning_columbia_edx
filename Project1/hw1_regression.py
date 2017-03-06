import numpy as np
import sys

np.set_printoptions(precision=16)

def RidgeRegression(X, y, lam=1, centering=False, standarize=False):
    """
    Applies Ridge Regression algorithm and returns the weights vector
    wRR
    """

    n_features = X.shape[1]

    # Preprocessing
    if centering:
        y = y - np.mean(y)
        xmean = np.mean(X, axis=0)


    if standarize:
        sigma = np.var(X, axis=0)
        X = (X - xmean)//sigma

    I = np.identity(n_features)
    # I[-1, -1] = 0

    A = np.linalg.solve(lam*I + np.dot(np.transpose(X[:, :]), X[:, :]), np.transpose(X[:, :]))
    
    wRR = np.dot(A, y)
    return wRR

def ActiveLearning(X_train, X_test, lam=1, sigma2=1):
    """
    Implements the Active Learning Scheme as seen in class.
    Returns a list of the first 10 chosen vectors in X_test
    """
    X = np.copy(X_train)
    D = np.copy(X_test)
    selected = []
    n_features = X.shape[1]
    I = np.identity(n_features)
    # I[-1, -1] = 0
    for i in xrange(10):
        
        sigma02_max = float('-Inf')
        idx_max = None
        for idx in range(len(D)):
            if idx not in selected:
                x0 = D[idx]
                cov = np.linalg.inv(lam*I + 1.0/sigma2*(np.dot(x0, np.transpose(x0)) + np.dot(np.transpose(X), X)))
                sigma02 = sigma2 + np.dot(np.transpose(x0), np.dot(cov, x0))
                if sigma02 > sigma02_max:
                    sigma02_max = sigma02
                    idx_max = idx

        selected.append(idx_max)
        X = np.concatenate((X, [D[idx_max]]))
    selected = [idx + 1 for idx in selected]
      
    return selected
        
        


if __name__ == '__main__':
    if len(sys.argv) != 6:
        print 'Usage: python hw1_regression.py <lambda> <sigma2> <X_train.csv> <y_train.csv> <X_test.csv>'
        sys.exit(0)

    lam = float(sys.argv[1])
    sigma2 = float(sys.argv[2])
    X_train = np.genfromtxt(sys.argv[3], delimiter=',', skip_header=0, names=None)
    y_train = np.genfromtxt(sys.argv[4], delimiter=',', skip_header=0, names=None)
    X_test = np.genfromtxt(sys.argv[5], delimiter=',', skip_header=0, names=None)
    wRR = RidgeRegression(X_train, y_train, lam)
    active =  ActiveLearning(X_train, X_test, lam, sigma2)

    with open("wRR_" + sys.argv[1] + ".csv", "w") as text_file:
        for w in wRR:
            text_file.write(str(w) + "\n")

    with open("active_" + sys.argv[1] + "_" + sys.argv[2] + ".csv", "w") as text_file:
        text_file.write(("%d") % active[0])
        for idx in range(1, 10):
            text_file.write((",%d") % active[idx])
