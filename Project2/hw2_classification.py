import sys
import numpy as np

def load_data(input_file, is_int=False):
    """
    Loads the dataset. It assumes a *.csv file without header, and the output variable
    in the last column 
    """
    if is_int:
        data = np.genfromtxt(input_file, delimiter=',', skip_header=0, names=None, dtype=int)
    else:
        data = np.genfromtxt(input_file, delimiter=',', skip_header=0, names=None)
    return data

def gauss(mu, cov, x):
    """
    Computes gaussian parametrized by mu and cov, given x. Make sure
    x dimensions are of correct size
    """
    d = len(x)
    den = np.sqrt(np.linalg.det(cov))*(2*np.pi)**(0.5*d)
    num = np.exp(-0.5 * np.dot(x - mu, np.linalg.solve(cov, np.transpose(x - mu))))
    return num/den



if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'Usage: python hw2_classification.py <X_train.csv> <y_train.csv> <X_test.csv>'
        sys.exit(0)

    # Read data
    X_train = load_data(sys.argv[1])
    y_train = load_data(sys.argv[2], True)
    X_test = load_data(sys.argv[3])

    # Class priors from data
    N = len(y_train)
    pi = dict()
    for i in y_train:
        pi[i] = pi.get(i, 0) + 1

    for key in pi.keys():
        pi[key] = pi[key]/float(N)

    n_classes = len(set(y_train))
    n_dim = X_train.shape[1]
    X_test_prob = []

    # TODO: This can be optimized, since mu and cov need to be computed just once
    for x_0 in X_test:
        prob = []
        for y in set(y_train):
            X_i = X_train[y_train == y, :]
            mu_i = np.mean(X_i, axis=0)
            cov_i = np.cov(X_i, rowvar=False)
            prob.append(gauss(mu_i, cov_i, x_0) * pi[y])

        # Factor for normalizing posterior
        norm = sum(prob)
        prob = [p / norm for p in prob]
        X_test_prob.append(prob)

    # Writing to output file
    N = X_test.shape[0]
    with open('probs_test.csv', 'w') as outfile:
        for i in xrange(N):
            for j in xrange(n_classes - 1):
                outfile.write(str(X_test_prob[i][j]) + ",")
            outfile.write(str(X_test_prob[i][j+1]) + "\n")

