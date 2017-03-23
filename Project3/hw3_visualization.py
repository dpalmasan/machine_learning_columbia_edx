from hw3_clustering import *
import pylab

X = load_data('X.csv')

# Play with the values of clusters
(c, mu, it) = KMeans(X, 3)

# Visualize
pylab.scatter(X[:, 0], X[:, 1], c=c)
pylab.scatter(mu[:, 0], mu[:, 1], c="black", marker="x")
pylab.show()

# TODO: Add EM-GMM visualization

