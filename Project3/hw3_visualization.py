from hw3_clustering import *
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

X = load_data('X.csv')

# Play with the values of clusters
(c, mu, it) = KMeans(X, 3)

# Visualize
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=c)
plt.scatter(mu[:, 0], mu[:, 1], c="red", marker="x", s=60)
plt.title("K-Means")


# This code was extracted from http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html
# Some modifications were made to adapt the code for my implemented function for EM-GMM
color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])
def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(1, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(min(X[:, 0] - 1), max(X[:, 0]) + 1)
    plt.ylim(min(X[:, 1] - 1), max(X[:, 1]) + 1)
    for mean in means:
        plt.scatter(mean[0], mean[1], c="red", marker="x", s=60)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

plt.figure()
(pi, mu, sigma, phi) = EM_GMM(X, 3)
plot_results(X, np.argmax(phi, axis=1), mu, sigma, 0, "EM-GMM")
plt.show()

