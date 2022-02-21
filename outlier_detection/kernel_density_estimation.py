from sklearn.neighbors import KernelDensity
import numpy as np

# Kernel Density Estimation
def KDE(args):

    X_train = args.train_data
    X_test = args.test_data

    kde_model = KernelDensity(kernel=args.KDE_kernel, bandwidth=args.KDE_bandwidth, algorithm=args.KDE_algorithm, metric=args.KDE_metric, breadth_first=args.KDE_breadth_first, leaf_size=args.KDE_leaf_size)
    kde_model.fit(X_train)

    score = - 1.0 * kde_model.score_samples(X_test)
    index_changes = np.where(score > np.percentile(score, args.percentile))[0]

    return index_changes
