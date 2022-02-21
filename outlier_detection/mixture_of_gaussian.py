from sklearn.mixture import GaussianMixture
import numpy as np

# Mixture of Gaussian
def MoG(args):

    X_train = args.train_data
    X_test = args.test_data

    gm = GaussianMixture(n_components=args.MoG_components, covariance_type=args.MoG_covariance, max_iter=args.MoG_max_iter, random_state=0).fit(X_train)
    score = -1.0* gm.predict_proba(X_test)
    index_changes = np.where(score[:,0] > np.percentile(score[:,0], args.percentile))[0]

    return index_changes
