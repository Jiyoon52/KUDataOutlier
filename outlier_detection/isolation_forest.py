from sklearn.ensemble import IsolationForest
import numpy as np

# Isolation Forest
def IF(args):

    X_train = args.train_data
    X_test = args.test_data

    if_model = IsolationForest(n_estimators=args.IF_estimators, max_samples=args.IF_max_samples, contamination=args.IF_contamination, max_features=args.IF_max_features, bootstrap=args.IF_bootstrap)
    if_model.fit(X_train)

    score = - 1.0 * if_model.score_samples(X_test)
    index_changes = np.where(score > np.percentile(score, args.percentile))[0]

    return index_changes
