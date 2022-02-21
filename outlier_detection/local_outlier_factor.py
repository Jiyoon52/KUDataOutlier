from sklearn.neighbors import LocalOutlierFactor
import numpy as np

# Local Outlier Factor
def LOF(args):
    
    X_train = args.train_data
    X_test = args.test_data

    lof_model = LocalOutlierFactor(n_neighbors=args.LOF_neighbors, novelty=True, algorithm=args.LOF_algorithm, leaf_size=args.LOF_leaf_size, metric=args.LOF_metric)
    lof_model.fit(X_train)

    score = - 1.0 * lof_model.score_samples(X_test)
    index_changes = np.where(score > np.percentile(score, args.percentile))[0]

    return index_changes
