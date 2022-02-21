from sklearn.impute import KNNImputer
import numpy as np

def KNN_imputer(args):

    data = args.nan_data
    imputer = KNNImputer(missing_values=args.KNN_missing_values, n_neighbors=args.KNN_neighbors, weights=args.KNN_weights, metric=args.KNN_metric)

    return imputer.fit_transform(data)

def Stats_imputer(args):

    data = args.nan_data

    if args.Stats_strategy == 'mean':

        num = np.mean(data)

    elif args.Stats_strategy == 'median':

        num = np.median(data)

    elif args.Stats_strategy == 'most_frequent':

        count = np.bincount(data)
        num = np.argmax(count)

    elif args.Stats_strategy == 'zero':

        num = 0

    elif args.Stats_strategy == 'one':

        num = 1

    data[np.isnan(data)] = num

    return data