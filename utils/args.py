import argparse
import numpy as np


def outlier_detection_argparser():
    parser = argparse.ArgumentParser(description='Outlier detection')
    parser.add_argument('--data_dir', help='Directory option', type=str, default='./data/uci_har_outlier_data.csv', dest='data_dir')
    parser.add_argument('--save_dir', help='Directory option', type=str, default='./result',dest='save_dir')
    parser.add_argument('--algorithm', help='Outlier detection algorithms', type=str, default='IF', dest='algorithm')
    parser.add_argument('--imputation', help='Imputation option', type=str, default='KNN', dest='imputation')
    parser.add_argument('--percentile', help='Percentile of outlier', type=float, default=95, dest='percentile')
    
    #Algorithms
    #Hyperparameters for SR
    parser.add_argument('--SR_series_window_size', help='Amp window size', type=int, default=24, dest='SR_series_window_size')
    parser.add_argument('--SR_spectral_window_size', help='Spectral window', type=int, default=24, dest='SR_spectral_window_size')
    parser.add_argument('--SR_score_window_size', help='Score window size(much bigger than period)', type=int, default=100, dest='SR_score_window_size')

    #Hyperparameters for MoG
    parser.add_argument('--MoG_threshold', help='Threshold for MoG', type=float, default=0.5, dest='MoG_threshold')
    parser.add_argument('--MoG_components', help='Number of components for MoG', type=int, default=2, dest='MoG_components')
    parser.add_argument('--MoG_covariance', help='Covariance type for MoG', type=str, default='full', dest='MoG_covariance')
    parser.add_argument('--MoG_max_iter', help='Max iteration of EM algorithm', type=int, default=100, dest='MoG_max_iter')

    #Hyperparameters for LOF
    parser.add_argument('--LOF_neighbors', help='Number of neighbors for LOF', type=int, default=5, dest='LOF_neighbors')
    parser.add_argument('--LOF_algorithm', help='Algorithm for LOF',type=str, default='auto', dest='LOF_algorithm')
    parser.add_argument('--LOF_leaf_size', help='Leaf size for LOF',type=int, default=30, dest='LOF_leaf_size')
    parser.add_argument('--LOF_metric', help='Metric for LOF',type=str, default='minkowski', dest='LOF_metric')

    #Hyperparameters for KDE
    parser.add_argument('--KDE_bandwidth', help='Bandwith for KDE', type=float, default=0.2, dest='KDE_bandwidth')
    parser.add_argument('--KDE_algorithm', help='Algorithm for KDE', type=str, default='auto', dest='KDE_algorithm')
    parser.add_argument('--KDE_kernel', help='Kernel for KDE', type=str, default='gaussian', dest='KDE_kernel')
    parser.add_argument('--KDE_metric', help='Metric for KDE', type=str, default='euclidean', dest='KDE_metric')
    parser.add_argument('--KDE_breadth_first', help='Breadth first', type=bool, default=True, dest='KDE_breadth_first')
    parser.add_argument('--KDE_leaf_size', help='Leaf size for KDE', type=int, default=40, dest='KDE_leaf_size')

    #Hyperparameters for IF
    parser.add_argument('--IF_estimators', help='Number of estimators for IF', type=int, default=100, dest='IF_estimators')
    parser.add_argument('--IF_max_samples', help='Max samples per one estimator', type=str, default='auto', dest='IF_max_samples')
    parser.add_argument('--IF_contatmination', help='Rate of outliers in data', type=str, default='auto', dest='IF_contamination')
    parser.add_argument('--IF_max_features', help='Max columns per one estimator', type=float, default=1.0, dest='IF_max_features')
    parser.add_argument('--IF_bootstrap', help='Boostrap option for IF', type=bool, default=False, dest='IF_bootstrap')
    
    #Imputation Methods
    #Hyperparameters for KNN imputation
    parser.add_argument('--KNN_missing_values', help='Type of values to impute', type=type, default=np.nan, dest='KNN_missing_values')
    parser.add_argument('--KNN_neighbors', help='Number of neighbors for KNN imputation', type=int, default=5, dest='KNN_neighbors')
    parser.add_argument('--KNN_weights', help='Weights of neighbors for KNN imputation', type=str, default='uniform', dest='KNN_weights')
    parser.add_argument('--KNN_metric', help='Metric for KNN imputation', type=str, default='nan_euclidean', dest='KNN_metric')

    #Hyperparameters for statistical imputation
    parser.add_argument('--Stats_missing_values', help='Type of values to impute', type=type, default=np.nan, dest='Stats_missing_values')
    parser.add_argument('--Stats_strategy', help='How to impute', type=str, default='mean', dest='Stats_strategy')

    return parser