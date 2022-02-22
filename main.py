import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import pickle

from utils.args import outlier_detection_argparser
# !pip install sranodec
from outlier_detection.spectral_residual import SR
from outlier_detection.local_outlier_factor import LOF
from outlier_detection.mixture_of_gaussian import MoG
from outlier_detection.kernel_density_estimation import KDE
from outlier_detection.isolation_forest import IF
from imputation.imputer import KNN_imputer, Stats_imputer

import argparse
import json
import warnings
warnings.filterwarnings("ignore")

def main():
    args.save_name = f'{args.algorithm}_{args.imputation}_{args.percentile}'
    
    data = pd.read_csv(args.data_dir).iloc[:,:-1] #[10299, 562] / 1028(outliers)
    columns_list = list(data.columns)
    
    replaced_data = pd.DataFrame()
    
    col_index = []
    row_index = []
    
    for col in tqdm(range(data.shape[1])):
        data_col = data.iloc[:,col]
        data_col = data_col.values.reshape(-1,1)
           
        args.train_data = data_col
        args.test_data = data_col
        
        '''
        Outlier Detection
        '''
        if args.algorithm == 'SR':
            indexes = SR(args)
        elif args.algorithm == 'LOF':
            indexes = LOF(args)
        elif args.algorithm == 'MoG':
            indexes = MoG(args)
        elif args.algorithm == 'KDE':
            indexes = KDE(args)
        elif args.algorithm == 'IF':
            indexes = IF(args)
    
        if args.imputation == None or indexes == []:
            pass
        else:
            data_col[indexes]=np.nan
            args.nan_data = data_col

        '''
        Imputation
        '''    
        if args.imputation == 'KNN':
            data_col = KNN_imputer(args)
        elif args.imputation == 'Stats':
            data_col = Stats_imputer(args)
        
        row_index.extend(indexes)
        col_index.extend([col]*len(indexes))
                
        data_col = pd.DataFrame(data_col, columns=[columns_list[col]])       
        replaced_data = pd.concat([replaced_data, data_col], axis=1)
        
    index_list = [[idx, col] for idx, col in zip(row_index, col_index)]
    
    replaced_data.to_csv(os.path.join(args.save_dir, f'{args.save_name}.csv'), index=False)
    with open(os.path.join(args.save_dir, f'{args.save_name}_outlier.pkl'), 'wb') as f:
        pickle.dump(index_list, f)
           
    return

if __name__ == "__main__":
    parser = outlier_detection_argparser()
    args = parser.parse_args()
    
    with open('config.json', 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
            
    main()