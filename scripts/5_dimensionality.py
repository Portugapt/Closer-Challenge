import time
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

timestr = time.strftime("%Y%m%d-%H%M%S")
script_dir = Path( __file__ ).parent.absolute()

### Import Dataset
dataset = pd.read_csv(f'{script_dir}/../data/Engineered_dataset.csv')
dfInsurance = dataset.copy()

pca_columns_1 = [column for column in dfInsurance.select_dtypes(include=['int64', 'float64']) \
               if (column not in ['cod_cust_id', 'Clusters_1', 'test_feature']) and \
               (('norm' in column) or \
               ('fe_' in column) or (column in ['flg_children','rt_premiums_year','rt_claims_year','atr_credit_score_proxy']) or \
               ('log_rt_' in column))]

pca = PCA(n_components=6)
X_pca = pca.fit_transform(dfInsurance[pca_columns_1])
dfInsurance[['PCA1_','PCA1_2','PCA1_3','PCA1_4','PCA1_5','PCA1_6']] = X_pca

pca_columns_2 = [column for column in dfInsurance.select_dtypes(include=['int64', 'float64']) \
               if (column not in ['cod_cust_id'])]

pca = PCA(n_components=6)
X_pca = pca.fit_transform(dfInsurance[pca_columns_2])
dfInsurance[['PCA2_','PCA2_2','PCA2_3','PCA2_4','PCA2_5','PCA2_6']] = X_pca

dfInsurance.to_csv(f'{script_dir}/../data/{timestr}_dataset.csv', index=False)