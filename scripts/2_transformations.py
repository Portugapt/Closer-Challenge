from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

script_dir = Path( __file__ ).parent.absolute()

### Import Dataset
dataset = pd.read_csv(f'{script_dir}/../data/filtered_dataset.csv')
dfInsurance = dataset.copy()

######
## [3] Add Column: Education Degree to Integer
## Application: Addition
######

# Apply lambda function to column
dfInsurance['atr_edu_deg'] = dfInsurance['dsc_edu_deg'].map(lambda x: str(x)[0])
# replace values 'n' from column atr_edu_deg for NaN
dfInsurance['atr_edu_deg'] = dfInsurance['atr_edu_deg'].replace('n', np.NaN)
# convert to numeric column atr_edu_deg
dfInsurance['atr_edu_deg'] = pd.to_numeric(dfInsurance['atr_edu_deg'])

######
## [5] Fill null values with closest neighbors values
## Application: Transformation
######

dfInsurance.reset_index(inplace=True, drop = True)
CONSIDER_COLUMNS = [column for column in dfInsurance.select_dtypes(include=['int64', 'float64']) if column not in ['cod_cust_id']]
X = dfInsurance.loc[:, dfInsurance.columns.isin(CONSIDER_COLUMNS)]
# define imputer
imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
# fit on the dataset
imputer.fit(X)
# transform the dataset
Xtrans = imputer.transform(X)

#imputer_column_names = ['dt_fpy', 'atr_cust_age', 'amt_gms', 'atr_gla', 'amt_cmv', 'rt_cr', 'amt_plob_motor', 'amt_plob_household', 'amt_plob_health', 'amt_plob_life', 'amt_plob_wcomp', 'atr_edu_deg']

dfX = pd.DataFrame(data=Xtrans, columns=CONSIDER_COLUMNS)
dfInsurance.update(dfX)

######
## [12] Data Types: Force columns to certain datatypes 
## Application: Transformation
######

dfInsurance['dt_fpy'] = dfInsurance['dt_fpy'].astype('int64')
dfInsurance['atr_cust_age'] = dfInsurance['atr_cust_age'].astype('int64')
dfInsurance['atr_gla'] = dfInsurance['atr_gla'].astype('int64')
dfInsurance['flg_children'] = dfInsurance['flg_children'].astype('int64')

dfInsurance.to_csv(f'{script_dir}/../data/transformed_dataset.csv', index=False)