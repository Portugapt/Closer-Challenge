### Read Libraries
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

### Import Dataset
dataset = pd.read_excel('../data/dataset.xlsx')
dfInsurance = dataset.copy()

np.random.seed(42)
if "Set" not in dfInsurance.columns:
    dfInsurance["Set"] = np.random.choice(["train", "test"], p =[.7, .3], size=(dfInsurance.shape[0],))

######
## [1] Remove Duplicates 
## Application: Filter
######

#Drops duplicates except for the first occurrence
dfInsurance = dfInsurance.drop_duplicates(subset=dfInsurance.columns.difference(['cod_cust_id']))

dfInsurance.duplicated(subset=dfInsurance.columns.difference(['cod_cust_id'])).sum()

######
## [2] Remove Impossible Values
## Application: Filter
######

dfInsurance = dfInsurance.loc[~((dfInsurance['dt_fpy'] > 2022) | 
                        (dfInsurance['atr_cust_age'] > 100))]

######
## [3] Add Outliers Columns
## Application: Addition
######


def interquartile_range(column):
    """Checks if data is 1.5 times the interquartile range greater than the third quartile (Q3) 
    or 1.5 times the interquartile range less than the first quartile (Q1)
    
    param column: a series or column from a dataset containing numerical data 
    output: returns the superior and inferior values where observations are below or above 1.5 times the interquartile range 
    """
    distance = 1.5 * (np.nanpercentile(column, 75) - np.nanpercentile(column, 25))
    lim_sup= distance + np.nanpercentile(column, 90)
    lim_inf= np.nanpercentile(column, 10) - distance
    
    return lim_sup, lim_inf

dfInsurance['outlier_candidate'] = ''
for column in dfInsurance.columns[1:-2]:
    if column not in('cod_cust_id', 'dt_fpy', 'atr_cust_age', 'dsc_edu_deg', 'atr_gla', 'flg_children'):
        
        lim_sup, lim_inf = interquartile_range(dfInsurance[column])

        dfInsurance['outlier_candidate'] = np.where((dfInsurance[column] > lim_sup) | (dfInsurance[column] < lim_inf), dfInsurance['outlier_candidate'].astype(str) +'%' + column, dfInsurance['outlier_candidate'])

dfInsurance['outlier_candidate'] = dfInsurance['outlier_candidate'].apply(lambda x: x.lstrip('%'))

######
## [4] Add Column: Education Degree to Integer
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

X = dfInsurance.loc[:, ~dfInsurance.columns.isin(['cod_cust_id', 'dsc_edu_deg', 'flg_children'])]

# define imputer
imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')

# fit on the dataset
imputer.fit(X)

# transform the dataset
Xtrans = imputer.transform(X)

imputer_column_names = ['dt_fpy', 'atr_cust_age', 'amt_gms', 'atr_gla', 'amt_cmv', 'rt_cr', 'amt_plob_motor', 'amt_plob_household', 'amt_plob_health', 'amt_plob_life', 'amt_plob_wcomp', 'atr_edu_deg']

dfX = pd.DataFrame(data=Xtrans, columns=imputer_column_names)
dfInsurance.update(dfX)

######
## [6] Add column: Total amount of premiums
## Application: Addition
######

dfInsurance['amt_premium_total'] = (dfInsurance['amt_plob_life'] + dfInsurance['amt_plob_household'] + dfInsurance['amt_plob_motor'] + 
                                    dfInsurance['amt_plob_health']+ dfInsurance['amt_plob_wcomp'])

######
## [7] Add column: First Policy Year to year of reference (1999)
## Application: Addition
######

dfInsurance['atr_fpy_to_date'] = pd.Series(1999 - dfInsurance['dt_fpy'], dtype="Int32")

######
## [8] Add columns: Rate of LOB Premium to Total Premium
## Application: Addition
######

dfInsurance['rt_plob_life'] = (dfInsurance['amt_plob_life'] / dfInsurance['amt_premium_total'])
dfInsurance['rt_plob_household'] = (dfInsurance['amt_plob_household'] / dfInsurance['amt_premium_total'])
dfInsurance['rt_plob_motor'] = (dfInsurance['amt_plob_motor'] / dfInsurance['amt_premium_total'])
dfInsurance['rt_plob_health'] = (dfInsurance['amt_plob_health'] / dfInsurance['amt_premium_total'])
dfInsurance['rt_plob_wcomp'] = (dfInsurance['amt_plob_wcomp'] / dfInsurance['amt_premium_total'])

######
## [9] Add columns: Create LOB Premium Classes in function of Total Premium
## Application: Addition
## Refer to EDA Notebook on why these were the defined classes
######

col = 'amt_plob_motor'
conditions = [dfInsurance[col] > 404, 
              (dfInsurance[col] > 225) & (dfInsurance[col] <= 404), 
              (dfInsurance[col] > 168) & (dfInsurance[col] <= 225),
              (dfInsurance[col] > 90) & (dfInsurance[col] <= 168),
              (dfInsurance[col] > 2.40) & (dfInsurance[col] <= 90),
              (dfInsurance[col] > 0) & (dfInsurance[col] <= 2.40),
              (dfInsurance[col] == 0),
              (dfInsurance[col] < 0)]
choices = [ "A", 'B', 'C', 'D', 'E', 'F', 'ZEROS','NEGATIVES']

dfInsurance["fe_bin_plob_motor"] = np.select(conditions, choices, default=np.nan)

col = 'amt_plob_life'
conditions = [dfInsurance[col] > 131.5, 
              (dfInsurance[col] > 86.4) & (dfInsurance[col] <= 131.5), 
              (dfInsurance[col] > 58) & (dfInsurance[col] <= 86.4),
              (dfInsurance[col] > 35.84) & (dfInsurance[col] <= 58),
              (dfInsurance[col] > 26.5) & (dfInsurance[col] <= 35.84),
              (dfInsurance[col] > 14.2) & (dfInsurance[col] <= 26.5),
              (dfInsurance[col] > 7.4) & (dfInsurance[col] <= 14.2),
              (dfInsurance[col] > 0) & (dfInsurance[col] <= 7.4),
              (dfInsurance[col] == 0),
              (dfInsurance[col] < 0)]
choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'ZEROS','NEGATIVES']

dfInsurance["fe_bin_plob_life"] = np.select(conditions, choices, default=np.nan)

col = 'amt_plob_health'
conditions = [dfInsurance[col] > 156, 
              (dfInsurance[col] > 95.3) & (dfInsurance[col] <= 156), 
              (dfInsurance[col] > 47.2) & (dfInsurance[col] <= 95.3),
              (dfInsurance[col] > 0) & (dfInsurance[col] <= 47.2),
              (dfInsurance[col] == 0),
              (dfInsurance[col] < 0)]
choices = ['A', 'B', 'C', 'D', 'ZEROS','NEGATIVES']

dfInsurance["fe_bin_plob_health"] = np.select(conditions, choices, default=np.nan)

col = 'amt_plob_wcomp'
conditions = [dfInsurance[col] > 69, 
              (dfInsurance[col] > 44.68) & (dfInsurance[col] <= 69), 
              (dfInsurance[col] > 22.11) & (dfInsurance[col] <= 44.68),
              (dfInsurance[col] > 6.33) & (dfInsurance[col] <= 22.11),
              (dfInsurance[col] > 0) & (dfInsurance[col] <= 6.33),
              (dfInsurance[col] == 0),
              (dfInsurance[col] < 0)]
choices = ['A', 'B', 'C', 'D', 'E', 'ZEROS','NEGATIVES']

dfInsurance["fe_bin_plob_wcomp"] = np.select(conditions, choices, default=np.nan)

col = 'amt_plob_household'
conditions = [dfInsurance[col] > 529.8, 
              (dfInsurance[col] > 368.1) & (dfInsurance[col] <= 529.8), 
              (dfInsurance[col] > 255.3) & (dfInsurance[col] <= 368.1),
              (dfInsurance[col] > 157.5) & (dfInsurance[col] <= 255.3),
              (dfInsurance[col] > 88.6) & (dfInsurance[col] <= 157.5),
              (dfInsurance[col] > 19.2) & (dfInsurance[col] <= 88.6),
              (dfInsurance[col] > 0) & (dfInsurance[col] <= 19.2),
              (dfInsurance[col] == 0),
              (dfInsurance[col] < 0)]
choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'ZEROS','NEGATIVES']

dfInsurance["fe_bin_plob_household"] = np.select(conditions, choices, default=np.nan)

######
## [10] Add Columns: Quadrants on Customer Monetary Value and Claims Rate 
## Application: Addition
## Refer to EDA Notebook on why these were the defined parameters [!]
######

m_slope = -525.5
rt_cut = 1
b_base = 500
conditions = [(dfInsurance['rt_cr'] >= rt_cut) & (dfInsurance['amt_cmv'] > ((m_slope)*dfInsurance['rt_cr']+b_base)), # Q1
              (dfInsurance['rt_cr'] >= rt_cut) & (dfInsurance['amt_cmv'] <= ((m_slope)*dfInsurance['rt_cr']+b_base)), # Q4
              (dfInsurance['rt_cr'] < rt_cut) & (dfInsurance['amt_cmv'] > ((m_slope)*dfInsurance['rt_cr']+b_base)), # Q2
              (dfInsurance['rt_cr'] < rt_cut) & (dfInsurance['amt_cmv'] <= ((m_slope)*dfInsurance['rt_cr']+b_base))] # Q3
choices = ['Q1', 'Q4', 'Q2', 'Q3']

dfInsurance["fe_cmv_cr_quadrant_Type1"] = np.select(conditions, choices, default=np.nan)


m_slope = -970
rt_cut = 0.6
b_base = 1050
conditions = [(dfInsurance['rt_cr'] >= rt_cut) & (dfInsurance['amt_cmv'] > ((m_slope)*dfInsurance['rt_cr']+b_base)), # Q1
              (dfInsurance['rt_cr'] >= rt_cut) & (dfInsurance['amt_cmv'] <= ((m_slope)*dfInsurance['rt_cr']+b_base)), # Q4
              (dfInsurance['rt_cr'] < rt_cut) & (dfInsurance['amt_cmv'] > ((m_slope)*dfInsurance['rt_cr']+b_base)), # Q2
              (dfInsurance['rt_cr'] < rt_cut) & (dfInsurance['amt_cmv'] <= ((m_slope)*dfInsurance['rt_cr']+b_base))] # Q3
choices = ['Q1', 'Q4', 'Q2', 'Q3']

dfInsurance["fe_cmv_cr_quadrant_Type2"] = np.select(conditions, choices, default=np.nan)


