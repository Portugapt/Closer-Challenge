### Read Libraries
import time
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

timestr = time.strftime("%Y%m%d-%H%M%S")

### Import Dataset
dataset = pd.read_excel('./data/dataset.xlsx')
dfInsurance = dataset.copy()

columns_map = {"Customer Identity":"cod_cust_id",
                "First Policy´s Year": "dt_fpy",
                "Customer Age": "atr_cust_age",
                "Educational Degree":"dsc_edu_deg",
                "Gross Monthly Salary": "amt_gms",
                "Geographic Living Area": "atr_gla",
                "Has Children (Y=1)":"flg_children",
                "Customer Monetary Value":"amt_cmv",
                "Claims Rate":"rt_cr",
                "Premiums in LOB: Motor":"amt_plob_motor",
                "Premiums in LOB: Household":"amt_plob_household",
                "Premiums in LOB: Health":"amt_plob_health",
                "Premiums in LOB:  Life":"amt_plob_life",
                "Premiums in LOB: Work Compensations":"amt_plob_wcomp"}

columns_map_reverse = {v: k for k, v in columns_map.items()}

dfInsurance = dfInsurance.rename(columns=columns_map)


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
## [4] Drop null values from children column
## Application: Transformation
######
dfInsurance.dropna(subset=['flg_children'], inplace=True)

######
## [5] Fill null values with closest neighbors values
## Application: Transformation
######

dfInsurance.reset_index(inplace=True, drop = True)

X = dfInsurance.loc[:, ~dfInsurance.columns.isin(['cod_cust_id', 'dsc_edu_deg', 'flg_children', 'Set'])]

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
## [6] Add Outliers Columns
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
    if column not in('cod_cust_id', 'dt_fpy', 'atr_cust_age', 'dsc_edu_deg', 'atr_gla', 'flg_children', 'Set'):
        
        lim_sup, lim_inf = interquartile_range(dfInsurance[column])

        dfInsurance['outlier_candidate'] = np.where((dfInsurance[column] > lim_sup) | (dfInsurance[column] < lim_inf), dfInsurance['outlier_candidate'].astype(str) +'%' + column, dfInsurance['outlier_candidate'])

dfInsurance['outlier_candidate'] = dfInsurance['outlier_candidate'].apply(lambda x: x.lstrip('%'))

######
## [7] Add column: Total amount of premiums
## Application: Addition
######

dfInsurance['amt_premium_total'] = (dfInsurance['amt_plob_life'] + dfInsurance['amt_plob_household'] + dfInsurance['amt_plob_motor'] + 
                                    dfInsurance['amt_plob_health']+ dfInsurance['amt_plob_wcomp'])


######
## [0] Data main cut, KEEP is usable data, and THROW is data we found is impossible to work with
## Application: Filter
######

dfInsurance['DATA_MAIN_CUT'] = np.where(((dfInsurance['dt_fpy'] > 2022) | 
                        (dfInsurance['atr_cust_age'] > 100) | 
                        (dfInsurance['amt_cmv'] < -160000) |
                        (dfInsurance['amt_gms']> 15000) |
                        (dfInsurance['rt_cr'] > 2) |
                        (dfInsurance['amt_plob_motor'] > 2000)|
                        (dfInsurance['amt_plob_household'] > 5000)|
                        (dfInsurance['amt_plob_health'] > 5000)|
                        (dfInsurance['amt_plob_wcomp'] > 500)|
                        (dfInsurance['amt_premium_total'] > 5000)), 'THROW','KEEP')

######
## [8] Add column: First Policy Year to year of reference (1999)
## Application: Addition
######

dfInsurance['atr_fpy_to_date'] = pd.Series(1999 - dfInsurance['dt_fpy'], dtype="Int32")

######
## [9] Add columns: Rate of LOB Premium to Total Premium
## Application: Addition
######

dfInsurance['rt_plob_life'] = (dfInsurance['amt_plob_life'] / dfInsurance['amt_premium_total'])
dfInsurance['rt_plob_household'] = (dfInsurance['amt_plob_household'] / dfInsurance['amt_premium_total'])
dfInsurance['rt_plob_motor'] = (dfInsurance['amt_plob_motor'] / dfInsurance['amt_premium_total'])
dfInsurance['rt_plob_health'] = (dfInsurance['amt_plob_health'] / dfInsurance['amt_premium_total'])
dfInsurance['rt_plob_wcomp'] = (dfInsurance['amt_plob_wcomp'] / dfInsurance['amt_premium_total'])

######
## [10] Add columns: Create LOB Premium Classes in function of Total Premium
## Application: Addition
## Refer to EDA Notebook on why these were the defined classes
######

col = 'amt_plob_motor'
conditions_motor = [dfInsurance[col] > 404, 
              (dfInsurance[col] > 225) & (dfInsurance[col] <= 404), 
              (dfInsurance[col] > 168) & (dfInsurance[col] <= 225),
              (dfInsurance[col] > 90) & (dfInsurance[col] <= 168),
              (dfInsurance[col] > 2.40) & (dfInsurance[col] <= 90),
              (dfInsurance[col] > 0) & (dfInsurance[col] <= 2.40),
              (dfInsurance[col] == 0),
              (dfInsurance[col] < 0)]
choices = [ "A", 'B', 'C', 'D', 'E', 'F', 'ZEROS','NEGATIVES']

dfInsurance["fe_bin_plob_motor"] = np.select(conditions_motor, choices, default=np.nan)

col = 'amt_plob_life'
conditions_life = [dfInsurance[col] > 131.5, 
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

dfInsurance["fe_bin_plob_life"] = np.select(conditions_life, choices, default=np.nan)

col = 'amt_plob_health'
conditions_health = [dfInsurance[col] > 156, 
              (dfInsurance[col] > 95.3) & (dfInsurance[col] <= 156), 
              (dfInsurance[col] > 47.2) & (dfInsurance[col] <= 95.3),
              (dfInsurance[col] > 0) & (dfInsurance[col] <= 47.2),
              (dfInsurance[col] == 0),
              (dfInsurance[col] < 0)]
choices = ['A', 'B', 'C', 'D', 'ZEROS','NEGATIVES']

dfInsurance["fe_bin_plob_health"] = np.select(conditions_health, choices, default=np.nan)

col = 'amt_plob_wcomp'
conditions_wcomp = [dfInsurance[col] > 69, 
              (dfInsurance[col] > 44.68) & (dfInsurance[col] <= 69), 
              (dfInsurance[col] > 22.11) & (dfInsurance[col] <= 44.68),
              (dfInsurance[col] > 6.33) & (dfInsurance[col] <= 22.11),
              (dfInsurance[col] > 0) & (dfInsurance[col] <= 6.33),
              (dfInsurance[col] == 0),
              (dfInsurance[col] < 0)]
choices = ['A', 'B', 'C', 'D', 'E', 'ZEROS','NEGATIVES']

dfInsurance["fe_bin_plob_wcomp"] = np.select(conditions_wcomp, choices, default=np.nan)

col = 'amt_plob_household'
conditions_household = [dfInsurance[col] > 529.8, 
              (dfInsurance[col] > 368.1) & (dfInsurance[col] <= 529.8), 
              (dfInsurance[col] > 255.3) & (dfInsurance[col] <= 368.1),
              (dfInsurance[col] > 157.5) & (dfInsurance[col] <= 255.3),
              (dfInsurance[col] > 88.6) & (dfInsurance[col] <= 157.5),
              (dfInsurance[col] > 19.2) & (dfInsurance[col] <= 88.6),
              (dfInsurance[col] > 0) & (dfInsurance[col] <= 19.2),
              (dfInsurance[col] == 0),
              (dfInsurance[col] < 0)]
choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'ZEROS','NEGATIVES']

dfInsurance["fe_bin_plob_household"] = np.select(conditions_household, choices, default=np.nan)

######
## [11] Add Columns: Quadrants on Customer Monetary Value and Claims Rate 
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


######
## [12] Data Types: Force columns to certain datatypes 
## Application: Transformation
######

dfInsurance['dt_fpy'] = dfInsurance['dt_fpy'].astype('int64')
dfInsurance['atr_cust_age'] = dfInsurance['atr_cust_age'].astype('int64')
dfInsurance['atr_gla'] = dfInsurance['atr_gla'].astype('int64')
dfInsurance['flg_children'] = dfInsurance['flg_children'].astype('int64')

######
## [13] Scale Tree Binned data
## Application: Addition
######
# https://stackoverflow.com/questions/5294955/how-to-scale-down-a-range-of-numbers-with-a-known-min-and-max-value
# https://stackoverflow.com/questions/67656988/group-by-minmaxscaler-in-pandas-dataframe

## Motor
choices = [6, 5, 4, 3, 2, 1, 0, -1]
dfInsurance["fe_int_plob_motor_b"] = np.select(conditions_motor, choices, default=np.nan)
dfInsurance["fe_int_plob_motor_a"] = dfInsurance["fe_int_plob_motor_b"]-1

g = dfInsurance.groupby(['DATA_MAIN_CUT', 'fe_bin_plob_motor'])['amt_plob_motor']
min_, max_ = g.transform('min'), g.transform('max')
dfInsurance['fe_amt_plob_motor' + '_scale'] = round((((dfInsurance["fe_int_plob_motor_b"]-dfInsurance["fe_int_plob_motor_a"])*(dfInsurance['amt_plob_motor'] - min_)) / 
                                                (max_ - min_)) + dfInsurance["fe_int_plob_motor_a"],5)
dfInsurance.drop(columns = ['fe_int_plob_motor_a', 'fe_int_plob_motor_b'], inplace=True)
dfInsurance['fe_amt_plob_motor' + '_scale'] = dfInsurance['fe_amt_plob_motor' + '_scale'].fillna(0)

## Life
choices = [8,7,6, 5, 4, 3, 2, 1, 0, -1]
dfInsurance["fe_int_plob_life_b"] = np.select(conditions_life, choices, default=np.nan)
dfInsurance["fe_int_plob_life_a"] = dfInsurance["fe_int_plob_life_b"]-1

g = dfInsurance.groupby(['DATA_MAIN_CUT', 'fe_bin_plob_life'])['amt_plob_life']
min_, max_ = g.transform('min'), g.transform('max')
dfInsurance['fe_amt_plob_life' + '_scale'] = round((((dfInsurance["fe_int_plob_life_b"]-dfInsurance["fe_int_plob_life_a"])*(dfInsurance['amt_plob_life'] - min_)) / 
                                                (max_ - min_)) + dfInsurance["fe_int_plob_life_a"],5)
dfInsurance.drop(columns = ['fe_int_plob_life_a', 'fe_int_plob_life_b'], inplace=True)
dfInsurance['fe_amt_plob_life' + '_scale'] = dfInsurance['fe_amt_plob_life' + '_scale'].fillna(0)

## Health
choices = [4, 3, 2, 1, 0, -1]
dfInsurance["fe_int_plob_health_b"] = np.select(conditions_health, choices, default=np.nan)
dfInsurance["fe_int_plob_health_a"] = dfInsurance["fe_int_plob_health_b"]-1

g = dfInsurance.groupby(['DATA_MAIN_CUT', 'fe_bin_plob_health'])['amt_plob_health']
min_, max_ = g.transform('min'), g.transform('max')
dfInsurance['fe_amt_plob_health' + '_scale'] = round((((dfInsurance["fe_int_plob_health_b"]-dfInsurance["fe_int_plob_health_a"])*(dfInsurance['amt_plob_health'] - min_)) / 
                                                (max_ - min_)) + dfInsurance["fe_int_plob_health_a"],5)
dfInsurance.drop(columns = ['fe_int_plob_health_a', 'fe_int_plob_health_b'], inplace=True)
dfInsurance['fe_amt_plob_health' + '_scale'] = dfInsurance['fe_amt_plob_health' + '_scale'].fillna(0)

## Work Compensation
choices = [5, 4, 3, 2, 1, 0, -1]
dfInsurance["fe_int_plob_wcomp_b"] = np.select(conditions_wcomp, choices, default=np.nan)
dfInsurance["fe_int_plob_wcomp_a"] = dfInsurance["fe_int_plob_wcomp_b"]-1

g = dfInsurance.groupby(['DATA_MAIN_CUT', 'fe_bin_plob_wcomp'])['amt_plob_wcomp']
min_, max_ = g.transform('min'), g.transform('max')
dfInsurance['fe_amt_plob_wcomp' + '_scale'] = round((((dfInsurance["fe_int_plob_wcomp_b"]-dfInsurance["fe_int_plob_wcomp_a"])*(dfInsurance['amt_plob_wcomp'] - min_)) / 
                                                (max_ - min_)) + dfInsurance["fe_int_plob_wcomp_a"],5)
dfInsurance.drop(columns = ['fe_int_plob_wcomp_a', 'fe_int_plob_wcomp_b'], inplace=True)
dfInsurance['fe_amt_plob_wcomp' + '_scale'] = dfInsurance['fe_amt_plob_wcomp' + '_scale'].fillna(0)

## Household
choices = [7, 6, 5, 4, 3, 2, 1, 0, -1]
dfInsurance["fe_int_plob_household_b"] = np.select(conditions_household, choices, default=np.nan)
dfInsurance["fe_int_plob_household_a"] = dfInsurance["fe_int_plob_household_b"]-1

g = dfInsurance.groupby(['DATA_MAIN_CUT', 'fe_bin_plob_household'])['amt_plob_household']
min_, max_ = g.transform('min'), g.transform('max')
dfInsurance['fe_amt_plob_household' + '_scale'] = round((((dfInsurance["fe_int_plob_household_b"]-dfInsurance["fe_int_plob_household_a"])*(dfInsurance['amt_plob_household'] - min_)) / 
                                                (max_ - min_)) + dfInsurance["fe_int_plob_household_a"],5)
dfInsurance.drop(columns = ['fe_int_plob_household_a', 'fe_int_plob_household_b'], inplace=True)
dfInsurance['fe_amt_plob_household' + '_scale'] = dfInsurance['fe_amt_plob_household' + '_scale'].fillna(0)

######
## [13] Scale Features to logarithm
## Application: Addition
######

dfInsurance['log_amt_plob_life'] = np.where(min(dfInsurance['amt_plob_life']) <= 0,
np.log(dfInsurance['amt_plob_life'] + abs(min(dfInsurance['amt_plob_life'])) + 1),
np.log(abs(dfInsurance['amt_plob_life'])))

dfInsurance['log_amt_plob_household'] = np.where(min(dfInsurance['amt_plob_household']) <= 0,
np.log(dfInsurance['amt_plob_household'] + abs(min(dfInsurance['amt_plob_household'])) + 1),
np.log(abs(dfInsurance['amt_plob_household'])))

dfInsurance['log_amt_plob_wcomp'] = np.where(min(dfInsurance['amt_plob_wcomp']) <= 0,
np.log(dfInsurance['amt_plob_wcomp'] + abs(min(dfInsurance['amt_plob_wcomp'])) + 1),
np.log(abs(dfInsurance['amt_plob_wcomp'])))

dfInsurance['log_amt_premium_total'] = np.where(min(dfInsurance['amt_premium_total']) <= 0,
np.log(dfInsurance['amt_premium_total'] + abs(min(dfInsurance['amt_premium_total'])) + 1),
np.log(abs(dfInsurance['amt_premium_total'])))

dfInsurance['log_rt_plob_life'] = np.where(min(dfInsurance['rt_plob_life']) <= 0,
np.log(dfInsurance['rt_plob_life'] + abs(min(dfInsurance['rt_plob_life'])) + 1),
np.log(abs(dfInsurance['rt_plob_life'])))

dfInsurance['log_rt_plob_household'] = np.where(min(dfInsurance['rt_plob_household']) <= 0,
np.log(dfInsurance['rt_plob_household'] + abs(min(dfInsurance['rt_plob_household'])) + 1),
np.log(abs(dfInsurance['rt_plob_household'])))

dfInsurance['log_rt_plob_motor'] = np.where(min(dfInsurance['rt_plob_motor']) <= 0,
np.log(dfInsurance['rt_plob_motor'] + abs(min(dfInsurance['rt_plob_motor'])) + 1),
np.log(abs(dfInsurance['rt_plob_motor'])))

dfInsurance['log_rt_plob_health'] = np.where(min(dfInsurance['rt_plob_health']) <= 0,
np.log(dfInsurance['rt_plob_health'] + abs(min(dfInsurance['rt_plob_health'])) + 1),
np.log(abs(dfInsurance['rt_plob_health'])))

dfInsurance['log_rt_plob_wcomp'] = np.where(min(dfInsurance['rt_plob_wcomp']) <= 0,
np.log(dfInsurance['rt_plob_wcomp'] + abs(min(dfInsurance['rt_plob_wcomp'])) + 1),
np.log(abs(dfInsurance['rt_plob_wcomp'])))

######
## [14] Scale Features with square root
## Application: Addition
######

dfInsurance['sqrt_amt_cmv'] = np.where(min(dfInsurance['amt_cmv']) <= 0, 
                                             np.sqrt(dfInsurance['amt_cmv'] + abs(min(dfInsurance['amt_cmv'])) + 1), 
                                             np.sqrt(dfInsurance['amt_cmv']))

######
## [15] Scale Features with MinMax
## Application: Addition
######


######
## [16] Scale Features with MinMax
## Application: Addition
######

dfInsurance['sqrt_amt_cmv'] = np.where(min(dfInsurance['amt_cmv']) <= 0,
np.sqrt(dfInsurance['amt_cmv'] + abs(min(dfInsurance['amt_cmv'])) + 1),
np.sqrt(dfInsurance['amt_cmv']))


######
## [17] Customer Claims Cost
## Application: Addition
######

dfInsurance['amt_gys'] = dfInsurance['amt_gms'] * 12

######
## [18] Customer Yearly Salary 
## Application: Addition
######

dfInsurance['amt_claims_total'] = dfInsurance['rt_cr'] * dfInsurance['amt_premium_total']

######
## [19] Customer Effort Rate - Cost of premiums in relation to customer income
## Application: Addition
######

dfInsurance['rt_premiums_year'] = dfInsurance['amt_premium_total'] / dfInsurance['amt_gys']
dfInsurance['rt_claims_year'] = dfInsurance['amt_claims_total'] / dfInsurance['amt_gys']


dfInsurance.to_csv(f'./data/{timestr}_dataset.csv', index=False)