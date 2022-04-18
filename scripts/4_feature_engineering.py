### Read Libraries
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

script_dir = Path( __file__ ).parent.absolute()

### Import Dataset
dataset = pd.read_csv(f'{script_dir}/../data/features_dataset.csv')
dfInsurance = dataset.copy()

scaler = MinMaxScaler()
sd_scaler = StandardScaler()

######
## LOB: MOTOR
######

## FEATURE 1 - Motor to predict Total Premium (Tree based Binning)
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

choices = [6, 5, 4, 3, 2, 1, 0, -1]
dfInsurance["fe_int_plob_motor_b"] = np.select(conditions_motor, choices, default=np.nan)
dfInsurance["fe_int_plob_motor_a"] = dfInsurance["fe_int_plob_motor_b"]-1

## FEATURE 2 - Scaling of Feature 1
g = dfInsurance.groupby(['fe_bin_plob_motor'])['amt_plob_motor']
min_, max_ = g.transform('min'), g.transform('max')
dfInsurance['fe_amt_plob_motor' + '_scale'] = round((((dfInsurance["fe_int_plob_motor_b"]-dfInsurance["fe_int_plob_motor_a"])*(dfInsurance['amt_plob_motor'] - min_)) / 
                                                (max_ - min_)) + dfInsurance["fe_int_plob_motor_a"],5)
dfInsurance.drop(columns = ['fe_int_plob_motor_a', 'fe_int_plob_motor_b'], inplace=True)
dfInsurance['fe_amt_plob_motor' + '_scale'] = dfInsurance['fe_amt_plob_motor' + '_scale'].fillna(0)

## FEATURE 3 - Motor to predict CMV (Tree based Binning)

col = 'amt_plob_motor'
conditions_motor = [dfInsurance[col] > 543, 
              (dfInsurance[col] > 425) & (dfInsurance[col] <= 543), 
              (dfInsurance[col] > 254) & (dfInsurance[col] <= 425),
              (dfInsurance[col] > 0) & (dfInsurance[col] <= 254),
              (dfInsurance[col] == 0),
              (dfInsurance[col] < 0)]
choices = [ "A", 'B', 'C', 'D', 'ZEROS','NEGATIVES']
dfInsurance["fe_bin_cmv_motor"] = np.select(conditions_motor, choices, default=np.nan)

## FEATURE 4 - Scaling of Feature 3
# https://stackoverflow.com/questions/5294955/how-to-scale-down-a-range-of-numbers-with-a-known-min-and-max-value
# https://stackoverflow.com/questions/67656988/group-by-minmaxscaler-in-pandas-dataframe
choices = [4, 3, 2, 1, 0, -1]
dfInsurance["fe_int_plob_motor_b"] = np.select(conditions_motor, choices, default=np.nan)
dfInsurance["fe_int_plob_motor_a"] = dfInsurance["fe_int_plob_motor_b"]-1
g = dfInsurance.groupby(['fe_bin_cmv_motor'])['amt_plob_motor']
min_, max_ = g.transform('min'), g.transform('max')
dfInsurance['fe_cmv_motor' + '_scale'] = round((((dfInsurance["fe_int_plob_motor_b"]-dfInsurance["fe_int_plob_motor_a"])*(dfInsurance['amt_plob_motor'] - min_)) / 
                                                (max_ - min_)) + dfInsurance["fe_int_plob_motor_a"],5)
dfInsurance.drop(columns = ['fe_int_plob_motor_a', 'fe_int_plob_motor_b'], inplace=True)
dfInsurance['fe_cmv_motor' + '_scale'] = dfInsurance['fe_cmv_motor' + '_scale'].fillna(0)

# FEATURE 5, 6 and 7 - Different types of rescaling
dfInsurance['log_amt_plob_motor'] = np.where(min(dfInsurance['amt_plob_motor']) <= 0,
np.log(dfInsurance['amt_plob_motor'] + abs(min(dfInsurance['amt_plob_motor'])) + 1),
np.log(abs(dfInsurance['amt_plob_motor'])))

dfInsurance['minmax_amt_plob_motor'] = scaler.fit_transform(dfInsurance[['amt_plob_motor']])
dfInsurance['norm_amt_plob_motor'] = sd_scaler.fit_transform(dfInsurance[['amt_plob_motor']])

# FEATURE 8 - Log of ratio
dfInsurance['log_rt_plob_motor'] = np.where(min(dfInsurance['rt_plob_motor']) <= 0,
np.log(dfInsurance['rt_plob_motor'] + abs(min(dfInsurance['rt_plob_motor'])) + 1),
np.log(abs(dfInsurance['rt_plob_motor'])))

######
## LOB: HOUSEHOLD
######

## FEATURE 1 - Household to predict Total Premium (Tree based Binning)
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

## FEATURE 2 - Scaling of Feature 1
choices = [7, 6, 5, 4, 3, 2, 1, 0, -1]
dfInsurance["fe_int_plob_household_b"] = np.select(conditions_household, choices, default=np.nan)
dfInsurance["fe_int_plob_household_a"] = dfInsurance["fe_int_plob_household_b"]-1
g = dfInsurance.groupby(['fe_bin_plob_household'])['amt_plob_household']
min_, max_ = g.transform('min'), g.transform('max')
dfInsurance['fe_amt_plob_household' + '_scale'] = round((((dfInsurance["fe_int_plob_household_b"]-dfInsurance["fe_int_plob_household_a"])*(dfInsurance['amt_plob_household'] - min_)) / 
                                                (max_ - min_)) + dfInsurance["fe_int_plob_household_a"],5)
dfInsurance.drop(columns = ['fe_int_plob_household_a', 'fe_int_plob_household_b'], inplace=True)
dfInsurance['fe_amt_plob_household' + '_scale'] = dfInsurance['fe_amt_plob_household' + '_scale'].fillna(0)

## FEATURE 3 - Household to predict CMV (Tree based Binning)
col = 'amt_plob_household'
conditions_household = [dfInsurance[col] > 143, 
              (dfInsurance[col] > 0) & (dfInsurance[col] <= 143), 
              (dfInsurance[col] == 0),
              (dfInsurance[col] < 0)]

choices = ['A', 'B', 'ZEROS','NEGATIVES']
dfInsurance["fe_bin_cmv_household"] = np.select(conditions_household, choices, default=np.nan)

## FEATURE 4 - Scaling of Feature 3
choices = [2, 1, 0, -1]
dfInsurance["fe_int_plob_household_b"] = np.select(conditions_household, choices, default=np.nan)
dfInsurance["fe_int_plob_household_a"] = dfInsurance["fe_int_plob_household_b"]-1

g = dfInsurance.groupby(['fe_bin_cmv_household'])['amt_plob_household']
min_, max_ = g.transform('min'), g.transform('max')
dfInsurance['fe_cmv_household' + '_scale'] = round((((dfInsurance["fe_int_plob_household_b"]-dfInsurance["fe_int_plob_household_a"])*(dfInsurance['amt_plob_household'] - min_)) / 
                                                (max_ - min_)) + dfInsurance["fe_int_plob_household_a"],5)
dfInsurance.drop(columns = ['fe_int_plob_household_a', 'fe_int_plob_household_b'], inplace=True)
dfInsurance['fe_cmv_household' + '_scale'] = dfInsurance['fe_cmv_household' + '_scale'].fillna(0)

## FEATURE 5, 6 and 7
dfInsurance['log_amt_plob_household'] = np.where(min(dfInsurance['amt_plob_household']) <= 0,
np.log(dfInsurance['amt_plob_household'] + abs(min(dfInsurance['amt_plob_household'])) + 1),
np.log(abs(dfInsurance['amt_plob_household'])))

dfInsurance['minmax_amt_plob_household'] = scaler.fit_transform(dfInsurance[['amt_plob_household']])
dfInsurance['norm_amt_plob_household'] = sd_scaler.fit_transform(dfInsurance[['amt_plob_household']])

# FEATURE 8 - Log of ratio

dfInsurance['log_rt_plob_household'] = np.where(min(dfInsurance['rt_plob_household']) <= 0,
np.log(dfInsurance['rt_plob_household'] + abs(min(dfInsurance['rt_plob_household'])) + 1),
np.log(abs(dfInsurance['rt_plob_household'])))

######
## LOB: LIFE
######

## FEATURE 1 - Life to predict Total Premium (Tree based Binning)
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

## FEATURE 2 - Scaling of Feature 1
choices = [8,7,6, 5, 4, 3, 2, 1, 0, -1]
dfInsurance["fe_int_plob_life_b"] = np.select(conditions_life, choices, default=np.nan)
dfInsurance["fe_int_plob_life_a"] = dfInsurance["fe_int_plob_life_b"]-1

g = dfInsurance.groupby(['fe_bin_plob_life'])['amt_plob_life']
min_, max_ = g.transform('min'), g.transform('max')
dfInsurance['fe_amt_plob_life' + '_scale'] = round((((dfInsurance["fe_int_plob_life_b"]-dfInsurance["fe_int_plob_life_a"])*(dfInsurance['amt_plob_life'] - min_)) / 
                                                (max_ - min_)) + dfInsurance["fe_int_plob_life_a"],5)
dfInsurance.drop(columns = ['fe_int_plob_life_a', 'fe_int_plob_life_b'], inplace=True)
dfInsurance['fe_amt_plob_life' + '_scale'] = dfInsurance['fe_amt_plob_life' + '_scale'].fillna(0)

## FEATURE 3 - Household to predict CMV (Tree based Binning)
col = 'amt_plob_life'
conditions_life = [dfInsurance[col] > 7, 
              (dfInsurance[col] > 0) & (dfInsurance[col] <= 7), 
              (dfInsurance[col] == 0),
              (dfInsurance[col] < 0)]
choices = ['A', 'B', 'ZEROS','NEGATIVES']
dfInsurance["fe_bin_cmv_life"] = np.select(conditions_life, choices, default=np.nan)

## FEATURE 4 - Scaling of Feature 3
choices = [2, 1, 0, -1]
dfInsurance["fe_int_plob_life_b"] = np.select(conditions_life, choices, default=np.nan)
dfInsurance["fe_int_plob_life_a"] = dfInsurance["fe_int_plob_life_b"]-1
g = dfInsurance.groupby(['fe_bin_cmv_life'])['amt_plob_life']
min_, max_ = g.transform('min'), g.transform('max')
dfInsurance['fe_cmv_life' + '_scale'] = round((((dfInsurance["fe_int_plob_life_b"]-dfInsurance["fe_int_plob_life_a"])*(dfInsurance['amt_plob_life'] - min_)) / 
                                                (max_ - min_)) + dfInsurance["fe_int_plob_life_a"],5)
dfInsurance.drop(columns = ['fe_int_plob_life_a', 'fe_int_plob_life_b'], inplace=True)
dfInsurance['fe_cmv_life' + '_scale'] = dfInsurance['fe_cmv_life' + '_scale'].fillna(0)

## FEATURE 5, 6 and 7
dfInsurance['log_amt_plob_life'] = np.where(min(dfInsurance['amt_plob_life']) <= 0,
np.log(dfInsurance['amt_plob_life'] + abs(min(dfInsurance['amt_plob_life'])) + 1),
np.log(abs(dfInsurance['amt_plob_life'])))
dfInsurance['minmax_amt_plob_life'] = scaler.fit_transform(dfInsurance[['amt_plob_life']])
dfInsurance['norm_amt_plob_life'] = sd_scaler.fit_transform(dfInsurance[['amt_plob_life']])

# FEATURE 8 - Log of ratio

dfInsurance['log_rt_plob_life'] = np.where(min(dfInsurance['rt_plob_life']) <= 0,
np.log(dfInsurance['rt_plob_life'] + abs(min(dfInsurance['rt_plob_life'])) + 1),
np.log(abs(dfInsurance['rt_plob_life'])))

## FEATURE 1 - Life to predict Total Premium (Tree based Binning)
## FEATURE 2 - Scaling of Feature 1
## FEATURE 3 - Household to predict CMV (Tree based Binning)
## FEATURE 4 - Scaling of Feature 3
## Feature 5, 6 and 7

######
## LOB: Health
######
## FEATURE 1 - Health to predict Total Premium (Tree based Binning)
col = 'amt_plob_health'
conditions_health = [dfInsurance[col] > 156, 
              (dfInsurance[col] > 95.3) & (dfInsurance[col] <= 156), 
              (dfInsurance[col] > 47.2) & (dfInsurance[col] <= 95.3),
              (dfInsurance[col] > 0) & (dfInsurance[col] <= 47.2),
              (dfInsurance[col] == 0),
              (dfInsurance[col] < 0)]
choices = ['A', 'B', 'C', 'D', 'ZEROS','NEGATIVES']
dfInsurance["fe_bin_plob_health"] = np.select(conditions_health, choices, default=np.nan)

## FEATURE 2 - Scaling of Feature 1
choices = [4, 3, 2, 1, 0, -1]
dfInsurance["fe_int_plob_health_b"] = np.select(conditions_health, choices, default=np.nan)
dfInsurance["fe_int_plob_health_a"] = dfInsurance["fe_int_plob_health_b"]-1

g = dfInsurance.groupby(['fe_bin_plob_health'])['amt_plob_health']
min_, max_ = g.transform('min'), g.transform('max')
dfInsurance['fe_amt_plob_health' + '_scale'] = round((((dfInsurance["fe_int_plob_health_b"]-dfInsurance["fe_int_plob_health_a"])*(dfInsurance['amt_plob_health'] - min_)) / 
                                                (max_ - min_)) + dfInsurance["fe_int_plob_health_a"],5)
dfInsurance.drop(columns = ['fe_int_plob_health_a', 'fe_int_plob_health_b'], inplace=True)
dfInsurance['fe_amt_plob_health' + '_scale'] = dfInsurance['fe_amt_plob_health' + '_scale'].fillna(0)

## FEATURE 3 - Health to predict CMV (Tree based Binning)
col = 'amt_plob_health'
conditions_health = [dfInsurance[col] > 248, 
              (dfInsurance[col] > 0) & (dfInsurance[col] <= 248), 
              (dfInsurance[col] == 0),
              (dfInsurance[col] < 0)]
choices = ['A', 'B', 'ZEROS','NEGATIVES']
dfInsurance["fe_bin_cmv_health"] = np.select(conditions_health, choices, default=np.nan)
## FEATURE 4 - Scaling of Feature 3
choices = [2, 1, 0, -1]
dfInsurance["fe_int_plob_health_b"] = np.select(conditions_health, choices, default=np.nan)
dfInsurance["fe_int_plob_health_a"] = dfInsurance["fe_int_plob_health_b"]-1
g = dfInsurance.groupby(['fe_bin_cmv_health'])['amt_plob_health']
min_, max_ = g.transform('min'), g.transform('max')
dfInsurance['fe_cmv_health' + '_scale'] = round((((dfInsurance["fe_int_plob_health_b"]-dfInsurance["fe_int_plob_health_a"])*(dfInsurance['amt_plob_health'] - min_)) / 
                                                (max_ - min_)) + dfInsurance["fe_int_plob_health_a"],5)
dfInsurance.drop(columns = ['fe_int_plob_health_a', 'fe_int_plob_health_b'], inplace=True)
dfInsurance['fe_cmv_health' + '_scale'] = dfInsurance['fe_cmv_health' + '_scale'].fillna(0)

## FEATURE 5, 6 and 7

dfInsurance['log_amt_plob_health'] = np.where(min(dfInsurance['amt_plob_health']) <= 0,
np.log(dfInsurance['amt_plob_health'] + abs(min(dfInsurance['amt_plob_health'])) + 1),
np.log(abs(dfInsurance['amt_plob_health'])))

dfInsurance['minmax_amt_plob_health'] = scaler.fit_transform(dfInsurance[['amt_plob_health']])

dfInsurance['norm_amt_plob_health'] = sd_scaler.fit_transform(dfInsurance[['amt_plob_health']])

# FEATURE 8 - Log of ratio

dfInsurance['log_rt_plob_health'] = np.where(min(dfInsurance['rt_plob_health']) <= 0,
np.log(dfInsurance['rt_plob_health'] + abs(min(dfInsurance['rt_plob_health'])) + 1),
np.log(abs(dfInsurance['rt_plob_health'])))


######
## LOB: Work Compensation
######
## FEATURE 1 - Life to predict Total Premium (Tree based Binning)
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

## FEATURE 2 - Scaling of Feature 1
choices = [5, 4, 3, 2, 1, 0, -1]
dfInsurance["fe_int_plob_wcomp_b"] = np.select(conditions_wcomp, choices, default=np.nan)
dfInsurance["fe_int_plob_wcomp_a"] = dfInsurance["fe_int_plob_wcomp_b"]-1

g = dfInsurance.groupby(['fe_bin_plob_wcomp'])['amt_plob_wcomp']
min_, max_ = g.transform('min'), g.transform('max')
dfInsurance['fe_amt_plob_wcomp' + '_scale'] = round((((dfInsurance["fe_int_plob_wcomp_b"]-dfInsurance["fe_int_plob_wcomp_a"])*(dfInsurance['amt_plob_wcomp'] - min_)) / 
                                                (max_ - min_)) + dfInsurance["fe_int_plob_wcomp_a"],5)
dfInsurance.drop(columns = ['fe_int_plob_wcomp_a', 'fe_int_plob_wcomp_b'], inplace=True)
dfInsurance['fe_amt_plob_wcomp' + '_scale'] = dfInsurance['fe_amt_plob_wcomp' + '_scale'].fillna(0)

## FEATURE 3 - Household to predict CMV (Tree based Binning)
col = 'amt_plob_wcomp'
conditions_wcomp = [dfInsurance[col] > 12, 
              (dfInsurance[col] > 0) & (dfInsurance[col] <= 12), 
              (dfInsurance[col] == 0),
              (dfInsurance[col] < 0)]

choices = ['A', 'B', 'ZEROS','NEGATIVES']
dfInsurance["fe_bin_cmv_wcomp"] = np.select(conditions_wcomp, choices, default=np.nan)

## FEATURE 4 - Scaling of Feature 3
choices = [2, 1, 0, -1]
dfInsurance["fe_int_plob_wcomp_b"] = np.select(conditions_wcomp, choices, default=np.nan)
dfInsurance["fe_int_plob_wcomp_a"] = dfInsurance["fe_int_plob_wcomp_b"]-1

g = dfInsurance.groupby(['fe_bin_cmv_wcomp'])['amt_plob_wcomp']
min_, max_ = g.transform('min'), g.transform('max')
dfInsurance['fe_cmv_wcomp' + '_scale'] = round((((dfInsurance["fe_int_plob_wcomp_b"]-dfInsurance["fe_int_plob_wcomp_a"])*(dfInsurance['amt_plob_wcomp'] - min_)) / 
                                                (max_ - min_)) + dfInsurance["fe_int_plob_wcomp_a"],5)
dfInsurance.drop(columns = ['fe_int_plob_wcomp_a', 'fe_int_plob_wcomp_b'], inplace=True)
dfInsurance['fe_cmv_wcomp' + '_scale'] = dfInsurance['fe_cmv_wcomp' + '_scale'].fillna(0)

## FEATURE 5, 6 and 7

dfInsurance['log_amt_plob_wcomp'] = np.where(min(dfInsurance['amt_plob_wcomp']) <= 0,
np.log(dfInsurance['amt_plob_wcomp'] + abs(min(dfInsurance['amt_plob_wcomp'])) + 1),
np.log(abs(dfInsurance['amt_plob_wcomp'])))

dfInsurance['minmax_amt_plob_wcomp'] = scaler.fit_transform(dfInsurance[['amt_plob_wcomp']])

dfInsurance['norm_amt_plob_wcomp'] = sd_scaler.fit_transform(dfInsurance[['amt_plob_wcomp']])

# FEATURE 8 - Log of ratio

dfInsurance['log_rt_plob_wcomp'] = np.where(min(dfInsurance['rt_plob_wcomp']) <= 0,
np.log(dfInsurance['rt_plob_wcomp'] + abs(min(dfInsurance['rt_plob_wcomp'])) + 1),
np.log(abs(dfInsurance['rt_plob_wcomp'])))

######
## Gross Monthly Salary
######

dfInsurance['minmax_amt_gms'] = scaler.fit_transform(dfInsurance[['amt_gms']])
dfInsurance['norm_amt_gms'] = sd_scaler.fit_transform(dfInsurance[['amt_gms']])
dfInsurance['sqrt_amt_gms'] = np.where(min(dfInsurance['amt_gms']) <= 0,
np.sqrt(dfInsurance['amt_gms'] + abs(min(dfInsurance['amt_gms'])) + 1),
np.sqrt(dfInsurance['amt_gms']))

######
## Gross Yearly Salary
######

dfInsurance['minmax_amt_gys'] = scaler.fit_transform(dfInsurance[['amt_gms']])
dfInsurance['norm_amt_gys'] = sd_scaler.fit_transform(dfInsurance[['amt_gms']])
dfInsurance['sqrt_amt_gys'] = np.where(min(dfInsurance['amt_gys']) <= 0,
np.sqrt(dfInsurance['amt_gys'] + abs(min(dfInsurance['amt_gys'])) + 1),
np.sqrt(dfInsurance['amt_gys']))

######
## Customer Age
######

dfInsurance['minmax_atr_cust_age'] = scaler.fit_transform(dfInsurance[['atr_cust_age']])
dfInsurance['norm_atr_cust_age'] = sd_scaler.fit_transform(dfInsurance[['atr_cust_age']])

dfInsurance['fe_bin_cust_age'] = pd.qcut(x = dfInsurance['atr_cust_age'], q = 4)
dfInsurance['fe_bin_cust_age'] = dfInsurance['fe_bin_cust_age'].astype(str)

######
## First Policy Year
######

dfInsurance['minmax_dt_fpy'] = scaler.fit_transform(dfInsurance[['dt_fpy']])
dfInsurance['norm_dt_fpy'] = sd_scaler.fit_transform(dfInsurance[['dt_fpy']])

######
## First Policy Year to Reference Year
######

dfInsurance['minmax_fpy_to_date'] = scaler.fit_transform(dfInsurance[['atr_fpy_to_date']])
dfInsurance['norm_fpy_to_date'] = sd_scaler.fit_transform(dfInsurance[['atr_fpy_to_date']])

######
## Customer Monetary Value
######

dfInsurance['sqrt_amt_cmv'] = np.where(min(dfInsurance['amt_cmv']) <= 0,
np.sqrt(dfInsurance['amt_cmv'] + abs(min(dfInsurance['amt_cmv'])) + 1),
np.sqrt(dfInsurance['amt_cmv']))
dfInsurance['minmax_amt_cmv'] = scaler.fit_transform(dfInsurance[['amt_cmv']])
dfInsurance['norm_amt_cmv'] = sd_scaler.fit_transform(dfInsurance[['amt_cmv']])


######
## Total Premium
######

dfInsurance['log_amt_premium_total'] = np.where(min(dfInsurance['amt_premium_total']) <= 0,
np.log(dfInsurance['amt_premium_total'] + abs(min(dfInsurance['amt_premium_total'])) + 1),
np.log(abs(dfInsurance['amt_premium_total'])))

######
## Claims Ratio vs Customer Monetary Value
######

m_slope = -525.5
rt_cut = 1
b_base = 500
conditions = [(dfInsurance['rt_cr'] >= rt_cut) & (dfInsurance['amt_cmv'] > ((m_slope)*dfInsurance['rt_cr']+b_base)), # Q1
              (dfInsurance['rt_cr'] >= rt_cut) & (dfInsurance['amt_cmv'] <= ((m_slope)*dfInsurance['rt_cr']+b_base)), # Q4
              (dfInsurance['rt_cr'] < rt_cut) & (dfInsurance['amt_cmv'] > ((m_slope)*dfInsurance['rt_cr']+b_base)), # Q2
              (dfInsurance['rt_cr'] < rt_cut) & (dfInsurance['amt_cmv'] <= ((m_slope)*dfInsurance['rt_cr']+b_base))] # Q3
choices_X = [1, 1, -1, -1]
choices_Y = [1, -1, 1, -1]

dfInsurance["fe_cmv_cr_Cut1_X"] = np.select(conditions, choices_X, default=np.nan)
dfInsurance["fe_cmv_cr_Cut1_Y"] = np.select(conditions, choices_Y, default=np.nan)

m_slope = -970
rt_cut = 0.6
b_base = 1050
conditions = [(dfInsurance['rt_cr'] >= rt_cut) & (dfInsurance['amt_cmv'] > ((m_slope)*dfInsurance['rt_cr']+b_base)), # Q1
              (dfInsurance['rt_cr'] >= rt_cut) & (dfInsurance['amt_cmv'] <= ((m_slope)*dfInsurance['rt_cr']+b_base)), # Q4
              (dfInsurance['rt_cr'] < rt_cut) & (dfInsurance['amt_cmv'] > ((m_slope)*dfInsurance['rt_cr']+b_base)), # Q2
              (dfInsurance['rt_cr'] < rt_cut) & (dfInsurance['amt_cmv'] <= ((m_slope)*dfInsurance['rt_cr']+b_base))] # Q3
choices_X = [1, 1, -1, -1]
choices_Y = [1, -1, 1, -1]

dfInsurance["fe_cmv_cr_Cut2_X"] = np.select(conditions, choices_X, default=np.nan)
dfInsurance["fe_cmv_cr_Cut2_Y"] = np.select(conditions, choices_Y, default=np.nan)

dfInsurance.to_csv(f'{script_dir}/../data/engineered_dataset.csv', index=False)