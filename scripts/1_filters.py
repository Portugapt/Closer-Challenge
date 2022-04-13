from pathlib import Path
import pandas as pd
import numpy as np

script_dir = Path( __file__ ).parent.absolute()

### Import Dataset
dataset = pd.read_excel(f'{script_dir}/../data/dataset.xlsx')
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
## [3] Data main cut, KEEP is usable data, and THROW is data we found is impossible to work with
## Application: Filter
######

dfInsurance['amt_premium_total'] = (dfInsurance['amt_plob_life'] + dfInsurance['amt_plob_household'] + dfInsurance['amt_plob_motor'] + 
                                    dfInsurance['amt_plob_health']+ dfInsurance['amt_plob_wcomp'])

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

dfInsurance = dfInsurance.loc[dfInsurance['DATA_MAIN_CUT'] == 'KEEP']
dfInsurance.drop(columns=['DATA_MAIN_CUT'], inplace=True)

######
## [4] Drop null values from children column
## Application: Transformation
######
dfInsurance.dropna(subset=['flg_children'], inplace=True)

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

dfInsurance.to_csv(f'{script_dir}/../data/filtered_dataset.csv', index=False)