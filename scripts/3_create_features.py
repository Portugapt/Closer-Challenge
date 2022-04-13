from pathlib import Path
import pandas as pd
import numpy as np

script_dir = Path( __file__ ).parent.absolute()

### Import Dataset
dataset = pd.read_csv(f'{script_dir}/../data/transformed_dataset.csv')
dfInsurance = dataset.copy()

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
## [18] Customer Claims Cost
## Application: Addition
######

dfInsurance['amt_gys'] = dfInsurance['amt_gms'] * 12

######
## [19] Customer Yearly Salary 
## Application: Addition
######

dfInsurance['amt_claims_total'] = dfInsurance['rt_cr'] * dfInsurance['amt_premium_total']

######
## [20] Customer Effort Rate - Cost of premiums in relation to customer income
## Application: Addition
######

dfInsurance['rt_premiums_year'] = dfInsurance['amt_premium_total'] / dfInsurance['amt_gys']
dfInsurance['rt_claims_year'] = dfInsurance['amt_claims_total'] / dfInsurance['amt_gys']

######
## [21] Customer "Credit Score" - Weird Pattern in Total Premium VS Claims Costs
## Application: Addition
######

conditions_credit_score_proxy = [(dfInsurance['amt_premium_total'] > 1000) & (dfInsurance['amt_claims_total'] < 740), 
                          (dfInsurance['amt_premium_total'] > 1000) & (dfInsurance['amt_claims_total'] > 740),]
choices = [1, -1]
dfInsurance["atr_credit_score_proxy"] = np.select(conditions_credit_score_proxy, choices, default=0)


dfInsurance.to_csv(f'{script_dir}/../data/features_dataset.csv', index=False)
