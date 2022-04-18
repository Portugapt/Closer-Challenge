dbscan_kwargs = {
            "eps":0.045,
            "metric": "euclidean",
            "algorithm": 'kd_tree',
            "min_samples": 17}
dbscan_model = DBSCAN(**dbscan_kwargs)
dbscan_model.fit(dfInsurance[['log_rt_plob_household', 'log_rt_plob_motor', 'rt_premiums_year', 'rt_plob_household', 'log_rt_plob_wcomp', 'rt_cr']])
dfInsurance['SCORER_DBSCAN_1'] = dbscan_model.labels_

conditions = [(dfInsurance['SCORER_DBSCAN_1'] == -1) & (dfInsurance['atr_cust_age'] < 49), 
              (dfInsurance['SCORER_DBSCAN_1'] == -1) & (dfInsurance['atr_cust_age'] >= 49), 
              (dfInsurance['SCORER_DBSCAN_1'] == 0) & (dfInsurance['amt_cmv'] < 250),
              (dfInsurance['SCORER_DBSCAN_1'] == 0) & (dfInsurance['amt_cmv'] >= 250)]
choices = [ "-1-A", '-1-B', '0-A', '0-C']

dfInsurance["SCORER_DBSCAN_1_DERIVADO"] = np.select(conditions, choices, default='OUTROS')
