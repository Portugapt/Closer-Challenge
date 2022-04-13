import os
import time

import logging

import argparse
import mlflow

logging.basicConfig(filename='kmeans_1.log', level=logging.DEBUG)

import pandas as pd
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
from sklearn.cluster import KMeans
# https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
from sklearn.metrics import silhouette_score, calinski_harabasz_score


## Define the experiment name
# KM_EXPERIMENT_ID = mlflow.create_experiment("Proof-of-concept KMeans")
mlflow.set_experiment(experiment_name = "Proof-of-concept KMeans")
#os.environ["MLFLOW_EXPERIMENT_ID"] = "Proof-of-concept-kmeans"
KM_EXPERIMENT = mlflow.get_experiment_by_name("Proof-of-concept KMeans")
# os.environ["MLFLOW_EXPERIMENT_ID"] = KM_EXPERIMENT.experiment_id
logging.info(f'Experiment Set. Experiment ID: {KM_EXPERIMENT.experiment_id}')
logging.debug(f'Artifact Location: {KM_EXPERIMENT.artifact_location}')
logging.debug(f'Tags: {KM_EXPERIMENT.tags}')
logging.debug(f'Lifecycle_stage: {KM_EXPERIMENT.lifecycle_stage}')

def parse_args():
    parser = argparse.ArgumentParser(description="Proof of concept")
    logging.info('Parser created')
    parser.add_argument(
        "--file_path",
        type=str,
        help="File path to csv data. Assumes it has a column named Set, to split the data"
    )
    parser.add_argument(
        "--n_klusters",
        type=int,
        help="Number of clusters for kmeans"
    )
    parser.add_argument(
        "--k_init",
        type=str,
        help="Type of Initialization"
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        help="Maximum iterations"
    )
    return parser.parse_args()

def load_data(file_path):
    logging.debug(f'load_data:data:{file_path}')
    dfInput = pd.read_csv(file_path)
    dfTrain = dfInput.loc[dfInput.Set == 'train']
    dfTest = dfInput.loc[dfInput.Set == 'test']
    return dfTrain, dfTest

def metrics_calculation(xData, labelData):
    sil_score = silhouette_score(xData, labelData, metric='euclidean')
    ch_score = calinski_harabasz_score(xData, labelData)
    return sil_score, ch_score

def main():
    args = parse_args()
    logging.debug(f'data: {args.file_path}')
    logging.debug(f'n clusters: {args.n_klusters}')
    logging.debug(f'type of init: {args.k_init}')
    
    listFeatures = ['fe_amt_plob_motor_scale','fe_amt_plob_life_scale',
                    'fe_amt_plob_health_scale','fe_amt_plob_wcomp_scale',
                    'fe_amt_plob_household_scale']

    if mlflow.active_run():
        mlflow.end_run()
    with mlflow.start_run():
        dfTrain, dfTest = load_data(args.file_path)
        ## Idea. Pass JSON of parameters, same as data path
        kmeans_params = {
            "n_clusters": args.n_klusters,
            "init": args.k_init,
            "n_init": 10,
            "max_iter": args.max_iter,
            "random_state": 42}

        model_kmeans = KMeans(**kmeans_params)
        model_kmeans.fit(dfTrain[listFeatures])

        predict_labelData = model_kmeans.predict(dfTest[listFeatures])
        sil, ch = metrics_calculation(dfTest[listFeatures], predict_labelData)

        mlflow.log_params(kmeans_params)
        mlflow.log_param('features', listFeatures)
        mlflow.log_metric('Silhouette', sil)
        mlflow.log_metric('Calinski Harabasz', ch)
        mlflow.sklearn.log_model(model_kmeans, artifact_path = 'poc_kmodel')

if __name__ == "__main__":
    main()