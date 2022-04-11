import os
import warnings
import sys

import mlflow

if __name__ == "__main__":
    
    string_to_save = sys.argv[1]
    with mlflow.start_run():
        mlflow.log_param("thestring", string_to_save)
        mlflow.log_param("x", 1)
        mlflow.log_metric("y", 2)