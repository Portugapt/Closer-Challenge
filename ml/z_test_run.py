import mlflow
import os

# Get the current working directory
cwd = os.getcwd()

print(cwd)
project_uri = "."
params = {"log_this_string": "Test_String"}

mlflow.run(project_uri, parameters=params, entry_point = 'mlflow_test')