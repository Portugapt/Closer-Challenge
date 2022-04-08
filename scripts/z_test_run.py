import mlflow
import os

# Get the current working directory
cwd = os.getcwd()

print(cwd)
project_uri = "C:\\Users\\JoãoMonteiro\\Documents\\dev\\formacao\\Closer-Challenge"
params = {"log_this_string": "Test_String"}

mlflow.run(project_uri, parameters=params, use_conda = True, entry_point = 'mlflow_test')