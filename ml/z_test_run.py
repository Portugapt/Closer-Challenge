import mlflow
import os

# Get the current working directory
cwd = os.getcwd()

print(cwd)
project_uri = "https://github.com/Portugapt/Closer-Challenge#ml"
params = {"log_this_string": "Test_String"}

mlflow.run(project_uri, version ='main', parameters=params, entry_point = 'main')