import os

from azureml.core import Workspace
from azureml.core.model import Model

from azureml.core.resource_configuration import ResourceConfiguration

EXPERIMENT_NAME = "experiment-sklearn"

if __name__ == "__main__":
    ws = Workspace.from_config()
    # Create a model folder in the current directory
    os.makedirs('./model', exist_ok=True)

    experiment = ws.experiments.get(EXPERIMENT_NAME)
    runs = experiment.get_runs()
    latest_run = None
    for run in runs:
        latest_run = run
        break
    
    if latest_run is None:
        raise RuntimeError(f"There was no training run for {EXPERIMENT_NAME}")

    # Download the model from run history
    latest_run.download_file(name='outputs/model.joblib', output_file_path='./model/model.joblib')

    model = Model.register(
        workspace=ws,
        model_name='my-topic-model',
        model_path='./model/model.joblib',
        model_framework=Model.Framework.SCIKITLEARN,  
        model_framework_version="1.0.0",  
        resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=1.0),
        description='Topic Model',
        #tags={'area': 'tabular', 'type': 'regression'}
    )

    print('Name:', model.name)
    print('Version:', model.version)