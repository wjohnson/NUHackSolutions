from azureml.core import Workspace
from azureml.core.model import Model

from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from azureml.core import Environment

if __name__ == "__main__":
    ws = Workspace.from_config()

    model = ws.models.get("my-topic-model")
    # set up pytorch environment
    env = Environment.get(ws, name='AzureML-sklearn-1.0-ubuntu20.04-py38-cpu')

    inference_config = InferenceConfig(
        entry_script='sklearnscore.py', environment=env, 
        source_directory="./sklearnmodel/scoring"
    )
    deployment_config = AciWebservice.deploy_configuration(
        cpu_cores = 1, memory_gb = 1)
    service = Model.deploy(ws, "mytopicservice", [model], 
        inference_config, deployment_config, overwrite=True)
    service.wait_for_deployment(show_output = True)
    print(service.state)