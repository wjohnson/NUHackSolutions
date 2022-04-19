import os
import json

import joblib
import numpy as np


def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.joblib")
    model = joblib.load(model_path)

def run(data):
    input_data = json.loads(data)
    np_arr = np.array(input_data)
    return model.transform(np_arr).tolist()