"""Run after xgbfast compile; imports no build-time ML libraries."""
import argparse

import numpy as np
from xgbfast import Predictor

parser = argparse.ArgumentParser()
parser.add_argument("artifact")
parser.add_argument("--data", required=True, help="2D .npy with the model's feature order")
args = parser.parse_args()

data = np.load(args.data, allow_pickle=False)
with Predictor.load(args.artifact, mode="native") as model:
    features = np.array(data[0], dtype=np.float32, order="C")
    score = model.predict(features)
    print("predict:", score)

with Predictor.load(args.artifact) as model:
    output = np.empty(1, dtype=np.float32)
    model.predict_into(features, output)
    print("predict_into:", float(output[0]))
    print("batch:", model.predict(data[:5]))
