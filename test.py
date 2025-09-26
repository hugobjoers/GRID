import pickle
import pandas as pd

obj = pd.read_pickle(r"logs/inference/runs/2025-09-16/16-17-06/pickle/merged_predictions.pkl")
print(obj[0].keys())
print(len(obj))
print(len(obj[10]["embedding"]))