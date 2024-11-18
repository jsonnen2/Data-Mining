import pandas as pd


feat = pd.read_csv("data\\train\\predict_recommendation\\features.csv")
target = pd.read_csv("data\\train\\predict_recommendation\\targets.csv")

target[target['recommendation'] == 1] = 'Recommended'
target[target['recommendation'] == 0] = 'Not Recommended'

feat.to_csv("data\\train\\predict_recommendation\\features.csv", index=False)
target.to_csv("data\\train\\predict_recommendation\\targets.csv", index=False)
