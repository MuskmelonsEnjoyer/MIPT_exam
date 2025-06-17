import pandas as pd

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", header=None)

df['Cat'] = df[1].map({"M": 0, "B":1})

X = df.drop(columns=[0, 1, 'Cat'])
y = df["Cat"]