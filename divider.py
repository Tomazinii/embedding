
import pandas as pd


df = pd.read_csv("/workspaces/tcc/data.csv")

data = df.head(50)

data.to_csv("/workspaces/tcc/10lines.csv", index=False)