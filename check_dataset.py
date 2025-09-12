import pandas as pd
df = pd.read_csv("cyberbullying_dataset.csv", encoding='utf-8', engine='python')
print(df.columns.tolist())
print(df.head(3))
