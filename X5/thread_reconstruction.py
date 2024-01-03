import pandas as pd
import numpy as np

df = pd.read_csv("f3352311.csv")
df["reply-to"] = df["reply-to"].astype(str)
code = input("Give thread id: ")
subdf = df[df["reply-to"] == code]
for index, row in subdf.iterrows():
    print(f"{row['Author']}: {row['Post']} ({row['Date']}), Lang = {row['Language']}")
