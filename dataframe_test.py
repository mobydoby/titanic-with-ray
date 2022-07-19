import pandas as pd 
import torch
import numpy as np     
from os.path import join

mydict = pd.read_csv("./titanic/data/train.csv").to_numpy()
print(mydict)
print("hey you", df.columns)
t = pd.concat([df.iloc[:,0], df.iloc[:, 2]], axis=1).to_numpy()
yo = torch.tensor(df.iloc[:,0])
print(t)
print(torch.tensor(t))

# torch_tensor = torch.tensor(df['targets'].values)

# print(torch_tensor)