# https://github.com/Sriya1008/IntroToDataScienceProject


import numpy as np
import json
import gzip
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

### Read in data
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)
    
data = parse('All_Beauty_5.json.gz')
print(data)
type(data)

# for obj in data:
#     print(obj)


### Read in data to pandas dataframe
def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('All_Beauty_5.json.gz')
print(df.dtypes)
df.dtypes
print(df.isnull().sum())
# print(df.dtypes)

#mac problems
with open('All_Beauty_5.json', 'r') as f:
  for line in f:
    review = json.loads(line.strip())
    print(review)


#### Splitting the data 
train, test = train_test_split(
    df, test_size=0.2, random_state=0)

print("df shape:", df.shape)
print("train shape:", train.shape)
print("test shape:", test.shape)



# dicMac = {}
# i = 0
# with open('All_Beauty_5.json', 'r') as f:
#   for line in f:
#     review = json.loads(line.strip())
#     dicMac[i] = line
#     i += 1
#     # print(review)
# dfmac = pd.DataFrame.from_dict(dicMac, orient='index')
# print(type(dfmac))
# print("dfmac shape:", dfmac.shape)

