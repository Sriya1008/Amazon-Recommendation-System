import numpy as np
import json
import gzip
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

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
    #print(review)

def parcemac(path):
  with open(path, 'r') as f:
    for line in f:
      # review = json.loads(line.strip())
      # print(review)
      # return review
      yield json.loads(line)
      
def getDFmac(path):
  i = 0
  df = {}
  for d in parcemac(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

dfmac = getDFmac('All_Beauty_5.json')
print(dfmac.dtypes)
print("dfmac shape:", dfmac.shape)


######################################################################
### Creating the user-item matrix as numpy array:

#Using item-based filtering

# Group by reviewerID and asin, taking the mean rating for duplicates
df = df.groupby(['reviewerID', 'asin'], as_index=False).agg({'overall': 'mean'})

# Now pivot to create the user-item matrix
user_item_matrix = df.pivot(index='reviewerID', columns='asin', values='overall')
user_item_matrix = user_item_matrix.fillna(0) # Make all NaN values 0
print("Sample of User-Item Matrix:\n", user_item_matrix.head())

#### Splitting the data for train and test

train, test = train_test_split(
    user_item_matrix, test_size=0.2, random_state=0)

print("df shape:", user_item_matrix.shape)
print("train shape:", train.shape)
print("test shape:", test.shape)

######################################################

# #Finding Cosine Similarity between items in training dataset
item_similarity_matrix = cosine_similarity(train.T)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
  
print("item_similarity_matrix shape:", item_similarity_matrix.shape)
# print("Item-Item Cosine Similarity Matrix:\n", item_similarity_matrix)
print("Item-Item Cosine Similarity Matrix First 10 Rows:\n", item_similarity_matrix[:10]) #Prints first 10 rows of the matrix

print(item_similarity_matrix[1,1])
print(item_similarity_matrix[1,6])

# print(train.sum(axis=0))  # Sum of ratings per item
# print(train.sum(axis=1))  # Sum of ratings per user



