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



#Using item-based filtering
#creating the user-item matrix:
# Group by reviewerID and asin, taking the mean rating for duplicates
df = df.groupby(['reviewerID', 'asin'], as_index=False).agg({'overall': 'mean'})

# Now pivot to create the user-item matrix
user_item_matrix = df.pivot(index='reviewerID', columns='asin', values='overall')
user_item_matrix = user_item_matrix.fillna(0)
print("Sample of User-Item Matrix:\n", user_item_matrix.head())

#### Splitting the data 
#for each user, randomly select 80% of his/her ratings as the training ratings, and use the remaining 20% ratings as testing ratings.

train, test = train_test_split(
    user_item_matrix, test_size=0.2, random_state=0)

print("df shape:", user_item_matrix.shape)
print("train shape:", train.shape)
print("test shape:", test.shape)

#Finding Cosine Similarity between items in training dataset
#Create a matrix that contains the similarity between each item
# Calculate the cosine similarity matrix between items
item_similarity_matrix = cosine_similarity(train.T)

print(type(item_similarity_matrix))
print("item_similarity_matrix shape:", item_similarity_matrix.shape)
# Print the item-item similarity matrix
print("Item-Item Cosine Similarity Matrix:\n", item_similarity_matrix)


print(train.sum(axis=0))  # Sum of ratings per item
print(train.sum(axis=1))  # Sum of ratings per user
