import numpy as np
import json
import gzip
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import sys

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
### Creating the user-item matrix as pandas dataframe:

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

### Item Similarity Matrix

# #Finding Cosine Similarity between items in training dataset as a numpy array
item_similarity_matrix = cosine_similarity(train.T)

item_similarity_matrix_pd = pd.DataFrame(item_similarity_matrix)
print(item_similarity_matrix_pd.head())
print("item_similarity_matrix_pd shape:", item_similarity_matrix_pd.shape)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
  
# print("item_similarity_matrix shape:", item_similarity_matrix.shape)
# # print("Item-Item Cosine Similarity Matrix:\n", item_similarity_matrix)
# print("Item-Item Cosine Similarity Matrix 0:10 Rows:\n", item_similarity_matrix[1:2]) #Prints first 10 rows of the matrix
# print("Item-Item Cosine Similarity Matrix 10:20 Rows:\n", item_similarity_matrix[10:20])
# print("Item-Item Cosine Similarity Matrix 20:20 Rows:\n", item_similarity_matrix[20:30])
# print("Item-Item Cosine Similarity Matrix 30:40 Rows:\n", item_similarity_matrix[30:40])
# print("Item-Item Cosine Similarity Matrix 40:50 Rows:\n", item_similarity_matrix[40:50])
# print("Item-Item Cosine Similarity Matrix 50:60 Rows:\n", item_similarity_matrix[50:60])
# print("Item-Item Cosine Similarity Matrix 60:70 Rows:\n", item_similarity_matrix[60:70])
# print("Item-Item Cosine Similarity Matrix 70:80 Rows:\n", item_similarity_matrix[70:80])
# print("Item-Item Cosine Similarity Matrix 80:85 Rows:\n", item_similarity_matrix[80:85])
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=np.inf)
print(sys.maxsize)
item_similarity_matrix.view()

print(item_similarity_matrix[1,1])
print(item_similarity_matrix[1,6]) 

# print(train.sum(axis=0))  # Sum of ratings per item
# print(train.sum(axis=1))  # Sum of ratings per user

##############################################
### Item Similarity Matrix

# #Finding Cosine Similarity between items in training dataset as a numpy array
user_similarity_matrix = cosine_similarity(train)

user_similarity_matrix_pd = pd.DataFrame(user_similarity_matrix)
print(user_similarity_matrix_pd.head())
print("user_similarity_matrix shape:", user_similarity_matrix_pd.shape)

print(type(user_similarity_matrix))

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=np.inf)
print(sys.maxsize)
user_similarity_matrix.view()

print(user_similarity_matrix[1,1])
print(user_similarity_matrix[1,6])


###############################################################
### Recomendation Score Matrix

# train = pd.DataFrame(train)

# # Function to calculate the recommendation score for an item
# def cal_score(history, similarities, avg_rating):
#     return np.sum((history - avg_rating) * similarities) / np.sum(similarities)

# # Initialize an empty DataFrame for user-item scores
# #data_ibs_user_score = pd.DataFrame(index=train.index, columns=train.columns)
# data_ibs_user_score = pd.DataFrame(data=train)
# print("[0,0]", data_ibs_user_score.iloc[0, 0])

# print(train.shape)
# print(user_similarity_matrix_pd.shape)

# print("iloc = ", train.iloc[1,0])

# train = train.iloc[:, 1:]

# # Iterate through users (rows)
# for i, user_row in train.iterrows():
#     # Get user's ratings
#     ratings = user_row[1:].to_numpy()  # Exclude user ID column
#     mean_avg = np.mean(ratings[ratings != 0])  # Average of non-zero ratings
    
#     # Store user ID in the score DataFrame
#     # data_ibs_user_score.iloc[i, 0] = user_row[0]  # User ID in the first column
#     print(i)
#     data_ibs_user_score.iloc[i, 0] = user_row.iloc[0]
    
#     # Iterate through items (columns)
#     for j, item in enumerate(train.columns[1:], start=1):  # Skip user ID column
#         # Check if the user has rated the item
#         if user_row[item] > 0:
#             # Already rated, assign -1
#             data_ibs_user_score.iloc[i, j] = -1
#         else:
#             # Get the top 10 similar items
#             top_n = user_similarity_matrix_pd[item].sort_values(ascending=False).iloc[1:11]
#             top_n_names = top_n.index
#             top_n_similarities = top_n.to_numpy()
            
#             # Get the user's rating history for the top N items
#             top_n_user_purchases = user_row[top_n_names].to_numpy()
            
#             # Calculate average ratings for the top N items
#             item_rating_avg = data_ibs[top_n_names].mean(axis=0).to_numpy()
            
#             # Compute the recommendation score
#             score = mean_avg + cal_score(top_n_user_purchases, top_n_similarities, item_rating_avg)
#             data_ibs_user_score.iloc[i, j] = score

# # View scores of each item for users
# print(data_ibs_user_score)

###############################################
def predict_ratings_user_based(user_item_matrix, user_similarity_matrix, user_means):
    """
    Predict ratings using user-based collaborative filtering.
    Args:
        user_item_matrix (pd.DataFrame): The user-item ratings matrix.
        user_similarity_matrix (np.ndarray): The user-user similarity matrix.
        user_means (pd.Series): The mean rating for each user.
    Returns:
        pd.DataFrame: Predicted ratings matrix.
    """
    # Convert to numpy for calculations
    ratings = user_item_matrix.values
    similarity = user_similarity_matrix
    user_means_array = user_means.values.reshape(-1, 1)

    # Calculate predictions
    weighted_sum = np.dot(similarity, ratings)
    similarity_sums = np.abs(similarity).sum(axis=1).reshape(-1, 1)

    # Avoid division by zero
    similarity_sums[similarity_sums == 0] = 1e-9

    # Add back user means
    predictions = user_means_array + (weighted_sum / similarity_sums)
    return pd.DataFrame(predictions, index=user_item_matrix.index, columns=user_item_matrix.columns)

# Compute user means for train data
user_means = train.mean(axis=1)

# Normalize the user-item matrix by subtracting user means
normalized_matrix = train.subtract(user_means, axis=0).fillna(0)

# Predict ratings for the user-item matrix
predicted_ratings_user_based = predict_ratings_user_based(train, user_similarity_matrix, user_means)

print("Predicted Ratings Matrix (User-Based):\n", predicted_ratings_user_based.head())

### Generate Recommendations
def recommend_items_user_based(user_id, user_item_matrix, predicted_ratings, top_n=10):
    """
    Recommend top N items for a user based on predicted ratings (user-based filtering).
    Args:
        user_id (str): The ID of the user.
        user_item_matrix (pd.DataFrame): Original user-item ratings matrix.
        predicted_ratings (pd.DataFrame): Predicted ratings matrix.
        top_n (int): Number of recommendations to return.
    Returns:
        list: Top N recommended item IDs.
    """
    # Get items the user has already rated
    user_ratings = user_item_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0].index

    # Get predicted ratings for the user
    user_predictions = predicted_ratings.loc[user_id]

    # Filter out already rated items
    recommendations = user_predictions.drop(index=rated_items)

    # Sort by predicted rating and return top N items
    top_items = recommendations.sort_values(ascending=False).head(top_n).index.tolist()
    return top_items

user_id = train.index[0]  # Replace with an actual user ID from your dataset
recommendations_user_based = recommend_items_user_based(user_id, train, predicted_ratings_user_based, top_n=10)
 
print(f"Top recommendations for user {user_id}:\n", recommendations_user_based)
 
###############################################################################################################################

# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity

# # Input data (assume it's a pandas DataFrame named `data`)
# # Columns: ["User", "Item1", "Item2", ..., "ItemN"]
# # Non-rated items have a value of 0.

# data = pd.DataFrame(user_item_matrix)
# print(data.head())
# print(data.shape)
# print(type(data))

# # print("iloc", data.iloc[1, 1:])

# # Normalize user ratings
# def normalize_ratings(data):
#     data_normalized = data.copy()
#     for i, row in data.iterrows():
#         ratings = row[1:]  # Exclude "User" column
#         avg_rating = ratings[ratings != 0].mean()  # Average of non-zero ratings

#         # Normalize ratings for the row
#         normalized_ratings = [r - avg_rating if r != 0 else 0 for r in ratings]

#         # Assign normalized ratings back to the corresponding row
#         print("i = ", i)
#         data_normalized.iloc[i, 1:] = normalized_ratings

#     return data_normalized

# data_normalized = normalize_ratings(data)

# # Prepare data frames for similarity calculations
# data_ibs = data.iloc[:, 1:]  # Exclude user column
# data_normalized_ibs = data_normalized.iloc[:, 1:]  # Exclude user column

# # Replace 0 with NaN in original data
# data_ibs = data_ibs.replace(0, np.nan)

# # Compute item-item cosine similarity
# def calculate_similarity(data):
#     similarity = cosine_similarity(data.T, data.T)  # Transpose for item-based similarity
#     similarity_df = pd.DataFrame(
#         similarity, index=data.columns, columns=data.columns
#     )
#     return similarity_df

# data_ibs_similarity = calculate_similarity(data_normalized_ibs)

# # Compute scores for item recommendations
# def calculate_scores(data_ibs, data_ibs_similarity, data):
#     data_ibs_user_score = pd.DataFrame(index=data.index, columns=data.columns)
#     for i, user_row in data.iterrows():
#         user_id = user_row["User"]
#         ratings = user_row[1:]  # Skip the "User" column
#         avg_rating = ratings[ratings != 0].mean()

#         data_ibs_user_score.loc[i, "User"] = user_id

#         for item in data.columns[1:]:
#             if user_row[item] > 0:
#                 # Already rated, assign -1
#                 data_ibs_user_score.loc[i, item] = -1
#             else:
#                 # Get top 10 similar items
#                 top_n = (
#                     data_ibs_similarity[item]
#                     .sort_values(ascending=False)
#                     .iloc[1:11]  # Skip self-similarity
#                 )
#                 top_n_items = top_n.index
#                 top_n_similarities = top_n.values

#                 # User's ratings for top N items
#                 top_n_user_ratings = user_row[top_n_items]

#                 # Average ratings for top N items
#                 item_avg_ratings = data_ibs[top_n_items].mean()

#                 # Calculate score
#                 numerator = np.sum(
#                     (top_n_user_ratings - item_avg_ratings) * top_n_similarities
#                 )
#                 denominator = np.sum(top_n_similarities)
#                 score = avg_rating + (numerator / denominator if denominator != 0 else 0)
#                 data_ibs_user_score.loc[i, item] = score
#     return data_ibs_user_score

# data_ibs_user_score = calculate_scores(data_ibs, data_ibs_similarity, data)

# # Generate top 100 recommended items for each user
# def generate_recommendations(data_ibs_user_score):
#     recommendations = pd.DataFrame(
#         index=data_ibs_user_score.index,
#         columns=["User"] + [f"Item{i}" for i in range(1, 101)],
#     )
#     for i, user_row in data_ibs_user_score.iterrows():
#         user_id = user_row["User"]
#         recommendations.loc[i, "User"] = user_id
#         # Sort items by score, exclude already rated items (-1)
#         sorted_items = (
#             user_row[1:]  # Skip "User" column
#             .sort_values(ascending=False)
#             .iloc[:100]
#             .index
#         )
#         recommendations.loc[i, 1:] = sorted_items
#     return recommendations

# data_user_scores_holder = generate_recommendations(data_ibs_user_score)

# # View the final recommendations
# print(data_user_scores_holder)
