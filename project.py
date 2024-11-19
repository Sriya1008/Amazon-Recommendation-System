import numpy as np
import json
import gzip
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import sys

### Read in data
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)
    
data = parse('All_Beauty_5.json.gz')
# print(data)
# type(data)

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
# print(df.dtypes)
# df.dtypes
# print(df.isnull().sum())
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

#Using user-based filtering

# Group by reviewerID and asin, taking the mean rating for duplicates
df = df.groupby(['reviewerID', 'asin'], as_index=False).agg({'overall': 'mean'})

# Now pivot to create the user-item matrix
user_item_matrix = df.pivot(index='reviewerID', columns='asin', values='overall')
user_item_matrix = user_item_matrix.fillna(0) # Make all NaN values 0
print("Sample of User-Item Matrix:\n", user_item_matrix.head())

#### Splitting the data for train and test###########################################

# Function for per-user train-test split
def user_based_train_test_split(user_item_matrix, test_size=0.2):
    """
    Splits the user-item matrix into training and testing datasets, ensuring
    an 80-20 split of ratings per user.

    Parameters:
    - user_item_matrix: pandas.DataFrame, the user-item matrix.
    - test_size: float, the proportion of ratings to use for testing.

    Returns:
    - train: pandas.DataFrame, the training user-item matrix.
    - test: pandas.DataFrame, the testing user-item matrix.
    """
    train = user_item_matrix.copy()
    test = user_item_matrix.copy()

    for user in user_item_matrix.index:
        user_ratings = user_item_matrix.loc[user]
        non_zero_ratings = user_ratings[user_ratings > 0]  # Only consider items the user rated
        test_indices = non_zero_ratings.sample(frac=test_size, random_state=42).index
        
        # Zero out test ratings in the training set
        train.loc[user, test_indices] = 0
        
        # Zero out training ratings in the test set
        # print(~test.index.isin(test_indices))
        # print(user)
        # print(test.loc[user,])
        # test.loc[user, ~test.index.isin(test_indices)] = 0
        test.loc[user, ~test.columns.isin(test_indices)] = 0


    return train, test

# Apply the per-user train-test split
train, test = user_based_train_test_split(user_item_matrix)

# Verify the split
print("Training set shape:", train.shape)
print("Testing set shape:", test.shape)
assert (train * test).sum().sum() == 0, "Train and test sets overlap!"
print("Train and test sets are properly separated.")


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
# print(sys.maxsize)
item_similarity_matrix.view()

# print(item_similarity_matrix[1,1])
# print(item_similarity_matrix[1,6]) 

# print(train.sum(axis=0))  # Sum of ratings per item
# print(train.sum(axis=1))  # Sum of ratings per user

##############################################
### Item Similarity Matrix

# #Finding Cosine Similarity between items in training dataset as a numpy array
user_similarity_matrix = cosine_similarity(train)

user_similarity_matrix_pd = pd.DataFrame(user_similarity_matrix)
print(user_similarity_matrix_pd.head())
print("user_similarity_matrix shape:", user_similarity_matrix_pd.shape)

# print(type(user_similarity_matrix))

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=np.inf)
# print(sys.maxsize)
# user_similarity_matrix.view()

# print(user_similarity_matrix[1,1])
# print(user_similarity_matrix[1,6])


###############################################################

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

print("len", len(predicted_ratings_user_based))

predicted_ratings_user_based_1_5 = predicted_ratings_user_based.copy()

for i in range (len(predicted_ratings_user_based)):
  for j in range (len(predicted_ratings_user_based.iloc[0])):
    if predicted_ratings_user_based.iat[i,j] < 0.2:
      predicted_ratings_user_based_1_5.iat[i,j] = 1
    elif predicted_ratings_user_based.iat[i,j] < 0.4:
      predicted_ratings_user_based_1_5.iat[i,j] = 2
    elif predicted_ratings_user_based.iat[i,j] < 0.6:
      predicted_ratings_user_based_1_5.iat[i,j] = 3
    elif predicted_ratings_user_based.iat[i,j] < 0.8:
      predicted_ratings_user_based_1_5.iat[i,j] = 4
    else:
      predicted_ratings_user_based_1_5.iat[i,j] = 5
    
# print(predicted_ratings_user_based_1_5.head())  


print("Predicted Ratings Matrix (User-Based):\n", predicted_ratings_user_based.head())

### Generate Recommendations
def recommend_items_for_all_users(user_item_matrix, predicted_ratings, top_n=10):
    """
    Generate recommendations for all users based on predicted ratings.
    Args:
        user_item_matrix (pd.DataFrame): Original user-item ratings matrix.
        predicted_ratings (pd.DataFrame): Predicted ratings matrix.
        top_n (int): Number of recommendations to return per user.
    Returns:
        dict: Dictionary with user IDs as keys and top N recommended item IDs as values.
    """
    recommendations = {}
    for user_id in user_item_matrix.index:
        # Get items the user has already rated
        user_ratings = user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index

        # Get predicted ratings for the user
        user_predictions = predicted_ratings.loc[user_id]

        # Filter out already rated items
        filtered_predictions = user_predictions.drop(index=rated_items)

        # Sort by predicted rating and get top N items
        top_items = filtered_predictions.sort_values(ascending=False).head(top_n).index.tolist()
        recommendations[user_id] = top_items

    return recommendations

# Generate and print recommendations for all users
all_user_recommendations = recommend_items_for_all_users(train, predicted_ratings_user_based, top_n=10)

# Print recommendations for each user
# for user, recommendations in all_user_recommendations.items():
#     print(f"Recommendations for user {user}:\n{recommendations}")


################################################################################################
#MAE and RMSE
# Flatten test and predicted matrices for comparison
test_values = test[test > 0].stack()  # Only consider non-zero ratings in the test set
predicted_values = predicted_ratings_user_based_1_5[test > 0].stack()


# print("Predicted Values Head:", predicted_values.head(10))
# print("Test Values Head:",test_values.head(10))

# Calculate MAE and RMSE
mae = mean_absolute_error(test_values, predicted_values)
rmse = np.sqrt(mean_squared_error(test_values, predicted_values))

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")  


# Function to calculate Precision, Recall, and F-measure for all users
def evaluate_recommendations(user_item_matrix, top_n_recommendations, test_data):
    precision_scores = []
    recall_scores = []
    f_measure_scores = []
    ndcg_scores = []

    # Iterate over all users
    for user_id in user_item_matrix.index:
        # Get the user's test set (items they haven't rated in the training set)
        user_test_items = test_data.loc[user_id][test_data.loc[user_id] > 0].index.tolist()
        
        # Get the top-n recommendations for this user
        recommended_items = top_n_recommendations.get(user_id, [])
        
        # Calculate Precision: Number of recommended items that are in the test set
        relevant_recommended = len(set(recommended_items) & set(user_test_items))
        precision = relevant_recommended / len(recommended_items) if recommended_items else 0
        
        # Calculate Recall: Number of recommended test items out of all test items
        recall = relevant_recommended / len(user_test_items) if user_test_items else 0
        
        # Calculate F-measure
        if precision + recall > 0:
            f_measure = 2 * precision * recall / (precision + recall)
        else:
            f_measure = 0

        # Calculate NDCG: Discounted Cumulative Gain at k (top N)
        dcg = 0
        idcg = 0
        for i, item in enumerate(recommended_items[:len(user_test_items)]):
            if item in user_test_items:
                dcg += 1 / np.log2(i + 2)  # i + 2 for 1-based index
            idcg += 1 / np.log2(i + 2)  # ideal DCG assumes all test items are relevant and in perfect order
        
        ndcg = dcg / idcg if idcg > 0 else 0
        
        # Store the results for this user
        precision_scores.append(precision)
        recall_scores.append(recall)
        f_measure_scores.append(f_measure)
        ndcg_scores.append(ndcg)

    # Calculate the average of all users' metrics
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f_measure = np.mean(f_measure_scores)
    avg_ndcg = np.mean(ndcg_scores)

    return avg_precision, avg_recall, avg_f_measure, avg_ndcg


# Evaluate the recommendation system
avg_precision, avg_recall, avg_f_measure, avg_ndcg = evaluate_recommendations(
    user_item_matrix, all_user_recommendations, test)

# Print the evaluation metrics
print(f"Average Precision: {avg_precision}")
print(f"Average Recall: {avg_recall}")
print(f"Average F-measure: {avg_f_measure}")
print(f"Average NDCG: {avg_ndcg}")

  
 

