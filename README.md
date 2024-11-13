https://github.com/Sriya1008/Amazon-Recommendation-System.git
https://towardsdatascience.com/comprehensive-guide-on-item-based-recommendation-systems-d67e40e2b75d


In the data set, the variables mean: 
reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B                                  Object
asin - ID of the product, e.g. 0000013714                                             Object
reviewerName - name of the reviewer                                                   Object
vote - helpful votes of the review                                                    Object
style - a disctionary of the product metadata, e.g., "Format" is "Hardcover"          Object
reviewText - text of the review                                                       Object
overall - rating of the product                                                       float64
summary - summary of the review                                                       Object
unixReviewTime - time of the review (unix time)                                       int64
reviewTime - time of the review (raw)                                                 Object
image - images that users post after they have received the product                   Object
verified                                                                              bool


------------------------------------------------------------------------------------------------------
pip install simplejson
pip install -U scikit-learn

from sklearn.metrics import mean_absolute_error as mae

#calculate MAE
mae(actual, pred)

#import necessary libraries
from sklearn.metrics import mean_squared_error
from math import sqrt

#calculate RMSE
sqrt(mean_squared_error(actual, pred)) 