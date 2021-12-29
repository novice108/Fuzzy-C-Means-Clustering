import pandas as pd
import numpy as np
import logging
from fuzzy_clustering import FCM
from visualization import draw_model_2d
from sklearn import preprocessing

"""
#from fuzzy_clustering import FCM
from fcmeans import FCM
from visualization import draw_model_2d
from sklearn import preprocessing
"""

dataset = pd.read_csv("AirlinesCluster.csv") #Importing the airlines data

dataset1 = dataset.copy() #Making a copy so that original data remains unaffected

dataset1 = dataset1[["Balance", "BonusMiles"]][:500] #Selecting only first 500 rows for faster computation


dataset1_standardized = preprocessing.scale(dataset1) #Standardizing the data to scale it between the upper and lower limit of 1 and 0

dataset1_standardized = pd.DataFrame(dataset1_standardized)

#fcm.set_logger(tostdout=False) #Telling the package class to stop the unnecessary output

fcm = FCM(n_clusters=5) #Defining k=5

fcm.fit(dataset1_standardized) #Training on data

predicted_membership = fcm.predict(np.array(dataset1_standardized)) #Testing on same data

draw_model_2d(fcm, data=np.array(dataset1_standardized), membership=predicted_membership) #Visualizing the data