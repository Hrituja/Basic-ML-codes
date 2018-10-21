#on boston house-prices dataset
#Samples Total 506
#Dimensionality 13
#Features real, positive
#targets real

import numpy as np
from sklearn import datasets, linear_model, metrics

# Load the boston dataset
(data, targets) = datasets.load_boston(return_X_y=True)    

trainingset=data[0:400,:]   #boston.data is numpy array 
trainingsettarget=targets[0:400]

testset=data[400:506,:]
testsettarget=targets[400:506]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(trainingset, trainingsettarget)

#Use the learnt model to predict
predictions=regr.predict(testset)

print("############### Predictions #################")
print(predictions)
print("#############################################")

print("MAE =", metrics.mean_absolute_error (testsettarget,predictions))

print("MSE = ", metrics.mean_squared_error(testsettarget,predictions))

