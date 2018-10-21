#K-means Clustering of IRIS Dataset

import numpy
from sklearn import datasets, metrics
from sklearn.cluster import KMeans

#load data
(data, targets)=datasets.load_iris(return_X_y=True); 

#create an instance of the model
model=KMeans(n_clusters=3)

#learn the model
model.fit(data)

#cluster numbers
prediction = model.labels_


print(prediction)

#prediction = numpy.choose(prediction, [0,1,2]).astype(numpy.int64)

#print(prediction)

