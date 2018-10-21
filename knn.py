import numpy
from sklearn import neighbors, datasets, metrics

(data, targets)=datasets.load_iris(return_X_y=True); 

trainingset=data[range(0,150,2),:]   
trainingsettarget=targets[range(0,150,2)]

testset=data[range(1,150,2),:]
testsettarget=targets[range(1,150,2)]


#creating instance of a classifier
clf = neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform') 
#or weights='distance'

#other argument is metric (default is 'minkowski'. There are many other possibilities including 'euclidean')

#Other useful argument is p which controls power of minkowski distance. Default is 2

#train the model
clf.fit(trainingset, trainingsettarget)

#predict using the learnt classifier
prediction = clf.predict(testset) 

print("############### Predictions #################")
print(prediction)
print("#############################################")

#print accuracy
print("Accuracy =", metrics.accuracy_score(testsettarget, prediction, normalize=True))