import numpy
from sklearn import datasets, metrics
from sklearn.tree import DecisionTreeClassifier

(data, targets)=datasets.load_iris(return_X_y=True); 

trainingset=data[range(0,150,2),:]
trainingsettarget=targets[range(0,150,2)]

testset=data[range(1,150,2),:]
testsettarget=targets[range(1,150,2)]

#creating instance of a classifier 
clf = DecisionTreeClassifier(criterion='entropy')
#default criterion is gini

#train the model
clf.fit(trainingset, trainingsettarget)

#predict using the learnt classifier
predictions = clf.predict(testset) 

print("############### Predictions #################")
print(predictions)
print("#############################################")

print("Accuracy =",metrics.accuracy_score(testsettarget, predictions, normalize=True))
