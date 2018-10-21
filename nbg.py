import numpy
from sklearn import datasets, metrics
from sklearn.naive_bayes import GaussianNB

(data, targets)=datasets.load_iris(return_X_y=True); 

trainingset=data[range(0,150,2),:]
trainingsettarget=targets[range(0,150,2)]

testset=data[range(1,150,2),:]
testsettarget=targets[range(1,150,2)]

#creating instance of a classifier 
clf = GaussianNB() 

#BernoulliNB() for BernouliNB
#MultinomialNB() for MultinomialNB()

#train the model
clf.fit(trainingset, trainingsettarget)

#predict using the learnt classifier
prediction = clf.predict(testset) 


print("############### Predictions #################")
print(prediction)
print("#############################################")

print("Accuracy =",metrics.accuracy_score(testsettarget, predictions, normalize=True))
