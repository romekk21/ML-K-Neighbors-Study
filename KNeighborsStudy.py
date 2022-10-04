import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot


bcdata = datasets.load_breast_cancer()
X, y = bcdata.data, bcdata.target
in_sample_err = []
error_rate = []

for i in range(7):
    k = pow(2, i)
    clf = KNeighborsClassifier(n_neighbors=k)
    error_rate.append(1- np.mean(cross_val_score(clf, X, y)))
    clf.fit(X, y)
    in_sample_err.append(1 - clf.score(X, y))


pyplot.plot(in_sample_err, label = 'In sample error')
pyplot.plot(error_rate, label = 'Out of sample error')
pyplot.xlabel("Number of neighbors (k=2^x)")
pyplot.ylabel("Error rate")
pyplot.legend()
pyplot.show()





