# Main Program File

from python.cli_options import args
from python.dataset_info import datasets

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from random import shuffle




dataset = datasets[args.dataset]
# if dataset["has_header"]: f.readline()  # skip the header

if dataset["filetype"] == "CSV":
	df = pd.read_csv(dataset["train_name"], header=0) #header = dataset["has_header"])
	

if dataset["filetype"] == "arff":
	data,meta= arff.loadarff(open(dataset["train_name"], 'rb'))
	df = pd.DataFrame(data)

data = df.as_matrix()
data_X = data[:, dataset["X_col"]]  
data_y = data[:, dataset["Y_col"]] 


X, X_val, y, y_val = train_test_split(data_X,data_y, test_size=0.2,stratify=data_y)
y = y.flatten()
y_val = y_val.flatten()


if args.algs == None:
	algs = ["svm", 'rf', 'mlp', 'knn', 'mnb']
else: algs = args.algs

p = {}
for a in algs:
	p[a] = {}

if "svm" in algs:
	if args.svmPar != None:
		p["svm"] = {
			"kernel" : args.svmPar[0],
			"C" : float(args.svmPar[1])
		}
	else:
		p["svm"] = {
			"kernel" : 'rbf',
			"C" : 1.0
		}

if "rf" in algs:
	if args.rfPar != None:
		p["rf"] = {
			"n" : int(args.rfPar[0]),
			"d" : int(args.rfPar[1])
		}
	else:
		p["rf"] = {
			"n" : 10,
			"d" : None
		}

if "mlp" in algs:
	if args.mlpPar != None:
		p["mlp"] = {
			"act" : args.mlpPar[0],
			"alpha" : float(args.mlpPar[1])
		}
	else:
		p["mlp"] = {
			"act" : 'relu',
			"alpha" : 0.0001
		}

if "knn" in algs:
	if args.knnPar != None:
		p["knn"] = {
			"k" : int(args.knnPar[0]),
			"weights" : args.knnPar[1]
		}
	else:
		p["knn"] = {
			"k" : 5,
			"weights" : "uniform"
		}



##### LEARNING ####

if "knn" in algs:
	neigh = KNeighborsClassifier(
		n_neighbors=p["knn"]["k"],
		weights=p["knn"]["weights"])
	scores = cross_val_score(neigh, X, y, cv=args.folds)
	p["knn"]["acc"] = scores.mean()
	p["knn"]["model"] = neigh.fit(X,y)
	

if "svm" in algs:
	clf = svm.SVC(kernel=p["svm"]["kernel"],
		C=p["svm"]["C"])
	scores = cross_val_score(clf, X, y, cv=args.folds)
	p["svm"]["acc"] = scores.mean()
	p["svm"]["model"] = clf.fit(X,y)


if "mnb" in algs:
	clf = MultinomialNB()
	scores = cross_val_score(clf, X, y, cv=args.folds)
	p["mnb"]["acc"] = scores.mean()
	p["mnb"]["model"] = clf.fit(X,y)

if "mlp" in algs:
	clf = MLPClassifier(activation=p["mlp"]["act"],
		alpha=p["mlp"]["alpha"])
	scores = cross_val_score(clf, X, y, cv=args.folds)
	p["mlp"]["acc"] = scores.mean()
	p["mlp"]["model"] = clf.fit(X,y)
	
if "rf" in algs:
	clf = RandomForestClassifier(max_depth=p["rf"]["d"],
		n_estimators=p["rf"]["n"],n_jobs=-1)
	scores = cross_val_score(clf, X, y, cv=args.folds)
	p["rf"]["acc"] = scores.mean()
	p["rf"]["model"] = clf.fit(X,y)


def majority(votes, p, weighted):
	l = []
	for a,v in votes.iteritems():
		l += [v]
	if not weighted:	
		shuffle(l)	
		return(max(set(l), key=l.count)) #compute mode
	else:
		s = set(l)
		l_votes = {}
		for label in s:
			l_votes[label] = 0
			for a in algs:
				if votes[a] == label:
					l_votes[label] += p[a]["acc"]
		max_v = 0
		for l,v in l_votes.iteritems():
			if v > max_v:
				max_l = l
				max_v = v
		return(max_l)




### Ensemble predictions
votes = {}		# votes for each sample

alg_preds = {} 	# predictions for all algorithms
for a in algs:
	alg_preds[a] = p[a]["model"].predict(X_val)

#############################################################################

predictions = [] # predictions of the ensemble (majority)
for i in range(len(X_val)):
	for a in algs:
		votes[a] = alg_preds[a][i]
	r = majority(votes, p, args.weighted)
	predictions += [r]




c = 0.0

for i in predictions:
	if predictions[i] == y_val[i]: c+=1.0

print(len(y_val))

print(c/len(predictions))