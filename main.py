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
from sklearn import preprocessing
from random import shuffle

# args.dataset = "bank"
# args.folds = 5

dataset = datasets[args.dataset]
# if dataset["has_header"]: f.readline()  # skip the header

if dataset["filetype"] == "CSV":
    df = pd.read_csv(dataset["train_name"])  # header = 'infer'])

if dataset["filetype"] == "arff":
    data, meta = arff.loadarff(open(dataset["train_name"], 'rb'))
    df = pd.DataFrame(data)

data = df.as_matrix()
data_X = data[:, dataset["X_col"]]
data_y = data[:, dataset["Y_col"]]

if (dataset["encode_labels"]):
    for j in range(data_X.shape[1]):
        le = preprocessing.LabelEncoder()
        X_col = np.reshape(data_X[:, j], (data_X.shape[0], 1))
        le.fit(X_col)
        data_X[:, j] = le.transform(X_col)

    le = preprocessing.LabelEncoder()
    le.fit(data_y)
    data_y = le.transform(data_y)

X, X_val, y, y_val = train_test_split(data_X, data_y, test_size=0.2, random_state=46)  # ,stratify=data_y)
y = y.flatten()
y_val = y_val.flatten()

if args.algs == None:
    algs = ["svm", 'rf', 'mlp', 'knn', 'mnb']
else:
    algs = args.algs

p = {}
for a in algs:
    p[a] = {}

if "svm" in algs:
    if args.svmPar != None:
        p["svm"] = {
            "kernel": args.svmPar[0],
            "C": float(args.svmPar[1])
        }
    else:
        p["svm"] = {
            "kernel": 'rbf',
            "C": 1.0
        }

if "rf" in algs:
    if args.rfPar != None:
        p["rf"] = {
            "n": int(args.rfPar[0]),
            "d": int(args.rfPar[1])
        }
    else:
        p["rf"] = {
            "n": 10,
            "d": None
        }

if "mlp" in algs:
    if args.mlpPar != None:
        p["mlp"] = {
            "act": args.mlpPar[0],
            "alpha": float(args.mlpPar[1])
        }
    else:
        p["mlp"] = {
            "act": 'relu',
            "alpha": 0.0001
        }

if "knn" in algs:
    if args.knnPar != None:
        p["knn"] = {
            "k": int(args.knnPar[0]),
            "weights": args.knnPar[1]
        }
    else:
        p["knn"] = {
            "k": 5,
            "weights": "uniform"
        }

##### LEARNING ####

if "knn" in algs:
    neigh = KNeighborsClassifier(
        n_neighbors=p["knn"]["k"],
        weights=p["knn"]["weights"])
    scores = cross_val_score(neigh, X, y, cv=args.folds)
    p["knn"]["acc"] = scores.mean()
    p["knn"]["model"] = neigh.fit(X, y)

if "svm" in algs:
    svclf = svm.SVC(kernel=p["svm"]["kernel"],
                    C=p["svm"]["C"])
    scores = cross_val_score(svclf, X, y, cv=args.folds)
    p["svm"]["acc"] = scores.mean()
    p["svm"]["model"] = svclf.fit(X, y)

if "mnb" in algs:
    mnbclf = MultinomialNB()
    scores = cross_val_score(mnbclf, X, y, cv=args.folds)
    p["mnb"]["acc"] = scores.mean()
    p["mnb"]["model"] = mnbclf.fit(X, y)

if "mlp" in algs:
    mlpclf = MLPClassifier(activation=p["mlp"]["act"],
                           alpha=p["mlp"]["alpha"])
    scores = cross_val_score(mlpclf, X, y, cv=args.folds)
    p["mlp"]["acc"] = scores.mean()
    p["mlp"]["model"] = mlpclf.fit(X, y)

if "rf" in algs:
    rfclf = RandomForestClassifier(max_depth=p["rf"]["d"],
                                   n_estimators=p["rf"]["n"], n_jobs=-1)
    scores = cross_val_score(rfclf, X, y, cv=args.folds)
    p["rf"]["acc"] = scores.mean()
    p["rf"]["model"] = rfclf.fit(X, y)


def mode(l):
    return (max(set(l), key=l.count))


def majority(votes, p, weighted):
    l = []
    for a, v in votes.items():
        l += [v]
    if not weighted:
        shuffle(l)
        return (mode(l))
    else:
        s = set(l)
        l_votes = {}
        for label in s:
            l_votes[label] = 0
            for a in algs:
                if votes[a] == label:
                    l_votes[label] += p[a]["acc"]
        max_v = 0
        for l, v in l_votes.items():
            if v > max_v:
                max_l = l
                max_v = v
        return (max_l)


### Ensemble predictions
votes = {}  # votes for each sample

alg_preds = {}  # predictions for all algorithms
for a in algs:
    alg_preds[a] = p[a]["model"].predict(X_val)

#############################################################################

predictions = []  # predictions of the ensemble (majority)
for i in range(len(X_val)):
    for a in algs:
        votes[a] = alg_preds[a][i]
    r = majority(votes, p, args.weighted)
    predictions += [r]

c = 0.0

svmr = 0.0
rfr = 0.0
mnbr = 0.0
knnr = 0.0
mlpr = 0.0

for i in range(len(predictions)):
    if predictions[i] == y_val[i]: c += 1.0

    if "svm" in alg_preds and alg_preds["svm"][i] == y_val[i]: svmr += 1.0
    if "rf" in alg_preds and alg_preds["rf"][i] == y_val[i]: rfr += 1.0
    if "mnb" in alg_preds and alg_preds["mnb"][i] == y_val[i]: mnbr+=1.0
    if "knn" in alg_preds and alg_preds["knn"][i] == y_val[i]: knnr += 1.0
    if "mlp" in alg_preds and alg_preds["mlp"][i] == y_val[i]: mlpr += 1.0

if "svm" in algs:
    print("\nsvm : accuracy " + str(svmr / len(predictions)))
if "rf" in algs:
    print("\nrf : accuracy " + str(rfr / len(predictions)))
if "mnb" in algs:
    print("\nmnb : accuracy "+str(mnbr/len(predictions)))
if "knn" in algs:
    print("\nknn : accuracy " + str(knnr / len(predictions)))
if "mlp" in algs:
    print("\nmlp : accuracy " + str(mlpr / len(predictions)))



print("\nEnsemble : accuracy " + str(c / len(predictions)))
