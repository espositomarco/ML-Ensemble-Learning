import itertools
import pandas as pd
import numpy as np
from python import MajorityClassifier as maj
from python.dataset_info import datasets
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time

_TUNED_PARAMETERS = {'svm': {'kernel' : 'rbf', 'C': 1000},
                     'rf': {'n_estimators': 100, 'max_depth': None},
                     'knn': {'n_neighbors': 3, 'weights': 'distance'},
                     'mnb': {},
                     'mlp': {'activation': 'relu', 'alpha': 1.0000000000000001e-05}}

def main():
    header = "algs,dataset,weighted,optimized,accuracy,precision,recall,fscore,cv_mean,cv_std,runtime"
    output = "results/res.csv"
    algs = list(maj._CLASSIFIERS.keys())
    alg_combinations = list(itertools.combinations(algs, 1))+list(itertools.combinations(algs, 3)) + list(itertools.combinations(algs, 5))
    total_iter = len(list(datasets.keys()))*2*2*len(alg_combinations)
    current_iter = 0
    #with open(output, "a") as f:
    #    f.write(header + '\n')

    for dataset in list(datasets.keys()):
        for comb in alg_combinations:
            for optimized in [True, False]:
                for weighted in [True, False]:
                    current_iter += 1
                    print("[%s]%d/%d" % (dataset, current_iter, total_iter))
                    if len(comb) == 1 and weighted:
                        continue
                    params = {}
                    if optimized:
                        for a in comb:
                            params[a] = _TUNED_PARAMETERS[a]

                    start_time = time.time()
                    data_X, data_y = read_dataset(datasets[dataset])
                    X, X_val, y, y_val = train_test_split(data_X.copy(), data_y.copy(), test_size=0.2)
                    clf = maj.MajorityClassifier(comb, params, weighted=weighted)
                    clf.fit(X, y)
                    y_pred = clf.predict(X_val)

                    runtime = time.time() - start_time
                    cv = cross_val_score(clf, data_X, data_y, cv=3)
                    acc = accuracy_score(np.array(y_val).flatten(), y_pred)
                    prf = precision_recall_fscore_support(np.array(y_val).flatten(), y_pred, average='macro')

                    clf_label = "(%s)%s%s" % ('_'.join(comb), '_w' if weighted else '', '_opt' if optimized else '')
                    res_str = "%s,%s,%s,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f" % (clf_label,
                                                                               dataset,
                                                                               weighted,
                                                                               optimized,
                                                                               acc,
                                                                               prf[0],
                                                                               prf[1],
                                                                               prf[2],
                                                                               cv.mean(),
                                                                               cv.std(),
                                                                               runtime)
                    with open(output, "a") as f:
                        f.write(res_str + '\n')


def read_dataset(dataset):
    df = pd.read_csv(dataset["train_name"])  # header = 'infer'])
    data_X = df.iloc[:, dataset["X_col"]].copy()
    data_y = df.iloc[:, dataset["Y_col"]].copy()
    assert(data_y.columns[0] == 'Class')
    return data_X, data_y


if __name__=='__main__':
    main()