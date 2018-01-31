import MajorityClassifier as maj
import itertools
import pandas as pd
import numpy as np
from scipy.io import arff
from python.dataset_info import datasets
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time


params = {'svm': {'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'], 'C': [1,10,100]},
          'rf': {'n_estimators': [10, 30, 50, 100], 'max_depth': [None,1,3]},
          'mlp': {'activation': ['tanh', 'identity', 'relu', 'logistic'], 'alpha': [0.0001, 0.001, 1.0000000000000001e-05, 9.9999999999999995e-07]},
          'knn': {'n_neighbors': [1,3,5,8,10,30], 'weights': ['uniform', 'distance']}}

opt_params = {'svm': {'kernel' : 'rbf', 'C': 1000},
              'rf': {'n_estimators': 50, 'max_depth': None},
              'knn' : {'n_neighbors': 3, 'weights': 'distance'},
              'mnb' : {},
              'mlp' : {'activation': 'relu', 'alpha': 1.0000000000000001e-05}}

def main():
    num_runs = 3
    header = "algs,dataset,weighted,accuracy,precision,recall,fscore,cv_mean,cv_std,runtime"
    output = "results/1_3_5_ensemble_default_params.csv"
    algs = list(maj._CLASSIFIERS.keys())
    alg_combinations = list(itertools.combinations(algs, 1))+list(itertools.combinations(algs, 3)) + list(itertools.combinations(algs, 5))


    with open(output, "a") as f:
        f.write(header+'\n')
    for weighted in [False, True]:
        for comb in alg_combinations:
            if comb in ['svm', 'knn', 'mlp', 'rf', 'mnb'] and weighted: break
            for dataset in list(datasets.keys()):
                for _ in range(0, num_runs):
                    start_time = time.time()
                    data_X, data_y = read_dataset(datasets[dataset])
                    X, X_val, y, y_val = train_test_split(data_X.copy(), data_y.copy(), test_size=0.2)
                    clf = maj.MajorityClassifier(comb, weighted=weighted)
                    clf.fit(X, y)
                    y_pred = clf.predict(X_val)

                    runtime = time.time()-start_time
                    cv = cross_val_score(clf, data_X, data_y, cv=5)
                    acc = accuracy_score(np.array(y_val).flatten(), y_pred)
                    prf = precision_recall_fscore_support(np.array(y_val).flatten(), y_pred,average='macro')


                    res_str = "%s,%s,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f" % (' '.join(comb),
                                                                   dataset,
                                                                   weighted,
                                                                   acc,
                                                                   prf[0],
                                                                   prf[1],
                                                                   prf[2],
                                                                   cv.mean(),
                                                                   cv.std(),
                                                                   runtime)
                    print(res_str)
                    with open(output, "a") as f:
                        f.write(res_str + '\n')

def read_dataset(dataset):
    if dataset["filetype"] == "CSV":
        df = pd.read_csv(dataset["train_name"])  # header = 'infer'])
    if (dataset["encode_labels"]):
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category').cat.codes

    data_X = df.iloc[:, dataset["X_col"]].copy()
    data_y = df.iloc[:, dataset["Y_col"]].copy()
    return data_X, data_y


if __name__=='__main__':
    main()