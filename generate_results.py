import MajorityClassifier as maj
import itertools
import pandas as pd
import numpy as np
from scipy.io import arff
from python.dataset_info import datasets
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time


def main():
    num_runs = 1
    header = "algs,dataset,accuracy,precision,recall,fscore,cv_mean,cv_std,runtime"
    output = "results/Majority_Weighted_New_sets_only_Default_params_All_combs.csv"
    algs = list(maj._CLASSIFIERS.keys())
    alg_combinations = list(itertools.combinations(algs, 3)) + list(itertools.combinations(algs, 5))


    with open(output, "a") as f:
        f.write(header+'\n')

    for comb in alg_combinations:
        for dataset in list(datasets.keys()):
            for _ in range(0, num_runs):
                start_time = time.time()
                data_X, data_y = read_dataset(datasets[dataset])
                X, X_val, y, y_val = train_test_split(data_X.copy(), data_y.copy(), test_size=0.2, random_state=46)
                clf = maj.MajorityClassifier(comb, weighted=True)
                clf.fit(X, y)
                y_pred = clf.predict(X_val)

                runtime = time.time()-start_time
                cv = cross_val_score(clf, data_X, data_y, cv=5)
                acc = accuracy_score(np.array(y_val).flatten(), y_pred)
                prf = precision_recall_fscore_support(np.array(y_val).flatten(), y_pred,average='macro')

                res_str = "%s,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f" % (' '.join(comb),
                                                               dataset,
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
    class_label = df.columns[dataset["Y_col"]]
    data_X = df.drop(class_label,axis=1)
    data_y = df[class_label].copy()
    return data_X, data_y


if __name__=='__main__':
    main()