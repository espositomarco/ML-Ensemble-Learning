import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

_CLASSIFIERS = {'svm': SVC,
                'knn': KNeighborsClassifier,
                'mnb': MultinomialNB,
                'rf': RandomForestClassifier,
                'mlp': MLPClassifier}

"""A refactored majority classifier that inherits from base classifiers and 
allows for better integration with other scikit tools"""
class MajorityClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, algs=None, params=None, weighted=False, folds=3):
        if not algs:
            algs = ['svm','knn','mnb','rf','mlp']
        if not params:
            params = {'svm': {}, 'knn': {}, 'mnb': {}, 'rf': {},'mlp': {}}
        self.classifiers = [_CLASSIFIERS[a](**(params[a])) for a in algs]
        self.weighted = weighted
        self.weights = [1]*len(algs)
        self.folds = folds

    def fit(self, X, y=None):
        for i, clf in enumerate(self.classifiers):
            clf.fit(X, np.ravel(y))
            if self.weighted:
                self.weights[i] = cross_val_score(clf, X, np.ravel(y), cv=self.folds).mean()

    def predict(self, X):
        return np.apply_along_axis(axis=1,
                                   arr=np.array(X),
                                   func1d=lambda x:
                                   self._count(np.concatenate([c.predict([x]) for c in self.classifiers])).argmax(),
                                   )

    def score(self, X, y=None, sample_weight=None):
        y_pred = self.predict(X)
        y_val = y.as_matrix().flatten()
        tp = 0
        for i in range(0,len(y)):
            tp += y_pred[i] == y_val[i]
        res = tp/len(y_val)
        return res

    def _count(self, votes):
        res = np.bincount(votes)
        if self.weighted:
            res = np.zeros((1, len(res)))
            for i, vote in enumerate(votes):
                res[0,vote] += self.weights[i]
        return res
