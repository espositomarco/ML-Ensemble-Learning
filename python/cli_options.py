import argparse

parser = argparse.ArgumentParser(description="Train an ensemble classifier.")

parser.add_argument('-D','--dataset', type=str, default="none",help="The training set" )
parser.add_argument('-A','--algs', type=str, 
	nargs='*', help="Ensemble algorithms. Any choice among {svm, rf, mlp, knn, mnb}" )

'''
svm: Support Vector Machine. Parameters: kernel, C, epsilon

rf: Random Forest. Parameters: number of trees, max depth of trees. USE njobs=-1

mlp: Multi-Layer Perceptron. Parameters: activation function in {'identity', 'logistic', 'tanh', 'relu'}, alpha

knn: KNN. Parameters: number of neighbors, weigths in {'uniform','distance'}

mnb: Multinomial Naive Bayes, no parameters.
'''


parser.add_argument('-W','--weighted', action="store_true" )
parser.add_argument('-CV','--cv', action="store_true" )
parser.add_argument('-F','--folds', type=int, default=1, help="CV folds")
parser.add_argument('-U','--reps', type=int, default=1, help="CV runs")







parser.add_argument('-S','--svmPar', type=str, 
	nargs="*",help="SVM parameters: [kernel, C]. Either give values for all of them or for none." )

parser.add_argument('-R','--rfPar', type=str, 
	nargs="*",help="Random Forest parameters: [n_trees,max_depth]. Either give values for all of them or for none." )

parser.add_argument('-M','--mlpPar', type=str, 
	nargs="*",help="SVM parameters: [activation, alpha]. Either give values for all of them or for none." )

parser.add_argument('-K','--knnPar', type=str, 
	nargs="*",help="SVM parameters: [k,weights]. Either give values for all of them or for none." )



args = parser.parse_args()

print(args)