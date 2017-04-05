import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, f_classif, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.cross_validation import StratifiedKFold
from scipy import interp
import time	
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

import random


def plot_ic_criterion(model, name, color):
	alpha_ = model.alpha_
	alphas_ = model.alphas_
	criterion_ = model.criterion_
	plt.plot(-np.log10(alphas_), criterion_, '--', color=color,
	         linewidth=3, label='%s criterion' % name)
	plt.axvline(-np.log10(alpha_), color=color, linewidth=3,
	            label='alpha: %s estimate' % name)
	plt.xlabel('-log(alpha)')
	plt.ylabel('criterion')

def print_result(name, res):
	print " & ".join([name, str(res['score']), str(res['roc_auc']), str(res['precision']), str(res['recall']),\
	 str(res['f1']), str(res['time'])])  + "\\\\"

def main(argv):
	bag_file = argv[0]
	class_file = argv[1]
	vocab_file = argv[2]
	X = np.loadtxt(bag_file, delimiter=',')
	X_orig = X
	print X.shape
	y = np.loadtxt(class_file)

	vocab_full = np.loadtxt(vocab_file, dtype=str)

	h = .02  # step size in the mesh


	dt = DecisionTreeClassifier(max_depth=20).fit(X,y)
	model = SelectFromModel(dt, prefit=True)
	X = model.transform(X)
	print X.shape

	indeces = model.get_support(indices=True)
	print indeces
	print sum(dt.feature_importances_[indeces])

	vocab = vocab_full[indeces]


	#Ranking top 10 features
	top_10_feat_selector = SelectKBest(chi2, k=10)
	top_10_feat_selector.fit(X, y)
	print top_10_feat_selector.get_support(True)
	top10_feats = [vocab[feat] for feat in top_10_feat_selector.get_support(True)]	
	print top10_feats

	top_10_feat_selector = SelectKBest(k=10)
	top_10_feat_selector.fit(X, y)
	print top_10_feat_selector.get_support(True)
	top10_feats = [vocab[feat] for feat in top_10_feat_selector.get_support(True)]	
	print top10_feats

	top_10_feat_selector = SelectKBest(mutual_info_regression, k=10)
	top_10_feat_selector.fit(X, y)
	print top_10_feat_selector.get_support(True)
	top10_feats = [vocab[feat] for feat in top_10_feat_selector.get_support(True)]	
	print top10_feats


	names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",# "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Multinomial Naive Bayes", "QDA"] #BernoulliNB, MultinomialNB, not GaussianNB


    #Sample param tuning code (was removed later)

	# cv_scores = []
	# for k in range(1,5,1):
	# 	knn = KNeighborsClassifier(n_neighbors=k)
	# 	scores = cross_val_score(knn, X, y, cv=4, scoring='accuracy')
	# 	cv_scores.append(scores.mean())
	# MSE = [1 - x for x in cv_scores]
	# # determining best k
	# n_neighbors = MSE.index(min(MSE)) + 1
	# print n_neighbors

	#Param tuning for Neural Net
	# for i in range(10):
	# 	alpha_range = np.logspace(-4, 3, 8)
	# 	param_grid = dict(alpha=alpha_range)
	# 	cv = StratifiedShuffleSplit(n_splits=4, test_size=0.25, random_state=random.randint(1,100))
	# 	grid = GridSearchCV(MultinomialNB(), param_grid=param_grid, cv=cv)
	# 	grid.fit(X, y)

	# 	print("The best parameters are %s with a score of %0.2f"
	# 	      % (grid.best_params_, grid.best_score_))


	classifiers = [
    KNeighborsClassifier(3), #use cross-val to fit params.,
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=0.1, C=10, probability=True), #RBF
    #GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=.1), #Neural Net
    AdaBoostClassifier(),
    MultinomialNB(alpha=0.0001),
    QuadraticDiscriminantAnalysis()]

	X_train, X_test, y_train, y_test = \
		train_test_split(X, y, test_size=1.0/4, random_state=13)


	results = {}

	i = 0

	precisions = []
	recalls = []

	confidences = np.zeros((X.shape[0], 2))

	plt.figure()
	lw = 2

	colormap = plt.cm.rainbow(np.linspace(0,1,len(classifiers)))
	
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')

	folds = 4
	cv = StratifiedKFold(y, folds)

	for name, clf in zip(names, classifiers):
		print name
		results[name] = {}
		res = results[name]
		'''
		clf = clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		score = clf.score(X_test, y_test)
		results[name]['score'] = round(score, 4)
		'''

		#For the ROC Curves
		mean_tpr = 0.0
		mean_fpr = np.linspace(0, 1, 100)
		
		for k,(train,test) in enumerate(cv):
			probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])

			fpr, tpr, threshold = roc_curve(y[test], probas_[:, 1])
    		mean_tpr += interp(mean_fpr, fpr, tpr)
    		mean_tpr[0] = 0.0
    
		tpr[-1] = 1.0
		

		#4 Fold Cross Validation on the data
		start = time.time()
		predicted = cross_val_predict(clf, X, y, cv=4)



		res['time'] = round(time.time() - start,4)

		confidences += cross_val_predict(clf, X, y, cv=folds, method='predict_proba')

		res['predictions'] = predicted
		res['score'] = round(accuracy_score(y, predicted), 4)

		res['precision'] = round(sum(cross_val_score(clf, X, y, cv=4, scoring='precision'))/folds,4)
		res['recall'] = round(sum(cross_val_score(clf, X, y, cv=4, scoring='recall'))/folds,4)
		res['f1'] = round(2.0 * res['precision'] * res['recall'] / (res['precision'] + res['recall']), 4)

		y_score = None
		try:
			y_score = clf.predict_proba(X)[:,1]
		except AttributeError:
				print "Uh oh!"

		fpr, tpr, threshold_array = roc_curve(y, y_score, pos_label=1)
		
		res['fpr']= fpr
		res['tpr'] = tpr
		res['roc_auc'] = round(auc(fpr, tpr), 4)
		
		plt.plot(fpr, tpr, color=colormap[i],
	         lw=lw, label=name)
		i = i + 1

	confidences /= len(classifiers)

	'''conf_class1 = confidences[:,1]
	conf_class1 = max(conf_class1, 1-conf_class1)
	ind = np.argpartition(conf_class1, -10)[-10:]

	print ind'''


	plt.legend(loc=4, prop={'size':10})

	for name in sorted(results, key= lambda k: results[k]['score'], reverse=True):
		res = results[name]
		print_result(name, res)
	plt.show()
	sys.exit()



if __name__ == "__main__":
	main(sys.argv[1:])