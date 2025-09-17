# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import BaggingClassifier
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import HistGradientBoostingClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import StackingClassifier
# from sklearn.ensemble import VotingClassifier
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.linear_model import SGDClassifier
# from sklearn.linear_model import RidgeClassifierCV
# from sklearn.linear_model import RidgeClassifier
# from sklearn.linear_model import PassiveAggressiveClassifier    
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LogisticRegressionCV
# from sklearn.linear_model import Perceptron
# from sklearn.mixture import DPGMM
# from sklearn.mixture import GMM 
# from sklearn.mixture import GaussianMixture
# from sklearn.mixture import VBGMM
# from sklearn.multiclass import OutputCodeClassifier
# from sklearn.multiclass import OneVsOneClassifier
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.multioutput import ClassifierChain
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.naive_bayes import BernoulliNB
# from sklearn.naive_bayes import GaussianNB
# from sklearn.naive_bayes import MultinomialNB  
# from sklearn.neighbors import RadiusNeighborsClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neighbors import NearestCentroid
# from sklearn.neural_network import MLPClassifier
# from sklearn.semi_supervised import LabelPropagation
# from sklearn.semi_supervised import LabelSpreading
# from sklearn.svm import OneClassSVM
# from sklearn.svm import LinearSVC
# from sklearn.svm import NuSVC
# from sklearn.svm import SVC
# from sklearn.tree import ExtraTreeClassifier
# from sklearn.tree import DecisionTreeClassifier

# import nltk
# import sklearn

# # print('The nltk version is {}.'.format(nltk.__version__))
# print('The scikit-learn version is {}.'.format(sklearn.__version__))

from sklearn.utils import all_estimators

estimators = all_estimators(type_filter='classifier') # 'classifier', 'regressor', 'transformer', 'cluster' or None

i = 0
for name, class_ in estimators:
    print(f'{i}. {class_.__name__}')
    i += 1






# AdaBoostClassifier
# BaggingClassifier
# BayesianGaussianMixture
# BernoulliNB
# CalibratedClassifierCV
# CategoricalNB
# ClassifierChain
# ComplementNB
# DecisionTreeClassifier
# DummyClassifier
# ExtraTreeClassifier
# ExtraTreesClassifier
# GaussianMixture
# GaussianNB
# GaussianProcessClassifier
# GradientBoostingClassifier
# GridSearchCV
# HalvingGridSearchCV
# HalvingRandomSearchCV
# HistGradientBoostingClassifier
# KNeighborsClassifier
# LabelPropagation
# LabelSpreading
# LinearDiscriminantAnalysis
# LogisticRegression
# LogisticRegressionCV
# MLPClassifier
# MultiOutputClassifier
# MultinomialNB
# NuSVC
# OneVsRestClassifier
# Pipeline
# QuadraticDiscriminantAnalysis
# RFE
# RFECV
# RadiusNeighborsClassifier
# RandomForestClassifier
# RandomizedSearchCV
# SGDClassifier
# SVC
# SelfTrainingClassifier
# StackingClassifier
# VotingClassifier