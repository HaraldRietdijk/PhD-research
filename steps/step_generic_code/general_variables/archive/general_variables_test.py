from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import FeatureAgglomeration
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import FastICA, PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectFwe, SelectKBest, SelectPercentile, VarianceThreshold, SelectFromModel, RFE
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, LinearRegression, SGDRegressor, TweedieRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC, SVC, OneClassSVM, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


FITTING_PARAMETERS={
        'LR' : [{'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                'penalty':['elasticnet'],
                'fit_intercept':[True, False],
                'l1_ratio':[0.2,0.35,0.5,0.65,0.8],
                'solver':['saga'] 
                },
                {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                'penalty':['l1'],
                'fit_intercept':[True, False],
                'solver':['saga'] 
                },
                {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                'penalty':['l1', 'l2'], # None and elasticnet
                'fit_intercept':[True, False],
                'solver':['liblinear']
                },
                {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                'penalty':['l2'],
                'fit_intercept':[True, False],
                'solver':['lbfgs','newton-cg', 'newton-cholesky', 'sag', 'saga'] 
                },
                {'penalty':[None],
                'fit_intercept':[True, False],
                'solver':['lbfgs','newton-cg', 'newton-cholesky', 'sag', 'saga'] 
                }
        ],
        'ADA' : [{'learning_rate':[0.1, 0.5, 1.0, 10.0],
                'n_estimators':[10, 50],
                'algorithm':['SAMME']
        }],
        'RF' : [{'criterion':['gini', 'entropy'],
                'max_features':[0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None],
                'n_estimators':[10, 50, 100, 500]
        }],
        'DT' : [{'criterion':['gini','entropy'],
                'max_features':['sqrt','log2']
        }],
        'NN' : [{'learning_rate':['constant', 'invscaling', 'adaptive'],
                'activation':['identity', 'logistic', 'tanh', 'relu'],
                'learning_rate_init':[0.01, 0.05, 0.1, 0.5, 1.0]
        }],
        'SGD' : [{'fit_intercept':[True, False],
                 'l1_ratio':[0, 0.15, 1.0],
                 'loss':['squared_error', 'squared_hinge', 'hinge', 'perceptron', 'log_loss', 'epsilon_insensitive', 'squared_epsilon_insensitive', 'modified_huber', 'huber']
        }],
        'SVC' : [{'kernel':['sigmoid','rbf'],
                 'probability':[True]
        }],
        'OCS' : [{'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
                 'degree':[3,5,7],
                 'shrinking':[True,False]
        }],
        'KNN' : [{'metric':['minkowski','euclidean','manhattan'],
                 'weights':['uniform','distance'],
                 'n_neighbors':[5, 6, 7, 8, 9]
        }]
}

CLASSIFIERS = [('ADA',AdaBoostClassifier(),True)]

CLASSIFIERS_WITH_PARAMETERS = [('ADA',AdaBoostClassifier(algorithm='SAMME'),True)]

ALGORITHM_PARAMETERS = { algorithm: [parameter for parameter in values[0]] for algorithm, values in FITTING_PARAMETERS.items()}

REGRESSORS = [('LINR', LinearRegression(), False, 1),
               ('DTR', DecisionTreeRegressor(), False, 1),
               ('ADAR', AdaBoostRegressor(), False, 1),
               ('LSVR', LinearSVR(), False, 1),
               ('RFR', RandomForestRegressor(), False, 1),
               ('TWDR', TweedieRegressor(), False, 1)]
