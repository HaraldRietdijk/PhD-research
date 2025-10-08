from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier,HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, LogisticRegressionCV,\
                                Perceptron, RidgeClassifier, SGDOneClassSVM
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
import itertools


FITTING_PARAMETERS={
        'LDA' : [{'solver':['lsqr', 'eigen'],
                'shrinkage':['auto',None]
                },
                {'solver':['svd']
                }],
        'ADA' : [{'learning_rate':[0.1, 0.5, 1.0, 10.0],
                'n_estimators':[5,10,15,20,25],
                'algorithm':['SAMME']
        }],
        'RF'  : [{'criterion':['gini', 'entropy'],
                'max_features':[0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None],
                'n_estimators':[5,10, 20,30]
        }],
        'GBC' : [{'loss':['log_loss', 'exponential'],
                'learning_rate':[0.1, 0.5, 1.0, 10.0],
                'n_estimators':[5,10, 20,30],
                'criterion':['friedman_mse', 'squared_error'],
                'min_samples_split':[2,10,20]
        }],
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
        'SGD' : [{'fit_intercept':[True, False],
                 'loss':['huber'],
                 'penalty':['l1', 'l2', None],
                 'alpha':[0.00001,0.0001],
                 'epsilon':[0.2,0.25,0.3],
                 'learning_rate':['constant', 'invscaling', 'adaptive'],
                 'max_iter':[2000],
                 'eta0':[0.5,1,2]
               },{'fit_intercept':[True, False],
                 'loss':['hinge', 'log_loss'],
                 'penalty':['l1', 'l2', None],
                 'alpha':[0.00001,0.0001],
                 'learning_rate':['constant', 'invscaling', 'adaptive'],
                 'max_iter':[2000],
                 'eta0':[0.5,1,2]
               }],
        'LSVC': [{'penalty':['l2'],
                  'loss':['hinge', 'squared_hinge'],
                  'C':[0.5,1,2],
                  'fit_intercept':[True,False],
                  'dual':['auto'],
                  'max_iter':[3000]
               },{'penalty':['l1'],
                  'loss':['squared_hinge'],
                  'C':[0.5,1,2],
                  'fit_intercept':[True,False],
                  'dual':[False],
                  'max_iter':[3000]
        }],
        'DT' : [{'criterion':['gini', 'entropy', 'log_loss'],
                 'splitter':['best', 'random'],
                 'max_depth':[5,10,None],
                 'min_samples_split':[2,5,10],
                 'min_samples_leaf':[2,5,10],
                 'max_features':['sqrt', 'log2'],
                 'max_leaf_nodes':[100,None]
                }]
}

CLASSIFIERS = [('LDA',LinearDiscriminantAnalysis(),True,28),
               ('ADA',AdaBoostClassifier(n_estimators=15),True,14),
               ('RF',RandomForestClassifier(n_estimators=10, max_depth=4, min_weight_fraction_leaf=0.1),True,14),
               ('GBC', GradientBoostingClassifier(max_depth=2, min_weight_fraction_leaf=0.1),True,16),
               ('LR', LogisticRegression(),True,32),
               ('SGD',SGDClassifier(),False,4),
               ('LSVC',LinearSVC(),True,12),
               ('DT',DecisionTreeClassifier(max_depth=3, min_weight_fraction_leaf=0.1),True,14)
               ]

ALGORITHM_PARAMETERS = { algorithm: [parameter for parameter in values[0]] for algorithm, values in FITTING_PARAMETERS.items()}

