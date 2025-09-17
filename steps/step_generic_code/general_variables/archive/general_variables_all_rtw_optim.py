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
        'LDA1' : [{'solver':['lsqr', 'eigen'],
                'shrinkage':['auto',None]
                },
                {'solver':['svd']
                }],
        'GBC1' : [{'loss':['log_loss', 'exponential'],
                'learning_rate':[0.1, 0.5, 1.0, 10.0],
                'n_estimators':[5,10, 20,30],
                'criterion':['friedman_mse', 'squared_error'],
                'min_samples_split':[2,10,20]
        }],
        'LR1' : [{'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 
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
        'LSVC1': [{'penalty':['l2'],
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
        }]
}

CLASSIFIERS = [('LDA1',LinearDiscriminantAnalysis(),True,28),
               ('GBC1', GradientBoostingClassifier(max_depth=2, min_weight_fraction_leaf=0.1),True,16),
               ('LR1', LogisticRegression(),True,32),
               ('LSVC1',LinearSVC(),True,12)]

CLASSIFIERS_WITH_PARAMETERS = [('LDA1',LinearDiscriminantAnalysis(),True),
               ('GBC1', GradientBoostingClassifier(),True),
               ('LR1', LogisticRegression(),True),
               ('LSVC1',LinearSVC(),True)]

ALGORITHM_PARAMETERS = { algorithm: [parameter for parameter in values[0]] for algorithm, values in FITTING_PARAMETERS.items()}
