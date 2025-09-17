from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier,HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, LogisticRegressionCV,\
                                Perceptron, RidgeClassifier, SGDOneClassSVM
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier


FITTING_PARAMETERS={
        'LDA' : [{'solver':['lsqr', 'eigen'],
                'shrinkage':['auto',None]
                },
                {'solver':['svd']
                }],
        'QDA' : [{'store_covariance':[True,False]
        }],
        'DUM' : [{'strategy':['constant'],
                  'constant':[1,0]
                },
                {'strategy':['most_frequent', 'prior', 'stratified', 'uniform']
                }],
        'BAC' : [{'n_estimators':[5,10,50]}],
        'ADA' : [{'learning_rate':[0.1, 0.5, 1.0, 10.0],
                'n_estimators':[10, 50],
                'algorithm':['SAMME']
        }],
        'ETSC' : [{'n_estimators':[5,10,50],
                   'criterion':['gini', 'entropy', 'log_loss'],
                   'min_samples_split':[2,10,50],
                   'max_features': [0.1, 0.25, 0.5, 0.75,'sqrt', 'log2', None]
        }],
        'RF'  : [{'criterion':['gini', 'entropy'],
                'max_features':[0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None],
                'n_estimators':[10, 50, 100, 500]
        }],
        'GBC' : [{'loss':['log_loss', 'exponential'],
                'learning_rate':[0.1, 0.5, 1.0, 10.0],
                'n_estimators':[10, 50, 100, 500],
                'criterion':['friedman_mse', 'squared_error'],
                'min_samples_split':[2,10,50]
        }],
        'HGB'  : [{'loss':['log_loss'],
                   'scoring':['accuracy','recall'],
                   'learning_rate':[0.1, 0.2],
                   'l2_regularization':[0.7,0.8,0.9],
                   'interaction_cst':['no_interactions']
        }],
        'GPC' : [{'kernel':[ConstantKernel(),None],
                  'optimizer':['fmin_l_bfgs_b', None]
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
        'LRCV' : [{'Cs':[1, 2], 
                'penalty':['elasticnet'],
                'fit_intercept':[True, False],
                'l1_ratios':[[0,0.25,0.5,0.75,1]],
                'solver':['saga'],
                'max_iter':[1000]
                },
                {'Cs':[1, 2], 
                'penalty':['l1', 'l2'],
                'fit_intercept':[True, False],
                'solver':['liblinear'],
                'max_iter':[1000]
                },
                {'Cs':[1,2], 
                'penalty':['l2'],
                'fit_intercept':[True, False],
                'solver':['lbfgs','newton-cg', 'newton-cholesky', 'sag'],
                'max_iter':[1000] 
                }
        ],
        'PAC' : [{'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                'fit_intercept':[True,False], 
                'loss':['squared_hinge', 'hinge'],
                'average':[True,False,10,50]

        }],
        'PER' : [{'penalty':['elasticnet'],
                'alpha':[0.00001,0.0001,0.001,0.01,0.1],
                'l1_ratio':[0.2,0.35,0.5,0.65,0.8],
                'fit_intercept':[True,False]
               },{'penalty':['l1', 'l2', None],
                'alpha':[0.00001,0.0001,0.001,0.01,0.1],
                'fit_intercept':[True,False]
        }],
        'RIC' : [{'alpha':[0.00001,0.0001,0.001,0.01,0.1],
                'fit_intercept':[True,False],
                'solver':['lbfgs'],
                'max_iter':[50,100],
                'positive':[True]
                },{'alpha':[0.00001,0.0001,0.001,0.01,0.1],
                'fit_intercept':[True,False],
                'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                'max_iter':[50,100],
                'positive':[False]
        }],
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
        'SGDOC': [{'nu':[0.2,0.35,0.5,0.65,0.8],
                 'fit_intercept':[True,False],
                 'learning_rate':['constant', 'invscaling', 'adaptive'],
                 'average':[True,False,10,50],
                 'eta0':[0.1,1,100]
                },{'nu':[0.2,0.35,0.5,0.65,0.8],
                 'fit_intercept':[True,False],
                 'learning_rate':['optimal'],
                 'average':[True,False,10,50]
        }],
        'BNB' : [{'alpha':[0.000001,0.0001,0.01,0.1,0.5],
                  'force_alpha':[False],
               },{'alpha':[0],
                  'force_alpha':[True]
        }],
        'CONB': [{'alpha':[0.000001,0.0001,0.01,0.1,0.5],
                  'force_alpha':[False],
                  'norm':[True,False]
               },{'alpha':[0],
                  'force_alpha':[True],
                  'norm':[True,False]
        }],
        'GNB' : [{'var_smoothing':[0.000000001,0.00001,0.1,0.5]
        }],
        'KNN' : [{'metric':['minkowski','euclidean','manhattan'],
                 'weights':['uniform','distance'],
                 'n_neighbors':[5, 6, 7, 8, 9],
                 'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
                 'leaf_size':[5,10,30,50]
        }],
        'NCC' : [{'metric':['euclidean','manhattan'],
                  'shrink_threshold':[0.1,1,10,50]

        }],
        'RNN'  : [{'radius':[1,5,10],
                   'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
                   'leaf_size':[5,10,30,50],
                   'metric':['minkowski','euclidean','manhattan']
        }],
        'NN'  : [{'learning_rate':['constant', 'invscaling', 'adaptive'],
                'activation':['identity', 'logistic', 'tanh', 'relu'],
                'learning_rate_init':[0.05, 0.1, 0.5, 1.0]
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
        # 'NSVC': [{'nu':[0.5],
        #           'kernel':['poly'],
        #           'degree':[3,4,5],
        #           'gamma':['scale', 'auto'],
        #           'probability':[True,False]
        #        }],
        'SVC' : [{'C':[0.9,1,1.1],
                  'kernel':['poly'],
                  'degree':[2,3,4,5],
                  'probability':[True,False]
               },{'C':[0.5,1,2],
                  'kernel':['rbf'],
                  'probability':[True,False]
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

CLASSIFIERS = [('LDA',LinearDiscriminantAnalysis(),True),
               ('QDA',QuadraticDiscriminantAnalysis(),True),
               ('DUM',DummyClassifier(),True),
               ('BAC',BaggingClassifier(),True),
               ('ADA',AdaBoostClassifier(),True),
               ('RF',RandomForestClassifier(),True),
               ('ETSC',ExtraTreesClassifier(),True),
               ('GBC', GradientBoostingClassifier(),True),
               ('HGB', HistGradientBoostingClassifier(),True),
               ('GPC',GaussianProcessClassifier(),True),
               ('PAC', PassiveAggressiveClassifier(),False),
               ('PER', Perceptron(),True),
               ('LRCV', LogisticRegressionCV(),True),
               ('LR', LogisticRegression(),True),
               ('RIC', RidgeClassifier(),True),
               ('SGD',SGDClassifier(),False),
               ('SGDOC',SGDOneClassSVM(),False),
               ('BNB',BernoulliNB(),True),
               ('CONB',ComplementNB(),True),
               ('GNB',GaussianNB(),True),
               ('KNN',KNeighborsClassifier(),True),
               ('NCC',NearestCentroid(),True),
               ('RNN',RadiusNeighborsClassifier(),True),
               ('NN',MLPClassifier(),True),
               ('LSVC',LinearSVC(),True),
        #        ('NSVC',NuSVC(),True),
               ('SVC',SVC(),True),
               ('DT',DecisionTreeClassifier(),True)]

CLASSIFIERS_WITH_PARAMETERS = [('LDA',LinearDiscriminantAnalysis(),True),
               ('QDA',QuadraticDiscriminantAnalysis(),True),
               ('DUM',DummyClassifier(),True),
               ('BAC',BaggingClassifier(),True),
               ('ADA',AdaBoostClassifier(),True),
               ('RF',RandomForestClassifier(),True),
               ('ETSC',ExtraTreesClassifier(),True),
               ('GBC', GradientBoostingClassifier(),True),
               ('HGB', HistGradientBoostingClassifier(),True),
               ('GPC',GaussianProcessClassifier(),True),
               ('PAC', PassiveAggressiveClassifier(),False),
               ('PER', Perceptron(),True),
               ('LRCV', LogisticRegressionCV(),True),
               ('LR', LogisticRegression(),True),
               ('RIC', RidgeClassifier(),True),
               ('SGD',SGDClassifier(),False),
               ('SGDOC',SGDOneClassSVM(),False),
               ('BNB',BernoulliNB(),True),
               ('CONB',ComplementNB(),True),
               ('GNB',GaussianNB(),True),
               ('KNN',KNeighborsClassifier(),True),
               ('NCC',NearestCentroid(),True),
               ('RNN',RadiusNeighborsClassifier(),True),
               ('NN',MLPClassifier(),True),
               ('LSVC',LinearSVC(),True),
        #        ('NSVC',NuSVC(),True),
               ('SVC',SVC(),True),
               ('DT',DecisionTreeClassifier(),True)]

ALGORITHM_PARAMETERS = { algorithm: [parameter for parameter in values[0]] for algorithm, values in FITTING_PARAMETERS.items()}
