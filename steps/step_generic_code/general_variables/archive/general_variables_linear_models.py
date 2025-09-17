
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, LogisticRegressionCV,\
                                Perceptron, RidgeClassifier, SGDOneClassSVM

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
        'LRCV' : [{'Cs':[1, 10, 100, 1000], 
                'penalty':['elasticnet'],
                'fit_intercept':[True, False],
                'l1_ratios':[[0.2,0.35,0.5,0.65,0.8]],
                'solver':['saga'],
                'scoring':['accuracy','recall'],
                'max_iter':[1000]
                },
                {'Cs':[1, 10, 100, 1000], 
                'penalty':['l1'],
                'fit_intercept':[True, False],
                'solver':['saga'],
                'scoring':['accuracy','recall'],
                'max_iter':[1000] 
                },
                {'Cs':[1, 10, 100, 1000], 
                'penalty':['l1', 'l2'], # None and elasticnet
                'fit_intercept':[True, False],
                'scoring':['accuracy','recall'],
                'solver':['liblinear'],
                'max_iter':[1000]
                },
                {'Cs':[1, 10, 100, 1000], 
                'penalty':['l2'],
                'fit_intercept':[True, False],
                'scoring':['accuracy','recall'],
                'solver':['lbfgs','newton-cg', 'newton-cholesky', 'sag', 'saga'],
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
                 'l1_ratio':[0.2,0.35,0.5,0.65,0.8],
                 'loss':['epsilon_insensitive', 'squared_epsilon_insensitive', 'huber'],
                 'penalty':['elasticnet'],
                 'alpha':[0.00001,0.0001,0.001,0.01,0.1],
                 'epsilon':[0.01,0.1,0.25,0.5],
                 'learning_rate':['constant', 'invscaling', 'adaptive'],
                 'average':[True,False,10,50],
                 'max_iter':[2000],
                 'eta0':[0.1,1,100]
               },{'fit_intercept':[True, False],
                 'l1_ratio':[0.2,0.35,0.5,0.65,0.8],
                 'loss':['squared_error', 'squared_hinge', 'hinge', 'perceptron', 'log_loss', 'modified_huber'],
                 'penalty':['elasticnet'],
                 'alpha':[0.00001,0.0001,0.001,0.01,0.1],
                 'learning_rate':['constant', 'invscaling', 'adaptive'],
                 'average':[True,False,10,50],
                 'max_iter':[2000],
                 'eta0':[0.1,1,100]
               },{'fit_intercept':[True, False],
                 'loss':['epsilon_insensitive', 'squared_epsilon_insensitive', 'huber'],
                 'penalty':['l1', 'l2', None],
                 'alpha':[0.00001,0.0001,0.001,0.01,0.1],
                 'epsilon':[0.01,0.1,0.25,0.5],
                 'learning_rate':['constant', 'invscaling', 'adaptive'],
                 'average':[True,False,10,50],
                 'max_iter':[2000],
                 'eta0':[0.1,1,100]
               },{'fit_intercept':[True, False],
                 'loss':['squared_error', 'squared_hinge', 'hinge', 'perceptron', 'log_loss', 'modified_huber'],
                 'penalty':['l1', 'l2', None],
                 'alpha':[0.00001,0.0001,0.001,0.01,0.1],
                 'learning_rate':['constant', 'invscaling', 'adaptive'],
                 'average':[True,False,10,50],
                 'max_iter':[2000],
                 'eta0':[0.1,1,100]
               },{'fit_intercept':[True, False],
                 'l1_ratio':[0.2,0.35,0.5,0.65,0.8],
                 'loss':['epsilon_insensitive', 'squared_epsilon_insensitive', 'huber'],
                 'penalty':['elasticnet'],
                 'alpha':[0.00001,0.0001,0.001,0.01,0.1],
                 'epsilon':[0.01,0.1,0.25,0.5],
                 'learning_rate':['optimal'],
                 'average':[True,False,10,50],
                 'max_iter':[2000]
               },{'fit_intercept':[True, False],
                 'l1_ratio':[0.2,0.35,0.5,0.65,0.8],
                 'loss':['squared_error', 'squared_hinge', 'hinge', 'perceptron', 'log_loss', 'modified_huber'],
                 'penalty':['elasticnet'],
                 'alpha':[0.00001,0.0001,0.001,0.01,0.1],
                 'learning_rate':['optimal'],
                 'average':[True,False,10,50],
                 'max_iter':[2000]
               },{'fit_intercept':[True, False],
                 'loss':['epsilon_insensitive', 'squared_epsilon_insensitive', 'huber'],
                 'penalty':['l1', 'l2', None],
                 'alpha':[0.00001,0.0001,0.001,0.01,0.1],
                 'epsilon':[0.01,0.1,0.25,0.5],
                 'learning_rate':['optimal'],
                 'average':[True,False,10,50],
                 'max_iter':[2000]
               },{'fit_intercept':[True, False],
                 'loss':['squared_error', 'squared_hinge', 'hinge', 'perceptron', 'log_loss', 'modified_huber'],
                 'penalty':['l1', 'l2', None],
                 'alpha':[0.00001,0.0001,0.001,0.01,0.1],
                 'learning_rate':['optimal'],
                 'average':[True,False,10,50],
                 'max_iter':[2000]
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
        }]
}

CLASSIFIERS = [('PAC', PassiveAggressiveClassifier(),False),
               ('PER', Perceptron(),True),
               ('LRCV', LogisticRegressionCV(),True),
               ('LR', LogisticRegression(),True),
               ('RIC', RidgeClassifier(),True),
               ('SGD',SGDClassifier(),False),
               ('SGDOC',SGDOneClassSVM(),False)]

CLASSIFIERS_WITH_PARAMETERS = [('PAC', PassiveAggressiveClassifier(),False),
               ('PER', Perceptron(),True),
               ('LRCV', LogisticRegressionCV(),True),
               ('LR', LogisticRegression(),True),
               ('RIC', RidgeClassifier(),True),
               ('SGD',SGDClassifier(),False),
               ('SGDOC',SGDOneClassSVM(),False)]

ALGORITHM_PARAMETERS = { algorithm: [parameter for parameter in values[0]] for algorithm, values in FITTING_PARAMETERS.items()}
