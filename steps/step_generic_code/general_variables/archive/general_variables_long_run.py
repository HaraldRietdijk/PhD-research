
from sklearn.linear_model import SGDClassifier, LogisticRegressionCV
from sklearn.svm import NuSVC, SVC
from sklearn.ensemble import HistGradientBoostingClassifier
FITTING_PARAMETERS={
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
        'NSVC': [{'nu':[0.5],
                  'kernel':['poly'],
                  'degree':[3,4,5],
                  'gamma':['scale', 'auto'],
                  'probability':[True,False]
               }],
        'SVC' : [{'C':[0.9,1,1.1],
                  'kernel':['poly'],
                  'degree':[2,3,4,5],
                  'probability':[True,False]
               },{'C':[0.5,1,2],
                  'kernel':['rbf'],
                  'probability':[True,False]
               }],
        'HGB'  : [{'loss':['log_loss'],
                   'scoring':['accuracy','recall'],
                   'learning_rate':[0.1, 0.2],
                   'l2_regularization':[0.7,0.8,0.9],
                   'interaction_cst':['no_interactions']
        }]
}

CLASSIFIERS = [('LRCV', LogisticRegressionCV(),True),
               ('SGD',SGDClassifier(),False),
               ('NSVC',NuSVC(),True),
               ('SVC',SVC(),True),
               ('HGB',HistGradientBoostingClassifier(),True)]

CLASSIFIERS_WITH_PARAMETERS = [('LRCV', LogisticRegressionCV(),True),
               ('SGD',SGDClassifier(),False),
               ('NSVC',NuSVC(),True),
               ('SVC',SVC(),True),
               ('HGB',HistGradientBoostingClassifier(),True)]

ALGORITHM_PARAMETERS = { algorithm: [parameter for parameter in values[0]] for algorithm, values in FITTING_PARAMETERS.items()}
