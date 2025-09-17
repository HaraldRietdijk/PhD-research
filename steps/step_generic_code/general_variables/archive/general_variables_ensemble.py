from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier,\
                        HistGradientBoostingClassifier, RandomForestClassifier



FITTING_PARAMETERS= {
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
        'GBC' : [{'loss':['exponential', 'log_loss'],
                'learning_rate':[0.1, 0.5, 1.0, 10.0],
                'n_estimators':[10, 50, 100, 500],
                'criterion':['friedman_mse', 'squared_error'],
                'min_samples_split':[2,10,50]
        }],
        'HGB'  : [{'loss':['log_loss'],
                   'scoring':['accuracy','f1','loss','precision'],
                   'learning_rate':[0.1, 0.5, 1.0, 10.0],
                   'l2_regularization':[0.2,0.35,0.5,0.65,0.8],
                   'interaction_cst':['pairwise', 'no_interactions']
        }]
}

CLASSIFIERS = [('BAC',BaggingClassifier(),True),
               ('ADA',AdaBoostClassifier(),True),
               ('RF',RandomForestClassifier(),True),
               ('ETSC',ExtraTreesClassifier(),True),
               ('GBC', GradientBoostingClassifier(),True),
               ('HGB', HistGradientBoostingClassifier(),True)]

CLASSIFIERS_WITH_PARAMETERS = [('BAC',BaggingClassifier(),True),
                               ('ADA',AdaBoostClassifier(),True),
                               ('RF',RandomForestClassifier(),True),
                               ('ETSC',ExtraTreesClassifier(),True),
                               ('GBC', GradientBoostingClassifier(),True),
                               ('HGB', HistGradientBoostingClassifier(),True)]

ALGORITHM_PARAMETERS = { algorithm: [parameter for parameter in values[0]] for algorithm, values in FITTING_PARAMETERS.items()}
