from sklearn.svm import SVC, NuSVC, LinearSVC



FITTING_PARAMETERS= {
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
        'NSVC': [{'nu':[0.4,0.5,0.6],
                  'kernel':['poly'],
                  'degree':[2,3,4,5],
                  'gamma':['scale', 'auto'],
                  'shrinking':[True,False],
                  'probability':[True,False]
               },{'nu':[0.4,0.5,0.6],
                  'kernel':['rbf', 'sigmoid'],
                  'gamma':['scale', 'auto'],
                  'shrinking':[True,False],
                  'probability':[True,False]
               },{'nu':[0.4,0.5,0.6],
                  'kernel':['linear'],
                  'shrinking':[True,False],
                  'probability':[True,False]
        }],
        'SVC' : [{'C':[0.5,1,2],
                  'kernel':['poly'],
                  'degree':[2,3,4,5],
                  'gamma':['scale', 'auto'],
                  'shrinking':[True,False],
                  'probability':[True,False]
               },{'C':[0.5,1,2],
                  'kernel':['rbf', 'sigmoid'],
                  'gamma':['scale', 'auto'],
                  'shrinking':[True,False],
                  'probability':[True,False]
               },{'C':[0.5,1,2],
                  'kernel':['linear'],
                  'shrinking':[True,False],
                  'probability':[True,False]
        }]
}

CLASSIFIERS = [('LSVC',LinearSVC(),True),
               ('NSVC',NuSVC(),True),
               ('SVC',SVC(),True)]

CLASSIFIERS_WITH_PARAMETERS = [('LSVC',LinearSVC(),True),
               ('NSVC',NuSVC(),True),
               ('SVC',SVC(),True)]

ALGORITHM_PARAMETERS = { algorithm: [parameter for parameter in values[0]] for algorithm, values in FITTING_PARAMETERS.items()}
