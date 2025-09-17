from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


FITTING_PARAMETERS= {
        'LDA' : [{'solver':['lsqr', 'eigen'],
                'shrinkage':['auto',None]
                },
                {'solver':['svd']
                }],
        'QDA' : [{'store_covariance':[True,False]
        }]
}

CLASSIFIERS = [('LDA',LinearDiscriminantAnalysis(),True),
               ('QDA',QuadraticDiscriminantAnalysis(),True)]

CLASSIFIERS_WITH_PARAMETERS = [('LDA',LinearDiscriminantAnalysis(),True),
                               ('QDA',QuadraticDiscriminantAnalysis(),True)]

ALGORITHM_PARAMETERS = { algorithm: [parameter for parameter in values[0]] for algorithm, values in FITTING_PARAMETERS.items()}
