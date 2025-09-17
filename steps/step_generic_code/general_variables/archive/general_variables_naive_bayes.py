from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB



FITTING_PARAMETERS= {
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
        }]
}

CLASSIFIERS = [('BNB',BernoulliNB(),True),
               ('CONB',ComplementNB(),True),
               ('GNB',GaussianNB(),True)]

CLASSIFIERS_WITH_PARAMETERS = [('BNB',BernoulliNB(),True),
               ('CONB',ComplementNB(),True),
               ('GNB',GaussianNB(),True)]

ALGORITHM_PARAMETERS = { algorithm: [parameter for parameter in values[0]] for algorithm, values in FITTING_PARAMETERS.items()}
