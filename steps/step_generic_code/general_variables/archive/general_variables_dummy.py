from sklearn.dummy import DummyClassifier


FITTING_PARAMETERS= {
        'DUM' : [{'strategy':['constant'],
                  'constant':[1,0]
                },
                {'strategy':['most_frequent', 'prior', 'stratified', 'uniform']
                }]
}

CLASSIFIERS = [('DUM',DummyClassifier(),True)]

CLASSIFIERS_WITH_PARAMETERS = [('DUM',DummyClassifier(),True)]

ALGORITHM_PARAMETERS = { algorithm: [parameter for parameter in values[0]] for algorithm, values in FITTING_PARAMETERS.items()}
