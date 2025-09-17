from sklearn.neural_network import MLPClassifier


FITTING_PARAMETERS= {
        'NN'  : [{'learning_rate':['constant', 'invscaling', 'adaptive'],
                'activation':['identity', 'logistic', 'tanh', 'relu'],
                'learning_rate_init':[0.01, 0.05, 0.1, 0.5, 1.0],
                'solver':['adam'],
                'alpha':[0.00001,0.001,0.1,0.25,0.5],
                'epsilon':[0.0001,0.000001,0.000000001],
                'beta_1':[0.8,0.9,0.95],
                'beta_2':[0.96,0.99,0.999]
               },{'learning_rate':['constant', 'invscaling', 'adaptive'],
                'activation':['identity', 'logistic', 'tanh', 'relu'],
                'learning_rate_init':[0.01, 0.05, 0.1, 0.5, 1.0],
                'solver':['lbfgs', 'sgd'],
                'alpha':[0.00001,0.001,0.1,0.25,0.5]
        }]
}

CLASSIFIERS = [('NN',MLPClassifier(),True)]

CLASSIFIERS_WITH_PARAMETERS = [('NN',MLPClassifier(),True)]

ALGORITHM_PARAMETERS = { algorithm: [parameter for parameter in values[0]] for algorithm, values in FITTING_PARAMETERS.items()}
