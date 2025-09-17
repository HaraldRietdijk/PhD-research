from sklearn.neural_network import MLPClassifier


FITTING_PARAMETERS= {
        'NN'  : [{'learning_rate':['constant', 'invscaling', 'adaptive'],
                'activation':['identity', 'logistic', 'tanh', 'relu'],
                'learning_rate_init':[0.05, 0.1, 0.5, 1.0]}]
}

CLASSIFIERS = [('NN',MLPClassifier(),True)]

CLASSIFIERS_WITH_PARAMETERS = [('NN',MLPClassifier(),True)]

ALGORITHM_PARAMETERS = { algorithm: [parameter for parameter in values[0]] for algorithm, values in FITTING_PARAMETERS.items()}
