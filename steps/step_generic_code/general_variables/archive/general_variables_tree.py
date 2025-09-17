from sklearn.tree import DecisionTreeClassifier


FITTING_PARAMETERS= {
        'DT' : [{'criterion':['gini', 'entropy', 'log_loss'],
                 'splitter':['best', 'random'],
                 'max_depth':[5,10,None],
                 'min_samples_split':[2,5,10],
                 'min_samples_leaf':[2,5,10],
                 'max_features':['sqrt', 'log2'],
                 'max_leaf_nodes':[100,None]
                }]
}

CLASSIFIERS = [('DT',DecisionTreeClassifier(),True)]

CLASSIFIERS_WITH_PARAMETERS = [('DT',DecisionTreeClassifier(),True)]

ALGORITHM_PARAMETERS = { algorithm: [parameter for parameter in values[0]] for algorithm, values in FITTING_PARAMETERS.items()}
