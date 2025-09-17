from sklearn.neighbors import NearestCentroid, KNeighborsClassifier,RadiusNeighborsClassifier



FITTING_PARAMETERS= {
        'KNN' : [{'metric':['minkowski','euclidean','manhattan'],
                 'weights':['uniform','distance'],
                 'n_neighbors':[5, 6, 7, 8, 9],
                 'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
                 'leaf_size':[5,10,30,50]
        }],
        'NCC' : [{'metric':['euclidean','manhattan'],
                  'shrink_threshold':[0.1,1,10,50]

        }],
        'RNN'  : [{'radius':[1,5,10],
                   'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
                   'leaf_size':[5,10,30,50],
                   'metric':['minkowski','euclidean','manhattan']
        }]
}

CLASSIFIERS = [('KNN',KNeighborsClassifier(),True),
               ('NCC',NearestCentroid(),True),
               ('RNN',RadiusNeighborsClassifier(),True)]

CLASSIFIERS_WITH_PARAMETERS = [('KNN',KNeighborsClassifier(),True),
               ('NCC',NearestCentroid(),True),
               ('RNN',RadiusNeighborsClassifier(),True)]

ALGORITHM_PARAMETERS = { algorithm: [parameter for parameter in values[0]] for algorithm, values in FITTING_PARAMETERS.items()}
