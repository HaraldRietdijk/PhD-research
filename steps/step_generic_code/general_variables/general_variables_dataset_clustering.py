from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering

CLASSES_VALUES={
        'ADA' : 0, 'BAC' : 1, 'RF': 2, 'GBC' :3, 'KNN' : 4, 'DT' : 5, 'LR' : 6
}

CLUSTERING_MODELS = { "4" : [('KMEANS4', KMeans(n_clusters=4, random_state=0, n_init="auto")),
                             ('AGGLCL4', AgglomerativeClustering(n_clusters=4)),
                             ('SPECCL4', SpectralClustering(n_clusters=4))],
                      "5" : [('KMEANS5', KMeans(n_clusters=5, random_state=0, n_init="auto")),
                             ('AGGLCL5', AgglomerativeClustering(n_clusters=5)),
                             ('SPECCL5', SpectralClustering(n_clusters=5))],
                      "7" : [('KMEANS7', KMeans(n_clusters=7, random_state=0, n_init="auto")),
                             ('AGGLCL7', AgglomerativeClustering(n_clusters=7)),
                             ('SPECCL7', SpectralClustering(n_clusters=7))],
                      "8" : [('KMEANS8', KMeans(n_clusters=8, random_state=0, n_init="auto")),
                             ('AGGLCL8', AgglomerativeClustering(n_clusters=8)),
                             ('SPECCL8', SpectralClustering(n_clusters=8))],
                      "9" : [('KMEANS9', KMeans(n_clusters=9, random_state=0, n_init="auto")),
                             ('AGGLCL9', AgglomerativeClustering(n_clusters=9)),
                             ('SPECCL9', SpectralClustering(n_clusters=9))]}

FITTING_PARAMETERS={
        'ADA' : [{'learning_rate':[0.1, 0.5, 1.0, 10.0],
                'n_estimators':[10, 50],
                'algorithm':['SAMME']
        }],
        'RF'  : [{'criterion':['gini', 'entropy'],
                'max_features':[0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None],
                'n_estimators':[10, 50, 100, 500]
        }],
        'DT' : [{'criterion':['gini', 'entropy', 'log_loss'],
                 'splitter':['best', 'random'],
                 'max_depth':[5,10,None],
                 'min_samples_split':[2,5,10],
                 'min_samples_leaf':[2,5,10],
                 'max_features':['sqrt', 'log2'],
                 'max_leaf_nodes':[100,None]
                }],
        'LR' : [{'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                'penalty':['elasticnet'],
                'fit_intercept':[True, False],
                'l1_ratio':[0.2,0.35,0.5,0.65,0.8],
                'solver':['saga'] 
                },
                {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                'penalty':['l1'],
                'fit_intercept':[True, False],
                'solver':['saga'] 
                },
                {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                'penalty':['l1', 'l2'], # None and elasticnet
                'fit_intercept':[True, False],
                'solver':['liblinear']
                },
                {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                'penalty':['l2'],
                'fit_intercept':[True, False],
                'solver':['lbfgs','newton-cg', 'newton-cholesky', 'sag', 'saga'] 
                },
                {'penalty':[None],
                'fit_intercept':[True, False],
                'solver':['lbfgs','newton-cg', 'newton-cholesky', 'sag', 'saga'] 
                }
        ]
}

CLASSIFIERS = [('DT',DecisionTreeClassifier(),True),
               ('ADA',AdaBoostClassifier(),True),
               ('RF',RandomForestClassifier(),True),
               ('LR', LogisticRegression(),True)]

ALGORITHM_PARAMETERS = { algorithm: [parameter for parameter in values[0]] for algorithm, values in FITTING_PARAMETERS.items()}
