from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier,HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, LogisticRegressionCV,\
                                Perceptron, RidgeClassifier, SGDOneClassSVM
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

CLASSES_VALUES={
        'ADA' : 0, 'BAC' : 1, 'RF': 2, 'GBC' :3, 'KNN' : 4, 'DT' : 5
}

FITTING_PARAMETERS={
        'ADA' : [{'learning_rate':[0.1, 0.5, 1.0, 10.0],
                'n_estimators':[10, 50],
                'algorithm':['SAMME']
        }],
        'BAC' : [{'n_estimators':[5,10,50]}],
        'RF'  : [{'criterion':['gini', 'entropy'],
                'max_features':[0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None],
                'n_estimators':[10, 50, 100, 500]
        }],
        'GBC' : [{'loss':['log_loss', 'exponential'],
                'learning_rate':[0.1, 0.5, 1.0, 10.0],
                'n_estimators':[10, 50, 100, 500],
                'criterion':['friedman_mse', 'squared_error'],
                'min_samples_split':[2,10,50]
        }],
        'KNN' : [{'metric':['minkowski','euclidean','manhattan'],
                 'weights':['uniform','distance'],
                 'n_neighbors':[5, 6, 7, 8, 9],
                 'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
                 'leaf_size':[5,10,30,50]
        }],
        'DT' : [{'criterion':['gini', 'entropy', 'log_loss'],
                 'splitter':['best', 'random'],
                 'max_depth':[5,10,None],
                 'min_samples_split':[2,5,10],
                 'min_samples_leaf':[2,5,10],
                 'max_features':['sqrt', 'log2'],
                 'max_leaf_nodes':[100,None]
                }]
}

CLASSIFIERS = [('BAC',BaggingClassifier(),True),
               ('ADA',AdaBoostClassifier(),True),
               ('RF',RandomForestClassifier(),True),
               ('GBC', GradientBoostingClassifier(),True),
               ('KNN',KNeighborsClassifier(),True),
               ('DT',DecisionTreeClassifier(),True)]

CLASSIFIERS_WITH_PARAMETERS = [('BAC',BaggingClassifier(),True),
               ('ADA',AdaBoostClassifier(),True),
               ('RF',RandomForestClassifier(),True),
               ('GBC', GradientBoostingClassifier(),True),
               ('KNN',KNeighborsClassifier(),True),
               ('DT',DecisionTreeClassifier(),True)]

ALGORITHM_PARAMETERS = { algorithm: [parameter for parameter in values[0]] for algorithm, values in FITTING_PARAMETERS.items()}
