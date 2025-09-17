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

FITTING_PARAMETERS={
        'RF1'  : [{'criterion':['gini', 'entropy'],
                'max_features':[0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None],
                'n_estimators':[50]
        }],
        'RF2'  : [{'criterion':['gini', 'entropy'],
                'max_features':[0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None],
                'n_estimators':[50]
        }],
        'RF3'  : [{'criterion':['gini', 'entropy'],
                'max_features':[0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None],
                'n_estimators':[500]
        }],
        'RF4'  : [{'criterion':['gini', 'entropy'],
                'max_features':[0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None],
                'n_estimators':[50]
        }],
        'RF5'  : [{'criterion':['gini', 'entropy'],
                'max_features':[0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None],
                'n_estimators':[100]
        }],
}

CLASSIFIERS = [('RF1',RandomForestClassifier(),True),
               ('RF2',RandomForestClassifier(),True),
               ('RF3',RandomForestClassifier(),True),
               ('RF4',RandomForestClassifier(),True),
               ('RF5',RandomForestClassifier(),True)
               ]

CLASSIFIERS_WITH_PARAMETERS = [('RF1',RandomForestClassifier(),True),
               ('RF2',RandomForestClassifier(),True),
               ('RF3',RandomForestClassifier(),True),
               ('RF4',RandomForestClassifier(),True),
               ('RF5',RandomForestClassifier(),True)]

ALGORITHM_PARAMETERS = { algorithm: [parameter for parameter in values[0]] for algorithm, values in FITTING_PARAMETERS.items()}
