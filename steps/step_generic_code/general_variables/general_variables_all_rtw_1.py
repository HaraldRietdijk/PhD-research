from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier,HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, LogisticRegressionCV,\
                                Perceptron, RidgeClassifier, SGDOneClassSVM
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
import itertools

FITTING_PARAMETERS={
        'ADA1' : [{'learning_rate':[0.1, 0.5, 1.0, 10.0],
                'n_estimators':[5,10,15,20,25],
                'algorithm':['SAMME']
        }],
        'ETSC1' : [{'n_estimators':[5,10,20],
                   'criterion':['gini', 'entropy', 'log_loss'],
                   'min_samples_split':[2,10,20],
                   'max_features': [0.1, 0.25, 0.5, 0.75,'sqrt', 'log2', None]
        }],
        'RF1'  : [{'criterion':['gini', 'entropy'],
                'max_features':[0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None],
                'n_estimators':[5,10, 20,30]
        }],
        'PAC1' : [{'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                'fit_intercept':[True,False], 
                'loss':['squared_hinge', 'hinge'],
                'average':[True,False,10,50]

        }],
        'PER1' : [{'penalty':['elasticnet'],
                'alpha':[0.00001,0.0001,0.001,0.01,0.1],
                'l1_ratio':[0.2,0.35,0.5,0.65,0.8],
                'fit_intercept':[True,False]
               },{'penalty':['l1', 'l2', None],
                'alpha':[0.00001,0.0001,0.001,0.01,0.1],
                'fit_intercept':[True,False]
        }],
        'RIC1' : [{'alpha':[0.00001,0.0001,0.001,0.01,0.1],
                'fit_intercept':[True,False],
                'solver':['lbfgs'],
                'max_iter':[50,100],
                'positive':[True]
                },{'alpha':[0.00001,0.0001,0.001,0.01,0.1],
                'fit_intercept':[True,False],
                'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                'max_iter':[50,100],
                'positive':[False]
        }],
        'SGD1' : [{'fit_intercept':[True, False],
                 'loss':['huber'],
                 'penalty':['l1', 'l2', None],
                 'alpha':[0.00001,0.0001],
                 'epsilon':[0.2,0.25,0.3],
                 'learning_rate':['constant', 'invscaling', 'adaptive'],
                 'max_iter':[2000],
                 'eta0':[0.5,1,2]
               },{'fit_intercept':[True, False],
                 'loss':['hinge', 'log_loss'],
                 'penalty':['l1', 'l2', None],
                 'alpha':[0.00001,0.0001],
                 'learning_rate':['constant', 'invscaling', 'adaptive'],
                 'max_iter':[2000],
                 'eta0':[0.5,1,2]
               }],
        'DT1' : [{'criterion':['gini', 'entropy', 'log_loss'],
                 'splitter':['best', 'random'],
                 'max_depth':[5,10,None],
                 'min_samples_split':[2,5,10],
                 'min_samples_leaf':[2,5,10],
                 'max_features':['sqrt', 'log2'],
                 'max_leaf_nodes':[100,None]
                }]
}

CLASSIFIERS = [('ADA1',AdaBoostClassifier(n_estimators=15),True,14),
               ('RF1',RandomForestClassifier(n_estimators=10, max_depth=4, min_weight_fraction_leaf=0.1),True,14),
               ('ETSC1',ExtraTreesClassifier(n_estimators=10, max_depth=4, min_weight_fraction_leaf=0.1),True,8),
               ('PAC1', PassiveAggressiveClassifier(),False,4),
               ('PER1', Perceptron(),True,4),
               ('RIC1', RidgeClassifier(),True,4),
               ('SGD1',SGDClassifier(),False,4),
               ('DT1',DecisionTreeClassifier(max_depth=3, min_weight_fraction_leaf=0.1),True,14)]

max_depth_options = [1, 2, 3]
min_samples_split_options = [0.1, 0.2, 0.3]
n_estimator_options = [5, 10, 15, 20, 25]
# max_depth_options = [1,3]
# min_samples_split_options = [0.1,0.3]
# n_estimator_options = [20, 25]
all_options = list(itertools.product(n_estimator_options, max_depth_options, min_samples_split_options))

CLASSIFIERS_FOR_TREE_OPT = [('ADA'+str(ne)+str(md)+str(ms)
                             ,AdaBoostClassifier(n_estimators=ne, estimator=DecisionTreeClassifier(max_depth=md, min_samples_split=ms))
                             ) for ne, md, ms in all_options]

CLASSIFIERS_WITH_PARAMETERS = [
               ('ADA1',AdaBoostClassifier(),True),
               ('RF1',RandomForestClassifier(),True),
               ('ETSC1',ExtraTreesClassifier(),True),
               ('PAC1', PassiveAggressiveClassifier(),False),
               ('PER1', Perceptron(),True),
               ('RIC1', RidgeClassifier(),True),
               ('SGD1',SGDClassifier(),False),
               ('DT1',DecisionTreeClassifier(),True)]

ALGORITHM_PARAMETERS = { algorithm: [parameter for parameter in values[0]] for algorithm, values in FITTING_PARAMETERS.items()}
