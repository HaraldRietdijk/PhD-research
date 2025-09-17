from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor,BaggingRegressor,\
        GradientBoostingRegressor,HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.linear_model import LinearRegression, SGDRegressor, PassiveAggressiveRegressor, Ridge
from sklearn.neighbors import KNeighborsRegressor,RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor


FITTING_PARAMETERS={
        'BAR' : [{'n_estimators':[5,10,50]}],
        'ADAR' : [{'learning_rate':[0.1, 0.5, 1.0, 10.0],
                'n_estimators':[10, 50]
        }],
        'ETSR' : [{'n_estimators':[5,10,50],
                   'criterion':['absolute_error', 'poisson', 'squared_error', 'friedman_mse'],
                   'min_samples_split':[2,10,50],
                   'max_features': [0.1, 0.25, 0.5, 0.75,'sqrt', 'log2', None]
        }],
        'RFR'  : [{'criterion':['absolute_error', 'squared_error', 'friedman_mse', 'poisson'],
                'max_features':[0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None],
                'n_estimators':[10, 50, 100, 500]
        }],
        'GBR' : [{'loss':['huber', 'quantile', 'absolute_error', 'squared_error'],
                'learning_rate':[0.1, 0.5, 1.0, 10.0],
                'n_estimators':[10, 50, 100, 500],
                'criterion':['friedman_mse', 'squared_error'],
                'min_samples_split':[2,10,50]
        }],
        'PAR' : [{'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                'fit_intercept':[True,False], 
                'loss':['squared_epsilon_insensitive', 'epsilon_insensitive'],
                'average':[True,False,10,50]

        }],
        'HGBR'  : [{'loss':['huber', 'quantile', 'absolute_error', 'squared_error'],
                   'scoring':['accuracy','recall'],
                   'learning_rate':[0.1, 0.2],
                   'l2_regularization':[0.7,0.8,0.9],
                   'interaction_cst':['no_interactions']
        }],
        'GPR' : [{'kernel':[ConstantKernel(),None],
                  'optimizer':['fmin_l_bfgs_b', None]
                }],
        'LINR' : [{'fit_intercept':[True,False]
                }],
        'RIDR' : [{'alpha':[0.00001,0.0001,0.001,0.01,0.1],
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
        'SGDR' : [{'fit_intercept':[True, False],
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
        'KNNR' : [{'metric':['minkowski','euclidean','manhattan'],
                 'weights':['uniform','distance'],
                 'n_neighbors':[5, 6, 7, 8, 9],
                 'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
                 'leaf_size':[5,10,30,50]
        }],
        'RNNR'  : [{'radius':[1,5,10],
                   'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
                   'leaf_size':[5,10,30,50],
                   'metric':['minkowski','euclidean','manhattan']
        }],
        'NNR'  : [{'learning_rate':['constant', 'invscaling', 'adaptive'],
                'activation':['identity', 'logistic', 'tanh', 'relu'],
                'learning_rate_init':[0.05, 0.1, 0.5, 1.0],
                'solver':['lbfsg','adam']
        }],
        'LSVR': [{'loss':['squared_epsilon_insensitive', 'epsilon_insensitive'],
                  'C':[0.5,1,2],
                  'fit_intercept':[True,False],
                  'dual':['auto'],
                  'max_iter':[3000]
               }],
        'NSVR': [{'nu':[0.5],
                  'kernel':['poly'],
                  'degree':[3,4,5],
                  'gamma':['scale', 'auto']
               }],
        'SVR' : [{'C':[0.9,1,1.1],
                  'kernel':['poly'],
                  'degree':[2,3,4,5]
                  },{'C':[0.5,1,2],
                  'kernel':['rbf']
                  }],
        'DTR' : [{'criterion':['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                 'splitter':['best', 'random'],
                 'max_depth':[2,3,4,5,6,7,8,9,10],
                 'min_samples_split':[2,5,10],
                 'min_samples_leaf':[2,5,10],
                 'max_features':['sqrt', 'log2', None],
                 'max_leaf_nodes':[10,20,30,40,50,100]
                }]
}

REGRESSORS = [('DTR', DecisionTreeRegressor(), False, 10), 
               ('ADAR', AdaBoostRegressor(), False, 10),
               ('LINR', LinearRegression(), False, 10),
               ('RFR', RandomForestRegressor(), False, 10),
               ('RIDR', Ridge(),True,10),
               ('PAR', PassiveAggressiveRegressor(),False,10),
               ('LSVR', LinearSVR(random_state=10), True, 10)]

NN_REGRESSORS = [('NNR',MLPRegressor(),True,1)]

REGRESSORS_WITH_PARAMETERS = [('DTR', DecisionTreeRegressor(max_depth=4), False, 10, 6), 
               ('ADAR', AdaBoostRegressor(), False, 10, 26),
               ('LINR', LinearRegression(), False, 10, 6),
               ('RFR', RandomForestRegressor(), False, 10, 22),
               ('RIDR', Ridge(),True,10, 16),
               ('PAR', PassiveAggressiveRegressor(),False,10, 14),
               ('LSVR', LinearSVR(random_state=10), True, 10, 30)]

ALGORITHM_PARAMETERS = { algorithm: [parameter for parameter in values[0]] for algorithm, values in FITTING_PARAMETERS.items()}
