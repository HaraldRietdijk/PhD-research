from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel


FITTING_PARAMETERS= {
        'GPC' : [{'kernel':[ConstantKernel(),None],
                  'optimizer':['fmin_l_bfgs_b', None]

                }]
}

CLASSIFIERS = [('GPC',GaussianProcessClassifier(),True)]

CLASSIFIERS_WITH_PARAMETERS = [('GPC',GaussianProcessClassifier(),True)]

ALGORITHM_PARAMETERS = { algorithm: [parameter for parameter in values[0]] for algorithm, values in FITTING_PARAMETERS.items()}
