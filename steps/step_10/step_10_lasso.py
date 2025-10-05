from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
import numpy as np

def do_lasso_fs(app, feature_names, X, y):
    clf = LassoCV().fit(X, y)
    importance = np.abs(clf.coef_)
    print(importance)

    idx_third = importance.argsort()[-3]
    threshold = importance[idx_third] + 0.01

    idx_features = (-importance).argsort()[:2]
    name_features = np.array(feature_names)[idx_features]
    print('Selected features: {}'.format(name_features))

    sfm = SelectFromModel(clf, threshold=threshold)
    sfm.fit(X, y)
    X_transform = sfm.transform(X)

    n_features = sfm.transform(X).shape[1]