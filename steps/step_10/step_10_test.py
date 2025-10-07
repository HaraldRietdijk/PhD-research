import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif  
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from steps.step_generic_code.general_variables.general_variables_all_shap import FITTING_PARAMETERS, CLASSIFIERS, ALGORITHM_PARAMETERS

def get_reduced_X(coefficients, features, X):
    X_sel = pd.DataFrame()
    for coefficient, feature in zip(coefficients, features):
        if coefficient>0:
            X_sel[feature]=X[feature]
    return X_sel

def do_test(dataframes, fitted_models, features):
    # do anova and chi2 selecting all features, and then select based on coef.
    scores = {}
    for name, classifier, _, _ in CLASSIFIERS:
        parameters=FITTING_PARAMETERS[name]
        scores[name]={'accuracy':[],'f1-score':[],'precision':[],'recall':[]}
        for k in range(5,6):
            X_anova = SelectKBest(f_classif,k=k)
            anova_svm = make_pipeline(X_anova, GridSearchCV(classifier, parameters, cv=2))
            anova_svm.fit(dataframes['X_train'],dataframes['Y_class_train'])
            print(name)
            if name in ['LDA','LSVC']:
                coefficients = anova_svm[-1].best_estimator_.coef_
            else:
                coefficients = list(anova_svm[-1].best_estimator_.feature_importances_)
            print(coefficients)
            anova_svm[:-1].inverse_transform(coefficients)[0]

            # feature_coefs = anova_svm[:-1].inverse_transform(coefficients)[0]
            # features_used = X_test.columns.tolist()
            # nr_features = len(features_used)
            # if nr_features in model_features.keys():
            #     if features_used in model_features[nr_features]:
            #         new_model = False
            #     else:
            #         model_features[nr_features].append(features_used)
            #         new_model = True
            # else:
            #     model_features[nr_features] = [features_used]
            #     new_model = True
            # if new_model:

            # if nr_features<k:
            #     estimator = GridSearchCV(classifier, parameters, cv=3).fit(X_train, dataframes['Y_class_train']).best_estimator_

    return 