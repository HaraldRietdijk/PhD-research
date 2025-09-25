import logging
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from steps.step_generic_code.general_variables.general_variables_all_shap import FITTING_PARAMETERS, CLASSIFIERS, ALGORITHM_PARAMETERS
from steps.step_10.step_10_generic_selection import save_pickle_and_metrics
      
def fit_models(X_train, Y_train):
    print("Step 10 CS: Fitting models")
    fitted_models = {}
    for name, regressor, normalized in CLASSIFIERS:
        print("Model: ", name)
        parameters=FITTING_PARAMETERS[name]
        log_text="Model {m}".format(m=name)
        logging.info(log_text)
        if name in ['OCS','SGDOC']:
            fitted_models[name] = GridSearchCV(regressor, parameters, scoring='accuracy', cv=2).fit(X_train, Y_train).best_estimator_
        else:
            newDF = X_train.copy()
            newDF.replace(np.nan, 'error', inplace =True)
            
            # # cast newDF in the same type of X_train
            # newDF = newDF.astype(type(X_train))
            newDF.to_csv('newDF_train.csv', sep=',')
            fitted_models[name] = GridSearchCV(regressor, parameters, cv=2).fit(X_train, Y_train).best_estimator_
    return fitted_models

def get_scores_for_prediction(fitted_models, X_set, Y_set):
    print("Step 10 CS: geting scores")
    models_scoring = {}
    for name, fitted_model in fitted_models.items():
        model_score = {}
        Y_pred = fitted_model.predict(X_set)
        model_score['accuracy_score'] = accuracy_score(Y_set,Y_pred)
        model_score['f1_score_micro'] = f1_score(Y_set,Y_pred,average='micro')
        model_score['f1_score_macro'] = f1_score(Y_set,Y_pred,average='macro')
        if name not in ['PAC','PER','RIC','SGD','PAC1','PER1','RIC1','SGD1','SGDOC','NCC','LSVC','LSVC1']:
            Y_pred_prob = fitted_model.predict_proba(X_set)
            if len(Y_pred_prob[0])==2:
                Y_pred_prob = Y_pred_prob[:,1]
            model_score['roc_auc_score_micro'] = roc_auc_score(Y_set,Y_pred_prob,average='micro',multi_class='ovr')
            model_score['roc_auc_score_macro'] = roc_auc_score(Y_set,Y_pred_prob,average='macro',multi_class='ovo')
        models_scoring[name] = model_score
    return models_scoring

def classifier_selection(app, folder, dataframes, nr_of_classes):
    fitted_models = fit_models(dataframes['X_train'], dataframes['Y_class_train'])
    models_scoring = {}
    models_scoring['Test'] = get_scores_for_prediction(fitted_models, dataframes['X_test'], dataframes['Y_class_test'])
    models_scoring['Train'] = get_scores_for_prediction(fitted_models, dataframes['X_train'], dataframes['Y_class_train'])
    save_pickle_and_metrics(app, folder, fitted_models, models_scoring, nr_of_classes, ALGORITHM_PARAMETERS)
    return fitted_models

