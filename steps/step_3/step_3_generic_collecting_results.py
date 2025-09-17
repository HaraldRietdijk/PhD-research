import time, logging
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

from steps.step_generic_code.general_variables.general_variables import FITTING_PARAMETERS, CLASSIFIERS
from steps.step_generic_code.dataframes_step_data import get_trainsets

def fit_model (treatment_id, use_algorithm_name, use_algorithm, X_train, Y_train):
    parameters=FITTING_PARAMETERS[use_algorithm_name]
    log_text="Model {m}, Treatment_id {t}".format(m=use_algorithm_name, t=treatment_id)
    logging.info(log_text)
    start = time.time()
    if use_algorithm_name in ['RF','DT']:
        fitted_model = GridSearchCV(use_algorithm, parameters,error_score=0.0, cv=5).fit(X_train, Y_train)
    elif use_algorithm_name in ['OCS','SGDOC']:
        fitted_model = GridSearchCV(use_algorithm, parameters, scoring='accuracy', cv=5).fit(X_train, Y_train)
    else:
        fitted_model = GridSearchCV(use_algorithm, parameters, cv=5).fit(X_train, Y_train)
    end = time.time()
    return fitted_model, (end-start)

def fit_models(df_treatment_id, df_dataframe, variable_threshold=False):
    print("Step 3: Fitting models")
    fitting_results_all_models={}
    for name, classifier, normalized in CLASSIFIERS:
        print("Model: ", name)
        fitting_results_for_model=dict()
        for treatment_id in df_treatment_id['treatment_id']:
            print(treatment_id, end=' ', flush=True)
            test_size = 0.3
            df_treatment = df_dataframe.loc[df_dataframe['treatment_id']==treatment_id]
            X_train_s,Y_train_s,X_train,Y_train = get_trainsets(df_treatment, test_size, variable_threshold)
            if Y_train['dailysteps_cat'].eq(Y_train['dailysteps_cat'].iloc[0]).all():
                print("Passing for {model} for {t_id}".format(model=name,t_id=treatment_id))
                fitting_time=0
                pass
            else:
                if normalized:
                    X_train = X_train_s
                fitted_model, fitting_time = fit_model(treatment_id, name, classifier, X_train, Y_train)
            fitting_results_for_model[str(treatment_id)]=(fitted_model,fitting_time)
        fitting_results_all_models[name]=fitting_results_for_model
        print('\n')
    return fitting_results_all_models

def make_predictions(df_treatment_id, df_dataframe, fitting_results, variable_threshold=False):
    print("Step 3: Predicting")
    Y_pred_all_models={}
    for treatment_id in df_treatment_id['treatment_id']:
        treatment_id_name=str(treatment_id)
        test_size = 0.3
        _, _, _,Y_train = get_trainsets(df_dataframe.loc[df_dataframe['treatment_id']==treatment_id],
                                       test_size, variable_threshold)
        if Y_train['dailysteps_cat'].eq(Y_train['dailysteps_cat'].iloc[0]).all():
            print("Passing for {model} for {t_id}".format(model=name,t_id=treatment_id))
            pass
        else:
            if variable_threshold:
                X_columns = ['weekday','hour','sum_steps','sum_steps_hour','threshold']
            else:
                # X_columns = ['weekday','hour','sum_steps','sum_steps_hour']
                X_columns = ['hour','sum_steps','sum_steps_hour']
            X = df_dataframe.loc[df_dataframe['treatment_id']==treatment_id][X_columns]
            Y = df_dataframe.loc[df_dataframe['treatment_id']==treatment_id][['dailysteps_cat']]
            Y_pred = {}
            for name, _, normalized in CLASSIFIERS:
                if normalized:
                    min_max_scaler = preprocessing.MinMaxScaler()
                    np_scaled = min_max_scaler.fit_transform(X)
                    X = pd.DataFrame(np_scaled)
                Y_pred[name] = fitting_results[name][treatment_id_name][0].predict(X)
            Y_pred_all_models[treatment_id_name] = (Y, Y_pred)
    return Y_pred_all_models
