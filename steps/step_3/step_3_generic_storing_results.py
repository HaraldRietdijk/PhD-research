import pickle, os
import pandas as pd

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sqlalchemy import and_

from database.models.hft_tables import HFT_ALGORITHM_T, HFT_MODEL_T, HFT_MODEL_PARAMETERS_T,\
                                       HFT_METRICS_T, HFT_PARAMETERS_T, HFT_FITTING_TIME_T

from steps.step_generic_code.general_variables.general_variables import CLASSIFIERS, ALGORITHM_PARAMETERS
from steps.step_generic_code.general_functions import pickle_destination
from steps.step_generic_code.dataframes_step_data import get_trainsets

def save_model_to_disk(dest, treatment_id, algorithm_name, fitting_results):
    pickle_model = str(treatment_id) + '_' + algorithm_name + '_' + 'model.pkl'
    fitted_model = fitting_results[algorithm_name][str(treatment_id)][0]
    pickle.dump(fitted_model.best_estimator_,open(os.path.join(dest, pickle_model),'wb'),protocol=4)
    return pickle_model

def save_pickle_and_metrics(app, run_id, df_treatment_id, df_dataframe, folder, fitting_results, predictions, random_seed=10):
    print("Step 3: Saving results")
    sub_folder='pkl_objects'
    dest = pickle_destination(folder,sub_folder)
    for treatment_id in df_treatment_id['treatment_id']:
        test_size = 0.3
        df_treatment = df_dataframe.loc[df_dataframe['treatment_id']==treatment_id]
        _, _ ,_ ,Y_train = get_trainsets(df_treatment, test_size)
        if Y_train['dailysteps_cat'].eq(Y_train['dailysteps_cat'].iloc[0]).all():
            pass
        else:
            for algorithm_name, _, _ in CLASSIFIERS:
                pickle_model = save_model_to_disk(dest, treatment_id, algorithm_name, fitting_results)
                insert_model_metrics_into_database(app, run_id, treatment_id, pickle_model, algorithm_name, dest, random_seed, 
                                                   df_dataframe, fitting_results, predictions)

def check_algorithm(app, algorithm_name):
    algorithm = app.session.query(HFT_ALGORITHM_T).filter(HFT_ALGORITHM_T.name==algorithm_name).first()
    if not algorithm:
        app.session.add(HFT_ALGORITHM_T(name = algorithm_name))
        app.session.commit()

def find_or_create_parameter_id (app, algorithm_name,parameter_name):
    check_algorithm(app, algorithm_name)
    parameter = app.session.query(HFT_PARAMETERS_T).filter(and_(HFT_PARAMETERS_T.hft_algorithm_t_name==algorithm_name,
                                                                HFT_PARAMETERS_T.name==parameter_name)).first()
    if not parameter:
        parameter = HFT_PARAMETERS_T(hft_algorithm_t_name = algorithm_name,
                                     name = parameter_name)
        app.session.add(parameter)
        app.session.commit()
    return parameter.id

def convert_list(list):
    string_value = '['
    first = True
    for value in list:
        if first:
            first = False
        else:
            string_value += ', '
        string_value += str(value)
    string_value += ']'
    return string_value

def insert_hyperparameters_into_database(app, model_id, treatment_id, algorithm_name, fitting_results):
    for parameter_name in ALGORITHM_PARAMETERS[algorithm_name]:
        parameter_id = find_or_create_parameter_id(app,algorithm_name,parameter_name)
        value=getattr(fitting_results[algorithm_name][str(treatment_id)][0].best_estimator_ , parameter_name)
        if type(value) is list:
            value = convert_list(value)
        app.session.add(HFT_MODEL_PARAMETERS_T(hft_model_id = model_id, 
                                               treatment_id = treatment_id,
                                               hft_parameters_t_id = parameter_id,
                                               value = value))
    app.session.commit()

def insert_model_and_parameters_into_database(app, pickle_model, algorithm_name, dest, treatment_id , random_seed, fitting_results):
    model=HFT_MODEL_T(name = pickle_model, 
                      algorithm = algorithm_name, 
                      destination = dest, 
                      random_seed = random_seed)
    app.session.add(model)
    app.session.commit()
    model_id = model.id
    insert_hyperparameters_into_database(app, model_id, treatment_id, algorithm_name, fitting_results)    
    return model.id

def insert_model_metrics_into_database(app, run_id, treatment_id, pickle_model, algorithm_name, dest, random_seed, 
                                       df_dataframe, fitting_results, predictions):
    model_id = insert_model_and_parameters_into_database(app, pickle_model, algorithm_name, dest, treatment_id,
                                                         random_seed, fitting_results)
    Y = predictions[str(treatment_id)][0]
    Y_pred = predictions[str(treatment_id)][1][algorithm_name]
    df_confusion_matrix = pd.DataFrame(confusion_matrix(Y,Y_pred))
    THRESHOLD = df_dataframe.loc[df_dataframe['treatment_id']==treatment_id]['threshold'].mean()
    TN = df_confusion_matrix.iloc[0,0]
    FP = df_confusion_matrix.iloc[0,1]
    FN = df_confusion_matrix.iloc[1,0]
    TP = df_confusion_matrix.iloc[1,1]
    F1 = f1_score(Y,Y_pred, average='macro')
    ACC = accuracy_score(Y, Y_pred)
    OBS = len(Y)
    WEEKDAY = random_seed
    app.session.add(HFT_METRICS_T(f1_score = F1, 
                                  true_negative = TN, 
                                  true_positive = TP,
                                  false_negative = FN, 
                                  false_positive = FP, 
                                  accuracy = ACC,
                                  hft_model_id = model_id, 
                                  hft_treatment_id = treatment_id, 
                                  threshold = THRESHOLD, 
                                  number_of_observations = OBS, 
                                  weekday = WEEKDAY,
                                  hft_run_id = run_id))
    app.session.commit()

def save_fitting_times(app, run_id, fitting_results_all_models, random_seed=10):
    for name, fitting_times_per_treatment in fitting_results_all_models.items():
        for treatment_id, fitting_results in fitting_times_per_treatment.items():
            app.session.add(HFT_FITTING_TIME_T(hft_treatment_id = int(treatment_id),
                                                algorithm = name, 
                                                fitting_time_sec = fitting_results[1],
                                                random_seed = random_seed,
                                                hft_run_id = run_id))
    app.session.commit()
