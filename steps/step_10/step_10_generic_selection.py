import os 
import pickle

from sqlalchemy import and_

from database.models.hft_tables import HFT_ALGORITHM_T, HFT_MODEL_T, HFT_MODEL_PARAMETERS_T, HFT_PARAMETERS_T
from database.models.hft_tables import HFT_ESTIMATOR_METRICS_T, HFT_METRICS_TYPES_T, HFT_ESTIMATOR_METRICS_RESULTS_T
from steps.step_generic_code.general_functions import pickle_destination, convert_list

def save_model_to_disk(dest, name, estimator):
    pickle_model = name+'_'+'model.pkl' 
    pickle.dump(estimator,open(os.path.join(dest,pickle_model),'wb'),protocol=4)
    return pickle_model

def check_algorithm(app, algorithm_name):
    algorithm = app.session.query(HFT_ALGORITHM_T).filter(HFT_ALGORITHM_T.name==algorithm_name).first()
    if not algorithm:
        app.session.add(HFT_ALGORITHM_T(name=algorithm_name))
        app.session.commit()

def find_or_create_parameter_id (app, algorithm_name,parameter_name):
    check_algorithm(app, algorithm_name)
    parameter=app.session.query(HFT_PARAMETERS_T).filter(and_(HFT_PARAMETERS_T.hft_algorithm_t_name==algorithm_name,
                                                       HFT_PARAMETERS_T.name==parameter_name)).first()
    if not parameter:
        parameter=HFT_PARAMETERS_T(hft_algorithm_t_name=algorithm_name,name=parameter_name)
        app.session.add(parameter)
        app.session.commit()
    return parameter.id

def insert_hyperparameters_into_database(app,model_id,algorithm_name, estimator, ALGORITHM_PARAMETERS):
    for parameter_name in ALGORITHM_PARAMETERS[algorithm_name]:
        parameter_id=find_or_create_parameter_id(app,algorithm_name,parameter_name)
        value=getattr(estimator, parameter_name)
        if type(value) is list:
            value = convert_list(value)
        app.session.add(HFT_MODEL_PARAMETERS_T(hft_model_id=model_id,treatment_id=0,
                                               hft_parameters_t_id=parameter_id,value=value))
    app.session.commit()

def insert_model_and_parameters_into_database(app, estimator, pickle_model, name, dest, ALGORITHM_PARAMETERS):
    model=HFT_MODEL_T(name=pickle_model,algorithm=name,destination=dest,random_seed=42)
    app.session.add(model)
    app.session.commit()
    model_id=model.id
    insert_hyperparameters_into_database(app,model_id,name, estimator, ALGORITHM_PARAMETERS)    
    return model.id

def store_metrics_result(app, estimator_metrics_id, name, value, test_or_train):
    metrics_type = app.session.query(HFT_METRICS_TYPES_T).filter(HFT_METRICS_TYPES_T.name==name).first()
    if not metrics_type:
        metrics_type = HFT_METRICS_TYPES_T(name=name)
        app.session.add(metrics_type)
        app.session.commit()
    estimator_metrics_result = HFT_ESTIMATOR_METRICS_RESULTS_T(estimator_metrics_id = estimator_metrics_id, test_or_train=test_or_train, 
                                                        metrics_type_id = metrics_type.id, value = value)
    app.session.add(estimator_metrics_result)
    app.session.commit()

def insert_model_metrics_into_database(app, regressor, pickle_model, name, dest, model_score, nr_of_classes, ALGORITHM_PARAMETERS):
    model_id = insert_model_and_parameters_into_database(app, regressor, pickle_model, name, dest, ALGORITHM_PARAMETERS)
    estimator_metrics = HFT_ESTIMATOR_METRICS_T(hft_model_id=model_id, nr_of_classes=nr_of_classes)
    app.session.add(estimator_metrics)
    app.session.commit()
    for score, value in model_score[0].items():
        store_metrics_result(app, estimator_metrics.id, score, value, 'test')
    for score, value in model_score[1].items():
        store_metrics_result(app, estimator_metrics.id, score, value, 'train')

def save_pickle_and_metrics(app, folder, fitted_models, models_scoring, nr_of_classes, ALGORITHM_PARAMETERS):
    print("Step 10 R/CS: Saving results")
    sub_folder='pkl_objects'
    dest = pickle_destination(folder,sub_folder)
    for name, estimator in fitted_models.items():
        pickle_model = save_model_to_disk(dest, name, estimator)
        scoring = (models_scoring['Test'][name], models_scoring['Train'][name])
        insert_model_metrics_into_database(app, estimator, pickle_model, name, dest, scoring, nr_of_classes, ALGORITHM_PARAMETERS)
