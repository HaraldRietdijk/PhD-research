from steps.step_generic_code.general_functions import start_logging, get_run_id, complete_run
from steps.step_generic_code.dataframes_step_data import get_cat, get_train_and_testsets
from steps.step_generic_code.dataframes_step_data import get_dataframe, get_dataframe_ns, get_threshold, get_threshold_ns

from steps.step_3.step_3_generic_storing_results import find_or_create_parameter_id, convert_list
from database.models.hft_tables import HFT_FITTING_PARAMETERS_T, HFT_DATA_CHARACTERISTIC_T, HFT_CHARACTERISTIC_CATEGORY_T, HFT_DATASET_CHARACTERISTIC_T
from steps.step_generic_code.general_variables.general_variables import FITTING_PARAMETERS

def save_parameter_scope(app, run_id):
    for algo, dicts in FITTING_PARAMETERS.items():
        for param_dict in dicts:
            for param, values in param_dict.items():
                for value in values:
                    if type(value) is list:
                        value = convert_list(value)
                    else:
                        value=str(value)
                    parameter_id=find_or_create_parameter_id(app, algo,param)
                    app.session.add(HFT_FITTING_PARAMETERS_T(hft_algorithm_t_name = algo, 
                                                             hft_parameters_t_id = parameter_id, 
                                                             value = value,
                                                             hft_run_id = run_id ))
    app.session.commit()

def check_category(app, name):
    category = app.session.query(HFT_CHARACTERISTIC_CATEGORY_T).filter(HFT_CHARACTERISTIC_CATEGORY_T.category_name==name).first()
    if not category:
        category=HFT_CHARACTERISTIC_CATEGORY_T(category_name = name)
        app.session.add(category)
        app.session.commit()
    return category.id

def find_or_create_characteristic_id (app, name, category):
    characteristic = app.session.query(HFT_DATA_CHARACTERISTIC_T).filter(HFT_DATA_CHARACTERISTIC_T.characteristic_name==name).first()
    if not characteristic:
        category_id=check_category(app,category)
        characteristic=HFT_DATA_CHARACTERISTIC_T(characteristic_name = name,description = name, category_id = category_id)
        app.session.add(characteristic)
        app.session.commit()
    return characteristic.id

def save_characteristic(app, category, name, treatment_id, value, random_seed, run_id):
    characteristic_id = find_or_create_characteristic_id(app, name, category)
    new_characteristic = HFT_DATASET_CHARACTERISTIC_T(hft_treatment_id = treatment_id, 
                                                      characteristic_id = characteristic_id,
                                                      value = value,
                                                      random_seed = random_seed,
                                                      hft_run_id = run_id)
    app.session.add(new_characteristic)
    app.session.commit()

def get_data_characteristics(app, df_treatment_id, df_dataframe, random_seed=10, run_id=0, variable_threshold=False):
    for treatment_id in df_treatment_id['treatment_id']:
        test_size = 0.3
        df_treatment = df_dataframe.loc[df_dataframe['treatment_id']==treatment_id]
        X_train,Y_train,X_test,Y_test = get_train_and_testsets(df_treatment, test_size, variable_threshold)
        if Y_train['dailysteps_cat'].eq(Y_train['dailysteps_cat'].iloc[0]).all():
            print("Passing for {t_id}".format(t_id=treatment_id))
            pass
        else:
            threshold = df_treatment['threshold'].mean()
            averages = X_train.mean()
            variance = X_train.var()
            covariance = X_train.cov()
            stand_dev = X_train.std()
            balance = Y_train.mean()
            pearson_hour_sum = covariance.iloc[0,1]/(stand_dev.iloc[0]*stand_dev.iloc[1])
            pearson_hour_sum_hour = covariance.iloc[0,2]/(stand_dev.iloc[0]*stand_dev.iloc[2])
            zeros = X_train.isin([0]).sum(axis=0)
            sparsity_hour_sum = zeros.iloc[1]/X_train.shape[0]
            sparsity_hour_sum_hour = zeros.iloc[2]/X_train.shape[0]
            save_characteristic(app,'classification', 'threshold', treatment_id, int(threshold),random_seed, run_id)
            save_characteristic(app,'classification', 'train_set_size', treatment_id, X_train.shape[0],random_seed, run_id)
            save_characteristic(app,'classification', 'number_of_observations', treatment_id, X_train.shape[0]+X_test.shape[0],random_seed, run_id)
            save_characteristic(app,'classification', 'class_balance', treatment_id, balance.iloc[0],random_seed, run_id)
            save_characteristic(app,'statistics', 'average_sum_steps', treatment_id, averages.iloc[1],random_seed, run_id)
            save_characteristic(app,'statistics', 'average_sum_steps_hour', treatment_id, averages.iloc[2],random_seed, run_id)
            save_characteristic(app,'statistics', 'variance_sum_steps', treatment_id, variance.iloc[1],random_seed, run_id)
            save_characteristic(app,'statistics', 'variance_sum_steps_hour', treatment_id, variance.iloc[2],random_seed, run_id)
            save_characteristic(app,'statistics', 'standard_deviation_sum_steps', treatment_id, stand_dev.iloc[1],random_seed, run_id)
            save_characteristic(app,'statistics', 'standard_deviation_sum_steps_hour', treatment_id, stand_dev.iloc[2],random_seed, run_id)
            save_characteristic(app,'statistics', 'pearson_correlation_hour_sum_steps', treatment_id, pearson_hour_sum,random_seed, run_id)
            save_characteristic(app,'statistics', 'pearson_correlation_hour_sum_steps_hour', treatment_id, pearson_hour_sum_hour,random_seed, run_id)
            save_characteristic(app,'statistics', 'sparsity_hour_sum_steps', treatment_id, sparsity_hour_sum,random_seed, run_id)
            save_characteristic(app,'statistics', 'sparsity_hour_sum_steps_hour', treatment_id, sparsity_hour_sum_hour,random_seed, run_id)
            save_characteristic(app,'statistics', 'coefficient_of_variation_sum_steps', treatment_id, stand_dev.iloc[1]/averages.iloc[1],random_seed, run_id)
            save_characteristic(app,'statistics', 'coefficient_of_variation_sum_steps_hour', treatment_id, stand_dev.iloc[2]/averages.iloc[2],random_seed, run_id)

def do_step_5(app):
    folder='results/article_5/characteristics/VFC'
    start_logging(folder)
    run_id = get_run_id(app, 'Getting dataframe size', 'definitive', 5, 'both')

    save_parameter_scope(app, run_id)

    df_treatment_id, df_dataframe = get_dataframe(app)
    print(df_dataframe)
    print(df_dataframe.shape)
    df_dataframe = get_threshold(df_dataframe)
    df_dataframe['dailysteps_cat'] = get_cat(df_dataframe, df_dataframe)
    get_data_characteristics(app, df_treatment_id, df_dataframe, random_seed=10, run_id=run_id)

    df_treatment_id, df_dataframe = get_dataframe_ns(app)
    threshold_type = 'LIN_W'
    df_dataframe = get_threshold_ns(app, threshold_type, df_dataframe)
    df_dataframe['dailysteps_cat'] = get_cat(df_dataframe, df_dataframe)
    get_data_characteristics(app, df_treatment_id, df_dataframe, random_seed=10, run_id=run_id)

    complete_run(app, run_id)
