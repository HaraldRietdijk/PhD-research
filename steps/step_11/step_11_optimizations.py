import sys

import pandas as pd

from sqlalchemy import and_

from database.models.feature_selection_data import SELECTION_METHOD, FEATURES_RANKING_GROUP, FEATURES_RANKING
from database.models.hft_tables import HFT_RUN_T

from steps.step_10.step_10_general_functions import append_scores_for_features, init_scores, save_method_results
from steps.step_generic_code.dataframe_colorectal_operations import get_dataframe as isala_dataframe
from steps.step_generic_code.dataframe_knee_operations import get_dataframe as NS_dataframe
from steps.step_generic_code.general_functions import complete_run, get_run_id
from steps.step_generic_code.general_variables.general_variables_all_shap import CLASSIFIERS

def get_query_filter(filters, optim):
    if optim:
        filters.append(SELECTION_METHOD.type=="optim")
    else:
        filters.append(SELECTION_METHOD.use_for_ordering==True) 
    return filters

def get_methods_and_features(app):
    def dataframe_to_dict(row):
        nonlocal method_features
        if not row["data_set"] in method_features.keys():
            method_features[row["data_set"]] = {}
        if not row["name"] in method_features[row["data_set"]].keys():
            method_features[row["data_set"]][row["name"]] = []
        method_features[row["data_set"]][row["name"]].append(row["feature"])
    
    method_features= {}
    methods_query = app.session.query(SELECTION_METHOD.id, SELECTION_METHOD.name, FEATURES_RANKING_GROUP.data_set,
                                      FEATURES_RANKING.feature, FEATURES_RANKING.ranking)\
                               .filter(SELECTION_METHOD.type=="optim", FEATURES_RANKING_GROUP.method_id==SELECTION_METHOD.id,
                                       FEATURES_RANKING.group_id==FEATURES_RANKING_GROUP.id)\
                               .order_by(SELECTION_METHOD.id, FEATURES_RANKING_GROUP.data_set, FEATURES_RANKING.ranking)\
                               .all()
    methods = pd.DataFrame([row for row in methods_query], columns=["method_id", "name", "data_set", "feature", "ranking"])
    methods.apply(dataframe_to_dict, axis=1)
    return method_features

def get_scores_for_optim_method(features, dataframes):
    scores = init_scores()
    for i in range(len(features)):
        method_features = features[:i+1]
        for name, classifier, _, _ in CLASSIFIERS:
            print(i+1,name)
            scores[name] = append_scores_for_features(scores, name, classifier, dataframes, method_features)
    return scores

def get_number_of_runs(app, method, data_set, step):
    return app.session.query(HFT_RUN_T).filter(and_(HFT_RUN_T.data_set==data_set, HFT_RUN_T.run_type==method,
                                                    HFT_RUN_T.description=="Optimization score run",
                                                    HFT_RUN_T.run_step==step, HFT_RUN_T.run_completed==True)).count()

def do_optim_method(app, dataframes, features, method, data_set):
    number_of_stored_runs = get_number_of_runs(app, method, data_set, step=11)
    number_of_runs_to_do = 30 - number_of_stored_runs
    if number_of_runs_to_do>0:
        for i in range(number_of_runs_to_do):
        # for i in range(3):
            print('Starting optim run: ', method, str(i+number_of_stored_runs), ' for ', data_set)
            run_id = get_run_id(app,"Optimization score run", method, 11, data_set)
            optim_methods_scores = {}
            optim_methods_scores[method] = get_scores_for_optim_method(features, dataframes)
            save_method_results(app, optim_methods_scores, run_id, 'optim')
            complete_run(app, run_id)

def do_optim_methods_runs(app):
    optim_methods_and_features = get_methods_and_features(app)
    dataframes = {}
    dataframes["isala"] = isala_dataframe(app)
    dataframes["NS"] = NS_dataframe(app,2)
    run_data_set = "all"
    if len(sys.argv)>=5: 
        run_data_set = sys.argv[4]
    print(run_data_set)
    for data_set, methods in optim_methods_and_features.items():
        if run_data_set=="all" or run_data_set==data_set:
            for method, features in methods.items():
                do_optim_method(app, dataframes[data_set], features, method, data_set)
