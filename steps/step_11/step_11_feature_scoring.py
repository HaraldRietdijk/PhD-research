import pandas as pd

from sqlalchemy import and_, func

from database.models.feature_selection_data import METHOD_RESULTS, METHOD_MODEL_AVERAGE, METHOD_MODEL_SCORING \
                                                    ,SELECTION_METHOD, METHOD_RESULTS_FEATURES,FEATURES_SCORING
from database.models.hft_tables import HFT_RUN_T
from steps.step_11.step_11_optimizations import get_query_filter

def get_model_scoring_and_average(app, optim):
    filters = get_query_filter([METHOD_MODEL_SCORING.method_id==METHOD_MODEL_AVERAGE.method_id,
               METHOD_MODEL_SCORING.model==METHOD_MODEL_AVERAGE.model,
               METHOD_MODEL_SCORING.data_set==METHOD_MODEL_AVERAGE.data_set], optim)
    method_model_metrics_query = app.session.query(METHOD_MODEL_SCORING.data_set, METHOD_MODEL_SCORING.model, METHOD_MODEL_SCORING.method_id,
                                                    METHOD_MODEL_SCORING.id, METHOD_MODEL_AVERAGE.accuracy, METHOD_MODEL_AVERAGE.nr_features)\
                                        .join(SELECTION_METHOD, SELECTION_METHOD.id==METHOD_MODEL_SCORING.method_id)\
                                        .filter(and_(*filters)).all()
    return pd.DataFrame([row for row in method_model_metrics_query], columns=["data_set", "model", "method_id", "method_model_id", "accuracy", "ranking"])

def collect_feature_ranking(app, optim):
    # Fill Feature_scoring, per dataset/method/model the averages of the features are stored here.
    # The model accuracy is the average of the model at the point the feature is added (ranking).
    # Accuracy delta and s_ordering contribution are added later.
    def create_or_update_feature_scoring(row):
        record = app.session.query(FEATURES_SCORING).filter(and_(FEATURES_SCORING.method_model_id==row["method_model_id"],
                                                                 FEATURES_SCORING.feature==row["feature"])).first()
        if record is None:
            record = FEATURES_SCORING(method_model_id = row["method_model_id"], feature = row["feature"])
        record.ranking = row["ranking"]
        record.model_accuracy = row["accuracy"]
        app.session.add(record)

    filters = get_query_filter([HFT_RUN_T.run_id==METHOD_RESULTS.run_id, METHOD_RESULTS.id==METHOD_RESULTS_FEATURES.result_id], optim)
    feature_ranking_query = app.session.query(HFT_RUN_T.data_set, METHOD_RESULTS.method_id, METHOD_RESULTS.model, 
                                              METHOD_RESULTS_FEATURES.feature, func.min(METHOD_RESULTS.nr_features))\
                                       .join(SELECTION_METHOD, SELECTION_METHOD.id==METHOD_RESULTS.method_id)\
                                       .filter(and_(*filters))\
                                       .group_by(HFT_RUN_T.data_set, METHOD_RESULTS.method_id, 
                                                 METHOD_RESULTS.model, METHOD_RESULTS_FEATURES.feature).all()
    feature_ranking = pd.DataFrame([row for row in feature_ranking_query], 
                                   columns=["data_set", "method_id", "model", "feature", "ranking"])
    model_metrics = get_model_scoring_and_average(app, optim)
    feature_ranking = pd.merge(feature_ranking, model_metrics, on=["data_set", "model", "method_id", "ranking"], how="inner")
    feature_ranking.apply(create_or_update_feature_scoring, axis=1)
    app.session.commit()
    #Still have to add mut_inf_c

def get_feature_list_per_model(app, optim):
    def create_feature_list_per_method_model(row):
        nonlocal feature_list_per_method_model
        if not row["method_model_id"] in feature_list_per_method_model.keys():
            feature_list_per_method_model[row["method_model_id"]] = {"dataset_model": (row["data_set"],row["model"]),
                                                                     "features" : []}
        feature_list_per_method_model[row["method_model_id"]]["features"].append(row["feature"])

    feature_list_per_method_model = {}
    filters = get_query_filter([FEATURES_SCORING.method_model_id==METHOD_MODEL_SCORING.id], optim)
    feature_scoring_query = app.session.query(FEATURES_SCORING.feature, FEATURES_SCORING.method_model_id,
                                              METHOD_MODEL_SCORING.model, METHOD_MODEL_SCORING.data_set)\
                                        .join(SELECTION_METHOD, SELECTION_METHOD.id==METHOD_MODEL_SCORING.method_id)\
                                        .filter(and_(*filters)).all()
    feature_scoring = pd.DataFrame([row for row in feature_scoring_query], 
                                   columns=["feature", "method_model_id", "model", "data_set"])
    feature_scoring.apply(create_feature_list_per_method_model, axis=1)
    return feature_list_per_method_model

def get_base_data(app):
    def collect_base_data_keys(row):
        nonlocal base_data
        if not row["data_set"] in base_data.keys():
            base_data[row["data_set"]]={}
        if not row["model"] in base_data[row["data_set"]].keys():
            base_data[row["data_set"]][row["model"]] = {"features" : [], "accuracy" : 0.0}

    all_features_query = app.session.query(METHOD_RESULTS.model, HFT_RUN_T.data_set, METHOD_RESULTS_FEATURES.feature,
                                           func.avg(METHOD_RESULTS.accuracy).label("accuracy"))\
                                     .join(SELECTION_METHOD, SELECTION_METHOD.id==METHOD_RESULTS.method_id)\
                                     .filter(and_(METHOD_RESULTS.run_id==HFT_RUN_T.run_id, SELECTION_METHOD.name=='base',
                                                  METHOD_RESULTS.id==METHOD_RESULTS_FEATURES.result_id,))\
                                     .group_by(METHOD_RESULTS.model, HFT_RUN_T.data_set, METHOD_RESULTS_FEATURES.feature).all()
    all_features =  pd.DataFrame([row for row in all_features_query], columns=["model", "data_set", "feature", "accuracy"])
    base_data={}
    all_features.apply(collect_base_data_keys, axis=1)
    for data_set, data in base_data.items():
        for model in data.keys():
            base_data[data_set][model]["features"] = all_features[(all_features["model"]==model) & 
                                         (all_features["data_set"]==data_set)]["feature"].tolist()
            base_data[data_set][model]["accuracy"] = all_features[(all_features["model"]==model) & 
                                         (all_features["data_set"]==data_set)]["accuracy"].mean()
    return base_data

def add_last_feature(app, optim):
    # For standard ranking the last feature is extracted from the base scoring run
    feature_list_per_method_model = get_feature_list_per_model(app, optim)
    base_data = get_base_data(app)
    feature_added = False
    for method_model_id, data in feature_list_per_method_model.items():
        all_features_list = base_data[data["dataset_model"][0]][data["dataset_model"][1]]["features"]
        for feature in all_features_list:
            if not feature in data["features"]:
                accuracy = base_data[data["dataset_model"][0]][data["dataset_model"][1]]["accuracy"]
                feature_scoring = FEATURES_SCORING(method_model_id=method_model_id, feature=feature, 
                                                   ranking=len(all_features_list), model_accuracy=accuracy)
                app.session.add(feature_scoring)
                feature_added = True
    if feature_added:
        app.session.commit()

def get_ordering_contribution(row, accuracy_delta):
    i = row["ranking"]
    f_m = row["optimal_nr_features"]
    d_i = abs(i - f_m) / (row["nr_features"] - f_m)
    p_i = 0 if (accuracy_delta > 0 & i < f_m) else 0 if (accuracy_delta < 0 & i > f_m) else -1
    return  p_i * d_i * accuracy_delta

def collect_accuracy_delta(app, optim):
    # Based on the feature scoring the accuracy deltas and s_ordering_contributions are calculated
    def determine_accuracy_delta(row):
        nonlocal current_method_model
        nonlocal current_accuracy 
        if row["method_model_id"] != current_method_model:
            current_accuracy = 0.5
            current_method_model = row["method_model_id"]
        record = app.session.query(FEATURES_SCORING).filter(FEATURES_SCORING.id==row["id"]).first()
        record.accuracy_delta = row["model_accuracy"] - current_accuracy
        record.s_ordering_contribution = get_ordering_contribution(row, record.accuracy_delta) 
        app.session.add(record)
        current_accuracy = row["model_accuracy"]

    filters = get_query_filter([FEATURES_SCORING.method_model_id==METHOD_MODEL_SCORING.id,
                                SELECTION_METHOD.id==METHOD_MODEL_SCORING.method_id], optim)
    features_scoring_query = app.session.query(FEATURES_SCORING).join(METHOD_MODEL_SCORING).join(SELECTION_METHOD)\
                                        .filter(and_(*filters)).order_by(FEATURES_SCORING.method_model_id, FEATURES_SCORING.ranking).all()
    columns = ["id", "method_model_id", "feature", "ranking", "accuracy_delta", "model_accuracy"]
    features_scoring = pd.DataFrame([[getattr(row, attr) for attr in columns]  for row in features_scoring_query], 
                                    columns = columns)
    method_model_data_query = app.session.query(METHOD_MODEL_SCORING.id, METHOD_MODEL_SCORING.nr_features, 
                                                func.count())\
                                         .join(FEATURES_SCORING).join(SELECTION_METHOD).filter(and_(*filters))\
                                         .group_by(METHOD_MODEL_SCORING.id, METHOD_MODEL_SCORING.nr_features).all()
    method_model_data = pd.DataFrame([row for row in method_model_data_query], columns=["method_model_id", "optimal_nr_features", "nr_features"])
    features_scoring = pd.merge(features_scoring, method_model_data, on=["method_model_id"], how='inner')
    current_method_model = -1
    current_accuracy = 0.5
    features_scoring.apply(determine_accuracy_delta, axis=1)
    app.session.commit()

def fill_feature_scoring(app, optim):
    print("Step 5: Collecting feature ranking")
    collect_feature_ranking(app, optim)
    if not optim:
        print("Step 6: Adding last feature")
        add_last_feature(app, optim)
    print("Step 7: Collecting accuracy delta")
    collect_accuracy_delta(app, optim)
