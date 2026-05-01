import pandas as pd

from sqlalchemy import and_, func, literal_column

from database.models.feature_selection_data import METHOD_MODEL_SCORING , SELECTION_METHOD, FEATURES_SCORING, \
                                                   FEATURES_RANKING_GROUP, FEATURES_RANKING
from steps.step_10.step_10_general_functions import get_method_id
from steps.step_11.step_11_optimizations import get_query_filter

def create_or_update_feature_ranking_group_id(app, row, metrics=False):
    commit = False
    feature_ranking_group = app.session.query(FEATURES_RANKING_GROUP)\
                                        .filter(and_(FEATURES_RANKING_GROUP.data_set==row["data_set"],
                                                    FEATURES_RANKING_GROUP.method_id==row["method_id"])).first()
    if feature_ranking_group is None:
        feature_ranking_group = FEATURES_RANKING_GROUP(data_set = row["data_set"], method_id = row["method_id"])
        commit = True
    if metrics:
        feature_ranking_group.max_accuracy = row["max_accuracy"]
        feature_ranking_group.s_ordering = row["s_ordering"]
        commit = True
    if commit:
        app.session.add(feature_ranking_group)
        app.session.commit()
    return feature_ranking_group.id

def create_or_update_rankings_info(app, optim):
    def create_or_update_feature_ranking_base(row):
        feature_ranking_group_id = create_or_update_feature_ranking_group_id(app, row, metrics=True)
        record = app.session.query(FEATURES_RANKING).filter(and_(FEATURES_RANKING.group_id==feature_ranking_group_id,
                                                                 FEATURES_RANKING.feature==row["feature"])).first()
        if record is None:
            record = FEATURES_RANKING(group_id = feature_ranking_group_id, feature = row["feature"])
        if not optim:
            record.ranking = int(row["ranking"])
            record.ranking_score = row["ranking"]
        record.model_accuracy = row["model_accuracy"]
        record.s_ordering_contribution = row["s_ordering_contribution"]
        record.accuracy_delta = row["accuracy_delta"]
        app.session.add(record)

    filters = get_query_filter([FEATURES_SCORING.method_model_id==METHOD_MODEL_SCORING.id,
                                METHOD_MODEL_SCORING.method_id==SELECTION_METHOD.id], optim)
    ranking_info_query = app.session.query(METHOD_MODEL_SCORING.data_set, METHOD_MODEL_SCORING.method_id,
                                           FEATURES_SCORING.feature, func.avg(FEATURES_SCORING.ranking),
                                           func.avg(METHOD_MODEL_SCORING.max_accuracy), func.avg(METHOD_MODEL_SCORING.s_ordering),
                                           func.avg(FEATURES_SCORING.model_accuracy), 
                                           func.avg(FEATURES_SCORING.s_ordering_contribution),
                                           func.avg(FEATURES_SCORING.accuracy_delta))\
                                    .join(SELECTION_METHOD)\
                                    .filter(and_(*filters))\
                                    .group_by(METHOD_MODEL_SCORING.data_set, METHOD_MODEL_SCORING.method_id,
                                              FEATURES_SCORING.feature).all()
    ranking_info = pd.DataFrame([row for row in ranking_info_query], 
                                 columns=["data_set", "method_id", "feature", "ranking", "max_accuracy", "s_ordering",
                                          "model_accuracy", "s_ordering_contribution", "accuracy_delta"])
    ranking_info.apply(create_or_update_feature_ranking_base, axis=1)
    app.session.commit()

def store_ordering(app, ordering_dataframe):
    def create_or_update_ordering(row):
        nonlocal data_set
        nonlocal rank
        feature_ranking_group_id = create_or_update_feature_ranking_group_id(app, row, metrics=False)
        record = app.session.query(FEATURES_RANKING).filter(and_(FEATURES_RANKING.group_id==feature_ranking_group_id,
                                                                 FEATURES_RANKING.feature==row["feature"])).first()
        if record is None:
            record = FEATURES_RANKING(group_id = feature_ranking_group_id, feature = row["feature"])
        if data_set!=row["data_set"]:
            data_set = row["data_set"]
            rank = 1
        record.ranking = rank
        record.ranking_score = row["ranking_score"]
        app.session.add(record)
        rank = rank + 1
    data_set = ""
    rank = 0
    ordering_dataframe.apply(create_or_update_ordering,axis=1)
    app.session.commit()

def simple_average(app):
    base_order_average_query = app.session.query(FEATURES_RANKING_GROUP.data_set, FEATURES_RANKING.feature,
                                                 func.avg(FEATURES_RANKING.ranking).label("ranking_score"))\
                                          .join(SELECTION_METHOD, SELECTION_METHOD.id==FEATURES_RANKING_GROUP.method_id)\
                                          .filter(and_(FEATURES_RANKING_GROUP.id==FEATURES_RANKING.group_id,
                                                       SELECTION_METHOD.use_for_ordering==True))\
                                          .group_by(FEATURES_RANKING_GROUP.data_set, FEATURES_RANKING.feature)\
                                          .order_by(FEATURES_RANKING_GROUP.data_set, literal_column("ranking_score")).all()
    base_order_average = pd.DataFrame([row for row in base_order_average_query],
                                      columns=["data_set", "feature", "ranking_score"])
    # avg+ includes mutal info classification base ordering
    # base_order_average["method_id"] = get_method_id(app, "simple_avg", "optim")
    base_order_average["method_id"] = get_method_id(app, "simple_av-", "optim")
    store_ordering(app, base_order_average)

def weighted_average(app):
    max_s_value_query = app.session.query(func.max(FEATURES_RANKING_GROUP.s_ordering).label("max_s_value"), 
                                          FEATURES_RANKING_GROUP.data_set)\
                                   .join(SELECTION_METHOD, SELECTION_METHOD.id==FEATURES_RANKING_GROUP.method_id)\
                                   .filter(SELECTION_METHOD.use_for_ordering==True)\
                                   .group_by(FEATURES_RANKING_GROUP.data_set).subquery()
    weighted_average_ordering_query = app.session.query(FEATURES_RANKING_GROUP.data_set, FEATURES_RANKING.feature,
                                               func.avg(((2-FEATURES_RANKING_GROUP.s_ordering)/(2-max_s_value_query.c.max_s_value))*FEATURES_RANKING.ranking).label("ranking_score"))\
                                          .join(SELECTION_METHOD, SELECTION_METHOD.id==FEATURES_RANKING_GROUP.method_id)\
                                          .join(max_s_value_query, max_s_value_query.c.data_set==FEATURES_RANKING_GROUP.data_set)\
                                          .filter(and_(FEATURES_RANKING_GROUP.id==FEATURES_RANKING.group_id,
                                                       SELECTION_METHOD.id==FEATURES_RANKING_GROUP.method_id,
                                                       max_s_value_query.c.data_set==FEATURES_RANKING_GROUP.data_set,
                                                       SELECTION_METHOD.use_for_ordering==True))\
                                          .group_by(FEATURES_RANKING_GROUP.data_set, FEATURES_RANKING.feature)\
                                          .order_by(FEATURES_RANKING_GROUP.data_set, literal_column("ranking_score")).all()
    weighted_average_ordering = pd.DataFrame([row for row in weighted_average_ordering_query],
                                             columns=["data_set", "feature", "ranking_score"])
    # weighted_average_ordering["method_id"] = get_method_id(app, "weight_avg", "optim")
    # avg+ includes mutal info classification base ordering 
    weighted_average_ordering["method_id"] = get_method_id(app, "weight_av-", "optim")
    store_ordering(app, weighted_average_ordering)

def accuracy_delta_based_ordering(app):
    accuracy_delta_ordering_query = app.session.query(FEATURES_RANKING_GROUP.data_set, FEATURES_RANKING.feature,
                                                      func.avg(FEATURES_RANKING.ranking*(1-FEATURES_RANKING_GROUP.max_accuracy)\
                                                               *(1-FEATURES_RANKING.accuracy_delta)/FEATURES_RANKING.model_accuracy).label("ranking_score"))\
                                              .join(SELECTION_METHOD, SELECTION_METHOD.id==FEATURES_RANKING_GROUP.method_id)\
                                              .filter(and_(FEATURES_RANKING_GROUP.id==FEATURES_RANKING.group_id,
                                                           SELECTION_METHOD.id==FEATURES_RANKING_GROUP.method_id,
                                                           SELECTION_METHOD.use_for_ordering==True))\
                                               .group_by(FEATURES_RANKING_GROUP.data_set, FEATURES_RANKING.feature)\
                                               .order_by(FEATURES_RANKING_GROUP.data_set, literal_column("ranking_score")).all()
    accuracy_delta_ordering = pd.DataFrame([row for row in accuracy_delta_ordering_query],
                                             columns=["data_set", "feature", "ranking_score"])
    accuracy_delta_ordering["method_id"] = get_method_id(app, "acc_delt-", "optim")
    store_ordering(app, accuracy_delta_ordering)

def s_ordering_contribution_weighted(app):
    distance_and_ranking = app.session.query(FEATURES_RANKING_GROUP.data_set, FEATURES_RANKING.feature,
                                             func.avg(FEATURES_RANKING.ranking).label("avg_ranking"),
                                             func.avg(FEATURES_RANKING.ranking - METHOD_MODEL_SCORING.nr_features).label("distance"))\
                                      .join(SELECTION_METHOD, SELECTION_METHOD.id==FEATURES_RANKING_GROUP.method_id)\
                                      .filter(and_(FEATURES_RANKING.group_id==FEATURES_RANKING_GROUP.id,
                                                   METHOD_MODEL_SCORING.data_set==FEATURES_RANKING_GROUP.data_set, 
                                                   METHOD_MODEL_SCORING.method_id==FEATURES_RANKING_GROUP.method_id,
                                                   SELECTION_METHOD.use_for_ordering==True))\
                                      .group_by(FEATURES_RANKING_GROUP.data_set,FEATURES_RANKING.feature).subquery()
    s_ordering_query = app.session.query(FEATURES_RANKING_GROUP.data_set, FEATURES_RANKING.feature,
                                         func.avg(distance_and_ranking.c.avg_ranking\
                                         -(FEATURES_RANKING.s_ordering_contribution*distance_and_ranking.c.distance\
                                         /METHOD_MODEL_SCORING.ordering_loss)).label("ranking_score"))\
                                  .join(SELECTION_METHOD, SELECTION_METHOD.id==FEATURES_RANKING_GROUP.method_id)\
                                  .filter(and_(FEATURES_RANKING_GROUP.id==FEATURES_RANKING.group_id,
                                          METHOD_MODEL_SCORING.method_id==FEATURES_RANKING_GROUP.method_id,
                                          METHOD_MODEL_SCORING.data_set==FEATURES_RANKING_GROUP.data_set,
                                          distance_and_ranking.c.data_set==FEATURES_RANKING_GROUP.data_set,
                                          distance_and_ranking.c.feature==FEATURES_RANKING.feature,
                                          SELECTION_METHOD.use_for_ordering==True))\
                                   .group_by(FEATURES_RANKING_GROUP.data_set,FEATURES_RANKING.feature)\
                                   .order_by(FEATURES_RANKING_GROUP.data_set, literal_column("ranking_score")).all()
    s_ordering = pd.DataFrame([row for row in s_ordering_query],
                                             columns=["data_set", "feature", "ranking_score"])
    s_ordering["method_id"] = get_method_id(app, "s_orde-", "optim")
    store_ordering(app, s_ordering)

def best_ordering(app):
    accuracy_delta_ordering_query = app.session.query(FEATURES_RANKING_GROUP.data_set, FEATURES_RANKING.feature,
                                                      func.avg(FEATURES_RANKING.ranking*(1-FEATURES_RANKING_GROUP.max_accuracy)\
                                                               *(1-FEATURES_RANKING.accuracy_delta)/FEATURES_RANKING.model_accuracy).label("ranking_score"))\
                                              .join(SELECTION_METHOD, SELECTION_METHOD.id==FEATURES_RANKING_GROUP.method_id)\
                                              .filter(and_(FEATURES_RANKING_GROUP.id==FEATURES_RANKING.group_id,
                                                           SELECTION_METHOD.id==FEATURES_RANKING_GROUP.method_id,
                                                           SELECTION_METHOD.use_for_ordering==True))\
                                               .group_by(FEATURES_RANKING_GROUP.data_set, FEATURES_RANKING.feature)\
                                               .order_by(FEATURES_RANKING_GROUP.data_set, literal_column("ranking_score")).all()
    accuracy_delta_ordering = pd.DataFrame([row for row in accuracy_delta_ordering_query],
                                             columns=["data_set", "feature", "ranking_score"])
    accuracy_delta_ordering["method_id"] = get_method_id(app, "acc_delt-", "optim")
    store_ordering(app, accuracy_delta_ordering)