import pandas as pd

from sqlalchemy import and_, func

from database.models.feature_selection_data import METHOD_MODEL_AVERAGE, METHOD_MODEL_SCORING, SELECTION_METHOD, FEATURES_SCORING
from steps.step_11.step_11_optimizations import get_query_filter

def collect_max_accuracy(app, optim):
    # Fill Method_model_scoring with the maximum average accuracy the model achieves for this method/dataset.
    # This is the max average accuracy per nr_of_feature. The corresponding nr_of_features is stored in nr_features
    def create_or_update_model_scoring(row):
        record = app.session.query(METHOD_MODEL_SCORING).filter(and_(METHOD_MODEL_SCORING.method_id==row["method_id"],
                                                               METHOD_MODEL_SCORING.data_set==row["data_set"],
                                                               METHOD_MODEL_SCORING.model==row["model"])).first()
        if record is None:
            record = METHOD_MODEL_SCORING(method_id = row["method_id"], data_set = row["data_set"],
                                          model = row["model"])
        record.max_accuracy = row["accuracy"]
        app.session.add(record)

    filters = get_query_filter([METHOD_MODEL_AVERAGE.method_id==SELECTION_METHOD.id], optim)
    method_model_averages_query = app.session.query(METHOD_MODEL_AVERAGE.method_id, METHOD_MODEL_AVERAGE.model, 
                                              METHOD_MODEL_AVERAGE.data_set, func.max(METHOD_MODEL_AVERAGE.accuracy))\
                                             .join(SELECTION_METHOD)\
                                             .filter(and_(*filters))\
                                             .group_by(METHOD_MODEL_AVERAGE.method_id, METHOD_MODEL_AVERAGE.model, 
                                              METHOD_MODEL_AVERAGE.data_set).all()
    method_model_averages = pd.DataFrame([row for row in method_model_averages_query], 
                                         columns=["method_id", "model", "data_set", "accuracy"])
    method_model_averages.apply(create_or_update_model_scoring, axis=1)
    app.session.commit()

def collect_optimal_nr_features(app, optim):
    # Fill the nr_features that corresponds to the maximum average accuracy.
    def update_model_scoring_features(row):
        record = app.session.query(METHOD_MODEL_SCORING).filter(and_(METHOD_MODEL_SCORING.method_id==row["method_id"],
                                                               METHOD_MODEL_SCORING.data_set==row["data_set"],
                                                               METHOD_MODEL_SCORING.model==row["model"])).first()
        record.nr_features = row["nr_features"]
        app.session.add(record)

    filters = get_query_filter([func.abs(METHOD_MODEL_AVERAGE.accuracy-METHOD_MODEL_SCORING.max_accuracy)<0.000001], optim)
    optimal_nr_features_query = app.session.query(METHOD_MODEL_AVERAGE.method_id, METHOD_MODEL_AVERAGE.model, 
                                              METHOD_MODEL_AVERAGE.data_set, func.min(METHOD_MODEL_AVERAGE.nr_features))\
                                           .join(METHOD_MODEL_SCORING, 
                                                 and_(METHOD_MODEL_AVERAGE.method_id==METHOD_MODEL_SCORING.method_id, 
                                                      METHOD_MODEL_AVERAGE.model==METHOD_MODEL_SCORING.model,
                                                      METHOD_MODEL_AVERAGE.data_set==METHOD_MODEL_SCORING.data_set))\
                                           .join(SELECTION_METHOD)\
                                           .filter(and_(*filters))\
                                           .group_by(METHOD_MODEL_AVERAGE.method_id, METHOD_MODEL_AVERAGE.model, 
                                              METHOD_MODEL_AVERAGE.data_set).all()
    optimal_nr_features = pd.DataFrame([row for row in optimal_nr_features_query], 
                                       columns=["method_id", "model", "data_set", "nr_features"])
    optimal_nr_features.apply(update_model_scoring_features, axis=1)
    app.session.commit()

def fill_s_ordering(app, optim):
    # Determine the s_ordering score for each combination of dataset/method/model is calculated
    # and stored in the Method_model_scoring table 
    def update_s_ordering(row):
        record = app.session.query(METHOD_MODEL_SCORING).filter(METHOD_MODEL_SCORING.id==row["id"]).first()
        max_score = row["max_accuracy"]**2
        record.s_ordering = max_score * (1 - row["sum_term"])
        record.ordering_loss = max_score - record.s_ordering
        app.session.add(record)

    contribution_sum = app.session.query(func.sum(FEATURES_SCORING.s_ordering_contribution)).filter(METHOD_MODEL_SCORING.id==FEATURES_SCORING.method_model_id)\
                                    .group_by(FEATURES_SCORING.method_model_id).label('sum_term')
    filters = get_query_filter([METHOD_MODEL_SCORING.method_id==SELECTION_METHOD.id], optim)
    model_scoring_query = app.session.query(METHOD_MODEL_SCORING.id, METHOD_MODEL_SCORING.max_accuracy, contribution_sum)\
                                     .join(SELECTION_METHOD)\
                                     .filter(and_(*filters)).all()
    model_scoring = pd.DataFrame([row for row in model_scoring_query], columns=["id", "max_accuracy", "sum_term"])
    model_scoring.apply(update_s_ordering, axis=1)
    app.session.commit()

def fill_model_scoring(app, optim):
    print("Step 3: Collecting maximum accuracy")
    collect_max_accuracy(app, optim)
    print("Step 4: Collecting optimal number of features")
    collect_optimal_nr_features(app, optim)
