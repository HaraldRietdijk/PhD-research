import pandas as pd

from sqlalchemy import and_, func, literal_column

from database.models.feature_selection_data import METHOD_RESULTS, METHOD_MODEL_AVERAGE, SELECTION_METHOD
from database.models.hft_tables import HFT_RUN_T
from steps.step_11.step_11_optimizations import get_query_filter

def collect_averages(app, optim):
    # Fill Method_model_average. This table has the average metrics per dataset/method/model/nr_features_used
    # based on the Method_Results data. This last table is filled during the scoring runs, 
    # and contains the data of at least 30 runs for each combination of dataset/method/model/nr_features_used 
    def create_or_update_average(row):
        record = app.session.query(METHOD_MODEL_AVERAGE).filter(and_(METHOD_MODEL_AVERAGE.data_set==row['data_set'],
                                                            METHOD_MODEL_AVERAGE.method_id==row['method_id'],
                                                            METHOD_MODEL_AVERAGE.model==row['model'],
                                                            METHOD_MODEL_AVERAGE.nr_features==row['nr_features'])).first()
        if record is None:
            record = METHOD_MODEL_AVERAGE(data_set = row['data_set'],method_id = row['method_id'],
                                          model = row['model'], nr_features = row['nr_features'])
        record.accuracy = row["accuracy"]
        record.f1_score = row["f1_score"]
        record.precision = row["precision"]
        record.recall = row["recall"]
        app.session.add(record)

    filters = get_query_filter([HFT_RUN_T.run_id==METHOD_RESULTS.run_id,SELECTION_METHOD.type!='base'], optim)
    averages_query = app.session.query(HFT_RUN_T.data_set, METHOD_RESULTS.method_id, METHOD_RESULTS.model, METHOD_RESULTS.nr_features,
                                func.avg(METHOD_RESULTS.accuracy).label('accuracy'),
                                func.avg(METHOD_RESULTS.f1_score).label('f1_score'),
                                func.avg(METHOD_RESULTS.precision).label('precision'),
                                func.avg(METHOD_RESULTS.recall).label('recall'))\
                            .join(SELECTION_METHOD, SELECTION_METHOD.id==METHOD_RESULTS.method_id)\
                            .filter(and_(*filters))\
                            .group_by(HFT_RUN_T.data_set, METHOD_RESULTS.method_id, METHOD_RESULTS.model, 
                                      METHOD_RESULTS.nr_features).all()
    averages = pd.DataFrame([row for row in averages_query], columns=["data_set", "method_id", "model", "nr_features",
                                                                      "accuracy","f1_score","precision","recall"])
    averages.apply(create_or_update_average, axis=1)
    app.session.commit()

def add_base_accuracy(app):
    # The scoring runs for the standard methods do not include the last features, so these values
    # are added from the base run wich has been done with all features. The metrics of the base run
    # are then added to the Method_model_average table with the last nr_of_features (i.e. 44 or 46)
    def create_or_update_base_average(row):
        record = app.session.query(METHOD_MODEL_AVERAGE).filter(and_(METHOD_MODEL_AVERAGE.method_id==row["method_id"],
                                                                     METHOD_MODEL_AVERAGE.model==row["model"],
                                                                     METHOD_MODEL_AVERAGE.data_set==row["data_set"],
                                                                     METHOD_MODEL_AVERAGE.nr_features==row["nr_features"])).first()
        if record is None:
            record = METHOD_MODEL_AVERAGE(method_id=row["method_id"], model=row["model"], data_set=row["data_set"],
                                          nr_features=row["nr_features"])
        record.accuracy = row["accuracy"]
        record.f1_score = row["f1_score"]
        record.precision = row["precision"]
        record.recall = row["recall"]
        app.session.add(record)

    nr_features_query = app.session.query(func.max(METHOD_RESULTS.nr_features).label("max_features"), HFT_RUN_T.data_set)\
                           .filter(METHOD_RESULTS.run_id==HFT_RUN_T.run_id, HFT_RUN_T.data_set.in_(["NS","isala"]))\
                           .group_by(HFT_RUN_T.data_set).subquery()    
    model_averages_query = app.session.query(METHOD_MODEL_AVERAGE.method_id, METHOD_MODEL_AVERAGE.model, 
                                             METHOD_MODEL_AVERAGE.data_set, 
                                             func.max(METHOD_MODEL_AVERAGE.nr_features).label("features"),
                                             func.max(nr_features_query.c.max_features).label("max_features"))\
                                      .join(nr_features_query, nr_features_query.c.data_set==METHOD_MODEL_AVERAGE.data_set)\
                                      .group_by(METHOD_MODEL_AVERAGE.method_id, METHOD_MODEL_AVERAGE.model, 
                                             METHOD_MODEL_AVERAGE.data_set)\
                                      .having(literal_column("features")<literal_column("max_features")).all()
    model_averages = pd.DataFrame([row for row in model_averages_query], 
                                  columns=["method_id", "model", "data_set", "features", "max_features"])
    if model_averages.shape[0]>0:
        base_accuracy_query = app.session.query(METHOD_RESULTS.model, HFT_RUN_T.data_set,
                                                METHOD_RESULTS.nr_features, func.avg(METHOD_RESULTS.accuracy).label("accuracy"),
                                                func.avg(METHOD_RESULTS.f1_score).label('f1_score'),
                                                func.avg(METHOD_RESULTS.precision).label('precision'),
                                                func.avg(METHOD_RESULTS.recall).label('recall'))\
                                        .join(SELECTION_METHOD, SELECTION_METHOD.id==METHOD_RESULTS.method_id)\
                                        .filter(and_(METHOD_RESULTS.run_id==HFT_RUN_T.run_id, SELECTION_METHOD.name=='base'))\
                                        .group_by(METHOD_RESULTS.method_id, METHOD_RESULTS.model, 
                                                HFT_RUN_T.data_set, METHOD_RESULTS.nr_features).all()
        base_accuracy = pd.DataFrame([row for row in base_accuracy_query], 
                                     columns=["model", "data_set", "nr_features", "accuracy", "f1_score", "precision", "recall"])
        all_averages = pd.merge(model_averages, base_accuracy, on=["model", "data_set"], how='inner')
        all_averages.apply(create_or_update_base_average, axis=1)
        app.session.commit()

def fill_model_averages(app, optim):
    print("Step 1: Collecting averages")
    collect_averages(app, optim)
    if not optim:
        print("Step 2: Adding base accuracy")
        add_base_accuracy(app)
