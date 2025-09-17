import pandas as pd
import numpy as np
import time

import os 
import pickle
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import Binarizer, MaxAbsScaler, MinMaxScaler, Normalizer, PolynomialFeatures, RobustScaler, StandardScaler

from sqlalchemy import and_, or_, not_, func

from steps.step_generic_code.dataframes_step_data import daily_steps_cat_f
from database.models.hft_tables import HFT_ALGORITHM_T, HFT_MODEL_T, HFT_MODEL_PARAMETERS_T, HFT_METRICS_T, HFT_PREDICTION_T
from database.models.hft_views import SumSteps
from steps.step_generic_code.general_variables.general_variables import FITTING_PARAMETERS, CLASSIFIERS, ALGORITHM_PARAMETERS

from database.models.hft_tables import HFT_METRICS_T
from steps.step_generic_code.general_variables.general_variables import CLASSIFIERS_WITH_PARAMETERS

def daily_steps_cat_f (steps_value,threshold):
    if (steps_value<threshold):
        #print('smaller then threshold')
        return 0
    if (steps_value>=threshold):
        #print('more then threshold')
        return 1
    
#function to determine the available models
def get_pickled_models_f (app,treatment_id, algorithm_name):
    pickle_model=str((treatment_id)+'_'+algorithm_name+'_'+'model.pkl')
    model = app.session.query(HFT_MODEL_T).filter(HFT_MODEL_T.name==pickle_model).first()
    print("Getting model for {alg} for {t_id} from {dest}"\
          .format(alg=algorithm_name,t_id=treatment_id,dest=model.destination))
    globals()['model%s' % treatment_id+algorithm_name] = pickle.load(open(os.path.join(model.destination,pickle_model), 'rb'))
    globals()['model_id%s' % treatment_id+algorithm_name]=model.id

#function to predict and save the results into the database
def predict_f(app, treatment_id,algorithm_name):
    dataframe_result = app.session.query(SumSteps.id, SumSteps.treatment_id, SumSteps.year, SumSteps.week, SumSteps.weekday,
                                SumSteps.hour, SumSteps.sum_steps, SumSteps.sum_steps_hour, SumSteps.daily_steps)\
                        .filter(and_(SumSteps.treatment_id==treatment_id,
                                        SumSteps.hour.in_([7,8,9,10,11,12,13,14,15,16,17,18]),
                                        not_(SumSteps.weekday.in_([5,6])),
                                        SumSteps.year==2015,
                                        or_(and_(SumSteps.research_group==2,SumSteps.week>15),
                                            and_(SumSteps.research_group==1,SumSteps.week>4))))\
                        .order_by(SumSteps.year,SumSteps.week,SumSteps.weekday,SumSteps.hour)    
    df= pd.DataFrame([row for row in dataframe_result],columns=['id','treatment_id','year','week','weekday','hour','sum_steps','sum_steps_hour','daily_steps'])
    
    y_pred = globals()['model%s' % treatment_id+algorithm_name].predict( df.iloc[:, 5:8].values)
    probs_all_model = globals()['model%s' % treatment_id+algorithm_name]
    if algorithm_name=="SGD":
        probs_all_model = CalibratedClassifierCV(probs_all_model)
    probs_all = probs_all_model.predict_proba(df.iloc[:, 5:8].values)
    #make a dataframe of  globals()['probs_all%s' % treatment_id] to use iloc
    probs_all=pd.DataFrame(probs_all)
    proba_all=probs_all.iloc[:,1]
    
    threshold_result = app.session.query(SumSteps.treatment_id, func.avg(SumSteps.sum_steps_hour))\
                        .filter(and_(SumSteps.treatment_id==treatment_id,
                                        SumSteps.hour==18,
                                        not_(SumSteps.weekday.in_([5,6])),
                                        SumSteps.year==2015,
                                        or_(and_(SumSteps.research_group==2,SumSteps.week>15),
                                            and_(SumSteps.research_group==1,SumSteps.week>4))))\
                        .group_by(SumSteps.treatment_id)
    df_threshold= pd.DataFrame([row for row in threshold_result],columns=['treatment_id','avg_daily_steps'])

  
    #use numpy vectorize to use function with more arguments and a dataframe determine if the threshold is met

    daily_steps=df['daily_steps']
    threshold=df_threshold['avg_daily_steps']
    df['dailysteps_cat']=np.vectorize(daily_steps_cat_f)(daily_steps, threshold)
    y= df.iloc[:,9].values
    
    #make a dataframe for all predictions
    df_prediction =pd.DataFrame({'id':df.id,'model_id':globals()['model_id%s' % treatment_id+algorithm_name],'probs':proba_all,'y':y,'y_pred': y_pred})
    #insert results into database, dataframe is based on alphabetical order.... so 
    for x in df_prediction.values:
        app.session.add(HFT_PREDICTION_T(hft_data_t_id=x[0],hft_model_t_id=x[1],probability=x[2],
                                         steps_cat=x[3],prediction_cat=x[4]))
    app.session.commit()

def do_step_4(app):
# select treatment id's for looping through all models etc...
    results = app.session.query(HFT_METRICS_T.hft_treatment_id).filter(HFT_METRICS_T.weekday==-1).distinct()
    #dataframe
    df_treatment_id=pd.DataFrame(results,columns=['treatment_id'])
    print(df_treatment_id)
    app.session.query(HFT_PREDICTION_T).delete()
    for i in df_treatment_id['treatment_id']:
        for algorithm_name, classifier, normalized in CLASSIFIERS_WITH_PARAMETERS:
            if algorithm_name=='SGD':
                continue
            else:
                get_pickled_models_f (app,str(i), algorithm_name)
                predict_f(app,int(i),algorithm_name)
