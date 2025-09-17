from sqlalchemy import text, or_, not_, and_
from database.models.hft_views import SumSteps
from database.models.hft_tables import HFT_MODEL_T, HFT_PARAMETERS_T
import pandas as pd
from steps.step_generic_code.general_variables.general_variables import FITTING_PARAMETERS

def test_sql(app):
    with app.engine.begin() as conn:
        sql = text('select * from hft_treatment_t')
        result  = conn.execute(sql)
        ids = [row.treatment_id for row in result]
        print(ids)
        sql = text('select id,treatment_id,year,week,weekday,hour,sum_steps, sum_steps_hour,daily_steps from hft_sum_steps_v where \
             hour in (7,8,9,10,11,12,13,14,15,16,17,18) \
             and weekday not in (5,6) and year=2015 and (case when research_group=2 and week>15 then 1 when research_group=1 and week>4 then 1 else 0 end = 1)\
             order by year,week,weekday,hour')
        set_time_out = text('SET GLOBAL connect_timeout=6000')
        app.session.query(set_time_out)
        sumsteps=app.session.query(SumSteps).all()
        print(len(sumsteps))

def test_sql_to_df(app):
    results=app.session.query(SumSteps.treatment_id).filter(and_(not_(SumSteps.weekday.in_([5,6])),
                                        or_(SumSteps.research_group==2,SumSteps.research_group==1)))\
                                        .distinct()
    df_treatment_id=pd.DataFrame(results,columns=['treatment_id'])
    for i in df_treatment_id['treatment_id']:
        treatment_id=int(i)
        print(treatment_id)

def test_create_id(app):
    model=HFT_MODEL_T(name="hoi",algorithm="algorithm_name",destination="dest")
    app.session.add(model)
    app.session.commit()
    print(model.id)

def find_or_create_parameter_id (app, algorithm_name,parameter_name):
    parameter=app.session.query(HFT_PARAMETERS_T).filter(and_(HFT_PARAMETERS_T.hft_algorithm_t_name==algorithm_name,
                                                       HFT_PARAMETERS_T.name==parameter_name)).first()
    if not parameter:
        parameter=HFT_PARAMETERS_T(hft_algorithm_t_name=algorithm_name,name=parameter_name)
        app.session.add(parameter)
        app.session.commit()
    return parameter.id

def list_keys():
    params = { algo: [key for key in values] for algo, values in FITTING_PARAMETERS.items()}
    print(params)

def do_test(app):
    # print(find_or_create_parameter_id(app,"LR","errors twee"))
    list_keys()

# select max f1 score:
# SELECT a.hft_treatment_id, a.f1_score, m.algorithm, mp.* 
# FROM hft_metrics_t a
# inner join hft_model_t m on a.hft_model_id = m.id
# inner join hft_model_parameters_t mp on m.id = mp.hft_model_id
# where f1_score = (
# select max(b.f1_score) from hft_metrics_t b
# 	where b.hft_treatment_id = a.hft_treatment_id
#     group by b.hft_treatment_id)
    
# # select max algorith with parameter values    
# SELECT m.algorithm, p.name, mp.value
# FROM hft_metrics_t a
# inner join hft_model_t m on a.hft_model_id = m.id
# inner join hft_model_parameters_t mp on m.id = mp.hft_model_id
# inner join hft_parameters_t p on p.id= mp.hft_parameters_t_id
# where accuracy = (
# select max(accuracy) from hft_metrics_t b
# 	where b.hft_treatment_id = a.hft_treatment_id
#     group by b.hft_treatment_id)
# group by m.algorithm, p.id, mp.value
# order by m.algorithm, p.id, mp.value
    
# Max accuracy per treatment_id    
# SELECT a.hft_treatment_id, a.f1_score, a.accuracy, m.algorithm -- , mp.* 
# FROM hft_metrics_t a
# inner join hft_model_t m on a.hft_model_id = m.id
# -- inner join hft_model_parameters_t mp on m.id = mp.hft_model_id
# where accuracy = (
# select max(b.accuracy) from hft_metrics_t b
# 	where b.hft_treatment_id = a.hft_treatment_id
#     group by b.hft_treatment_id)
# order by algorithm