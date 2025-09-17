import pandas as pd
from sqlalchemy import and_, func

from database.models.hft_tables import HFT_DATA_T
from database.models.patient_data import THRESHOLD_TYPES, OPERATION_DATA, THRESHOLDS
from database.models.ns_views import SumStepsNS

def check_threshold_type(app, id, description):
    threshold_type=app.session.query(THRESHOLD_TYPES).filter(THRESHOLD_TYPES.id==id).first()
    if threshold_type is None:
        threshold_type = THRESHOLD_TYPES(id=id,description=description)
        app.session.add(threshold_type)
        app.session.commit()

def average_per_week(app, df_data, pre_or_post):
    print('Calculating average per week')
    threshold_type = 'AVG_W'
    check_threshold_type(app, threshold_type, 'Average per week')
    for name, groups in df_data.groupby(['treatment_id','year','week']):
        week_threshold = groups['steps'].mean()
        for weekday in groups['weekday']:
            threshold = app.session.query(THRESHOLDS).filter(and_(THRESHOLDS.treatment_id==name[0], 
                                                                    THRESHOLDS.threshold_type==threshold_type,
                                                                    THRESHOLDS.pre_or_post==pre_or_post,
                                                                    THRESHOLDS.year==name[1], 
                                                                    THRESHOLDS.week==name[2],
                                                                    THRESHOLDS.weekday==weekday))\
                                                        .first()
            if threshold is None:
                threshold = THRESHOLDS(treatment_id=name[0], threshold_type=threshold_type, 
                                       pre_or_post=pre_or_post, year=name[1], week=name[2], weekday=weekday)
            threshold.threshold = week_threshold
            app.session.add(threshold)        
    app.session.commit()

def linear_progressive_per_week(app, df_data, pre_or_post):
    print('Calculating linear progressive per week')
    threshold_type = 'LIN_W'
    check_threshold_type(app, threshold_type, 'Linear progressive per day based on weekly maximum, assuming this will be on the last day of the week')
    for treatment, treatment_data in df_data.groupby('treatment_id'):
        last_week_end = 0
        grouped_data = treatment_data.groupby(['year','week'])
        for name, groups in grouped_data:
            this_week_end = groups['steps'].max()
            average_increase = (this_week_end-last_week_end)/7
            if (average_increase<0):
                average_increase = 0
                last_week_end = (this_week_end + last_week_end)/2
            weekdays = groups.groupby(['weekday'])
            for weekday_char, steps in weekdays:
                weekday=int(weekday_char[0])
                sum_steps = steps['steps'].sum()
                if sum_steps>0:
                    threshold = app.session.query(THRESHOLDS).filter(and_(THRESHOLDS.treatment_id==treatment, 
                                                                        THRESHOLDS.threshold_type==threshold_type,
                                                                        THRESHOLDS.pre_or_post==pre_or_post,
                                                                        THRESHOLDS.year==name[0], 
                                                                        THRESHOLDS.week==name[1],
                                                                        THRESHOLDS.weekday==weekday))\
                                                            .first()
                    if threshold is None:
                        threshold = THRESHOLDS(treatment_id=treatment, threshold_type=threshold_type, 
                                            pre_or_post=pre_or_post, year=name[0], week=name[1], weekday=weekday)
                    threshold.threshold = last_week_end + average_increase*(weekday+1)
                    app.session.add(threshold)
            last_week_end = this_week_end
    app.session.commit()

def calculate_thresholds(app):
    all_data = app.session.query(SumStepsNS.daily_steps, SumStepsNS.mdate, SumStepsNS.year, 
                                 SumStepsNS.week, SumStepsNS.weekday, SumStepsNS.treatment_id, SumStepsNS.operation_date)
    all_data_dataframe = pd.DataFrame([row for row in all_data],
                          columns=['steps', 'mdate', 'year', 'week', 'weekday', 'treatment_id', 'operation_date'])
    pre_operation_dataframe = all_data_dataframe[all_data_dataframe.mdate<all_data_dataframe.operation_date] 
    post_operation_dataframe = all_data_dataframe[all_data_dataframe.mdate>all_data_dataframe.operation_date] 
    average_per_week(app, pre_operation_dataframe, 'PRE')
    linear_progressive_per_week(app, pre_operation_dataframe, 'PRE')
    average_per_week(app, post_operation_dataframe, 'POST')
    linear_progressive_per_week(app, post_operation_dataframe, 'POST')
