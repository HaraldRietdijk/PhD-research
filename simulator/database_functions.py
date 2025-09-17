import pandas as pd
from database.models.hft_tables import HFT_DATA_T, HFT_TREATMENT_T
from database.models.simulator_data import SETTINGS_DRIFT,MOVEMENT_PATTERN, PATTERN_PEAK, PEAK_WIDTH, MOVEMENT_PATTERN_USED

def add_treatment_and_run_id(app, settings, run_id):
    treatment = HFT_TREATMENT_T(age = settings.age, gender = settings.gender, research_group = settings.research_group)
    app.session.add(treatment)
    app.session.commit()
    settings.run_id = run_id
    settings.treatment_id = treatment.treatment_id
    app.session.commit()
    return settings

def get_drift_settings(app, settings):
    stored_drifts = app.session.query(SETTINGS_DRIFT).filter(SETTINGS_DRIFT.id==settings.id).all()
    lower_bound = 0
    last_percentage = 0
    drift_settings = []
    for bound in stored_drifts:
        upper_bound = int(bound.boundary * settings.max_simulation_time / 100)
        if lower_bound>0:
            drift_settings.append((lower_bound, upper_bound, last_percentage))
        lower_bound = upper_bound
        last_percentage = bound.percentage
    drift_settings.append((lower_bound, settings.max_simulation_time, last_percentage))
    settings.drift_settings = drift_settings
    return settings

def get_movement_patterns(app):
    # {'afternoon_three': [4, 750, [[12, 8, 11, 4, 7], [18, 12, 15, 2, 4], [24, 18, 21, 4, 7]]], 
    #  etc.} : pattern : [pattern_id, step_adjustment, [boundaries (hour_before, hour_from, hour_to, width from, width_to)]]
    stored_patterns = app.session.query(MOVEMENT_PATTERN.id, MOVEMENT_PATTERN.name, MOVEMENT_PATTERN.step_adjustment,
                                          PATTERN_PEAK.hour_before, PATTERN_PEAK.hour_from, PATTERN_PEAK.hour_to,
                                          PEAK_WIDTH.width_from, PEAK_WIDTH.width_to)\
                                    .join(PATTERN_PEAK, PATTERN_PEAK.pattern_id == MOVEMENT_PATTERN.id)\
                                    .join(PEAK_WIDTH, PEAK_WIDTH.id == PATTERN_PEAK.width_id)\
                                    .order_by(MOVEMENT_PATTERN.name, PATTERN_PEAK.hour_before).all()
    movement_patterns = {}
    patterns_by_id = {}
    for row in stored_patterns:
        if not(row[1] in movement_patterns.keys()):
            movement_patterns[row[1]] = [row[0], row[2], []]
            patterns_by_id[row[0]] = row[1]
        pattern_data = [row[3], row[4], row[5], row[6], row[7]]
        patterns = movement_patterns[row[1]][2]
        patterns.append(pattern_data)
        movement_patterns[row[1]] = [row[0], row[2], patterns]
    return movement_patterns, patterns_by_id

def store_daily_steps(app, daily_steps, settings):
    def add_record_to_session(row):
        year = row["current_day"].year
        week = row["current_day"].isocalendar().week
        weekday = row["current_day"].weekday()
        data_record = HFT_DATA_T(steps = row["steps"], mdate = row["current_day"], year = year, 
                                 week = week, weekday = weekday, hour = row["hour"],
                                 treatment_id=settings.treatment_id)
        app.session.add(data_record) 

    df = pd.DataFrame(daily_steps, columns = ["current_day", "hour", "steps"]) 
    df.apply(add_record_to_session, axis=1)
    app.session.commit()

def store_daily_patterns(app, daily_patterns, settings):
    for pattern in daily_patterns:
        data_record = MOVEMENT_PATTERN_USED(settings_id = settings.id,
                                            pattern_id = pattern[1],    
                                            mdate = pattern[0])
        app.session.add(data_record)
    app.session.commit()