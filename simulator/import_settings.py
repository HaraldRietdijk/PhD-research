from datetime import datetime
import random
import numpy as np
from database.models.simulator_data import SETTINGS_DRIFT, SETTINGS_INPUT, SETTINGS_MOTIVATION

def get_line_as_dict(line):
    line_values={}
    line_stripped_split = line.replace(" ","").strip().split(",")
    for idx, name in enumerate(["age", "gender","research_group","start_date", "max_simulation_time", "get_threshold", 
                                "movement_pattern", "movement_intensity", "bounds", "drift_percentages", 
                                "motivation_times", "motivation_types", "motivation_step_adjustments"]):
        line_values[name] = None if line_stripped_split[idx]=='' else line_stripped_split[idx]
    return line_values

def create_settings_input_record(app, line, movement_patterns):
    if line["age"] == None:
        line["age"] = np.random.randint(18,67)
    if line["gender"] == None:
        line["gender"] = random.choice(['f', 'm', 'u'])
    if line["research_group"] == None:
        line["research_group"] = 5
    if line["start_date"] == None:
        line["start_date"] = datetime.today().strftime('%Y-%m-%d')
    movement_pattern = movement_patterns[line["movement_pattern"]][0]
    settings=SETTINGS_INPUT(age=line["age"], gender=line["gender"], research_group=line["research_group"],
                            start_date=line["start_date"], max_simulation_time=line["max_simulation_time"],
                            get_threshold=(line["get_threshold"]=='1'), movement_pattern=movement_pattern, 
                            movement_intensity=line["movement_intensity"])
    app.session.add(settings)
    app.session.commit()
    return settings.id

def create_drift_records(app, settings_id, line):
    boundaries = line["bounds"].replace("[","").replace("]","").split("-")
    percentages = line["drift_percentages"].replace("[","").replace("]","").split("-")
    for idx in range(len(boundaries)):
        new_drift = SETTINGS_DRIFT(settings_id=settings_id, boundary=boundaries[idx], percentage=percentages[idx])
        app.session.add(new_drift)
    app.session.commit()

def create_motivation_records(app, settings_id, line):
    hours = line["motivation_times"].replace("[","").replace("]","").split("-")
    effects = line["motivation_types"].replace("[","").replace("]","").split("-")
    adjustments = line["motivation_step_adjustments"].replace("[","").replace("]","").split("-")
    for idx in range(len(hours)):
        effect= None if effects[idx]=='' else effects[idx]
        adjustment = None if adjustments[idx]=='' else adjustments[idx]
        new_motivation = SETTINGS_MOTIVATION(settings_id=settings_id, hour=hours[idx], effect=effect, 
                                             step_adjustment=adjustment)
        app.session.add(new_motivation)
    app.session.commit()

def read_from_file(app, filename, movement_patterns):
    with open(filename) as input_data:
        for line in input_data:
            if line.strip()[0] != '#':
                line_values = get_line_as_dict(line)
                settings_id = create_settings_input_record(app, line_values, movement_patterns)
                create_drift_records(app, settings_id, line_values)
                create_motivation_records(app, settings_id, line_values)
    input_data.closed
    
def get_settings_input(app, filename, movement_patterns):
    read_from_file(app, filename, movement_patterns)
    return app.session.query(SETTINGS_INPUT).filter(SETTINGS_INPUT.treatment_id == None).all()