import math, random, sys
import numpy as np
from datetime import timedelta

def get_move_pattern_array(max_simulation_time, main_pattern, movement_patterns):
    """
    This function generates and returns an array with the length of the max number of simulations days. 
    The biggest portion of this array will be filled with the main personality given as an argument. The rest 
    of the days will be filled with the other six personalities, randomly drawn, and some days with no registered activity.
    """
    percent = np.random.randint(65,80)
    days_main_pattern = int((max_simulation_time / 100) * percent)
    movement_patterns_all_days = [main_pattern for _ in range(days_main_pattern)]
    pattern_choices = [value for value in movement_patterns.keys() if (value!=main_pattern and value!="zero_steps")]
    movement_patterns_all_days.extend([random.choice(pattern_choices) for _ in range(days_main_pattern, max_simulation_time)])
    random.shuffle(movement_patterns_all_days)
    number_of_days_with_no_activity = int((max_simulation_time / 100) * np.random.randint(5,15))
    days_with_no_steps = random.sample(range(0, (max_simulation_time - 1)), number_of_days_with_no_activity)
    for day in range(number_of_days_with_no_activity):
        movement_patterns_all_days[days_with_no_steps[day]] = "zero_steps"
    return movement_patterns_all_days

def get_peak_pattern(movement_pattern):
    peak_pattern = []
    for pattern in movement_pattern:
        time_before = pattern[0]
        peak_moment = random.choice([choice for choice in range(pattern[1], pattern[2])])
        peak_width = random.choice([choice for choice in range(pattern[3], pattern[4])])
        peak_pattern.append([time_before, peak_moment, peak_width])
    return peak_pattern

def get_normal_steps(hour, current_peaks, step_adjustment):
    search_time_before = True
    idx = 0
    while search_time_before:
        if (current_peaks[idx][0]>hour):
            search_time_before = False
        else:
            idx+=1
    moment = current_peaks[idx][1]
    width = current_peaks[idx][2]
    before_e = 1 / (width * math.sqrt(2.0*math.pi))
    e_power = -1/2 * (((hour-moment) / width) * ((hour-moment) / width))
    probability = before_e * math.exp(e_power)
    return int(probability*step_adjustment)

def get_jitter(movement_intensity):
    jitterManipulation1 = [0.05, 0.1, 0.5, 1.0, 2.5, 5.0] 
    jitter = 0
    if movement_intensity == "low":
        jitter = int(np.random.randint(-20,20)*random.choice(jitterManipulation1))
    elif movement_intensity == "average":
        jitter = int(np.random.randint(-10,35)*random.choice(jitterManipulation1))
    elif movement_intensity == "high":
        jitter = int(np.random.randint(-5,50)*random.choice(jitterManipulation1))
    else:
        sys.exit("Unrecognized movement intensity.\nExiting program.")
    return int(jitter*random.choice(jitterManipulation1))

def add_jitter_and_randomness(steps, movement_intensity, hour):
    steps = steps + get_jitter(movement_intensity)
    if steps < 0:
        steps = 0
    n = random.random()
    if 0 <= hour < 7:
        if n < 0.7:
            steps = 0
        else:
            steps = int(steps * 0.25)
    elif 7 < hour <= 18:
        steps = int((steps * 3) * n)
    elif 18 < hour:
        steps = int((steps * 1) * n)    
    return steps

def add_drift(steps, day, settings):
    for bound in settings.drift_settings:
        if bound[0] <= day < bound[1]: 
            steps = steps - int((steps / 100) * bound[2])
    return steps

def get_steps_for_hour(day, hour, settings, current_peaks, step_adjustment):
    steps = get_normal_steps(hour, current_peaks, step_adjustment)
    steps = add_jitter_and_randomness(steps, settings.movement_intensity, hour)
    if 6 < hour < 19:
        steps = add_drift(steps, day, settings)
    return steps

def get_steps_for_day(movement_pattern, day, settings, current_day):
    current_peaks = get_peak_pattern(movement_pattern[2])
    step_adjustment = movement_pattern[1]
    this_day_steps = []
    for hour in range(24):
        steps = get_steps_for_hour(day, hour, settings, current_peaks, step_adjustment)
        this_day_steps.append([current_day, hour, steps])
    return this_day_steps

def data_generation(settings, movement_patterns):
    daily_steps = []
    daily_patterns = []
    movement_patterns_all_days = get_move_pattern_array(settings.max_simulation_time, settings.movement_pattern_name, movement_patterns)
    print("starting for treatment_id:",settings.treatment_id)
    for day in range(settings.max_simulation_time):
        current_day = settings.start_date + timedelta(days=day)
        current_day_pattern = movement_patterns_all_days[day]
        if current_day_pattern=='zero_steps':
            this_day_steps = [[current_day, hour, 0] for hour in range(24)]
        else:
            this_day_steps = get_steps_for_day(movement_patterns[current_day_pattern], day, settings, current_day)
        daily_steps.extend(this_day_steps)
        daily_patterns.append([current_day, movement_patterns[current_day_pattern][0]])
    return daily_steps, daily_patterns
