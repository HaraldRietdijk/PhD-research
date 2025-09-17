# Import libraries for the data generation part
import pandas as pd
import numpy as np
import math, sys
import random

# Get steps by following the normal distribution function
# time = current hour, mean = the peak of the distribution, std = standard deviation, 
def getStepsNormal(time, mean, std, stepAdjustment):
    """
    This function uses the normal distribution function to retrieve the probabilty at the current time, so the x-axis is in 
    time, and returns the steps for that moment in time. After retrieving the probabilty from the function it is multiplies 
    by maxSteps to get a high enough number of steps for the current time.
    Parameters:
        time, is an integer which is the current     hour to calculate the number of set steps for.
        mean, an integer representing the median needed for the normal distribution function.
        std, an integer representing the standard deviation for the normal distribution function.
        stepAdjustment, an integer that modifies the final size of the returned step.
    """
    # Get the variables needed to feed the exponential
    before_e = 1 / (std * math.sqrt(2.0*math.pi))
    e_power = -1/2 * (((time-mean) / std) * ((time-mean) / std))
    
    probability = before_e * math.exp(e_power)
    
    # Probability is between 0 and 1, adjust with a maxSteps
    # This will always result is a perfect curve
    # Steps are integers, so we correct for the floating points with the int()
    return int(probability*stepAdjustment)

def addRandomness(step, moveIntensity, time):
    """
    This function adds a slightly random number to increase variabilty in steps per hour. The randomness is movement 
    intensity dependend as well as time dependend.
    Parameters:
        step, an integere representing the steps per hour.
        moveIntensity, a string representing how active the person is. This variable has effect on the randomness.
        time, an integer representing the current hour for which the step are being modified.
    """
    # To make the step adjustment more random the program will randomly select from an array of numbers as well as use
    # an adjustment linked to how active a person is
    jitterManipulation1 = [0.05, 0.1, 0.5, 1.0, 2.5, 5.0] 
    jitter = 0
    
    # First round of jitter generation
    if moveIntensity == "low":
        jitter = int(np.random.randint(-20,20)*random.choice(jitterManipulation1))
    elif moveIntensity == "average":
        jitter = int(np.random.randint(-10,35)*random.choice(jitterManipulation1))
    elif moveIntensity == "high":
        jitter = int(np.random.randint(-5,50)*random.choice(jitterManipulation1))
    else:
        sys.exit("Unrecognized movement intensity.\nExiting program.")
    # Second round of jitter generation
    jitter = int(jitter*random.choice(jitterManipulation1))
    
    # Adding the jitter to step
    step = step + jitter
    
    # Steps cannot be less than 0!
    if step < 0:
        step = 0
        
    # Adding a random multiplication based on time of day, idea from Talko  
    if 0 <= time < 7:
        n = random.random()
        if n < 0.7:
            step = 0
        else:
            step = int(step * 0.25)
    elif 7 < time <= 18:
        n = random.random()
        step = int((step * 3) * n)
    elif 18 < time:
        n = random.random()
        step = int((step * 1) * n)
    
    return step

def addDrift(step, day, maxDays, boundOne, boundTwo, boundThree, boundFour, driftPercentageOne, driftPercentageTwo, driftPercentageThree, driftPercentageFour):
    '''
    Drift following percental decrease with four boundaries.
    Parameters:
        step, an integer which is the number of steps at the moment for which drift needs to be added.
        day, the current day. Depending on the boundaries and the current day the step may be changed little or a lot.
        maxDays, the simulation time in days. 
        boundOne/Two/Three/Four, the boundaries when drift starts to happen (boundOne) or increases (boundTwo, boundThree, 
            boundFour).
        driftPercentageOne/Two/Three/Four, the percentage of the steps reduced from steps due to drift.
    '''
    # Get max number of days in 1% for easy calculation
    maxDaysPercentage = maxDays / 100
    
    # Get the boundaries for drift
    boundaryOne = int(boundOne * maxDaysPercentage)     # The first boundIncrement% of the days nothing happens
    boundaryTwo = int(boundTwo * maxDaysPercentage)     # Drift starts appearing around this time
    boundaryThree = int(boundThree * maxDaysPercentage) # Drift is increasing around this time
    boundaryFour = int(boundFour * maxDaysPercentage)   # Drift increases even more around this time
    
    # Based on where in time we are (based on days) apply drift
    if boundaryOne <= day < boundaryTwo: # The first time to add drift, drift starts
        step = step - int((step / 100) * driftPercentageOne)
    elif boundaryTwo <= day < boundaryThree: # The drift increases
        step = step - int((step / 100) * driftPercentageTwo)
    elif boundaryThree <= day < boundaryFour: # The drift increases some more
        step = step - int((step / 100) * driftPercentageThree)
    elif boundaryFour <= day: # Maximum drift at the end of the simulation time
        step = step - int((step / 100) * driftPercentageFour)
    
    return step

def getMovePatternArray(mp, maxDays):
    """
    This function generates and returns an array with the length of the max number of simulations days (derived from 
    tmax). The biggest portion of this array will be filled with the main personality given as an argument. The rest 
    of the days will be filled with the other six personalities, randomly drawn.
    Parameters:
        mp, the main movement pattern (personality).
        maxDays, the maximum time to generate data for in days.
    """
    # Based on how many days will be simulated, calculate how many of these should the main personality
    # Get a random percentage to work with and get the number of days associated with that percentage
    percent = np.random.randint(65,80)
    daysMain = int((maxDays / 100) * percent)
    
    # Start filling the array
    mpArray = []
    for day in range(maxDays):
        # Fill the first part of the array, so the first days, with the main movement pattern
        if day < daysMain:
            mpArray.append(mp)
        else:
            # Now start adding the other possible movement patterns, randomly selected
            mpTwo = random.choice(["morning_two","morning_three","afternoon_three","evening_two","evening_three","day_one","day_three"])
            # Only use the other six, not the main
            while mpTwo == mp:
                mpTwo = random.choice(["morning_two","morning_three","afternoon_three","evening_two","evening_three","day_one","day_three"])
            mpArray.append(mpTwo)

    # Now shuffle the array so that the first days do not have to be the main movement pattern/personality
    random.shuffle(mpArray)
    
    return mpArray

def getPatternBasedStep(mp, time, stepAdjustment, peakOne, peakTwo, peakThree, widthOne, widthTwo):
    """
    This retrieves and returns the correct number for steps per hour based on the movement pattern given and the current 
    time.
    Parameters:
        mp, the movement pattern applied at the moment of calling this function.
        time, the currentHour for which the number of step per hour needs to be retrieved.
        stepAdjustment, an arbitrary integer that modifies the final size of the returned step.
        peakOne, location of the first peak in the movement pattern.
        peakTwo, location of the second peak in the movement pattern.
        peakThree, location of the third peak in the movement pattern if there is a third peak.
        widthOne, the width of the main peak of the movement pattern.
        widtTwo, the width of the other peak(s) of the movement pattern.
    """
    step = 0
        
    # Based on the movement pattern of activity we need different shapes of the normal distribution, 
    # so check for movement pattern value and currentHour to apply the right values to the normal distribution
    # First check which movement pattern to use, than check the currentHour
    if mp == "morning_three":
        #mean highest peak = 9
        # Get the locations of the peaks of the current day
        # After resetting the day, the currentHour is set to zero. So only then we need to adjust the peaks and width
        if time == 0: 
            peakOne = random.choice([8,9,10])
            peakTwo = random.choice([13,14,15])
            peakThree = random.choice([18,19,20])
            widthOne = random.choice([2,3])
            widthTwo = random.choice([4,5,6])
        # Check the time following the tripleNormal idea to get the steps for currentHour
        if time < 12:
            #step = getStepsNormal(time, 9, 3, stepAdjustment)
            step = getStepsNormal(time, peakOne, widthOne, stepAdjustment)
        elif time < 18:
            #step = getStepsNormal(time, 14, 4, stepAdjustment)
            step = getStepsNormal(time, peakTwo, widthTwo, stepAdjustment)
        else:
            #step = getStepsNormal(time, 19, 5, stepAdjustment)
            step = getStepsNormal(time, peakThree, widthTwo, stepAdjustment)
    elif mp == "morning_two":
        #mean highest around 9h, second peak is wider and lower
        # Get the locations of the peaks of the current day
        # After resetting the day, the currentHour is set to zero. So only then we need to adjust the peaks and width
        if time == 0: 
            peakOne = random.choice([8,9,10])
            peakTwo = random.choice([17,18,19])
            widthOne = random.choice([2,3])
            widthTwo = random.choice([4,5,6])
        # Get the steps for currentHour
        if time < 13:
            #step = getStepsNormal(time, 9, 3, stepAdjustment)
            step = getStepsNormal(time, peakOne, widthOne, stepAdjustment)
        else:
            #step = getStepsNormal(time, 18, 5, stepAdjustment)
            step = getStepsNormal(time, peakTwo, widthTwo, stepAdjustment)
    elif mp == "afternoon_three":
        #mean highest peak = 15
        # Get the locations of the peaks of the current day
        # After resetting the day, the currentHour is set to zero. So only then we need to adjust the peaks and width
        if time == 0: 
            peakOne = random.choice([8,9,10])
            peakTwo = random.choice([12,13,14])
            peakThree = random.choice([18,19,20])
            widthOne = random.choice([2,3])
            widthTwo = random.choice([4,5,6])
        # Get the steps for currentHour
        if time < 12:
            #step = getStepsNormal(time, 9, 4, stepAdjustment)
            step = getStepsNormal(time, peakOne, widthTwo, stepAdjustment)
        elif time < 18:
            #step = getStepsNormal(time, 13, 3, stepAdjustment)
            step = getStepsNormal(time, peakTwo, widthOne, stepAdjustment)
        else:
            #step = getStepsNormal(time, 20, 6, stepAdjustment)
            step = getStepsNormal(time, peakThree, widthTwo, stepAdjustment)
    elif mp == "evening_three":
        #mean highest peak = 20
        # Get the locations of the peaks of the current day
        # After resetting the day, the currentHour is set to zero. So only then we need to adjust the peaks and width
        if time == 0: 
            peakOne = random.choice([8,9,10])
            peakTwo = random.choice([12,13,14])
            peakThree = random.choice([19,20,21])
            widthOne = random.choice([2,3])
            widthTwo = random.choice([4,5,6])
        # Get the steps for currentHour
        if time < 12:
            #step = getStepsNormal(time, 9, 4, stepAdjustment)
            step = getStepsNormal(time, peakOne, widthTwo, stepAdjustment)
        elif time < 17:
            #step = getStepsNormal(time, 15, 4, stepAdjustment)
            step = getStepsNormal(time, peakTwo, widthTwo, stepAdjustment)
        else:
            #step = getStepsNormal(time, 20, 2, stepAdjustment)
            step = getStepsNormal(time, peakThree, widthOne, stepAdjustment)
    elif mp == "evening_two":
        #mean highest around 19h, second peak is wider and lower
        # Get the locations of the peaks of the current day
        # After resetting the day, the currentHour is set to zero. So only then we need to adjust the peaks and width
        if time == 0: 
            peakOne = random.choice([9,10,11])
            peakTwo = random.choice([18,19,20])
            widthOne = random.choice([2,3])
            widthTwo = random.choice([4,5,6])
        # Get the steps for currentHour
        if time < 15:
            #step = getStepsNormal(time, 10, 5, stepAdjustment)
            step = getStepsNormal(time, peakOne, widthTwo, stepAdjustment)
        else:
            #step = getStepsNormal(time, 19, 3, stepAdjustment)
            step = getStepsNormal(time, peakTwo, widthOne, stepAdjustment)
    elif mp == "day_three":
        #3 peaks with the same height
        # Get the locations of the peaks of the current day
        # After resetting the day, the currentHour is set to zero. So only then we need to adjust the peaks and width
        if time == 0: 
            peakOne = random.choice([8,9,10])
            peakTwo = random.choice([12,13,14])
            peakThree = random.choice([18,19,20])
            widthOne = random.choice([3,4,5])
        # Get the steps for currentHour
        if time < 11:
            #step = getStepsNormal(time, 8, 4, stepAdjustment)
            step = getStepsNormal(time, peakOne, widthOne, stepAdjustment)
        elif time < 17:
            #step = getStepsNormal(time, 13, 4, stepAdjustment)
            step = getStepsNormal(time, peakTwo, widthOne, stepAdjustment)
        else:
            #step = getStepsNormal(time, 19, 4, stepAdjustment)
            step = getStepsNormal(time, peakThree, widthOne, stepAdjustment)
    elif mp == "day_one":
        # Get the locations of the peaks of the current day
        # After resetting the day, the currentHour is set to zero. So only then we need to adjust the peaks and width
        if time == 0: 
            peakOne = random.choice([12,13,14])
            widthOne = random.choice([9,10,11])
        # Get the steps for currentHour
        # only one peak around 13 which is very wide and not moment of day dependent
        #step = getStepsNormal(time, 13, 10, stepAdjustment)
        step = getStepsNormal(time, peakOne, widthOne, stepAdjustment)
    else:
        print("Current movement pattern is: " + mp + ".")
        sys.exit("Incorrect movement pattern detected.\nExiting program.")
    
    # Now return the steps per hour for the given time and the peaks and width (in case they were updated)
    return ([step, peakOne, peakTwo, peakThree, widthOne, widthTwo])

def dataGeneration(treatment_id, settings, datasetVersion = 0):
    """
    This function generates the number of steps an individual walked per hour. The steps are 
    retrieved following a probability distribution. At the end of the function, the data 
    retrieved is exported to an csv-file following the name of the individual.
    """
    # Set variables relevant for the simulation loop
    tMax = settings["maxSimulationTime"]*24           # The program simulates per hour so tMAx is in hours
    tCounter = 0
    currentHour = 0
    currentDay = 1
    weekDay = "Monday"
    weekDays = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    # We start the weekDays list with Sunday so that we can apply the module operator for updating the weekDay
    currentWeek = 1
    RT = 0                      # Running total per day
    data = []
    
    # Variables for the distributions, they will be adjusted based on the movement pattern of the first day
    peakOne = 9
    peakTwo = 13
    peakThree = 18
    widthOne = 2
    widthTwo = 4
    
    # Variable to adjust the probability to an actual number
    stepAdjustment = 1
    
    # Retrieve on which days which movement pattern should be used
    MP = getMovePatternArray(settings["MovePattern"], settings["maxSimulationTime"]) #MP is the move patterns array in days
    
    # Add some days where there was no movement at all, 
    # which is a percentage of the total number of days for which data will be generated
    nDaysZeroSteps = int((settings["maxSimulationTime"] / 100) * np.random.randint(5,15))
    DaysZeroSteps = random.sample(range(0, (settings["maxSimulationTime"] - 1)), nDaysZeroSteps)
    for day in range(nDaysZeroSteps):
        MP[DaysZeroSteps[day]] = "zero_steps"
    
    print("Starting the data generation loop.")
    
    # Simulation loop to generate the steps per hour
    while tCounter < tMax:
        # First get the right movement pattern for the current day and the drift to apply
        currentMP = MP[(currentDay - 1)] # -1 because we start counting days with 1, but list index start with 0
        
        # If only one distribution is used (day as movement pattern, one peak), the number of steps needs to be increased more by stepAdjustment
        if currentMP == "day_one":
            stepAdjustment = 2250
        else:
            stepAdjustment = 750
        
        step = 0
        # Update steps if it is not a day on which there should be zero steps
        if currentMP != "zero_steps":
            # get the step of currentHour for current movement pattern and the (new) values for peak and width
            step, peakOne, peakTwo, peakThree, widthOne, widthTwo = getPatternBasedStep(currentMP, currentHour, stepAdjustment, peakOne, peakTwo, peakThree, widthOne, widthTwo)
        
            # The distributions give standard the same number, add jitter/randomness to break the perfection of the data set
            step = addRandomness(step, settings["MovementIntensity"], currentHour)
            
            # Apply drift to the work hours 7h-18h, total of 12 hours
            if 6 < currentHour < 19:
                step = addDrift(step, currentDay, settings["maxSimulationTime"], 
                                settings["boundOne"], settings["boundTwo"], settings["boundThree"], settings["boundFour"], 
                                settings["driftPercentageOne"], settings["driftPercentageTwo"], settings["driftPercentageThree"], settings["driftPercentageFour"])
         
        # Update running total
        if currentHour > 6:
            RT = RT + step
        
        # Add the new data to the list
        data.append([treatment_id, currentWeek, weekDay, currentDay, currentHour, step, RT, currentMP])
        
        # Update counter and time/date
        tCounter = tCounter + 1
        currentHour = currentHour + 1
        # Check if the day needs to be updated
        # If a new day started we also have to reset RT to 0
        if currentHour == 24:
            currentDay = currentDay + 1
            weekDay = weekDays[currentDay%7]
            currentHour = 0
            RT = 0
            # Check if the week needs to be updated, Do we need weeks?
            # We follow a normal week. If the weekend data is not needed it can be ommitted in the analysis
            if currentDay % 7 == 1: 
                currentWeek = currentWeek + 1
                #currentDay = 1 # We need the day to keep going for the movement pattern array
    
    # After the simulation loop we can change the list to a df
    # Working in a pandas df in the simulation loop is not good programming, it is significant slower
    df = pd.DataFrame(data, columns = ['treatment ID', 'week', 'weekday', 'day', 'hour', 'steps', 'running total', 'Todays movement pattern']) 
    
    print("Calculating the data dependend variables.")
    
    # Time to add step dependend variables
    df_18=df.query('hour == 18')
    average_at_18 =df_18['running total'].mean()
    df['average_at_18'] =int(average_at_18) # Average number of steps at the end of the workday
    
    df_merge_18 = df_18[['week', 'day','running total']]
    df_merge_18_2 = df_merge_18.rename(columns = {'running total':'steps_at_18_this_day'})
    df = df.merge(df_merge_18_2, on = ['week', 'day'], how = 'right') # Total of steps during working hours that day
    
    # Add if the threshold was met to the dataframe (1 is met, 0 is not met)
    df['met_threshold'] = ((df['steps_at_18_this_day'] >= df['average_at_18']))
    df['met_threshold'] = df['met_threshold'].astype(int)
    
    # Add most of the parameters from the input file, each in their own column
    # movementPattern, movementIntensity, boundOne, boundTwo, boundThree, boundFour, driftPercentageOne, 
    # driftPercentageTwo, driftPercentageThree, driftPercentageFour, motivationTime, motivationType, 
    # motivationStepAdjustment
    df['movementPattern'] = settings["MovePattern"]
    # Move 'Todays movement pattern' to after movementPattern
    df = df[['treatment ID', 'week', 'weekday', 'day', 'hour', 'steps', 'running total', 'average_at_18', 'steps_at_18_this_day', 'met_threshold', 'movementPattern', 'Todays movement pattern']]
    df['movementIntensity'] = settings["MovementIntensity"]
    df['boundOne'] = settings["boundOne"]
    df['boundTwo'] = settings["boundTwo"]
    df['boundThree'] = settings["boundThree"]
    df['boundFour'] = settings["boundFour"]
    df['driftPercentageOne'] = settings["driftPercentageOne"]
    df['driftPercentageTwo'] = settings["driftPercentageTwo"]
    df['driftPercentageThree'] = settings["driftPercentageThree"]
    df['driftPercentageFour'] = settings["driftPercentageFour"]
    df['motivationTime'] = settings["motivationTime"]
    df['motivationType'] = settings["motivationType"]
    df['motivationStepAdjustment'] = settings["motivationStepAdjustment"]
    
    if settings["getThreshold"]:
        # Get threshold met per day, on one row per day for easy reading
        # Generate a subset of the dataframe 
        df_met = df[df.columns[[1,2,3,4,9]]] # Get week, weekday, day, hour and threshold_met
        # Select for one hour to get one line
        df_met = df_met.query('hour == 23')
        df_met = df_met[df_met.columns[[0,1,2,4]]] #Drop hour
        #print(df_met.iloc[:1])
        # Export to csv    
        exportMet = "Thresholds/" + str(treatment_id) + "_threshold.csv"
        df_met.to_csv(exportMet, index = False)
        print("Exported threshold met per day, on one row per day for easy reading to " + exportMet + ".")
    
    # Export the data to a file named after the individual/treatment ID
    if datasetVersion == 1:
        exportName = "DataVFC/" + str(treatment_id) + ".csv"
    elif datasetVersion == 2:
        exportName = "DataVFC2/" + str(treatment_id) + ".csv"
    else:
        exportName = str(treatment_id) + ".csv"
    print("Exporting data to " + exportName + ".")
    df.to_csv(exportName, index = False)

