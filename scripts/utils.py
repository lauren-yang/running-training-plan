# Function to find the appropriate age range for a given age
def find_age_range(age, gender):
    for (age_range, g), (mean, std) in population_stats.items():
        if age_range[0] <= age <= age_range[1] and g == gender:
            return mean, std
    return None, None

# Function to calculate fitness score
def calculate_fitness_score(predicted_5k_time, age, gender):
    mean_5k_time, std_5k_time = find_age_range(age, gender)
    if mean_5k_time is None or std_5k_time is None:
        raise ValueError("No population statistics available for the specified age and gender group.")
    fitness_score = 50 + 10 * (mean_5k_time - predicted_5k_time) / std_5k_time
    return fitness_score


# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)

# Define the objective function
def objective_function(feature_values_scaled):
    prediction = model.predict([feature_values_scaled])[0]
    return np.abs(prediction - desired_increase)



#function to convert time to seconds
def time_to_seconds(time_str):
    # Split the time string into minutes, seconds, and milliseconds
    parts = time_str.split(':')
    
    if len(parts) != 3:
        raise ValueError("Time must be in the format MINUTES:SECONDS:MILLISECONDS")
    
    # Convert each part to an integer
    minutes = int(parts[0])
    seconds = int(parts[1])
    milliseconds = int(parts[2])
    
    # Calculate the total time in seconds
    total_seconds = minutes * 60 + seconds + milliseconds / 1000
    
    return total_seconds
