#REGRESSION MODEL TO PREDICT 5k TIMES
#slightly lower r^2 value, doesn't impute values so there is more data to work with.

#1

import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np

# Import data
dat = pd.read_csv('activities.csv')

# Filter for running activities
dat = dat[dat['Activity Type'] == 'Run']

# Specify the date format
date_format = "%b %d, %Y, %I:%M:%S %p"

# Create a datetime column
dat['Activity Date'] = pd.to_datetime(dat['Activity Date'], format=date_format)

# Sort data by date
dat = dat.sort_values('Activity Date')

# Create cumulative metrics
dat['Cumulative Distance'] = dat['Distance'].cumsum()
dat['Cumulative Duration'] = dat['Moving Time'].cumsum()

# Select features and target variable
features = ['Distance', 'Average Speed', 'Average Heart Rate', 'Average Cadence', 'Cumulative Distance', 'Cumulative Duration']
target = 'Best 5k'

# Separate features and target for the rows with target values
dat_with_target = dat.dropna(subset=[target])
X = dat_with_target[features]
y = dat_with_target[target]

# Impute missing values in the features
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Normalize/standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions using the testing data
y_pred = model.predict(X_test)

# Calculate and print the mean squared error and R^2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean squared error (MSE): {mse}")
print(f"Coefficient of determination (R^2): {r2}")

# Compute feature importances using permutation importance
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)

# Feature importance
feature_importances = pd.DataFrame(result.importances_mean, index=features, columns=['Importance']).sort_values('Importance', ascending=False)
print("Feature Importances:")
print(feature_importances)

# Impute and scale the features for the entire original dataset
X_original = dat[features]
X_original_imputed = imputer.transform(X_original)
X_original_scaled = scaler.transform(X_original_imputed)

# Predict 5K times for the entire original dataset
predicted_5k_times_all = model.predict(X_original_scaled)

# Add predicted 5K times to the original dataframe
dat['Predicted 5K Time'] = predicted_5k_times_all

# Save the predictions in a new dataframe
predicted_5k_times_all_df = dat[['Activity Date', 'Distance', 'Average Speed', 'Average Heart Rate', 'Average Cadence', 'Cumulative Distance', 'Cumulative Duration', 'Predicted 5K Time']]

# Display the number of rows in the new dataframe
print(f"Number of rows in the predicted_5k_times_all_df DataFrame: {len(predicted_5k_times_all)}")

# Scatter plot of actual vs predicted values for the test set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted 5K Time')
plt.xlabel('Actual 5K Time (minutes)')
plt.ylabel('Predicted 5K Time (minutes)')
plt.legend()
plt.grid(True)
plt.show()


###################################################

#ASSIGN FITNESS SCORE

#2

#Fitness score is based on how your predicted 5k time compares to other in your age and gender group.

import pandas as pd
import numpy as np

# Example population statistics (mean and std) for different age ranges and gender groups
# Format: {((min_age, max_age), gender): (mean_5k_time, std_5k_time)}
population_stats = {
    ((15, 19), 'F'): (1850, 343.334),
    ((20, 24), 'F'): (1880, 312.122),
    ((25, 29), 'F'): (1920, 323.046),
    ((30, 34), 'F'): (1960, 335.531),
    ((35, 39), 'F'): (1990, 391.713),
    ((40, 44), 'F'): (2070, 388.592),
    ((45, 49), 'F'): (2530, 380.669),
    ((50, 54), 'F'): (2880, 400.548),
    ((55, 59), 'F'): (3000, 400.093),
    ((60, 64), 'F'): (3540, 234.092),
    ((65, 69), 'F'): (3600, 234.092),
    ((15, 19), 'M'): (1440, 269.205),
    ((20, 24), 'M'): (1530, 327.728),
    ((25, 29), 'M'): (1590, 280.909),
    ((30, 34), 'M'): (1655, 312.122),
    ((35, 39), 'M'): (1675, 312.122),
    ((40, 44), 'M'): (1695, 327.728),
    ((45, 49), 'M'): (1740, 339.432),
    ((50, 54), 'M'): (1980, 480.667),
    ((55, 59), 'M'): (2000, 411.22),
    ((60, 64), 'M'): (2200, 475.986),
    ((65, 69), 'M'): (2200, 312.122),
    # Add more age ranges and gender groups as needed
}


# Ask the user for their age and gender
age = int(input("Enter your age: "))
gender = input("Enter your gender (M/F): ")

# Calculate fitness scores for each activity
dat['Fitness Score'] = dat['Predicted 5K Time'].apply(lambda x: calculate_fitness_score(x, age, gender))

#Ensure the fitness score is a value between 1 and 100
dat['Fitness Score'] = dat['Fitness Score'].clip(1, 100)

# Save the DataFrame with fitness scores to a new CSV file
dat.to_csv('activities_with_fitness_scores.csv', index=False)

print("The data with fitness scores has been saved to 'activities_with_fitness_scores.csv'")


# Print the most recent fitness score
most_recent_fitness_score = dat.iloc[-1]['Fitness Score']
print("Your most recent fitness score is:", most_recent_fitness_score)


##########################################


#SORTING BY WEEKS, WITH ADDITIONAL CONSIDERATIONS FOR HEART RATE NULL ACTIVITIES

#3


import pandas as pd

# Load the data
activities = pd.read_csv('activities_with_fitness_scores.csv')

# Classify each activity
def classify_activity(row, difficulty_dict, distance_dict, pace_dict):
    difficulty = ''
    distance = ''
    
    if not pd.isnull(row['Average Heart Rate']):
        for key, value in difficulty_dict.items():
            if value['min'] <= row['Average Heart Rate'] < value['max']:
                difficulty = key
                break
    else:
        for time_range, paces in pace_dict.items():
            lower, upper = map(int, time_range.split('-'))
            if lower <= row['Predicted 5K Time'] < upper:
                for pace_key, pace_value in paces.items():
                    if pace_value[0] <= row['Average Speed'] < pace_value[1]:
                        difficulty = pace_key
                        break
                break

    for key, value in distance_dict.items():
        if value['min'] <= row['Distance'] < value['max']:
            distance = key
            break
            
    return difficulty, distance

#dictionaries for classification
difficulty_dict = {
    'easy': {'min': 0, 'max': 171},
    'threshold': {'min': 171, 'max': 183},
    'hard': {'min': 183, 'max': float('inf')}
}

pace_dict = {
    "900-1020": {
        "easy": (0, 3.6),
        "threshold": (3.6, 5.9),
        "hard": (5.9, float('inf'))
    },
    "1020-1080": {
        "easy": (0, 3.6),
        "threshold": (3.6, 5.25),
        "hard": (5.25, float('inf'))
    },
    "1080-1200": {
        "easy": (0, 3.1),
        "threshold": (3.1, 4.5),
        "hard": (4.5, float('inf'))
    },
    "1200-1320": {
        "easy": (0, 3.3),
        "threshold": (3.3, 3.8),
        "hard": (3.8, float('inf'))
    },
    "1320-1440": {
        "easy": (0, 7.5),
        "threshold": (5.5, 6.5),
        "hard": (4.5, float('inf'))
    },
    "1440-1560": {
        "easy": (0, 7.5),
        "threshold": (5.5, 6.5),
        "hard": (4.5, float('inf'))
    },
    "1560-2000000000000": {
        "easy": (6.5, 7.5),
        "threshold": (5.5, 6.5),
        "hard": (4.5, float('inf'))
    },
    "1501-2000000000000": {
        "easy": (6.5, 7.5),
        "threshold": (5.5, 6.5),
        "hard": (4.5, float('inf'))
    },
    # Add more ranges as needed
}

distance_dict = {
    'short': {'min': 0, 'max': 5},
    'medium': {'min': 5, 'max': 10},
    'long': {'min': 10, 'max': float('inf')}
}

activities[['Difficulty', 'Distance Type']] = activities.apply(lambda row: classify_activity(row, difficulty_dict, distance_dict, pace_dict), axis=1, result_type='expand')

# Save the new csv with classifications
activities.to_csv('activities_wfs_types.csv', index=False)

# Extract the week from the date
activities['Week'] = pd.to_datetime(activities['Activity Date']).dt.to_period('W')

# Group by week and summarize
def summarize_week(group):
    summary = {
        'easy_long_runs': len(group[(group['Difficulty'] == 'easy') & (group['Distance Type'] == 'long')]),
        'threshold_long_runs': len(group[(group['Difficulty'] == 'threshold') & (group['Distance Type'] == 'long')]),
        'easy_medium_runs': len(group[(group['Difficulty'] == 'easy') & (group['Distance Type'] == 'medium')]),
        'threshold_medium_runs': len(group[(group['Difficulty'] == 'threshold') & (group['Distance Type'] == 'medium')]),
        'hard_long_runs': len(group[(group['Difficulty'] == 'hard') & (group['Distance Type'] == 'long')]),
        'hard_medium_runs': len(group[(group['Difficulty'] == 'hard') & (group['Distance Type'] == 'medium')]),
        'hard_short_runs': len(group[(group['Difficulty'] == 'hard') & (group['Distance Type'] == 'short')]),
        'threshold_short_runs': len(group[(group['Difficulty'] == 'threshold') & (group['Distance Type'] == 'short')]),
        'easy_short_runs': len(group[(group['Difficulty'] == 'easy') & (group['Distance Type'] == 'short')]),
    }
    for key in distance_dict.keys():
        for diff in difficulty_dict.keys():
            run_type = f"{diff}_{key}"
            summary[f"cumulative_distance_{run_type}"] = group[(group['Difficulty'] == diff) & (group['Distance Type'] == key)]['Distance'].sum()
            summary[f"cumulative_time_{run_type}"] = group[(group['Difficulty'] == diff) & (group['Distance Type'] == key)]['Moving Time'].sum()
    return pd.Series(summary)

# Initialize an empty list to hold monthly summaries
weekly_summaries = []


# Group activities by week
grouped_activities = activities.groupby('Week')

# Iterate through each group and apply the summarization function
for name, group in grouped_activities:
    summary = summarize_week(group)
    summary['Week'] = name
    weekly_summaries.append(summary)

# Convert the list of summaries to a DataFrame
weekly_training = pd.DataFrame(weekly_summaries)

# Save the weekly training summary to a new csv
weekly_training.to_csv('weekly_training.csv', index=False)

# Save the monthly training summary to a new csv
#monthly_training.to_csv('monthly_training.csv', index=False)

########################

# Extract the week from the date
activities['Week'] = pd.to_datetime(activities['Activity Date']).dt.to_period('W')

# Group by week and get the first and last fitness scores
fitness_score_changes = activities.groupby('Week')['Fitness Score'].agg(['first', 'last']).reset_index()
fitness_score_changes['Change in Fitness Score'] = fitness_score_changes['last'] - fitness_score_changes['first']

# Merge the fitness score changes with the weekly training summary
weekly_training = pd.merge(weekly_training, fitness_score_changes[['Week', 'Change in Fitness Score']], on='Week', how='left')

# Save the updated weekly training summary to a new csv
weekly_training.to_csv('weekly_training.csv', index=False)

print("Updated weekly training summary with fitness score changes saved.")


##########################################

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('monthly_training.csv')

# Adjust the features list based on available columns
features = [
    'cumulative_distance_easy_short', 'cumulative_time_easy_short',
    'cumulative_distance_easy_medium', 'cumulative_time_easy_medium',
    'cumulative_distance_easy_long', 'cumulative_time_easy_long',
    'cumulative_distance_threshold_short', 'cumulative_time_threshold_short',
    'cumulative_distance_threshold_medium', 'cumulative_time_threshold_medium',
    'cumulative_distance_threshold_long', 'cumulative_time_threshold_long',
    'cumulative_distance_hard_short', 'cumulative_time_hard_short',
    'cumulative_distance_hard_medium', 'cumulative_time_hard_medium',
    'cumulative_distance_hard_long', 'cumulative_time_hard_long',
    'easy_short_runs', 'easy_medium_runs', 'easy_long_runs',
    'threshold_short_runs', 'threshold_medium_runs', 'threshold_long_runs',
    'hard_short_runs', 'hard_medium_runs', 'hard_long_runs'
]

# Ensure that only existing columns are used as features
features = [col for col in features if col in data.columns]
print("Using features:", features)

target = 'Change in Fitness Score'

# Drop rows with missing target values
data = data.dropna(subset=[target])
print(f"Number of data points after dropping nulls in target: {len(data)}")

# Separate features and target
X = data[features]
y = data[target]

# Normalize/standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Generate polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Create a Ridge Regression model
model = Ridge()

# Define hyperparameters to tune
param_grid = {
    'alpha': [0.01, 0.1, 1, 10, 100]
}



# Train the model using the training data
grid_search.fit(X_train, y_train)

# Get the best estimator
best_model = grid_search.best_estimator_

# Make predictions using the testing data
y_pred = best_model.predict(X_test)

# Calculate and print the mean squared error and R^2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Best Model Parameters: {grid_search.best_params_}")
print(f"Mean squared error (MSE): {mse}")
print(f"Coefficient of determination (R^2): {r2}")

# Scatter plot of actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Fitness Score Increase')
plt.xlabel('Actual Fitness Score Increase')
plt.ylabel('Predicted Fitness Score Increase')
plt.legend()
plt.grid(True)
plt.show()


######################################

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load the weekly summarized data
weekly_training = pd.read_csv('weekly_training.csv')
dat = pd.read_csv('activities_wfs_types.csv')

# Select features and target variable
features = ['cumulative_distance_easy_short', 'cumulative_time_easy_short',
            'cumulative_distance_easy_medium', 'cumulative_time_easy_medium',
            'cumulative_distance_easy_long', 'cumulative_time_easy_long',
            'cumulative_distance_threshold_short', 'cumulative_time_threshold_short',
            'cumulative_distance_threshold_medium', 'cumulative_time_threshold_medium',
            'cumulative_distance_threshold_long', 'cumulative_time_threshold_long',
            'cumulative_distance_hard_short', 'cumulative_time_hard_short',
            'cumulative_distance_hard_medium', 'cumulative_time_hard_medium',
            'cumulative_distance_hard_long', 'cumulative_time_hard_long',
            'easy_short_runs', 'easy_medium_runs', 'easy_long_runs',
            'threshold_short_runs', 'threshold_medium_runs', 'threshold_long_runs',
            'hard_short_runs', 'hard_medium_runs', 'hard_long_runs']
target = 'Change in Fitness Score'



# Load the CSV file
activities_forpred = pd.read_csv('activities_wfs_types.csv')


# Sort by Activity Date in descending order to get the latest activities first
activities_sorted = activities.sort_values(by='Activity Date', ascending=False)


# Retrieve the latest predicted 5k time
if not activities_sorted.empty and 'Predicted 5K Time' in activities_sorted.columns:
    latest_predicted_5k_time = activities_sorted.iloc[0]['Predicted 5K Time']
    latest_5k_date = activities_sorted.iloc[0]['Activity Date']

print('What is your goal 5k time? (use format MN:SS:MS')
goal_5k = input()
goal_5k = time_to_seconds(goal_5k)


#Convert to fitness score
desired_increase = calculate_fitness_score(latest_predicted_5k_time - goal_5k, age, gender)


# Separate features and target for the rows with target values
data_with_target = weekly_training.dropna(subset=[target])
X = data_with_target[features]
y = data_with_target[target]

# Normalize/standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create a Ridge regression model
model = Ridge()
model.fit(X_train, y_train)



# Divide the desired increase by 4 for a one-month plan
desired_increase_per_week = desired_increase / 4

# Set initial values for the optimization (mean of each feature)
initial_values = np.mean(X_scaled, axis=0)

# Run the optimization
result = minimize(objective_function, initial_values, method='BFGS')

# Get the optimized feature values
optimized_feature_values_scaled = result.x

# Inverse transform to get the original scale values
optimized_feature_values = scaler.inverse_transform([optimized_feature_values_scaled])[0]

# Map the optimized feature values back to the feature names
optimized_feature_values_dict = dict(zip(features, optimized_feature_values))

# Define upper and lower bounds for each parameter
bounds = {
    'cumulative_distance_easy_short': (0, 20),
    'cumulative_time_easy_short': (0, 1000),
    
    'cumulative_distance_easy_medium': (0, 50),
    'cumulative_time_easy_medium': (0, 2000),
    
    'cumulative_distance_easy_long': (10, 30),
    'cumulative_time_easy_long': (0, 3000),
    
    'cumulative_distance_threshold_short': (0, 10),
    'cumulative_time_threshold_short': (0, 1000),
    
    'cumulative_distance_threshold_medium': (9, 20),
    'cumulative_time_threshold_medium': (0, 2000),
    
    'cumulative_distance_threshold_long': (0, 30),
    'cumulative_time_threshold_long': (0, 3000),
    
    'cumulative_distance_hard_short': (5, 10),
    'cumulative_time_hard_short': (0, 1000),
    
    'cumulative_distance_hard_medium': (0, 20),
    'cumulative_time_hard_medium': (0, 2000),
    
    'cumulative_distance_hard_long': (0, 30),
    'cumulative_time_hard_long': (0, 3000),

    #number of each run
    'easy_short_runs': (0, 10),
    'easy_medium_runs': (1, 10),
    'easy_long_runs': (1, 10),
    'threshold_short_runs': (0, 10),
    'threshold_medium_runs': (2, 10),
    'threshold_long_runs': (0, 10),
    'hard_short_runs': (1, 10),
    'hard_medium_runs': (0, 10),
    'hard_long_runs': (0, 10),
}

# Adjust the predicted values to fall within the specified ranges and distribute over 4 weeks with progression
def adjust_and_distribute(values, bounds, weeks=4):
    adjusted_values = {}
    for key, value in values.items():
        lower, upper = bounds[key]
        adjusted_value = np.clip(value, lower, upper)
        adjusted_values[key] = adjusted_value

    # Define progression factors to increase difficulty each week
    progression_factors = [0.242, 0.253, 0.258, 0.247]
    progression_factors = np.array(progression_factors) / np.sum(progression_factors)

    # Initialize a list to hold the weekly plans
    weekly_plans = []

    # Ensure each plan has at least two easy long runs and three short hard runs over the 4 weeks
    min_easy_long_runs = 2
    min_hard_short_runs = 3

    # Distribute the values over the weeks with progression
    for week in range(weeks):
        week_plan = {k: (v * progression_factors[week]) for k, v in adjusted_values.items()}

        # Adjust to ensure the minimum number of runs
        if week == 0:
            week_plan['easy_long_runs'] = max(week_plan.get('easy_long_runs', 0), min_easy_long_runs)
            week_plan['hard_short_runs'] = max(week_plan.get('hard_short_runs', 0), min_hard_short_runs)

        weekly_plans.append(pd.Series(week_plan).apply(np.round))

    training_plan = pd.concat(weekly_plans, axis=1).T
    return training_plan

# Generate a training plan
weeks = 4
weekly_plan = adjust_and_distribute(optimized_feature_values_dict, bounds, weeks)

# Convert the training plan to a DataFrame for better readability
training_plan_df = pd.DataFrame(weekly_plan)

# Aggregate cumulative distances for each run type
aggregated_plan = training_plan_df[[
    'cumulative_distance_easy_short', 'cumulative_distance_easy_medium', 'cumulative_distance_easy_long',
    'cumulative_distance_threshold_short', 'cumulative_distance_threshold_medium', 'cumulative_distance_threshold_long',
    'cumulative_distance_hard_short', 'cumulative_distance_hard_medium', 'cumulative_distance_hard_long'
]]

# Rename columns for better readability
aggregated_plan.columns = [
    'Easy Short', 'Easy Medium', 'Easy Long',
    'Threshold Short', 'Threshold Medium', 'Threshold Long',
    'Hard Short', 'Hard Medium', 'Hard Long'
]

# Plot the aggregated plan
fig, ax = plt.subplots(figsize=(12, 8))
aggregated_plan.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('Recommended Training Plan: Cumulative Distances')
ax.set_xlabel('Week')
ax.set_ylabel('Distance (km)')
plt.tight_layout()
plt.show()

print("Recommended training plan (cumulative distances):")
print(aggregated_plan)

# Save the aggregated training plan to a CSV file
aggregated_plan.to_csv('aggregated_recommended_training_plan.csv', index=False)

# Calculate total distance per week
total_distance_per_week = aggregated_plan.sum(axis=1)
print("Total distance per week:")
print(total_distance_per_week)

print("Training plan saved to 'aggregated_recommended_training_plan.csv'.")




   





