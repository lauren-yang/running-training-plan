# main.py
import pandas as pd
from regression_model import train_regression_model, adjust_and_distribute

# Load data
dat = pd.read_csv('../data/activities.csv')

# Train regression model and get predicted 5k times
dat, predicted_5k_times_all_df = train_regression_model(dat)

# Save predicted 5k times to CSV
predicted_5k_times_all_df.to_csv('../output/predicted_5k_times_all.csv', index=False)

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

# Population statistics
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
}

# Ask the user for their age and gender
age = int(input("Enter your age: "))
gender = input("Enter your gender (M/F): ")

# Calculate fitness scores for each activity
dat['Fitness Score'] = dat['Predicted 5K Time'].apply(lambda x: calculate_fitness_score(x, age, gender))

# Ensure the fitness score is a value between 1 and 100
dat['Fitness Score'] = dat['Fitness Score'].clip(1, 100)

# Save the DataFrame with fitness scores to a new CSV file
dat.to_csv('../output/activities_with_fitness_scores.csv', index=False)

print("The data with fitness scores has been saved to 'activities_with_fitness_scores.csv'")

# Print the most recent fitness score
most_recent_fitness_score = dat.iloc[-1]['Fitness Score']
print("Your most recent fitness score is:", most_recent_fitness_score)

# Function to classify activities
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

# Mock dictionaries for classification
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
}

distance_dict = {
    'short': {'min': 0, 'max': 5},
    'medium': {'min': 5, 'max': 10},
    'long': {'min': 10, 'max': float('inf')}
}

dat[['Difficulty', 'Distance Type']] = dat.apply(lambda row: classify_activity(row, difficulty_dict, distance_dict, pace_dict), axis=1, result_type='expand')

# Save the new csv with classifications
dat.to_csv('../output/activities_wfs_types.csv', index=False)

# Extract the week from the date
dat['Week'] = pd.to_datetime(dat['Activity Date']).dt.to_period('W')

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
    for key in ['short', 'medium', 'long']:
        for diff in ['easy', 'threshold', 'hard']:
            run_type = f"{diff}_{key}"
            summary[f"cumulative_distance_{run_type}"] = group[(group['Difficulty'] == diff) & (group['Distance Type'] == key)]['Distance'].sum()
            summary[f"cumulative_time_{run_type}"] = group[(group['Difficulty'] == diff) & (group['Distance Type'] == key)]['Moving Time'].sum()
    return pd.Series(summary)

# Initialize an empty list to hold weekly summaries
weekly_summaries = []

# Group activities by week
grouped_activities = dat.groupby('Week')

# Iterate through each group and apply the summarization function
for name, group in grouped_activities:
    summary = summarize_week(group)
    summary['Week'] = name
    weekly_summaries.append(summary)

# Convert the list of summaries to a DataFrame
weekly_training = pd.DataFrame(weekly_summaries)

# Save the weekly training summary to a new CSV
weekly_training.to_csv('../output/weekly_training.csv', index=False)

# Load the weekly summarized data
weekly_training = pd.read_csv('../output/weekly_training.csv')

# Extract the week from the date
dat['Week'] = pd.to_datetime(dat['Activity Date']).dt.to_period('W')

# Group by week and get the first and last fitness scores
fitness_score_changes = dat.groupby('Week')['Fitness Score'].agg(['first', 'last']).reset_index()
fitness_score_changes['Change in Fitness Score'] = fitness_score_changes['last'] - fitness_score_changes['first']

# Merge the fitness score changes with the weekly training summary
weekly_training = pd.merge(weekly_training, fitness_score_changes[['Week', 'Change in Fitness Score']], on='Week', how='left')

# Save the updated weekly training summary to a new csv
weekly_training.to_csv('../output/weekly_training.csv', index=False)

print("Updated weekly training summary with fitness score changes saved.")

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

# Define the objective function
def objective_function(feature_values_scaled):
    prediction = model.predict([feature_values_scaled])[0]
    return np.abs(prediction - desired_increase)

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
aggregated_plan.to_csv('../output/aggregated_recommended_training_plan.csv', index=False)

# Calculate total distance per week
total_distance_per_week = aggregated_plan.sum(axis=1)
print("Total distance per week:")
print(total_distance_per_week)

print("Training plan saved to 'aggregated_recommended_training_plan.csv'.")
