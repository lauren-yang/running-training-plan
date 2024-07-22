import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from fitness_score import calculate_fitness_score
from weekly_summary import classify_activity, summarize_week
from regression_model import train_regression_model, adjust_and_distribute

# 1. Load data
data_path = '../data/activities.csv'
dat = pd.read_csv(data_path)

# 2. Filter for running activities
dat = dat[dat['Activity Type'] == 'Run']

# 3. Create a datetime column
date_format = "%b %d, %Y, %I:%M:%S %p"
dat['Activity Date'] = pd.to_datetime(dat['Activity Date'], format=date_format)

# 4. Sort data by date
dat = dat.sort_values('Activity Date')

# 5. Create cumulative metrics
dat['Cumulative Distance'] = dat['Distance'].cumsum()
dat['Cumulative Duration'] = dat['Moving Time'].cumsum()

# 6. Select features and target variable
features = ['Distance', 'Average Speed', 'Average Heart Rate', 'Average Cadence', 'Cumulative Distance', 'Cumulative Duration']
target = 'Best 5k'

# 7. Separate features and target for the rows with target values
dat_with_target = dat.dropna(subset=[target])
X = dat_with_target[features]
y = dat_with_target[target]

# 8. Train regression model and make predictions
X_scaled, X_original_scaled, dat = train_regression_model(X, y, dat, features, target)

# 9. Calculate fitness scores
age = int(input("Enter your age: "))
gender = input("Enter your gender (M/F): ")
dat['Fitness Score'] = dat['Predicted 5K Time'].apply(lambda x: calculate_fitness_score(x, age, gender))
dat['Fitness Score'] = dat['Fitness Score'].clip(1, 100)
dat.to_csv('../output/activities_with_fitness_scores.csv', index=False)

# 10. Classify each activity and create weekly summaries
activities = pd.read_csv('../output/activities_with_fitness_scores.csv')
difficulty_dict = {
    'easy': {'min': 0, 'max': 171},
    'threshold': {'min': 171, 'max': 183},
    'hard': {'min': 183, 'max': float('inf')}
}
pace_dict = {
    "900-1020": {"easy": (0, 3.6), "threshold": (3.6, 5.9), "hard": (5.9, float('inf'))},
    "1020-1080": {"easy": (0, 3.6), "threshold": (3.6, 5.25), "hard": (5.25, float('inf'))},
    "1080-1200": {"easy": (0, 3.1), "threshold": (3.1, 4.5), "hard": (4.5, float('inf'))},
    "1200-1320": {"easy": (0, 3.3), "threshold": (3.3, 3.8), "hard": (3.8, float('inf'))},
    "1320-1440": {"easy": (0, 7.5), "threshold": (5.5, 6.5), "hard": (4.5, float('inf'))},
    "1440-1560": {"easy": (0, 7.5), "threshold": (5.5, 6.5), "hard": (4.5, float('inf'))},
    "1560-2000000000000": {"easy": (6.5, 7.5), "threshold": (5.5, 6.5), "hard": (4.5, float('inf'))},
    "1501-2000000000000": {"easy": (6.5, 7.5), "threshold": (5.5, 6.5), "hard": (4.5, float('inf'))},
}
distance_dict = {
    'short': {'min': 0, 'max': 5},
    'medium': {'min': 5, 'max': 10},
    'long': {'min': 10, 'max': float('inf')}
}

activities[['Difficulty', 'Distance Type']] = activities.apply(
    lambda row: classify_activity(row, difficulty_dict, distance_dict, pace_dict), axis=1, result_type='expand')
activities.to_csv('../output/activities_wfs_types.csv', index=False)

# 11. Create weekly summaries
weekly_training = pd.read_csv('../output/activities_wfs_types.csv')
grouped_activities = weekly_training.groupby('Week')
weekly_summaries = [summarize_week(group) for name, group in grouped_activities]
weekly_training = pd.DataFrame(weekly_summaries)
weekly_training.to_csv('../output/weekly_training.csv', index=False)

# 12. Update weekly summaries with fitness score changes
fitness_score_changes = activities.groupby('Week')['Fitness Score'].agg(['first', 'last']).reset_index()
fitness_score_changes['Change in Fitness Score'] = fitness_score_changes['last'] - fitness_score_changes['first']
weekly_training = pd.merge(weekly_training, fitness_score_changes[['Week', 'Change in Fitness Score']], on='Week', how='left')
weekly_training.to_csv('../output/weekly_training.csv', index=False)

# 13. Optimize training plan
goal_5k = input("What is your goal 5k time? (use format MM:SS:MS): ")
goal_5k = time_to_seconds(goal_5k)
latest_predicted_5k_time = activities_sorted.iloc[0]['Predicted 5K Time']
desired_increase = calculate_fitness_score(latest_predicted_5k_time - goal_5k, age, gender)
training_plan_df, aggregated_plan, total_distance_per_week = optimize_training_plan(weekly_training, features, target, desired_increase)

# 14. Save the final training plan
aggregated_plan.to_csv('../output/aggregated_recommended_training_plan.csv', index=False)
print("Recommended training plan (cumulative distances):")
print(aggregated_plan)
print("Total distance per week:")
print(total_distance_per_week)
