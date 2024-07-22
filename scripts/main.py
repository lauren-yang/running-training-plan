# main.py
import pandas as pd
from regression_model import train_regression_model, adjust_and_distribute

# Load data
dat = pd.read_csv('../data/activities.csv')

# Train regression model and get predicted 5k times
dat, predicted_5k_times_all_df = train_regression_model(dat)

# Save predicted 5k times to CSV
predicted_5k_times_all_df.to_csv('../output/predicted_5k_times_all.csv', index=False)

# Additional processing...

# Calculate total distance per week
# Ensure the fitness score is a value between 1 and 100
dat['Fitness Score'] = dat['Fitness Score'].clip(1, 100)

# Save the DataFrame with fitness scores to a new CSV file
dat.to_csv('../output/activities_with_fitness_scores.csv', index=False)

# Print the most recent fitness score
most_recent_fitness_score = dat.iloc[-1]['Fitness Score']
print("Your most recent fitness score is:", most_recent_fitness_score)

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

# Adjust the predicted values to fall within the specified ranges and distribute over 4 weeks with progression
weekly_plan = adjust_and_distribute(optimized_feature_values_dict, bounds)

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
