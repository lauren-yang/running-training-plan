import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import numpy as np

def train_regression_model(dat):
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

    return dat

# Function to calculate fitness score
def calculate_fitness_score(predicted_5k_time, age, gender):
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
    }

    # Function to find the appropriate age range for a given age
    def find_age_range(age, gender):
        for (age_range, g), (mean, std) in population_stats.items():
            if age_range[0] <= age <= age_range[1] and g == gender:
                return mean, std
        return None, None

    mean_5k_time, std_5k_time = find_age_range(age, gender)
    if mean_5k_time is None or std_5k_time is None:
        raise ValueError("No population statistics available for the specified age and gender group.")
    fitness_score = 50 + 10 * (mean_5k_time - predicted_5k_time) / std_5k_time
    return fitness_score

# Function to convert time to seconds
def time_to_seconds(time_str):
    parts = time_str.split(':')
    if len(parts) != 3:
        raise ValueError("Time must be in the format MINUTES:SECONDS:MILLISECONDS")
    minutes = int(parts[0])
    seconds = int(parts[1])
    milliseconds = int(parts[2])
    total_seconds = minutes * 60 + seconds + milliseconds / 1000
    return total_seconds

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
