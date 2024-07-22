import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import numpy as np
from scipy.optimize import minimize

def train_regression_model(X, y, dat, features, target):
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

    return X_scaled, X_original_scaled, dat

def optimize_training_plan(weekly_training, features, target, desired_increase):
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

    # Calculate total distance per week
    total_distance_per_week = aggregated_plan.sum(axis=1)

    return training_plan_df, aggregated_plan, total_distance_per_week
