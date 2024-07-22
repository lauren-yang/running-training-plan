# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Function to train and save the regression model
def train_and_save_model(data_path, model_path, imputer_path, scaler_path):
    # Load your dataset
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

    # Save the model, imputer, and scaler
    joblib.dump(model, model_path)
    joblib.dump(imputer, imputer_path)
    joblib.dump(scaler, scaler_path)

# Train and save the model
train_and_save_model('data/activities.csv', 'models/trained_model.pkl', 'models/imputer.pkl', 'models/scaler.pkl')
