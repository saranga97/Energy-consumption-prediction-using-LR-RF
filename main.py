import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def linear_regression_model(X_train, y_train, X_test):
    # Train a Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train) # Train the Liner Regression model with the best parameters
    lr_predictions = lr_model.predict(X_test) # Make predictions using the trained Liner Regression model
    return lr_model, lr_predictions # Return the best parameters

def random_forest_model(X_train, y_train, X_test, param_grid):
    # Train a Random Forest Regression model
    rf = RandomForestRegressor(random_state=42) # Initialize Random Forest Regressor
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3) # Perform grid search to find the best hyperparameters
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_ # Get the best parameters found by GridSearchCV
    rf_best_model = RandomForestRegressor(**best_params, random_state=42) # Initialize Random Forest Regressor with the best parameters
    rf_best_model.fit(X_train, y_train) # Train the Random Forest model with the best parameters
    rf_best_predictions = rf_best_model.predict(X_test) # Make predictions using the trained Random Forest model
    return rf_best_model, rf_best_predictions, best_params # Return the best parameters

def save_model(model, filename):
    # Save the model to disk
    joblib.dump(model, filename)

def load_model(filename):
    # Load the model from disk
    return joblib.load(filename)

def predict_energy_consumption(model, X):
    # Predict energy consumption using the trained model
    return model.predict(X)

# Load the dataset
df = pd.read_csv('Data/Energy_consumption.csv')

# Drop the 'Timestamp' column as it's not needed
df.drop('Timestamp', inplace=True, axis=1)

# Perform one-hot encoding for categorical variables
df_encoded = pd.get_dummies(data=df)

# Calculate correlation matrix
corr_matrix = df_encoded.corr()

# Select features with highest correlation with the target variable
target_var = 'EnergyConsumption'
best_corr_cols = corr_matrix[target_var].abs().sort_values(ascending=False)[1:]
best_corr_col_names = best_corr_cols.index.tolist()

# Prepare features (X) and target variable (y)
X = df_encoded[best_corr_col_names]
y = df_encoded['EnergyConsumption']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate Linear Regression model
lr_model, lr_predictions = linear_regression_model(X_train_scaled, y_train, X_test_scaled)
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)

# Define hyperparameters grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

# Train and evaluate Random Forest model
rf_model, rf_predictions, best_params = random_forest_model(X_train_scaled, y_train, X_test_scaled, param_grid)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

# Print evaluation metrics
print("Linear Regression:")
print(f'Mean Squared Error: {lr_mse}')
print(f'R-squared: {lr_r2}')

print("\nRandom Forest Regression:")
print(f'Mean Squared Error: {rf_mse}')
print(f'R-squared: {rf_r2}')

# Save the best Random Forest model
save_model(rf_model, "random_forest_model.pkl")

# Load the saved model
loaded_rf_model = load_model("random_forest_model.pkl")

# Example usage: Make predictions using the loaded model
example_predictions = predict_energy_consumption(loaded_rf_model, X_test_scaled)
print("\nExample Predictions using the loaded Random Forest model:")
print(example_predictions)
