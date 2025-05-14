# model_development.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Model Development Script
# Trains multiple regression models and evaluates them.

def main():
    # Ensure models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')

    # Load the feature-engineered dataset
    df = pd.read_csv('data/amazon_delivery_features.csv')

    # Define features and target
    X = df[['Agent_Age', 'Agent_Rating', 'Distance_km', 'Hour', 'DayOfWeek',
            'Weather', 'Traffic', 'Vehicle', 'Area', 'Category']]
    y = df['Delivery_Time']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Preprocessing: scale numeric and one-hot encode categorical features
    numeric_features = ['Agent_Age', 'Agent_Rating', 'Distance_km', 'Hour', 'DayOfWeek']
    categorical_features = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    # Define regression pipelines
    pipelines = {
        'Linear_Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ]),
        'Random_Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        'XGBoost': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor(n_estimators=100,
                                      objective='reg:squarederror',
                                      random_state=42))
        ])
    }

    # Train and evaluate each model
    for name, model in pipelines.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"{name} -- RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")

        # Save the trained model
        model_filename = f"models/{name.lower()}_model.pkl"
        joblib.dump(model, model_filename)
        print(f"Saved {name} model to {model_filename}")

if __name__ == '__main__':
    main()
