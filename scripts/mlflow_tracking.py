# mlflow_tracking.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

# MLflow Tracking Script
# Logs model parameters, metrics, and models to MLflow.

def main():
    # Set experiment name
    mlflow.set_experiment("Amazon_Delivery_Time_Prediction")

    # Load the feature-engineered dataset
    df = pd.read_csv('data/amazon_delivery_features.csv')
    X = df[['Agent_Age', 'Agent_Rating', 'Distance_km', 'Hour', 'DayOfWeek',
            'Weather', 'Traffic', 'Vehicle', 'Area', 'Category']]
    y = df['Delivery_Time']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Preprocessing (same as model development)
    numeric_features = ['Agent_Age', 'Agent_Rating', 'Distance_km', 'Hour', 'DayOfWeek']
    categorical_features = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    # Define models to track
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state=42)
    }

    for name, clf in models.items():
        with mlflow.start_run(run_name=name):
            # Create pipeline with preprocessing and model
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', clf)
            ])
            # Train the model
            pipeline.fit(X_train, y_train)
            # Predict on test set
            preds = pipeline.predict(X_test)
            # Calculate metrics
            # rmse = mean_squared_error(y_test, preds, squared=False)
            # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            # Log parameters and metrics to MLflow
            mlflow.log_param("model_type", name)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            # Log the model artifact
            mlflow.sklearn.log_model(pipeline, f"{name.replace(' ', '_')}_model")
            print(f"Logged {name} to MLflow (RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f})")

if __name__ == '__main__':
    main()

