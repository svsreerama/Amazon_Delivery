# Feature engineering script
# feature_engineering.py
import pandas as pd
import math
import os

# Feature Engineering Script
# Computes new features like distance and time-based features, then saves the enhanced dataset.

def haversine(lat1, lon1, lat2, lon2):
    # Calculate the great-circle distance between two points on Earth.
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def main():
    # Ensure data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    # Load the cleaned dataset
    df = pd.read_csv('data/amazon_delivery_cleaned.csv')

    # Calculate distance (in km) between store and drop locations
    df['Distance_km'] = df.apply(lambda row: haversine(
        row['Store_Latitude'], row['Store_Longitude'],
        row['Drop_Latitude'], row['Drop_Longitude']), axis=1)

    # Combine Order_Date and Order_Time if not already done
    if 'Order_DateTime' not in df.columns:
        df['Order_DateTime'] = pd.to_datetime(df['Order_Date'] + ' ' + df['Order_Time'])

    # Extract time-based features: hour of day and day of week
    df['Hour'] = pd.to_datetime(df['Order_DateTime']).dt.hour
    df['DayOfWeek'] = pd.to_datetime(df['Order_DateTime']).dt.dayofweek

    # Save the feature-enhanced dataset
    df.to_csv('data/amazon_delivery_features.csv', index=False)
    print("Feature-engineered data saved to 'data/amazon_delivery_features.csv'.")

if __name__ == '__main__':
    main()

