# Data loading and cleaning script
# data_preparation.py
import pandas as pd
import os

# Data Preparation Script
# Loads raw data, cleans it, and saves the cleaned dataset.

def main():
    # Ensure the data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    # Load the raw dataset
    df = pd.read_csv('data/amazon_delivery.csv')

    # Drop rows with missing values in critical columns
    df = df.dropna(subset=['Agent_Rating', 'Weather'])

    # Convert date and time columns to datetime objects
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y')
    # Combine Order_Date and Order_Time into a single datetime if needed
    df['Order_DateTime'] = pd.to_datetime(df['Order_Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Order_Time'])
    df['Pickup_DateTime'] = pd.to_datetime(df['Order_Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Pickup_Time'])

    # Save the cleaned dataset to a new CSV file
    df.to_csv('data/amazon_delivery_cleaned.csv', index=False)
    print("Cleaned data saved to 'data/amazon_delivery_cleaned.csv'.")

if __name__ == '__main__':
    main()

