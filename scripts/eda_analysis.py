# EDA script
# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

# Exploratory Data Analysis Script
# Loads cleaned data, creates plots to analyze delivery time distribution
# and the impact of various features.

def haversine(lat1, lon1, lat2, lon2):
    # Calculate the great-circle distance between two points on Earth (in km).
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def main():
    # Create output directory for plots
    output_dir = 'eda_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the cleaned dataset
    df = pd.read_csv('data/amazon_delivery_cleaned.csv')

    # Compute distance between store and drop locations
    df['Distance'] = df.apply(lambda row: haversine(
        row['Store_Latitude'], row['Store_Longitude'],
        row['Drop_Latitude'], row['Drop_Longitude']), axis=1)

    # Plot 1: Distribution of Delivery Time
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Delivery_Time'], kde=True, bins=30, color='skyblue')
    plt.title('Distribution of Delivery Time')
    plt.xlabel('Delivery Time (minutes)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'delivery_time_distribution.png'))
    plt.close()

    # Plot 2: Delivery Time by Weather
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Weather', y='Delivery_Time', data=df)
    plt.title('Delivery Time by Weather')
    plt.savefig(os.path.join(output_dir, 'delivery_time_by_weather.png'))
    plt.close()

    # Plot 3: Delivery Time by Traffic
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Traffic', y='Delivery_Time', data=df)
    plt.title('Delivery Time by Traffic Level')
    plt.savefig(os.path.join(output_dir, 'delivery_time_by_traffic.png'))
    plt.close()

    # Plot 4: Delivery Time vs Agent Rating
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Agent_Rating', y='Delivery_Time', data=df)
    plt.title('Delivery Time vs Agent Rating')
    plt.xlabel('Agent Rating')
    plt.ylabel('Delivery Time (minutes)')
    plt.savefig(os.path.join(output_dir, 'delivery_time_vs_agent_rating.png'))
    plt.close()

    # Plot 5: Delivery Time vs Distance
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Distance', y='Delivery_Time', data=df)
    plt.title('Delivery Time vs Distance')
    plt.xlabel('Distance (km)')
    plt.ylabel('Delivery Time (minutes)')
    plt.savefig(os.path.join(output_dir, 'delivery_time_vs_distance.png'))
    plt.close()

    print(f"EDA plots saved in the '{output_dir}' directory.")

if __name__ == '__main__':
    main()

