import warnings; warnings.filterwarnings('ignore')

# Data manipulation
import numpy as np
import pandas as pd; pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', 4)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns; color_pal = sns.color_palette("husl", 9); plt.style.use('fivethirtyeight')

# Utilities
from datetime import datetime, timedelta
import os
import re

# Load and preprocess energy data
consumptions = pd.read_csv(r"C:\Users\ashis\OneDrive\Desktop\Monthly-Daily-Energy-Forecasting-Docker-API\data\raw\energy\household_power_consumption.zip", 
                           sep=';', header=0, na_values='?', 
                           dtype={'Date': str, 'Time': str, 'Global_active_power': np.float64}, 
                           infer_datetime_format=False)

# Standardize column names using lowercase
consumptions.rename(
    columns={
        'Date': 'date',
        'Time': 'time',
        'Global_active_power': 'total_consumption'
    },
    inplace=True
)

# Define the dataframe index based on the timestamp (date-time)
consumptions.index = pd.to_datetime(
    consumptions.date + "-" + consumptions.time,
    format="%d/%m/%Y-%H:%M:%S"
)

# Drop the date and time variables that are now redundant with the index
consumptions.drop(columns=['date', 'time'], inplace=True)

# Resample daily data
consumptions_df = consumptions.resample('D').sum()

# Adjust years to range from 2020 to 2024
start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 12, 31)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
consumptions_df.index = date_range[:len(consumptions_df)]

# Data inspection
print(consumptions_df.shape)
consumptions_df.info()

# Visualization: Electric Power Consumption Over Time
plt.figure(figsize=(20, 5))
plt.title('Electric Power Consumption Over Time')
plt.xlabel('Date')
plt.ylabel('Total Consumption (kWh)')
plt.plot(consumptions_df['total_consumption'])
plt.show()

# Function to calculate the rho association metric
def calculate_rho(grouped_data, overall_mean):
    sum_of_squares_within = sum(grouped_data.apply(lambda x: len(x) * (x.mean() - overall_mean)**2))
    total_sum_of_squares = sum((consumptions_df_copy['total_consumption'] - overall_mean)**2)
    rho = sum_of_squares_within / total_sum_of_squares
    return rho

# Copy the data for transformations
consumptions_df_copy = consumptions_df.copy()
consumptions_df_copy['dayofweek'] = consumptions_df_copy.index.dayofweek
consumptions_df_copy['month'] = consumptions_df_copy.index.month
consumptions_df_copy['quarter'] = consumptions_df_copy.index.quarter
consumptions_df_copy['year'] = consumptions_df_copy.index.year

# Overall mean of total consumption
overall_mean = consumptions_df_copy['total_consumption'].mean()

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(20, 8))

# List of categories
categories = ['dayofweek', 'month', 'quarter', 'year']
category_labels = [
    ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    ['Q1', 'Q2', 'Q3', 'Q4'],
    range(consumptions_df_copy['year'].min(), consumptions_df_copy['year'].max() + 1)
]

# Plot for each category
for i, (category, labels) in enumerate(zip(categories, category_labels)):
    ax = axes[i // 2, i % 2]
    sns.boxplot(data=consumptions_df_copy, x=category, y='total_consumption', ax=ax, palette=color_pal)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Calculate the rho value for the category
    grouped = consumptions_df_copy.groupby(category)['total_consumption']
    rho = calculate_rho(grouped, overall_mean)
    
    # Add the rho value as text on the plot
    ax.text(0.95, 0.95, f'ρ = {rho:.2f}',
            transform=ax.transAxes, 
            horizontalalignment='right',
            verticalalignment='top',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.5))
    
    # Add a red line for the overall mean
    ax.axhline(overall_mean, color='red', linestyle='--')
    
    ax.set_title(f'Electric Power Consumption by {category.capitalize()}')
    ax.set_xlabel(category.capitalize())
    ax.set_ylabel('Total Consumption (kWh)')
    ax.set_xticklabels(labels)
    ax.tick_params(axis='both', which='major', labelsize=10)
plt.tight_layout()
plt.show()

# Save the adjusted dataset
consumptions_df.to_csv(r"C:\Users\ashis\OneDrive\Desktop\Monthly-Daily-Energy-Forecasting-Docker-API\data\processed\adjusted_consumptions.csv", index=True)
