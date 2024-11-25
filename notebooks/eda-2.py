import warnings; warnings.filterwarnings('ignore')

# Data manipulation
import numpy as np
import pandas as pd; pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', 4)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns; color_pal = sns.color_palette("husl", 9); plt.style.use('fivethirtyeight')

# Utilities
from datetime import datetime, date
import math
import os
import re
import missingno as msno

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
consumptions_df.tail(3)

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

# **Skip Missing Value Visualization for Electricity Data**
# msno.matrix(consumptions_df)

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

# Data inspection and weather processing
correlation_matrix = consumptions_df.corr()

# **Skip Correlation Matrix Heatmap**
# plt.figure(figsize=(6, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Correlation Matrix of Features")
# plt.show()

# Target correlations
threshold = 0.5
target_correlations = correlation_matrix['total_consumption'].drop('total_consumption', axis=0).abs()
highly_correlated_features = target_correlations[target_correlations > threshold]
print("Most correlated features with 'total_consumption':")
print(highly_correlated_features.sort_values(ascending=False))

# Process French holidays
french_holidays_df = pd.read_csv(r"C:\Users\ashis\OneDrive\Desktop\Monthly-Daily-Energy-Forecasting-Docker-API\data\raw\holidays\jours_feries_metropole.csv",
                                 parse_dates=['date'])
french_holidays_df.head(3)

# Load and process weather data
weather_dictionary = {}
data_directory = r"C:\Users\ashis\OneDrive\Desktop\Monthly-Daily-Energy-Forecasting-Docker-API\data\raw\weather"
for file_name in os.listdir(data_directory):
    if file_name.endswith(".csv"):
        weather_dictionary[file_name] = pd.read_csv(os.path.join(data_directory, file_name),
                                                    parse_dates=['datetime', 'sunrise', 'sunset'],
                                                    index_col='datetime')
print(weather_dictionary.keys())
weather_df = pd.concat([weather_df for df_name, weather_df in weather_dictionary.items()], axis=0)

# **Skip Missing Value Visualization for Weather Data**
# msno.matrix(weather_df)
# plt.figure(figsize=(5, 3))
# plt.show()

# Functions to preprocess weather data
def clean_string(s):
    return re.sub(r'[^a-zA-Z0-9\s]', '', s.replace(' ', '_')).lower()

def calculate_day_length(df, sunrise_col='sunrise', sunset_col='sunset'):
    df[sunrise_col] = pd.to_datetime(df[sunrise_col], format='%H:%M:%S').dt.time
    df[sunset_col] = pd.to_datetime(df[sunset_col], format='%H:%M:%S').dt.time
    df['day_length'] = ((pd.to_datetime(df[sunset_col].astype(str)) - pd.to_datetime(df[sunrise_col].astype(str))).dt.total_seconds()) / 3600.0
    return df.drop([sunrise_col, sunset_col], axis=1)

def preprocess_weather_data(df, start_date, end_date, columns_to_keep, column_to_encode):
    df.sort_index(inplace=True)
    df_selected = df[columns_to_keep].copy()
    df_filtered = df_selected[(df_selected.index >= start_date) & (df_selected.index <= end_date)].copy()
    df_filtered[column_to_encode] = df_filtered[column_to_encode].apply(clean_string)
    dummies = pd.get_dummies(df_filtered[column_to_encode], prefix=column_to_encode)
    df_encoded = pd.concat([df_filtered, dummies], axis=1).drop(column_to_encode, axis=1)
    return calculate_day_length(df_encoded)

columns_to_keep = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 
                   'dew', 'humidity', 'precip', 'precipprob', 'precipcover', 'snow', 'snowdepth', 
                   'windgust', 'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 
                   'visibility', 'sunrise', 'sunset', 'moonphase', 'conditions']
start_date = '2006-12-16'
end_date = '2010-11-26'
column_to_encode = 'conditions'
processed_weather_df = preprocess_weather_data(df=weather_df, 
                                               start_date=start_date, 
                                               end_date=end_date, 
                                               columns_to_keep=columns_to_keep, 
                                               column_to_encode=column_to_encode)

# Merge weather and energy data
weather_and_consumption_df = pd.merge(consumptions_df, processed_weather_df, left_index=True, right_index=True)
french_holidays_set = set(french_holidays_df.date)
weather_and_consumption_df['is_holiday'] = weather_and_consumption_df.index.isin(french_holidays_set)

# Save the merged dataset
weather_and_consumption_df.to_csv(r"C:\Users\ashis\OneDrive\Desktop\Monthly-Daily-Energy-Forecasting-Docker-API\data\processed\weather_and_consumption.csv", index=True)
