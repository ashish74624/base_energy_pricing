#!/usr/bin/env python
# coding: utf-8

# <a href="https://github.com/labrijisaad/Energy-Forecast-API/blob/main/notebooks/Machine Learning - Energy_Forecasting.ipynb" target="_blank">
#   <img src="https://img.shields.io/badge/Open%20in-GitHub-blue.svg" alt="Open In GitHub"/>
# </a>

# ## <center><a><span style="color:blue">`Machine Learning` - Energy Forecasting Dataset</span></a></center>

# Now, we will create our Machine Learning flow to train the model for future prediction. The **critical question** at this stage is: **how far into the future are we aiming to predict?** It's important to note that the farther ahead we predict, the greater the challenge for the model to make accurate predictions.
# 
# In our scenario, we'll focus on two distinct prediction horizons:
# 
# - **Short-term (`1-Day`)**: We aim to do **next-day** predictions. [Explore `1-Day` Future Predictions](#1-Day).
#   
# - **Long-term (`30-Day`)**:We aim to do a **month-ahead** predictions. [Explore `30-Day` Future Predictions](#30-Day).

# ### Import the needed libraries

# In[21]:


# Supress warnings
import warnings ; warnings.filterwarnings('ignore')

# Data manipulation
import numpy as np
import pandas as pd ; pd.set_option('display.max_columns', None) ; pd.set_option('display.max_rows', 4)

# Visualization
import matplotlib.pyplot as plt ; import matplotlib.dates as mdates
import seaborn as sns ; color_pal = sns.color_palette("husl", 9) ; plt.style.use('fivethirtyeight')
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from colorama import Fore

# Machine Learning
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# XGBoost
import xgboost as xgb

# Utilities
from datetime import datetime, date
import math
import os
import re

# Data missingness visualization
import missingno as msno

# Progress bar
from tqdm import tqdm


# #### Load Processed Data

# In[2]:


weather_and_consumption_df = pd.read_csv('../data/processed/weather_and_consumption.csv', index_col=0, parse_dates=True)
weather_and_consumption_df.head(1)


# In[3]:


weather_and_consumption_df.columns


# #### Copy the Data

# In[4]:


df = weather_and_consumption_df.copy()


# ## <center><a class="anchor" id="1-Day"><span style="color:green">Training Models for `1-Day` Future Predictions</span></a></center>

# ### <a><span style="color:red">Feature Engineering based on the `EDA`</span></a>

# In[5]:


def create_features(df, column_names, lags, window_sizes):
    """
    Create time series features based on time series index and add lag and rolling features for specified columns.
    """
    # List to store created feature names
    created_features = []

    # Basic time series features
    basic_features = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear']
    for feature in basic_features:
        # Add basic time series features to the DataFrame
        df[feature] = getattr(df.index, feature)
        created_features.append(feature)
    
    for column_name in column_names:
        # Lag features for each specified column
        for lag in lags:
            lag_feature_name = f'{column_name}_lag_{lag}'
            df[lag_feature_name] = df[column_name].shift(lag)
            created_features.append(lag_feature_name)
        
        # Rolling window features for each specified column
        for window in window_sizes:
            rolling_mean_name = f'{column_name}_rolling_mean_{window}'
            df[rolling_mean_name] = df[column_name].shift(1).rolling(window=window).mean()
            created_features.append(rolling_mean_name)
        
    return df, created_features


# In[6]:


# Apply the Feature Engineering Function 
df, created_features = create_features(df, 
                                     column_names=['total_consumption', 'Global_intensity', 'Sub_metering_3', 'Sub_metering_1', 
                                                    'temp', 'day_length', 'tempmax', 'feelslike', 'feelslikemax', 'feelslikemin', 'tempmin'], 
                                     lags=[1, 2, 3, 4, 5, 6, 7, 30, 90, 365], 
                                     window_sizes=[2, 3, 4, 5, 6, 7, 30, 90, 365])


# ### <a><span style="color:red">Train / Test Split</span></a>
# #### Define The Features and the Target

# In[7]:


# External Features that we managed to add
EXTERNAL_FEATURES = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew', 
                     'humidity', 'precip', 'precipprob', 'precipcover', 'snow', 'snowdepth', 'windgust', 'windspeed', 
                     'winddir', 'sealevelpressure', 'cloudcover', 'visibility', 'moonphase', 'conditions_clear', 
                     'conditions_overcast', 'conditions_partiallycloudy', 'conditions_rain', 'conditions_rainovercast',
                     'conditions_rainpartiallycloudy', 'conditions_snowovercast', 'conditions_snowpartiallycloudy', 
                     'conditions_snowrain', 'conditions_snowrainovercast', 'conditions_snowrainpartiallycloudy', 
                     'day_length', 'is_holiday']

# Features got with feature engineering
FEATURES = created_features

# Target Column
TARGET = 'total_consumption'


# In[8]:


df.tail(2)


# Let's first train the model on historical data and evaluate its performance in the future. We can split the data as follows:
# - Training data consists of records where the date is before `2010-05-17`.
# - Testing data consists of records where the date is on or after `2010-05-17`
# 
# ##### Define the Train / Test Data

# In[9]:


# Define a Threshold
threshold = '2010-05-17'

# Splitting the data into train and test sets based on the defined Threshold
train_df = df.loc[df.index < threshold].copy()
test_df = df.loc[df.index >= threshold].copy()

# Define the X_train / y_train
X_train = train_df[FEATURES+EXTERNAL_FEATURES]
y_train = train_df[TARGET]

# Define the X_test / y_test
X_test = test_df[FEATURES+EXTERNAL_FEATURES]
y_test = test_df[TARGET]


# ##### Plot the Train / Test Data

# In[10]:


# Plotly graph objects for training and test sets
trace1 = go.Scatter(x=train_df.index, y=train_df.total_consumption, mode='lines', name='Training Set')
trace2 = go.Scatter(x=test_df.index, y=test_df.total_consumption, mode='lines', name='Test Set')

# Add a vertical line for the train-test split date based 
vline = go.layout.Shape(type="line", x0=threshold, y0=0, x1=threshold, y1=max(df.total_consumption),
                        line=dict(color="Black", width=2, dash="dash"))

layout = go.Layout(title='Data Train/Test Split',
                   xaxis=dict(title='Date'),
                   yaxis=dict(title='Total Consumption'),
                   shapes=[vline])

fig = go.Figure(data=[trace1, trace2], layout=layout)

# Show plot
fig.show()


# ### <a><span style="color:red">Training Models for Predicting `1-Day` Ahead</span></a>
# 
# In this scenario, our goal is to train a model to forecast values one day into the future. This task involves using a **1-day lag**, which makes the challenge quite straightforward. We expect the 1-day lag feature to be crucial after training.
# 
# To evaluate performance, we'll develop and train several types of models: a **Random Forest**, an **XGBoost regressor** models.s.

# In[11]:


X_train.columns


# #### Instantiate Models

# In[12]:


# Instantiate Random Forest Regressor
rfr = RandomForestRegressor(n_estimators=600, max_depth=3)

# Instantiate XGBoost Regressor
import xgboost as xgb
xgb = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)


# #### Train Models

# In[13]:


# Train the Random Forest Regressor model
rfr.fit(X_train, y_train)


# In[14]:


# Train the XGBoost Regressor
xgb.fit(X_train, y_train, verbose=100,
        eval_set=[(X_train, y_train), (X_train, y_train)])


# #### Plot the Feature Importance

# In[15]:


# Create DataFrame for feature importances
feature_data_rfr = pd.DataFrame({
    'Feature': X_train.columns, 
    'Importance': rfr.feature_importances_,
    'Model': 'Random Forest'
})

feature_data_xgb = pd.DataFrame({
    'Feature': X_train.columns, 
    'Importance': xgb.feature_importances_,
    'Model': 'XGBoost'
})

# Combine the DataFrames
feature_data_combined = pd.concat([feature_data_rfr, feature_data_xgb])

# Sort each DataFrame by importance and select the top 10 features
top_features_rfr = feature_data_rfr.sort_values(by='Importance', ascending=False).head(10)
top_features_xgb = feature_data_xgb.sort_values(by='Importance', ascending=False).head(10)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(20, 6))  # 1 line, 2 plots

# Random Forest
sns.barplot(data=top_features_rfr, x='Importance', y='Feature', palette='viridis', ax=axs[0])
axs[0].set_title('Random Forest: Top 10 Features', fontsize=16)
axs[0].set_xlabel('Feature Importance', fontsize=12)
axs[0].set_ylabel('Feature', fontsize=12)

# XGBoost
sns.barplot(data=top_features_xgb, x='Importance', y='Feature', palette='viridis', ax=axs[1])
axs[1].set_title('XGBoost: Top 10 Features', fontsize=16)
axs[1].set_xlabel('Feature Importance', fontsize=12)
axs[1].set_ylabel('Feature', fontsize=12)

# Add main title
plt.suptitle("1-Day Future Prediction", fontsize=20)

# Adjust layout before setting the super title
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the plot as an image file
plt.savefig('../results/top_features_comparison_1-Day_Future_Prediction.png')

# Display the plot
plt.show()


# #### Analyse the Models Performances

# In[16]:


# Adding Random Forest predictions to the test set DataFrame
test_df['RandomForest_Prediction'] = rfr.predict(X_test)
test_df['XGBoost_Prediction'] = xgb.predict(X_test)

df_final = df.merge(test_df[['RandomForest_Prediction', 'XGBoost_Prediction']], how='left', left_index=True, right_index=True)

# Plotly graph for truth data and predictions
train_data = go.Scatter(x=train_df.index, y=train_df['total_consumption'], mode='lines', name='Train Data', line=dict(color='Blue'))
test_data = go.Scatter(x=test_df.index, y=test_df['total_consumption'], mode='lines', name='Test Data', line=dict(color='ForestGreen'))
random_forest_predictions = go.Scatter(x=df_final.index, y=df_final['RandomForest_Prediction'], mode='markers', name='Random Forest Predictions', marker=dict(color='Red'))
xgboost_predictions = go.Scatter(x=df_final.index, y=df_final['XGBoost_Prediction'], mode='markers', name='XGBoost Prediction Predictions', marker=dict(color='Orange'))

# Adding a vertical line for the train-test split date
vline = dict(
    type="line", x0=threshold, y0=0, x1=threshold, y1=1, line=dict(color="Black", width=2, dash="dash"), xref='x', yref='paper'
)
# Update layout for a better visual presentation
layout = go.Layout(
    title="Real Data and Predictions Comparison for '1-Day' Future Prediction",
    xaxis=dict(title='Index/Date'),
    yaxis=dict(title='Total Consumption/Predictions'),
    legend_title='Legend',
    shapes=[vline]
)

fig = go.Figure(data=[train_data, test_data, random_forest_predictions, xgboost_predictions], layout=layout)

# Show the interactive plot
fig.show()

# Calculate the RMSE for the Random Forest Model on the Test Data
y_pred_rfr = rfr.predict(X_test)
rmse_rfr = np.sqrt(mean_squared_error(y_test, y_pred_rfr))
print(f"Random Forest RMSE: {rmse_rfr}")

# Calculate the RMSE for the XGBoost Model on the Test Data
y_pred_xgb = xgb.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f"XGBoost RMSE: {rmse_xgb}")

# Save the interactive plot as HTML file
pio.write_html(fig, file='../results/Real Data and Predictions Comparison for 1-Day Future Prediction.html')


# [🔍 Open Figure](https://htmlpreview.github.io/?https://github.com/labrijisaad/Energy-Forecast-API/blob/main/results/Real%20Data%20and%20Predictions%20Comparison%20for%201-Day%20Future%20Prediction.html)

# ## <center><a class="anchor" id="30-Day"><span style="color:green">Training Models for `30-Day` Future Predictions</span></a></center>
# 
# In our second scenario, we focus on developing a model to forecast values 30 days into the future. This presents a significantly more challenging task. To enhance the model's performance, we will emplo**y Time Series Cross-Validati**on. This approach involves repeatedly training the model on the dataset using **a sliding window techniq****ue, aiming for predictions spanning 30 days. For clarity and insight, these predictions will be visualized through plottingIWe plan to build and train two types of models: **e Random Fore**st and a**n XGBoost regress**or. Additionally, we will visualize the feature importance to identify which factors most influence the model's predictions.

# ### <a><span style="color:red">Time Series `Cross Validation`</span></a>

# ##### Prepare the Data Create Features for the 30-Day Future Predictions

# In[17]:


# Copy the Original Dataset
df = weather_and_consumption_df.copy()

# Apply the Feature Engineering Function 
df, created_features = create_features(df, 
                         column_names=['total_consumption', 'Global_intensity', 'Sub_metering_3', 'Sub_metering_1', 
                                                    'temp', 'day_length', 'tempmax', 'feelslike', 'feelslikemax', 'feelslikemin', 'tempmin'], 
                         lags=[30, 40, 50, 60, 90, 365], 
                         window_sizes=[])

# Features got with feature engineering
FEATURES = created_features

# External Features that we managed to add
EXTERNAL_FEATURES = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin',
                     'feelslike', 'dew', 'humidity', 'precip', 'precipprob', 'precipcover',
                     'snow', 'snowdepth', 'windgust', 'windspeed', 'winddir',
                     'sealevelpressure', 'cloudcover', 'visibility', 'moonphase',
                     'conditions_clear', 'conditions_overcast', 'conditions_partiallycloudy',
                     'conditions_rain', 'conditions_rainovercast',
                     'conditions_rainpartiallycloudy', 'conditions_snowovercast',
                     'conditions_snowpartiallycloudy', 'conditions_snowrain',
                     'conditions_snowrainovercast', 'conditions_snowrainpartiallycloudy',
                     'day_length', 'is_holiday']

# Target Column
TARGET = 'total_consumption'


# ##### Split Data into Train / Test sets 

# In[18]:


# Define a Threshold
threshold = '2010-05-17'

# Splitting the data into train and test sets based on the defined Threshold
train_df_cv = df.loc[df.index < threshold].copy()
test_df_cv = df.loc[df.index >= threshold].copy()

# Define the X_train / y_train  and X_test / y_test 
X_train_cv = train_df_cv[FEATURES+EXTERNAL_FEATURES]
y_train_cv = train_df_cv[TARGET]

X_test_cv = test_df_cv[FEATURES+EXTERNAL_FEATURES]
y_test_cv = test_df_cv[TARGET]


# ##### Viz the Train / Test data

# In[19]:


# Creating Plotly graph objects for training and test sets
trace1 = go.Scatter(x=train_df_cv.index, y=train_df_cv.total_consumption, mode='lines', name='Training Set')
trace2 = go.Scatter(x=test_df_cv.index, y=test_df_cv.total_consumption, mode='lines', name='Test Set')

# Adding a vertical line for the train-test split date
vline = go.layout.Shape(type="line", x0=threshold, y0=0, x1=threshold, y1=max(df.total_consumption),
                        line=dict(color="Black", width=2, dash="dash"))

layout = go.Layout(title='Data Train/Test Split',
                   xaxis=dict(title='Date'),
                   yaxis=dict(title='Total Consumption'),
                   shapes=[vline])

fig = go.Figure(data=[trace1, trace2], layout=layout)

# Show plot
fig.show()


# ##### Set up the 7-folds Time Series Split on the Train data

# In[22]:


tss = TimeSeriesSplit(n_splits=7, test_size=30)

fig, axs = plt.subplots(7, 1, figsize=(20, 15), sharex=True)

# Define a color palette
color_palette = plt.get_cmap('Set1')

for fold, (train_idx, val_idx) in enumerate(tss.split(train_df_cv)):
    train_cv = df.iloc[train_idx]
    test_cv = df.iloc[val_idx]
    
    # Plot Training Set
    axs[fold].plot(train_cv.index, train_cv[TARGET], label='Training Set', linewidth=2, color=color_palette(0))
    
    # Plot Test Set
    axs[fold].plot(test_cv.index, test_cv[TARGET], label='Validation Set', color=color_palette(1), linewidth=2)
    
    # Mark the beginning of the test set
    axs[fold].axvline(test_cv.index.min(), color='gray', ls='--', lw=2)
    
    # Set title for each subplot
    axs[fold].set_title(f'Fold {fold+1}', fontsize=14, fontweight='bold')
    
    # Format the x-axis with dates
    axs[fold].xaxis.set_major_locator(mdates.AutoDateLocator())
    axs[fold].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(axs[fold].xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=12)

    # Improve readability
    axs[fold].tick_params(axis='y', labelsize=12)
    axs[fold].grid(True, which='major', linestyle='--', linewidth='0.5', color='gray')
    axs[fold].legend(fontsize=12, loc='upper left')

# Improve layout to prevent label overlap and set a shared xlabel
fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])
fig.subplots_adjust(hspace=0.4)  
fig.text(0.5, 0.02, 'Date', ha='center', va='center', fontsize=16, fontweight='bold')
fig.text(0.01, 0.5, 'Total Consumption', ha='center', va='center', rotation='vertical', fontsize=16, fontweight='bold')

# Figure title
fig.suptitle("7-Fold Time Series Split on Train Data", fontsize=24, fontweight='bold', y=0.98)

# Save the plot as an image file
plt.savefig('../results/7-Fold Time Series Split on Train Data.png')

plt.show()


# ##### Viz the Features that will be used in Training

# In[23]:


X_train_cv.columns


# #### Instantiate Models

# In[24]:


# Define the Random Forest Regressor
rfr_cv = RandomForestRegressor(n_estimators=600, max_depth=3)

# Define the XGBoost Regressor
import xgboost as xgb
xgb_cv = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                          n_estimators=2000,
                          early_stopping_rounds=50,
                          objective='reg:squarederror',
                          max_depth=3,
                          learning_rate=0.05)


# #### Train Models

# In[25]:


preds = []
scores = []

for fold, (train_idx, val_idx) in tqdm(enumerate(tss.split(X_train_cv))):
    X_train_fold = X_train_cv.iloc[train_idx]
    y_train_fold = y_train_cv.iloc[train_idx]
    X_val_fold = X_train_cv.iloc[val_idx]
    y_val_fold = y_train_cv.iloc[val_idx]

    # Fit the Random Forest model
    rfr_cv.fit(X_train_fold, y_train_fold)
    # Predict on the validation set
    y_pred_rfr = rfr_cv.predict(X_val_fold)
    # Calculate and store the score for Random Forest
    score_rfr = np.sqrt(mean_squared_error(y_val_fold, y_pred_rfr))
    print(f"Fold {fold}: Random Forest Regressor RMSE = {score_rfr}")
    
    # Fit the XGBoost model with early stopping
    xgb_cv.fit(X_train_fold, y_train_fold, verbose=100,
               eval_set=[(X_val_fold, y_val_fold)])
    # Predict on the validation set using the best iteration
    y_pred_xgb = xgb_cv.predict(X_val_fold)
    # Calculate and store the score for XGBoost
    score_xgb = np.sqrt(mean_squared_error(y_val_fold, y_pred_xgb))
    print(f"Fold {fold}: XGBoost Regressor RMSE = {score_xgb}")
    
    # Storing predictions and scores
    preds.append({'RF': y_pred_rfr, 'XGB': y_pred_xgb})
    scores.append({'RF': score_rfr, 'XGB': score_xgb})

# Optionally, calculate and print the average score across all folds for each model
avg_score_rfr = np.mean([score['RF'] for score in scores])
avg_score_xgb = np.mean([score['XGB'] for score in scores])
print(f"Random Forest Regressor Average RMSE across all folds: {avg_score_rfr}")
print(f"XGBoost Regressor Average RMSE across all folds: {avg_score_xgb}")


# #### Analyse the Models Performances RMSE while Training using Cross Validation

# In[26]:


# Extract RMSE scores for each model
rf_rmse_scores = [score['RF'] for score in scores]
xgb_rmse_scores = [score['XGB'] for score in scores]
folds = list(range(1, len(rf_rmse_scores) + 1))

# Plotting
plt.figure(figsize=(20, 4))
plt.plot(folds, rf_rmse_scores, marker='o', label='Random Forest RMSE')
plt.plot(folds, xgb_rmse_scores, marker='s', label='XGBoost RMSE')
plt.xlabel('Fold')
plt.ylabel('RMSE')
plt.title('Evolution of RMSE across folds')
plt.legend()
plt.grid(True)
plt.xticks(folds)

# Save the plot as an image file
plt.savefig('../results/Evolution of RMSE across folds.png')

# Diplay the figure
plt.show()


# #### Plot the Feature Importance

# In[27]:


# Create DataFrame for Random Forest feature importances
feature_data_rfr = pd.DataFrame({
    'Feature': X_train_cv.columns, 
    'Importance': rfr_cv.feature_importances_
}).sort_values(by='Importance', ascending=False).head(10)

# Create DataFrame for XGBoost feature importances
feature_data_xgb = pd.DataFrame({
    'Feature': X_train_cv.columns, 
    'Importance': xgb_cv.feature_importances_
}).sort_values(by='Importance', ascending=False).head(10)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(20, 6))  # 1 line, 2 plots

# Random Forest
sns.barplot(data=feature_data_rfr, x='Importance', y='Feature', palette='viridis', ax=axs[0])
axs[0].set_title('Random Forest: Top 10 Features after TimeSeries Cross val', fontsize=16)
axs[0].set_xlabel('Feature Importance', fontsize=12)
axs[0].set_ylabel('Feature', fontsize=12)

# XGBoost
sns.barplot(data=feature_data_xgb, x='Importance', y='Feature', palette='viridis', ax=axs[1])
axs[1].set_title('XGBoost: Top 10 Features after TimeSeries Cross val', fontsize=16)
axs[1].set_xlabel('Feature Importance', fontsize=12)
axs[1].set_ylabel('Feature', fontsize=12)

# Add main title
plt.suptitle("30-Day Future Prediction", fontsize=20)

# Adjust layout
plt.tight_layout()

# Save the plot as an image file
plt.savefig('../results/top_features_comparison_30-Day_Future_Prediction.png')

# Display the plot
plt.show()


# #### Analyse the Models Performances on Test data after Cross Validation

# In[28]:


# Adding Random Forest predictions to the test set DataFrame
test_df_cv['RandomForest_Prediction_cv'] = rfr_cv.predict(X_test_cv)
test_df_cv['XGBoost_Prediction_cv'] = xgb_cv.predict(X_test_cv)


df_final = df.merge(test_df_cv[['RandomForest_Prediction_cv', 'XGBoost_Prediction_cv']], how='left', left_index=True, right_index=True)

# Plotly graph for truth data and predictions
train_data = go.Scatter(x=train_df_cv.index, y=train_df_cv['total_consumption'], mode='lines', name='Train Data', line=dict(color='Blue'))
test_data = go.Scatter(x=test_df_cv.index, y=test_df_cv['total_consumption'], mode='lines', name='Test Data', line=dict(color='ForestGreen'))
random_forest_predictions = go.Scatter(x=df_final.index, y=df_final['RandomForest_Prediction_cv'], mode='markers', name='Random Forest CV Predictions', marker=dict(color='Red'))
xgboost_predictions = go.Scatter(x=df_final.index, y=df_final['XGBoost_Prediction_cv'], mode='markers', name='XGBoost Prediction CV Predictions', marker=dict(color='Orange'))

# Adding a vertical line for the train-test split date
vline = dict(
    type="line", x0=threshold, y0=0, x1=threshold, y1=1, line=dict(color="Black", width=2, dash="dash"), xref='x', yref='paper'
)

# Update layout for a better visual presentation
layout = go.Layout(
    title="Real Data and Predictions Comparison for '30-Day' Future Prediction",
    xaxis=dict(title='Index/Date'),
    yaxis=dict(title='Total Consumption/Predictions'),
    legend_title='Legend',
    shapes=[vline]  # Adding the vertical line to the layout
)

fig = go.Figure(data=[train_data, test_data, random_forest_predictions, xgboost_predictions], layout=layout)

# Show the interactive plot
fig.show()

# Calculate the RMSE for the Random Forest Model on the Test Data
y_pred_rfr_cv = rfr_cv.predict(X_test_cv)
rmse_rfr_cv = np.sqrt(mean_squared_error(y_test_cv, y_pred_rfr_cv))
print(f"Random Forest RMSE: {rmse_rfr_cv}")

# Calculate the RMSE for the XGBoost Model on the Test Data
y_pred_xgb_cv = xgb_cv.predict(X_test_cv)
rmse_xgb_cv = np.sqrt(mean_squared_error(y_test_cv, y_pred_xgb_cv))
print(f"XGBoost RMSE: {rmse_xgb_cv}")

# Save the interactive plot as HTML file
pio.write_html(fig, file='../results/Real Data and Predictions Comparison for 30-Day Future Prediction.html')


# [🔍 Open Figure](https://htmlpreview.github.io/?https://github.com/labrijisaad/Energy-Forecast-API/blob/main/results/Real%20Data%20and%20Predictions%20Comparison%20for%2030-Day%20Future%20Prediction.html)

# ## Connect with me 🌐
# <div align="center">
#   <a href="https://www.linkedin.com/in/labrijisaad/">
#     <img src="https://img.shields.io/badge/LinkedIn-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" style="margin-bottom: 5px;"/>
#   </a>
#   <a href="https://github.com/labrijisaad">
#     <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" style="margin-bottom: 5px;"/>
#   </a>
# </div>
