import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go

# Load data
weather_and_consumption_df = pd.read_csv(
    r"C:\Users\ashis\OneDrive\Desktop\Monthly-Daily-Energy-Forecasting-Docker-API\data\processed\weather_and_consumption.csv",
    index_col=0, parse_dates=True
)

# Data preparation
df = weather_and_consumption_df.copy()
threshold = '2010-05-17'
train_df = df.loc[df.index < threshold].copy()
test_df = df.loc[df.index >= threshold].copy()
TARGET = 'total_consumption'

# Train/Test Data Split Graph (Plotly)
trace1 = go.Scatter(x=train_df.index, y=train_df[TARGET], mode='lines', name='Training Set')
trace2 = go.Scatter(x=test_df.index, y=test_df[TARGET], mode='lines', name='Test Set')
vline = go.layout.Shape(
    type="line", x0=threshold, y0=0, x1=threshold, y1=max(df[TARGET]),
    line=dict(color="Black", width=2, dash="dash")
)
layout = go.Layout(
    title='Data Train/Test Split',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Total Consumption'),
    shapes=[vline]
)
fig = go.Figure(data=[trace1, trace2], layout=layout)
fig.show()

# Real vs. Predicted Consumption (Plotly)
# For demonstration, we'll create mock predictions. Replace with actual model predictions if available.
test_df['RandomForest_Prediction'] = test_df[TARGET] * 0.9  # Mock predictions
test_df['XGBoost_Prediction'] = test_df[TARGET] * 1.1       # Mock predictions
df_final = df.merge(
    test_df[['RandomForest_Prediction', 'XGBoost_Prediction']],
    how='left', left_index=True, right_index=True
)

train_data = go.Scatter(x=train_df.index, y=train_df[TARGET], mode='lines', name='Train Data', line=dict(color='Blue'))
test_data = go.Scatter(x=test_df.index, y=test_df[TARGET], mode='lines', name='Test Data', line=dict(color='ForestGreen'))
random_forest_predictions = go.Scatter(x=df_final.index, y=df_final['RandomForest_Prediction'], mode='markers', name='Random Forest Predictions', marker=dict(color='Red'))
xgboost_predictions = go.Scatter(x=df_final.index, y=df_final['XGBoost_Prediction'], mode='markers', name='XGBoost Predictions', marker=dict(color='Orange'))

layout = go.Layout(
    title="Real Data and Predictions Comparison",
    xaxis=dict(title='Index/Date'),
    yaxis=dict(title='Total Consumption/Predictions'),
    legend_title='Legend',
    shapes=[vline]
)
fig = go.Figure(data=[train_data, test_data, random_forest_predictions, xgboost_predictions], layout=layout)
fig.show()

# 7-Fold Time Series Split Visualization (Matplotlib)
from sklearn.model_selection import TimeSeriesSplit

tss = TimeSeriesSplit(n_splits=7, test_size=30)
fig, axs = plt.subplots(7, 1, figsize=(20, 15), sharex=True)
color_palette = plt.get_cmap('Set1')

for fold, (train_idx, val_idx) in enumerate(tss.split(train_df)):
    train_cv = df.iloc[train_idx]
    test_cv = df.iloc[val_idx]
    axs[fold].plot(train_cv.index, train_cv[TARGET], label='Training Set', linewidth=2, color=color_palette(0))
    axs[fold].plot(test_cv.index, test_cv[TARGET], label='Validation Set', color=color_palette(1), linewidth=2)
    axs[fold].axvline(test_cv.index.min(), color='gray', ls='--', lw=2)
    axs[fold].set_title(f'Fold {fold+1}', fontsize=14, fontweight='bold')
    axs[fold].xaxis.set_major_locator(mdates.AutoDateLocator())
    axs[fold].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(axs[fold].xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=12)
    axs[fold].tick_params(axis='y', labelsize=12)
    axs[fold].grid(True, which='major', linestyle='--', linewidth='0.5', color='gray')
    axs[fold].legend(fontsize=12, loc='upper left')

fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])
fig.subplots_adjust(hspace=0.4)
fig.text(0.5, 0.02, 'Date', ha='center', va='center', fontsize=16, fontweight='bold')
fig.text(0.01, 0.5, 'Total Consumption', ha='center', va='center', rotation='vertical', fontsize=16, fontweight='bold')
fig.suptitle("7-Fold Time Series Split on Train Data", fontsize=24, fontweight='bold', y=0.98)
plt.show()
