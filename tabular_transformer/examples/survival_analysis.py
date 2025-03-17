#!/usr/bin/env python
# TTML Survival Analysis Examples - Simplified

import sys
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from sklearn.model_selection import train_test_split

# Import TTML modules
from tabular_transformer.models import TabularTransformer
from tabular_transformer.models.task_heads import SurvivalHead, CompetingRisksHead
from tabular_transformer.utils.config import TransformerConfig
from tabular_transformer.data.dataset import TabularDataset

# Import data utilities
from tabular_transformer.examples.data_utils import download_support_dataset

# Part 1: Basic Survival Analysis
# Download SUPPORT dataset
support_df = download_support_dataset(save_csv=False)
print("SUPPORT dataset shape:", support_df.shape)
print("\nFeature types:")
print(support_df.dtypes)
print("\nEvent distribution:")
print(support_df['death'].value_counts(normalize=True))

# Identify numeric and categorical columns
numeric_features = support_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = support_df.select_dtypes(include=['object']).columns.tolist()

# Remove time and event columns from features
time_column = 'time'
event_column = 'death'

if time_column in numeric_features:
    numeric_features.remove(time_column)
if event_column in numeric_features:
    numeric_features.remove(event_column)
if time_column in categorical_features:
    categorical_features.remove(time_column)
if event_column in categorical_features:
    categorical_features.remove(event_column)

print("\nFeatures to be used:")
print("Numeric:", numeric_features)
print("Categorical:", categorical_features)

# Train/test split
train_df, test_df = train_test_split(support_df, test_size=0.2, random_state=42)

print("\nTrain set shape:", train_df.shape)
print("Test set shape:", test_df.shape)

# Prepare dummy survival curves (just for demonstration)
# In a real implementation, you would train a model to predict these
print("\nSimulating model predictions for survival analysis...")

# Example: Create synthetic risk scores based on some features
risk_scores = 0.5 * np.ones(len(test_df))
# Add some influence from features
risk_scores += 0.1 * test_df['age'].values / test_df['age'].max()
risk_scores += 0.2 * test_df['num_comorbidities'].values / test_df['num_comorbidities'].max()
risk_scores -= 0.1 * (test_df['sex'] == 1).values  # Lower risk for one gender

# Convert to risk groups
risk_groups = pd.qcut(risk_scores, q=3, labels=['Low', 'Medium', 'High'])

# Plot Kaplan-Meier survival curves by risk group
print("\nGenerating Kaplan-Meier curves by risk group...")
plt.figure(figsize=(10, 6))
kmf = KaplanMeierFitter()

for group in ['Low', 'Medium', 'High']:
    mask = risk_groups == group
    if mask.sum() > 0:  # Check that we have samples in this group
        kmf.fit(
            test_df.loc[mask, time_column],
            test_df.loc[mask, event_column],
            label=f'{group} Risk'
        )
        kmf.plot()

plt.title('Survival Curves by Risk Group')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.grid(True)
plt.show()

# Part 2: Competing Risks Analysis (Simplified)
# Create a synthetic cause column
print("\nCreating synthetic cause column for competing risks demonstration...")
support_df_cr = support_df.copy()
cause_column = 'cause'
support_df_cr[cause_column] = 0  # Default to censored

# For death=1, assign random cause 1, 2, or 3
np.random.seed(42)
event_indices = support_df_cr[event_column] == 1
support_df_cr.loc[event_indices, cause_column] = np.random.randint(1, 4, size=event_indices.sum())

# Show cause distribution
print("\nCause distribution:")
print(support_df_cr[cause_column].value_counts())

# Split into train/test sets
train_cr, test_cr = train_test_split(support_df_cr, test_size=0.2, random_state=42)

# Generate dummy cumulative incidence curves for competing risks
print("\nGenerating dummy cumulative incidence functions...")

# Create time points for plotting
time_points = np.linspace(0, support_df_cr[time_column].max(), 100)

# Create synthetic CIF curves for three causes
plt.figure(figsize=(12, 6))

# Cause 1: High early risk that plateaus
cif1 = 0.3 * (1 - np.exp(-0.05 * time_points))
# Cause 2: Linear risk increase
cif2 = 0.2 * time_points / time_points.max()
# Cause 3: Late increasing risk
cif3 = 0.15 * (1 - np.exp(-0.02 * (time_points - 20)))
cif3[time_points < 20] = 0

plt.plot(time_points, cif1, label='Cause 1')
plt.plot(time_points, cif2, label='Cause 2')
plt.plot(time_points, cif3, label='Cause 3')

plt.title('Simulated Cumulative Incidence Functions')
plt.xlabel('Time')
plt.ylabel('Cumulative Incidence')
plt.legend()
plt.grid(True)
plt.show()

# Individual Patient Analysis (Demonstration)
print("\nIndividual Patient Analysis:")
sample_indices = np.random.choice(len(test_cr), 3, replace=False)

for i, idx in enumerate(sample_indices):
    patient = test_cr.iloc[idx]
    
    # Show patient details
    print(f"\nPatient {i+1}:")
    print(f"Observed time: {patient[time_column]:.1f}")
    print(f"Event occurred: {bool(patient[event_column])}")
    if patient[event_column]:
        print(f"Cause: {patient[cause_column]}")
    
    # Show key features for this patient
    print("\nKey patient characteristics:")
    for feature in numeric_features[:5]:  # Show first 5 features
        print(f"{feature}: {patient[feature]:.2f}")
    
    # Simulate feature importance (random values)
    print("\nExample feature importance values:")
    sample_features = np.random.choice(numeric_features, 3, replace=False)
    for feature in sample_features:
        importance = np.random.uniform(-1, 1)
        direction = "increases" if importance > 0 else "decreases"
        print(f"{feature}: {abs(importance):.4f} ({direction} risk)")

print("\nNote: This is a simplified demonstration. In a real implementation, you would:")
print("1. Train actual survival and competing risks models")
print("2. Use model predictions for analysis")
print("3. Calculate actual feature importance using methods like SHAP or permutation importance")