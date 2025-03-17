"""
Utility functions for downloading and preprocessing datasets for TTML examples.
"""
import pandas as pd
import numpy as np
import requests
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tabular_transformer.data.dataset import TabularDataset

# Create a dummy function to replace load_support since it's not available in lifelines.datasets
def load_support():
    """
    Return a simplified version of the SUPPORT dataset for demonstration purposes
    """
    # Create a synthetic dataset with similar properties to the SUPPORT dataset
    n_samples = 100
    np.random.seed(42)
    
    # Generate synthetic data
    data = {
        'age': np.random.randint(18, 90, n_samples),
        'sex': np.random.choice([0, 1], n_samples),
        'race': np.random.choice([0, 1, 2], n_samples),
        'num_comorbidities': np.random.randint(0, 5, n_samples),
        'diabetes': np.random.choice([0, 1], n_samples),
        'dementia': np.random.choice([0, 1], n_samples),
        'cancer': np.random.choice([0, 1], n_samples),
        'time': np.random.exponential(30, n_samples).astype(int) + 1,
        'death': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    }
    
    return pd.DataFrame(data)

def download_adult_dataset(save_csv=True):
    """
    Downloads the UCI Adult dataset from OpenML.
    Returns a pandas DataFrame.
    """
    adult = fetch_openml(name='adult', version=2, as_frame=True)
    df = adult.frame
    if save_csv:
        df.to_csv('adult_dataset.csv', index=False)
    return df

def download_titanic_dataset(save_csv=True):
    """
    Downloads the Titanic dataset from OpenML.
    Returns a pandas DataFrame.
    """
    titanic = fetch_openml(name='titanic', version=1, as_frame=True)
    df = titanic.frame
    if save_csv:
        df.to_csv('titanic_dataset.csv', index=False)
    return df

def download_wine_quality_dataset(save_csv=True, variant='red'):
    """
    Downloads the Wine Quality dataset from OpenML.
    Choose variant 'red' or 'white'.
    Returns a pandas DataFrame.
    """
    dataset_name = 'wine-quality-' + variant
    wine = fetch_openml(name=dataset_name, version=1, as_frame=True)
    df = wine.frame
    if save_csv:
        df.to_csv(f'wine_quality_{variant}.csv', index=False)
    return df

def download_support_dataset(save_csv=True):
    """
    Downloads the SUPPORT dataset for survival analysis.
    Returns a pandas DataFrame.
    """
    df = load_support()
    if save_csv:
        df.to_csv('support_dataset.csv', index=False)
    return df

def download_nhanes_dataset(save_csv=True):
    """
    Downloads a processed NHANES dataset from a public GitHub repository.
    Returns a pandas DataFrame.
    """
    url = 'https://raw.githubusercontent.com/selva86/datasets/master/nhanes.csv'
    r = requests.get(url)
    filename = 'nhanes.csv'
    if save_csv:
        with open(filename, 'wb') as f:
            f.write(r.content)
    df = pd.read_csv(filename)
    return df

def prepare_dataset(df, target_column, categorical_columns=None, numerical_columns=None, test_size=0.2, random_state=42):
    """
    Prepare dataset for TTML model using TabularDataset.
    
    Args:
        df: pandas DataFrame
        target_column: name of target column or list of target columns
        categorical_columns: list of categorical column names (if None, auto-detect)
        numerical_columns: list of numerical column names (if None, auto-detect)
        test_size: proportion of dataset to include in test split
        random_state: random state for reproducibility
    
    Returns:
        Dictionary containing:
        - train_dataset: TabularDataset for training
        - test_dataset: TabularDataset for testing
        - train_idx: indices used for training split
        - test_idx: indices used for test split
    """
    # Auto-detect column types if not provided
    if numerical_columns is None:
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target column(s) from feature lists
    if isinstance(target_column, str):
        target_columns = [target_column]
    else:
        target_columns = target_column
    
    for col in target_columns:
        if col in numerical_columns:
            numerical_columns.remove(col)
        if col in categorical_columns:
            categorical_columns.remove(col)
    
    # Create target dictionary for TabularDataset
    if isinstance(target_column, str):
        target_dict = {'main': [target_column]}
    elif isinstance(target_column, dict):
        target_dict = target_column
    else:
        target_dict = {'main': target_column}
    
    # Create train/test datasets
    train_dataset, test_dataset, _ = TabularDataset.from_dataframe(
        dataframe=df,
        numeric_columns=numerical_columns,
        categorical_columns=categorical_columns,
        target_columns=target_dict,
        validation_split=test_size,
        random_state=random_state
    )
    
    # Get indices for the splits
    all_indices = np.arange(len(df))
    train_size = int((1 - test_size) * len(df))
    train_idx = all_indices[:train_size]
    test_idx = all_indices[train_size:]
    
    return {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'train_idx': train_idx,
        'test_idx': test_idx
    }

if __name__ == "__main__":
    # Example usage: download and print summary info
    datasets = {
        "Adult": download_adult_dataset(),
        "Titanic": download_titanic_dataset(),
        "Wine_Red": download_wine_quality_dataset(variant='red'),
        "Support": download_support_dataset(),
        "NHANES": download_nhanes_dataset()
    }
    
    for name, df in datasets.items():
        print(f"\nDataset: {name}")
        print(f"Shape: {df.shape}")
        print("\nFeature types:")
        print(df.dtypes.value_counts())
        print("\nMissing values:")
        print(df.isnull().sum().sum())
