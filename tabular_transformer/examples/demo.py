"""
Demo script for tabular transformer.

This script demonstrates the usage of the tabular transformer package for
multi-task learning on tabular data with a variational transformer model.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

from tabular_transformer.data import TabularDataset
from tabular_transformer.models import TabularTransformer
from tabular_transformer.models.task_heads import RegressionHead, ClassificationHead
from tabular_transformer.training import Trainer
from tabular_transformer.utils.config import ModelConfig, TransformerConfig, TaskConfig
from tabular_transformer.inference import Predictor, UncertaintySimulator


def create_demo_dataset():
    """
    Create a demo dataset with both regression and classification tasks.
    
    Returns:
        Tuple of (df, numeric_cols, categorical_cols, task_cols)
    """
    print("Creating demo dataset...")
    
    # Load California housing dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # Create a dataframe
    columns = housing.feature_names
    df = pd.DataFrame(X, columns=columns)
    df['price'] = y
    
    # Add some categorical features
    # Binned median income
    kb = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    df['income_category'] = kb.fit_transform(df[['MedInc']])
    
    # Ocean proximity based on longitude
    df['ocean_proximity'] = pd.cut(
        df['Longitude'],
        bins=5,
        labels=['INLAND', 'NEAR BAY', 'NEAR OCEAN', '<1H OCEAN', 'ISLAND']
    )
    
    # Binned house values for classification task
    df['price_category'] = pd.qcut(df['price'], q=4, labels=False)
    
    # Identify column types
    numeric_cols = [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
        'Population', 'AveOccup', 'Latitude', 'Longitude'
    ]
    categorical_cols = ['income_category', 'ocean_proximity']
    task_cols = {
        'regression': ['price'],
        'classification': ['price_category']
    }
    
    return df, numeric_cols, categorical_cols, task_cols


def create_model_config(numeric_cols, categorical_cols, task_cols):
    """
    Create a model configuration.
    
    Args:
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        task_cols: Dict mapping task names to target column names
        
    Returns:
        ModelConfig instance
    """
    # Create transformer config
    transformer_config = TransformerConfig(
        input_dim=8,  # Will be overridden based on preprocessor
        embed_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        variational=True,
        beta=0.1,
    )
    
    # Create task configs
    task_configs = {
        'regression': TaskConfig(
            name='regression',
            type='regression',
            output_dim=1,
            target_columns=task_cols['regression'],
            hidden_dims=[64, 32],
            dropout=0.1,
            weight=1.0,
        ),
        'classification': TaskConfig(
            name='classification',
            type='classification',
            output_dim=4,  # 4 price categories
            target_columns=task_cols['classification'],
            hidden_dims=[64, 32],
            dropout=0.1,
            weight=1.0,
        ),
    }
    
    # Create model config
    model_config = ModelConfig(
        transformer=transformer_config,
        tasks=task_configs,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        batch_size=64,
        learning_rate=1e-3,
        weight_decay=1e-5,
        max_epochs=20,
        early_stopping_patience=5,
    )
    
    return model_config


def main(args):
    """
    Main function to run the demo.
    
    Args:
        args: Command-line arguments
    """
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create demo dataset
    df, numeric_cols, categorical_cols, task_cols = create_demo_dataset()
    
    # Split into train, validation, and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=args.seed)
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=args.seed)
    
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    # Create model configuration
    config = create_model_config(numeric_cols, categorical_cols, task_cols)
    
    # Create task target dictionaries
    train_targets = {
        task_name: task_config.target_columns
        for task_name, task_config in config.tasks.items()
    }
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = TabularDataset.from_dataframe(
        dataframe=pd.concat([train_df, val_df, test_df]),
        numeric_columns=config.numeric_columns,
        categorical_columns=config.categorical_columns,
        target_columns=train_targets,
        validation_split=len(val_df) / (len(train_df) + len(val_df)),
        test_split=len(test_df) / (len(train_df) + len(val_df) + len(test_df)),
        random_state=args.seed,
    )
    
    # Get preprocessor from training dataset
    preprocessor = train_dataset.preprocessor
    
    # Create data loaders
    train_loader = train_dataset.create_dataloader(
        batch_size=config.batch_size, shuffle=True
    )
    val_loader = val_dataset.create_dataloader(
        batch_size=config.batch_size, shuffle=False
    )
    test_loader = test_dataset.create_dataloader(
        batch_size=config.batch_size, shuffle=False
    )
    
    # Set input dimension based on preprocessor
    config.transformer.input_dim = preprocessor.get_total_embedding_dim()
    
    # Create model components
    encoder = TabularTransformer(
        numeric_dim=len(config.numeric_columns),
        categorical_dims=preprocessor.categorical_nunique,
        categorical_embedding_dims=preprocessor.categorical_embedding_dims,
        config=config.transformer,
    )
    
    # Create task heads
    task_heads = {}
    for task_name, task_config in config.tasks.items():
        if task_config.type == 'regression':
            task_heads[task_name] = RegressionHead(
                name=task_name,
                input_dim=config.transformer.embed_dim,
                output_dim=task_config.output_dim,
                hidden_dims=task_config.hidden_dims,
                dropout=task_config.dropout,
                uncertainty=True,  # Enable uncertainty estimation
            )
        elif task_config.type == 'classification':
            task_heads[task_name] = ClassificationHead(
                name=task_name,
                input_dim=config.transformer.embed_dim,
                num_classes=task_config.output_dim,
                hidden_dims=task_config.hidden_dims,
                dropout=task_config.dropout,
            )
    
    # Create trainer
    trainer = Trainer(
        encoder=encoder,
        task_head=task_heads,
        config=config,
    )
    
    # Train model
    if not args.skip_training:
        print("Training model...")
        training_results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.max_epochs,
            early_stopping_patience=config.early_stopping_patience,
            model_dir=args.output_dir,
        )
        
        # Print training results
        print("Training completed.")
        print("Best validation metrics:")
        for metric, value in training_results["best_metrics"].items():
            if "val" in metric:
                print(f"  {metric}: {value:.4f}")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader, prefix="test")
    print("Test metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Create predictor for inference
    predictor = Predictor(
        encoder=encoder,
        task_head=task_heads,
        preprocessor=preprocessor,
    )
    
    # Run predictions on test set
    print("Generating predictions on test set...")
    predictions = predictor.predict(test_loader)
    
    # Print prediction shape for each task
    for task_name, task_preds in predictions.items():
        if task_name != "latent_representations":
            print(f"Task '{task_name}' predictions:")
            for key, value in task_preds.items():
                print(f"  {key}: shape {value.shape}")
    
    # Demonstrate uncertainty quantification
    if args.show_uncertainty:
        print("Running uncertainty simulation...")
        simulator = UncertaintySimulator(predictor, num_samples=20)
        
        # Take a subset of the test set for demonstration
        small_df = test_df.iloc[:5].reset_index(drop=True)
        
        # Generate predictions with uncertainty
        # Regression task - use get_prediction_intervals for a simplified interface
        mean, lower, upper = simulator.get_prediction_intervals(
            data=small_df, 
            task_name="regression",
            prediction_key="mean",  # Key in the regression head output
            interval_width=0.95,    # 95% prediction interval
        )
        
        # Print results
        print("\nRegression predictions with 95% confidence intervals:")
        print("  Row | Mean  | Lower | Upper | Actual")
        print("  ---------------------------------")
        for i in range(len(small_df)):
            print(f"  {i}   | {mean[i, 0]:.2f} | {lower[i, 0]:.2f} | {upper[i, 0]:.2f} | {small_df['price'].iloc[i]:.2f}")
        
        # Classification task - use full simulation API for more complex uncertainty analysis
        stats = simulator.simulate_and_calculate_statistics(
            data=small_df,
            task_names=["classification"],
        )
        
        # Get class probabilities and their standard deviations
        probs = stats["classification"]["probabilities"]["mean"]
        stds = stats["classification"]["probabilities"]["std"]
        
        # Print results
        print("\nClassification probability predictions with uncertainty:")
        print("  Row | Predicted Class | Probability | Std Dev | Actual Class")
        print("  ---------------------------------------------------")
        for i in range(len(small_df)):
            pred_class = np.argmax(probs[i])
            prob = probs[i, pred_class]
            std = stds[i, pred_class]
            actual = int(small_df['price_category'].iloc[i])
            print(f"  {i}   | {pred_class}             | {prob:.2f}       | {std:.2f}    | {actual}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Tabular Transformer Demo")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="./model_checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training and just demonstrate prediction")
    parser.add_argument("--show-uncertainty", action="store_true",
                        help="Demonstrate uncertainty quantification")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not args.skip_training:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the demo
    main(args)
