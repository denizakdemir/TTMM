# Tabular Transformer (TTML)

A PyTorch-based package for applying transformer architectures to tabular data. This package provides a comprehensive implementation of a transformer-based model for tabular data with multiple task heads, variational inference for uncertainty quantification, and support for handling missing values.

> ⚠️ **Important API Note**: This package uses the parameter name `task_head` (singular) consistently in its API, even when passing multiple task heads as a dictionary. Previous versions or documentation may reference `task_heads` (plural), which is no longer supported.

## Features

- **Handles Mixed Data Types**: Processes both numeric and categorical features in tabular data
- **Missing Value Handling**: Learns structural missingness patterns and handles random missing values
- **Multi-Task Learning**: Common transformer-based encoder with task-specific heads
- **Supported Task Types**:
  - **Supervised**: Classification, regression, survival (right-censored), competing risks, and count outcomes
  - **Unsupervised**: Clustering
- **Uncertainty Quantification**: Variational inference for probabilistic predictions
- **Preprocessing Utilities**: Feature scaling, categorical encoding, and more
- **Training & Evaluation**: Complete training loop with multi-task loss aggregation and model evaluation

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Pandas
- scikit-learn
- tqdm

### Install from source

```bash
git clone https://github.com/yourusername/tabular_transformer.git
cd tabular_transformer
pip install -e .
```

For development or examples:

```bash
pip install -e ".[dev,examples]"
```

## Quick Start

```python
import pandas as pd
import torch
from tabular_transformer.data import TabularDataset
from tabular_transformer.models import TabularTransformer
from tabular_transformer.models.task_heads import RegressionHead, ClassificationHead
from tabular_transformer.training import Trainer
from tabular_transformer.inference import Predictor

# 1. Prepare your data
df = pd.read_csv("your_data.csv")
numeric_cols = ["feature1", "feature2", "feature3"]
categorical_cols = ["category1", "category2"]

# 2. Create datasets
train_dataset, val_dataset, _ = TabularDataset.from_dataframe(
    dataframe=df,
    numeric_columns=numeric_cols,
    categorical_columns=categorical_cols,
    target_columns={"regression": ["target"]},
    validation_split=0.2,
)

# 3. Create model components
preprocessor = train_dataset.preprocessor
encoder = TabularTransformer(
    numeric_dim=len(numeric_cols),
    categorical_dims=preprocessor.categorical_nunique,
    categorical_embedding_dims=preprocessor.categorical_embedding_dims,
    config=transformer_config,  # See examples/demo.py for config setup
)

task_heads = {
    "regression": RegressionHead(
        name="regression",
        input_dim=encoder.config.embed_dim,
        output_dim=1,
    )
}

# 4. Create trainer and train
trainer = Trainer(encoder=encoder, task_head=task_heads)
trainer.train(
    train_loader=train_dataset.create_dataloader(batch_size=64),
    val_loader=val_dataset.create_dataloader(batch_size=64),
    num_epochs=10,
)

# 5. Create predictor and generate predictions
predictor = Predictor(encoder=encoder, task_head=task_heads, preprocessor=preprocessor)
predictions = predictor.predict_dataframe(df)
```

See the [demo script](tabular_transformer/examples/demo.py) for a more complete example.

## Architecture

The Tabular Transformer package consists of several key components:

1. **Data Processing**: Handles feature preprocessing, dataset creation, and batching
2. **Transformer Encoder**: Implements a transformer architecture for tabular data
3. **Task Heads**: Specialized heads for different prediction tasks
4. **Training**: Training loop with multi-task learning support
5. **Inference**: Prediction utilities and uncertainty quantification
6. **Explainability**: Tools for model interpretability and explanation

### Transformer Encoder

The core of the model is a transformer encoder that processes tabular data:
- Embeds categorical features using learned embeddings
- Processes numeric features through projection layers
- Applies self-attention mechanisms to capture feature interactions
- Optionally implements variational inference for uncertainty estimation

### Task Heads

Task-specific heads connect to the encoder output:
- **Classification**: For binary and multi-class problems
- **Regression**: For continuous outcomes with optional uncertainty
- **Survival**: For time-to-event data with right censoring
- **Competing Risks**: For time-to-event data with multiple competing events
- **Count**: For count outcomes (Poisson, Negative Binomial, Binomial)
- **Clustering**: For unsupervised learning

## Explainability

The package includes comprehensive model explainability components:

- **Global Explanations**: Understand overall model behavior
  - Feature importance calculation using permutation methods
  - Model-agnostic interpretation techniques
  
- **Local Explanations**: Understand individual predictions
  - LIME (Local Interpretable Model-agnostic Explanations)
  - Counterfactual explanations to explore "what-if" scenarios
  
- **Visualization Tools**: Visual representation of model behavior
  - Partial Dependence Plots (PDP) to show feature-target relationships
  - Individual Conditional Expectation (ICE) plots for per-instance analysis
  - Calibration plots for evaluating prediction accuracy
  
- **Sensitivity Analysis**: Measure prediction stability
  - Quantifies how predictions change with feature variations
  - Tornado plots for visualizing sensitivity rankings

- **Report Generation**: Create comprehensive explainability reports
  - HTML dashboards with interactive visualizations
  - Combined explainability metrics in a single interface

See the [explainability demo](tabular_transformer/examples/explainability_demo.py) for a complete example.

## Missing Value Handling

The model handles missing values in two ways:
1. **Random Missingness**: Uses masks to ignore missing values during training
2. **Structural Missingness**: Learns patterns of missingness through special embeddings

## Uncertainty Quantification

For uncertainty estimation, the package provides:
1. **Variational Inference**: Using a latent Gaussian layer
2. **Monte Carlo Simulation**: To generate prediction intervals and uncertainty estimates

## Development

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
