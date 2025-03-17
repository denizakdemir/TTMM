"""
Setup script for the tabular_transformer package.
"""

from setuptools import setup, find_packages

setup(
    name="tabular_transformer",
    version="0.1.0",
    author="TTML Team",
    author_email="example@example.com",
    description="A tabular transformer model for multi-task learning on tabular data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tabular_transformer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "torch>=1.7.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.50.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=20.8b1",
            "isort>=5.7.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
        ],
        "examples": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "jupyter>=1.0.0",
        ],
        "explainability": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "shap>=0.40.0",
            "lime>=0.2.0",
            "scikit-learn>=0.24.0",
        ],
        "dashboard": [
            "dash>=2.0.0",
            "streamlit>=1.0.0",
            "plotly>=5.0.0",
        ],
    },
)
