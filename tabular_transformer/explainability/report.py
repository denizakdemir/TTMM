"""
Reporting and dashboard utilities for tabular transformer explainability.

This module provides classes and functions for generating comprehensive
reports and interactive dashboards for model explainability.
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import base64
from io import BytesIO

from tabular_transformer.utils.logger import LoggerMixin
from tabular_transformer.inference.predict import Predictor
from tabular_transformer.data.dataset import TabularDataset
from tabular_transformer.explainability.global_explanations import (
    GlobalExplainer,
    PermutationImportance,
    SHAPExplainer,
    AttentionExplainer,
)
from tabular_transformer.explainability.local_explanations import (
    LocalExplainer,
    LIMEExplainer,
    CounterfactualExplainer,
)
from tabular_transformer.explainability.visualizations import (
    ExplainabilityViz,
    PDPlot,
    ICEPlot,
    CalibrationPlot,
)
from tabular_transformer.explainability.sensitivity import SensitivityAnalyzer


class Report(LoggerMixin):
    """
    Report generator for tabular transformer explainability.
    
    This class generates comprehensive reports combining multiple
    explainability methods for a complete model overview.
    """
    
    def __init__(
        self,
        predictor: Predictor,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize report generator.
        
        Args:
            predictor: Predictor instance with trained model
            feature_names: Optional list of feature names (will use preprocessor column names if not provided)
        """
        self.predictor = predictor
        self.encoder = predictor.encoder
        self.task_heads = predictor.task_heads
        self.preprocessor = predictor.preprocessor
        self.device = predictor.device
        
        # Set feature names
        if feature_names is None:
            numeric_cols = self.preprocessor.numeric_columns
            categorical_cols = self.preprocessor.categorical_columns
            self.feature_names = numeric_cols + categorical_cols
        else:
            self.feature_names = feature_names
        
        # Initialize explainers
        self.global_explainer = GlobalExplainer(predictor, self.feature_names)
        self.perm_importance = PermutationImportance(predictor, self.feature_names)
        self.shap_explainer = SHAPExplainer(predictor, self.feature_names)
        self.attention_explainer = AttentionExplainer(predictor, self.feature_names)
        
        self.lime_explainer = LIMEExplainer(predictor, self.feature_names)
        self.counterfactual_explainer = CounterfactualExplainer(predictor, self.feature_names)
        
        self.pd_plot = PDPlot(predictor, self.feature_names)
        self.ice_plot = ICEPlot(predictor, self.feature_names)
        self.calibration_plot = CalibrationPlot(predictor, self.feature_names)
        
        self.sensitivity_analyzer = SensitivityAnalyzer(predictor, self.feature_names)
    
    def generate_global_explanations(
        self,
        data: Union[pd.DataFrame, TabularDataset],
        target_columns: Dict[str, str],
        task_names: Optional[List[str]] = None,
        top_features: int = 10,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        """
        Generate global explanations for the model.
        
        Args:
            data: Input data
            target_columns: Dict mapping task names to target column names
            task_names: Optional list of task names to explain (all by default)
            top_features: Number of top features to include in explanations
            batch_size: Batch size for prediction
            
        Returns:
            Dict containing global explanations and visualizations
        """
        self.logger.info("Generating global explanations...")
        
        # Determine which tasks to explain
        if task_names is None:
            task_names = list(self.task_heads.keys())
        
        # Generate permutation importance
        permutation_importance = self.perm_importance.compute_importance(
            data=data,
            target_columns=target_columns,
            task_names=task_names,
            batch_size=batch_size,
        )
        
        # Create feature importance plots
        importance_fig = self.global_explainer.plot_feature_importance(
            importance_scores=permutation_importance,
            top_n=top_features,
        )
        
        # Store results
        return {
            "permutation_importance": permutation_importance,
            "importance_fig": importance_fig,
        }
    
    def generate_local_explanations(
        self,
        instances: pd.DataFrame,
        task_name: str,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        """
        Generate local explanations for specific instances.
        
        Args:
            instances: DataFrame containing instances to explain
            task_name: Name of the task to explain
            batch_size: Batch size for prediction
            
        Returns:
            Dict containing local explanations and visualizations
        """
        self.logger.info("Generating local explanations...")
        
        # Limit to a manageable number of instances
        if len(instances) > 5:
            self.logger.info(f"Limiting local explanations to first 5 instances (from {len(instances)})")
            instances = instances.iloc[:5].copy()
        
        # Generate LIME explanations
        lime_explanations = []
        lime_figures = []
        
        for i in range(len(instances)):
            instance = instances.iloc[i]
            
            # Generate LIME explanation
            lime_explanation = self.lime_explainer.explain_instance(
                instance=instance,
                task_name=task_name,
            )
            
            # Create visualization
            lime_fig = self.lime_explainer.plot_explanation(lime_explanation)
            
            lime_explanations.append(lime_explanation)
            lime_figures.append(lime_fig)
        
        # Generate counterfactual explanations
        cf_explanations = []
        cf_figures = []
        
        for i in range(len(instances)):
            instance = instances.iloc[i]
            
            # Generate counterfactual explanation
            cf_explanation = self.counterfactual_explainer.explain_instance(
                instance=instance,
                task_name=task_name,
            )
            
            # Create visualization
            cf_fig = self.counterfactual_explainer.plot_explanation(cf_explanation)
            
            cf_explanations.append(cf_explanation)
            cf_figures.append(cf_fig)
        
        # Store results
        return {
            "instances": instances,
            "lime_explanations": lime_explanations,
            "lime_figures": lime_figures,
            "counterfactual_explanations": cf_explanations,
            "counterfactual_figures": cf_figures,
        }
    
    def generate_visualization_plots(
        self,
        data: Union[pd.DataFrame, TabularDataset],
        task_name: str,
        target_column: str,
        features: Optional[List[str]] = None,
        n_features: int = 5,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        """
        Generate visualization plots for model behavior.
        
        Args:
            data: Input data
            task_name: Name of the task to visualize
            target_column: Column name for the target variable
            features: Optional list of features to visualize (top n important features by default)
            n_features: Number of features to visualize if features not provided
            batch_size: Batch size for prediction
            
        Returns:
            Dict containing visualization plots
        """
        self.logger.info("Generating visualization plots...")
        
        # Determine which features to visualize
        if features is None:
            # Use permutation importance to select top features
            importance = self.perm_importance.compute_importance(
                data=data,
                target_columns={task_name: target_column},
                task_names=[task_name],
                batch_size=batch_size,
            )
            
            # Get top n numeric features
            feature_importance = importance[task_name]
            feature_df = pd.DataFrame({
                "Feature": self.feature_names,
                "Importance": feature_importance,
            })
            
            # Filter to numeric features
            numeric_features = [
                f for f in feature_df["Feature"] 
                if f in self.preprocessor.numeric_columns
            ]
            
            # Sort and select top n
            numeric_feature_df = feature_df[feature_df["Feature"].isin(numeric_features)]
            numeric_feature_df = numeric_feature_df.sort_values("Importance", ascending=False)
            features = numeric_feature_df["Feature"].head(n_features).tolist()
        
        # Generate PDP for each feature
        pdp_results = {}
        pdp_figures = {}
        
        for feature in features:
            # Skip if not a valid feature
            if feature not in self.feature_names:
                continue
                
            # Generate PDP
            pdp_result = self.pd_plot.compute_partial_dependence(
                data=data,
                feature=feature,
                task_name=task_name,
                batch_size=batch_size,
            )
            
            # Create visualization
            pdp_fig = self.pd_plot.plot_partial_dependence(pdp_result)
            
            pdp_results[feature] = pdp_result
            pdp_figures[feature] = pdp_fig
        
        # Generate ICE plots
        ice_results = {}
        ice_figures = {}
        
        for feature in features:
            # Skip if not a valid feature
            if feature not in self.feature_names:
                continue
                
            # Generate ICE
            ice_result = self.ice_plot.compute_ice_curves(
                data=data,
                feature=feature,
                task_name=task_name,
            )
            
            # Create visualization
            ice_fig = self.ice_plot.plot_ice_curves(ice_result)
            
            ice_results[feature] = ice_result
            ice_figures[feature] = ice_fig
        
        # Generate calibration plot
        calibration_result = self.calibration_plot.compute_calibration_curve(
            data=data,
            task_name=task_name,
            target_column=target_column,
            batch_size=batch_size,
        )
        
        calibration_fig = self.calibration_plot.plot_calibration_curve(calibration_result)
        
        # Store results
        return {
            "pdp_results": pdp_results,
            "pdp_figures": pdp_figures,
            "ice_results": ice_results,
            "ice_figures": ice_figures,
            "calibration_result": calibration_result,
            "calibration_fig": calibration_fig,
        }
    
    def generate_sensitivity_analysis(
        self,
        instance: pd.Series,
        task_name: str,
        features: Optional[List[str]] = None,
        n_features: int = 5,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        """
        Generate sensitivity analysis for a specific instance.
        
        Args:
            instance: Instance to analyze
            task_name: Name of the task to analyze
            features: Optional list of features to analyze (top n numeric features by default)
            n_features: Number of features to analyze if features not provided
            batch_size: Batch size for prediction
            
        Returns:
            Dict containing sensitivity analysis results
        """
        self.logger.info("Generating sensitivity analysis...")
        
        # Determine which features to analyze
        if features is None:
            # Use all numeric features but limit to top n
            features = self.preprocessor.numeric_columns[:n_features]
        
        # Generate sensitivity analysis
        sensitivity_result = self.sensitivity_analyzer.compute_sensitivity(
            instance=instance,
            task_name=task_name,
            features_to_analyze=features,
            batch_size=batch_size,
        )
        
        # Create visualization
        sensitivity_fig = self.sensitivity_analyzer.plot_sensitivity(sensitivity_result)
        tornado_fig = self.sensitivity_analyzer.plot_tornado(sensitivity_result)
        
        # Store results
        return {
            "sensitivity_result": sensitivity_result,
            "sensitivity_fig": sensitivity_fig,
            "tornado_fig": tornado_fig,
        }
    
    def generate_report(
        self,
        data: Union[pd.DataFrame, TabularDataset],
        target_columns: Dict[str, str],
        task_names: Optional[List[str]] = None,
        sample_instances: Optional[pd.DataFrame] = None,
        top_features: int = 10,
        output_path: Optional[str] = None,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive explainability report.
        
        Args:
            data: Input data
            target_columns: Dict mapping task names to target column names
            task_names: Optional list of task names to explain (all by default)
            sample_instances: Optional DataFrame with sample instances for local explanations
            top_features: Number of top features to include in explanations
            output_path: Optional path to save the report (HTML or JSON)
            batch_size: Batch size for prediction
            
        Returns:
            Dict containing all report components
        """
        # Determine which tasks to explain
        if task_names is None:
            task_names = list(self.task_heads.keys())
        
        # Generate global explanations
        global_exp = self.generate_global_explanations(
            data=data,
            target_columns=target_columns,
            task_names=task_names,
            top_features=top_features,
            batch_size=batch_size,
        )
        
        # Generate local explanations if sample instances provided
        local_exp = {}
        if sample_instances is not None:
            for task_name in task_names:
                local_exp[task_name] = self.generate_local_explanations(
                    instances=sample_instances,
                    task_name=task_name,
                    batch_size=batch_size,
                )
        
        # Generate visualization plots
        viz_plots = {}
        for task_name in task_names:
            viz_plots[task_name] = self.generate_visualization_plots(
                data=data,
                task_name=task_name,
                target_column=target_columns[task_name],
                n_features=min(5, top_features),
                batch_size=batch_size,
            )
        
        # Generate sensitivity analysis if sample instances provided
        sensitivity = {}
        if sample_instances is not None and len(sample_instances) > 0:
            for task_name in task_names:
                sensitivity[task_name] = self.generate_sensitivity_analysis(
                    instance=sample_instances.iloc[0],
                    task_name=task_name,
                    batch_size=batch_size,
                )
        
        # Compile report
        report = {
            "global_explanations": global_exp,
            "local_explanations": local_exp,
            "visualization_plots": viz_plots,
            "sensitivity_analysis": sensitivity,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model_info": {
                    "n_features": len(self.feature_names),
                    "feature_names": self.feature_names,
                    "numeric_features": self.preprocessor.numeric_columns,
                    "categorical_features": self.preprocessor.categorical_columns,
                    "tasks": list(self.task_heads.keys()),
                },
            },
        }
        
        # Save report if output path provided
        if output_path is not None:
            extension = os.path.splitext(output_path)[1].lower()
            
            if extension == ".html":
                # Save as HTML report
                self._save_html_report(report, output_path)
            elif extension == ".json":
                # Save as JSON
                self._save_json_report(report, output_path)
            else:
                self.logger.warning(f"Unsupported report format: {extension}")
        
        return report
    
    def _figure_to_base64(self, fig: plt.Figure) -> str:
        """
        Convert matplotlib figure to base64 string.
        
        Args:
            fig: Matplotlib figure
            
        Returns:
            Base64 encoded string
        """
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close(fig)
        return img_str
    
    def _save_html_report(self, report: Dict[str, Any], output_path: str) -> None:
        """
        Save report as HTML.
        
        Args:
            report: Report dict
            output_path: Path to save HTML report
        """
        self.logger.info(f"Saving HTML report to {output_path}")
        
        # Convert matplotlib figures to base64 strings
        html_content = []
        
        # Add header
        html_content.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Tabular Transformer Explainability Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                h1, h2, h3 { color: #333; }
                .section { margin-bottom: 30px; }
                .figure { margin: 20px 0; text-align: center; }
                .figure img { max-width: 100%; }
                .footer { margin-top: 50px; color: #777; font-size: 0.8em; }
            </style>
        </head>
        <body>
            <h1>Tabular Transformer Explainability Report</h1>
        """)
        
        # Add metadata
        metadata = report["metadata"]
        html_content.append(f"""
            <div class="section">
                <h2>Model Information</h2>
                <p>Generated: {metadata['timestamp']}</p>
                <p>Features: {len(metadata['model_info']['feature_names'])}</p>
                <p>Tasks: {', '.join(metadata['model_info']['tasks'])}</p>
            </div>
        """)
        
        # Add global explanations
        html_content.append("""
            <div class="section">
                <h2>Global Explanations</h2>
        """)
        
        # Add feature importance plot
        if "importance_fig" in report["global_explanations"]:
            fig = report["global_explanations"]["importance_fig"]
            img_str = self._figure_to_base64(fig)
            html_content.append(f"""
                <div class="figure">
                    <h3>Feature Importance</h3>
                    <img src="data:image/png;base64,{img_str}" alt="Feature Importance">
                </div>
            """)
        
        html_content.append("</div>")  # Close global explanations section
        
        # Add task-specific sections
        for task_name in report["visualization_plots"]:
            html_content.append(f"""
                <div class="section">
                    <h2>Task: {task_name}</h2>
            """)
            
            # Add PDP plots
            html_content.append("""
                <div class="subsection">
                    <h3>Partial Dependence Plots</h3>
            """)
            
            viz_plots = report["visualization_plots"][task_name]
            
            for feature, fig in viz_plots["pdp_figures"].items():
                img_str = self._figure_to_base64(fig)
                html_content.append(f"""
                    <div class="figure">
                        <h4>PDP for {feature}</h4>
                        <img src="data:image/png;base64,{img_str}" alt="PDP {feature}">
                    </div>
                """)
            
            html_content.append("</div>")  # Close PDP subsection
            
            # Add calibration plot
            if "calibration_fig" in viz_plots:
                fig = viz_plots["calibration_fig"]
                img_str = self._figure_to_base64(fig)
                html_content.append(f"""
                    <div class="figure">
                        <h3>Calibration Plot</h3>
                        <img src="data:image/png;base64,{img_str}" alt="Calibration Plot">
                    </div>
                """)
            
            # Add local explanations if available
            if task_name in report["local_explanations"]:
                html_content.append("""
                    <div class="subsection">
                        <h3>Local Explanations</h3>
                """)
                
                local_exp = report["local_explanations"][task_name]
                
                for i, fig in enumerate(local_exp["lime_figures"]):
                    img_str = self._figure_to_base64(fig)
                    html_content.append(f"""
                        <div class="figure">
                            <h4>LIME Explanation for Instance {i+1}</h4>
                            <img src="data:image/png;base64,{img_str}" alt="LIME {i+1}">
                        </div>
                    """)
                
                html_content.append("</div>")  # Close local explanations subsection
            
            # Add sensitivity analysis if available
            if task_name in report["sensitivity_analysis"]:
                html_content.append("""
                    <div class="subsection">
                        <h3>Sensitivity Analysis</h3>
                """)
                
                sensitivity = report["sensitivity_analysis"][task_name]
                
                if "sensitivity_fig" in sensitivity:
                    fig = sensitivity["sensitivity_fig"]
                    img_str = self._figure_to_base64(fig)
                    html_content.append(f"""
                        <div class="figure">
                            <h4>Sensitivity Analysis</h4>
                            <img src="data:image/png;base64,{img_str}" alt="Sensitivity">
                        </div>
                    """)
                
                if "tornado_fig" in sensitivity:
                    fig = sensitivity["tornado_fig"]
                    img_str = self._figure_to_base64(fig)
                    html_content.append(f"""
                        <div class="figure">
                            <h4>Tornado Plot</h4>
                            <img src="data:image/png;base64,{img_str}" alt="Tornado Plot">
                        </div>
                    """)
                
                html_content.append("</div>")  # Close sensitivity subsection
            
            html_content.append("</div>")  # Close task section
        
        # Add footer
        html_content.append("""
            <div class="footer">
                <p>Generated by Tabular Transformer Explainability Module</p>
            </div>
        </body>
        </html>
        """)
        
        # Write HTML to file
        with open(output_path, "w") as f:
            f.write("".join(html_content))
    
    def _save_json_report(self, report: Dict[str, Any], output_path: str) -> None:
        """
        Save report as JSON.
        
        Args:
            report: Report dict
            output_path: Path to save JSON report
        """
        self.logger.info(f"Saving JSON report to {output_path}")
        
        # Create a serializable version of the report
        json_report = {
            "metadata": report["metadata"],
            "global_explanations": {
                "permutation_importance": {
                    task: importance.tolist() 
                    for task, importance in report["global_explanations"]["permutation_importance"].items()
                }
            },
        }
        
        # Write JSON to file
        with open(output_path, "w") as f:
            json.dump(json_report, f, indent=2)


class DashboardBuilder(LoggerMixin):
    """
    Interactive dashboard builder for tabular transformer explainability.
    
    This class helps create interactive dashboards for exploring model behavior
    and explanations.
    """
    
    def __init__(
        self,
        report_generator: Report,
    ):
        """
        Initialize dashboard builder.
        
        Args:
            report_generator: Report generator instance
        """
        self.report_generator = report_generator
        self.logger.info("Dashboard builder initialized. Note: For fully interactive dashboards, additional libraries like Dash or Streamlit are required.")
    
    def build_dashboard(
        self,
        data: Union[pd.DataFrame, TabularDataset],
        target_columns: Dict[str, str],
        output_directory: str,
        task_names: Optional[List[str]] = None,
    ) -> str:
        """
        Build a static dashboard for exploring model explanations.
        
        Args:
            data: Input data
            target_columns: Dict mapping task names to target column names
            output_directory: Directory to save dashboard files
            task_names: Optional list of task names to include (all by default)
            
        Returns:
            Path to main dashboard HTML file
        """
        self.logger.info(f"Building static dashboard in {output_directory}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        
        # Generate comprehensive report
        report = self.report_generator.generate_report(
            data=data,
            target_columns=target_columns,
            task_names=task_names,
            sample_instances=data.iloc[:5] if len(data) >= 5 else data,
        )
        
        # Save HTML report as main dashboard page
        main_output_path = os.path.join(output_directory, "dashboard.html")
        self.report_generator._save_html_report(report, main_output_path)
        
        self.logger.info(f"Dashboard built successfully at {main_output_path}")
        return main_output_path
    
    def build_interactive_dashboard(
        self,
        data: Union[pd.DataFrame, TabularDataset],
        target_columns: Dict[str, str],
        host: str = "127.0.0.1",
        port: int = 8050,
        task_names: Optional[List[str]] = None,
    ) -> None:
        """
        Build and launch an interactive dashboard (requires Dash or Streamlit).
        
        Args:
            data: Input data
            target_columns: Dict mapping task names to target column names
            host: Host to bind server to
            port: Port to bind server to
            task_names: Optional list of task names to include (all by default)
        """
        self.logger.info("Interactive dashboard requires Dash or Streamlit to be installed.")
        self.logger.info("This is a placeholder for the interactive dashboard feature.")
        
        # Check if Dash is installed
        try:
            import dash
            self.logger.info("Dash is available. You can extend this method to create a Dash dashboard.")
        except ImportError:
            self.logger.info("Dash is not installed. Consider installing it with 'pip install dash'.")
        
        # Check if Streamlit is installed
        try:
            import streamlit
            self.logger.info("Streamlit is available. You can extend this method to create a Streamlit dashboard.")
        except ImportError:
            self.logger.info("Streamlit is not installed. Consider installing it with 'pip install streamlit'.")
        
        self.logger.info("For fully interactive dashboards, consider implementing a custom solution using Dash or Streamlit.")
