"""
Experiment tracking utilities
Integrates with MLflow and Weights & Biases for experiment tracking.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Any
import json


class ExperimentTracker:
    """Unified experiment tracking interface"""
    
    def __init__(self, tracking_uri: Optional[str] = None, use_mlflow: bool = True, use_wandb: bool = False):
        self.use_mlflow = use_mlflow
        self.use_wandb = use_wandb
        self.mlflow_run = None
        self.wandb_run = None
        
        if use_mlflow:
            try:
                import mlflow
                mlflow.set_tracking_uri(tracking_uri or "file:./mlruns")
                self.mlflow = mlflow
            except ImportError:
                print("MLflow not available")
                self.use_mlflow = False
        
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("Weights & Biases not available")
                self.use_wandb = False
    
    def start_run(self, experiment_name: str, run_name: Optional[str] = None):
        """Start a new experiment run"""
        if self.use_mlflow:
            self.mlflow_run = self.mlflow.start_run(run_name=run_name)
        
        if self.use_wandb:
            self.wandb_run = self.wandb.init(project="dgx-spark-nlp", name=run_name)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        if self.use_mlflow and self.mlflow_run:
            self.mlflow.log_params(params)
        
        if self.use_wandb and self.wandb_run:
            self.wandb.config.update(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        if self.use_mlflow and self.mlflow_run:
            self.mlflow.log_metrics(metrics, step=step)
        
        if self.use_wandb and self.wandb_run:
            self.wandb.log(metrics, step=step)
    
    def log_artifact(self, filepath: str):
        """Log an artifact"""
        if self.use_mlflow and self.mlflow_run:
            self.mlflow.log_artifact(filepath)
        
        if self.use_wandb and self.wandb_run:
            self.wandb.log_artifact(filepath)
    
    def end_run(self):
        """End the current run"""
        if self.use_mlflow and self.mlflow_run:
            self.mlflow.end_run()
            self.mlflow_run = None
        
        if self.use_wandb and self.wandb_run:
            self.wandb.finish()
            self.wandb_run = None


def setup_tracking(experiment_name: str, use_mlflow: bool = True, use_wandb: bool = False):
    """Setup experiment tracking"""
    tracker = ExperimentTracker(use_mlflow=use_mlflow, use_wandb=use_wandb)
    tracker.start_run(experiment_name)
    return tracker

