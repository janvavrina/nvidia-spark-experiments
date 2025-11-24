"""
Experiment runner with dependency resolution and checkpointing
"""

import os
import sys
import json
import time
import subprocess
import traceback
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import yaml


@dataclass
class Experiment:
    """Represents a single experiment"""
    id: str
    name: str
    script: str
    dependencies: List[str]
    enabled: bool = True
    timeout: Optional[int] = None  # seconds
    retries: int = 0


@dataclass
class ExperimentResult:
    """Result of running an experiment"""
    experiment_id: str
    status: str  # "success", "failed", "skipped"
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    error: Optional[str] = None
    output_file: Optional[str] = None


class ExperimentRunner:
    """Manages experiment execution with dependency resolution"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.results: Dict[str, ExperimentResult] = {}
        self.experiments: Dict[str, Experiment] = {}
        
    def load_experiments(self, config_path: str):
        """Load experiments from YAML config"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        for exp_config in config.get("experiments", []):
            exp = Experiment(
                id=exp_config["id"],
                name=exp_config.get("name", exp_config["id"]),
                script=exp_config["script"],
                dependencies=exp_config.get("dependencies", []),
                enabled=exp_config.get("enabled", True),
                timeout=exp_config.get("timeout"),
                retries=exp_config.get("retries", 0),
            )
            self.experiments[exp.id] = exp
    
    def load_checkpoint(self):
        """Load previous results from checkpoint"""
        checkpoint_file = self.checkpoint_dir / "results.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                for exp_id, result_data in data.items():
                    result = ExperimentResult(**result_data)
                    self.results[exp_id] = result
    
    def save_checkpoint(self):
        """Save current results to checkpoint"""
        checkpoint_file = self.checkpoint_dir / "results.json"
        data = {
            exp_id: asdict(result)
            for exp_id, result in self.results.items()
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_execution_order(self) -> List[str]:
        """Topological sort to determine execution order"""
        # Build dependency graph
        in_degree = {exp_id: 0 for exp_id in self.experiments.keys()}
        graph = {exp_id: [] for exp_id in self.experiments.keys()}
        
        for exp_id, exp in self.experiments.items():
            if not exp.enabled:
                continue
            for dep in exp.dependencies:
                if dep in self.experiments:
                    in_degree[exp_id] += 1
                    graph[dep].append(exp_id)
        
        # Topological sort
        queue = [exp_id for exp_id, degree in in_degree.items() 
                if degree == 0 and self.experiments[exp_id].enabled]
        order = []
        
        while queue:
            exp_id = queue.pop(0)
            order.append(exp_id)
            
            for neighbor in graph[exp_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for cycles
        if len(order) < sum(1 for exp in self.experiments.values() if exp.enabled):
            remaining = set(self.experiments.keys()) - set(order)
            raise ValueError(f"Circular dependencies detected or missing dependencies: {remaining}")
        
        return order
    
    def is_completed(self, exp_id: str) -> bool:
        """Check if experiment already completed successfully"""
        if exp_id in self.results:
            return self.results[exp_id].status == "success"
        return False
    
    def dependencies_satisfied(self, exp_id: str) -> bool:
        """Check if all dependencies are satisfied"""
        exp = self.experiments[exp_id]
        for dep_id in exp.dependencies:
            if not self.is_completed(dep_id):
                return False
        return True
    
    def run_experiment(self, exp_id: str) -> ExperimentResult:
        """Run a single experiment"""
        exp = self.experiments[exp_id]
        
        print(f"\n{'='*60}")
        print(f"Running experiment: {exp.name} ({exp_id})")
        print(f"{'='*60}")
        
        start_time = time.time()
        result = ExperimentResult(
            experiment_id=exp_id,
            status="failed",
            start_time=start_time,
        )
        
        try:
            # Check if script exists
            script_path = Path(exp.script)
            if not script_path.exists():
                raise FileNotFoundError(f"Script not found: {exp.script}")
            
            # Run the script
            env = os.environ.copy()
            env["EXPERIMENT_ID"] = exp_id
            env["EXPERIMENT_NAME"] = exp.name
            
            cmd = [sys.executable, str(script_path)]
            
            print(f"Command: {' '.join(cmd)}")
            print(f"Working directory: {os.getcwd()}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            
            try:
                stdout, stderr = process.communicate(timeout=exp.timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                raise TimeoutError(f"Experiment timed out after {exp.timeout} seconds")
            
            if process.returncode == 0:
                result.status = "success"
                print(f"✓ Experiment completed successfully")
            else:
                result.error = stderr
                print(f"✗ Experiment failed with return code {process.returncode}")
                print(f"Error output:\n{stderr}")
            
            # Save output
            output_file = self.checkpoint_dir / f"{exp_id}_output.txt"
            with open(output_file, 'w') as f:
                f.write(f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}")
            result.output_file = str(output_file)
            
        except Exception as e:
            result.error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            print(f"✗ Experiment failed with exception: {e}")
        
        finally:
            end_time = time.time()
            result.end_time = end_time
            result.duration = end_time - start_time
            self.results[exp_id] = result
            self.save_checkpoint()
        
        return result
    
    def run_all(self, resume: bool = True, skip_completed: bool = True):
        """Run all experiments in dependency order"""
        if resume:
            self.load_checkpoint()
        
        execution_order = self.get_execution_order()
        
        print(f"\n{'='*60}")
        print(f"Experiment Execution Plan")
        print(f"{'='*60}")
        print(f"Total experiments: {len(execution_order)}")
        print(f"Execution order:")
        for i, exp_id in enumerate(execution_order, 1):
            status = "✓" if self.is_completed(exp_id) else "○"
            print(f"  {i}. {status} {self.experiments[exp_id].name} ({exp_id})")
        
        print(f"\nStarting execution...")
        
        for i, exp_id in enumerate(execution_order, 1):
            exp = self.experiments[exp_id]
            
            if not exp.enabled:
                print(f"\n[{i}/{len(execution_order)}] Skipping disabled experiment: {exp.name}")
                continue
            
            if skip_completed and self.is_completed(exp_id):
                print(f"\n[{i}/{len(execution_order)}] Skipping completed experiment: {exp.name}")
                continue
            
            if not self.dependencies_satisfied(exp_id):
                print(f"\n[{i}/{len(execution_order)}] Dependencies not satisfied for: {exp.name}")
                result = ExperimentResult(
                    experiment_id=exp_id,
                    status="skipped",
                    start_time=time.time(),
                    end_time=time.time(),
                    duration=0.0,
                    error="Dependencies not satisfied",
                )
                self.results[exp_id] = result
                continue
            
            # Retry logic
            max_attempts = exp.retries + 1
            for attempt in range(max_attempts):
                if attempt > 0:
                    print(f"\nRetry attempt {attempt + 1}/{max_attempts} for {exp.name}")
                
                result = self.run_experiment(exp_id)
                
                if result.status == "success":
                    break
                
                if attempt < max_attempts - 1:
                    print(f"Waiting before retry...")
                    time.sleep(5)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print execution summary"""
        print(f"\n{'='*60}")
        print(f"Execution Summary")
        print(f"{'='*60}")
        
        total = len(self.results)
        successful = sum(1 for r in self.results.values() if r.status == "success")
        failed = sum(1 for r in self.results.values() if r.status == "failed")
        skipped = sum(1 for r in self.results.values() if r.status == "skipped")
        
        print(f"Total: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Skipped: {skipped}")
        
        if failed > 0:
            print(f"\nFailed experiments:")
            for exp_id, result in self.results.items():
                if result.status == "failed":
                    print(f"  - {self.experiments[exp_id].name} ({exp_id})")
                    if result.error:
                        print(f"    Error: {result.error[:200]}...")

