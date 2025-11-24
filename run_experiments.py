#!/usr/bin/env python3
"""
Main experiment orchestration script
Run all experiments sequentially with dependency resolution and checkpointing.
"""

import argparse
import sys
from pathlib import Path
from orchestrator.experiment_runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(
        description="Run DGX Spark NLP experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments.yaml",
        help="Path to experiment configuration YAML file",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from checkpoint (start fresh)",
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Rerun experiments that already completed",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoint files",
    )
    
    args = parser.parse_args()
    
    # Check if config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        print(f"Please create {config_path} with experiment definitions.")
        sys.exit(1)
    
    # Create runner
    runner = ExperimentRunner(checkpoint_dir=args.checkpoint_dir)
    
    # Load experiments
    try:
        runner.load_experiments(str(config_path))
        print(f"Loaded {len(runner.experiments)} experiments from {config_path}")
    except Exception as e:
        print(f"Error loading experiments: {e}")
        sys.exit(1)
    
    # Run experiments
    try:
        runner.run_all(
            resume=not args.no_resume,
            skip_completed=not args.rerun_completed,
        )
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
        print("Progress saved to checkpoint. Use --no-resume to start fresh.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Check if all succeeded
    failed = sum(1 for r in runner.results.values() if r.status == "failed")
    if failed > 0:
        print(f"\n⚠️  {failed} experiment(s) failed. Check logs for details.")
        sys.exit(1)
    else:
        print("\n✓ All experiments completed successfully!")


if __name__ == "__main__":
    main()

