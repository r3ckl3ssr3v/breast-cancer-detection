import os
import argparse
import subprocess
import time
import pandas as pd
from datetime import datetime

def run_command(command, description=None):
    """Run a command and print its output."""
    if description:
        print(f"\n{'='*80}\n{description}\n{'='*80}")
    
    print(f"Running command: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
        return False
    
    return True

def main(args):
    # Create project directories if they don't exist
    os.makedirs("c:/SDP_Project_P3/data/raw", exist_ok=True)
    os.makedirs("c:/SDP_Project_P3/data/processed", exist_ok=True)
    os.makedirs("c:/SDP_Project_P3/models", exist_ok=True)
    os.makedirs("c:/SDP_Project_P3/results", exist_ok=True)
    os.makedirs("c:/SDP_Project_P3/logs", exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"c:/SDP_Project_P3/logs/run_{timestamp}.log"
    
    print(f"Starting project run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file}")
    
    # Download datasets if requested
    if args.download_data:
        run_command("python c:/SDP_Project_P3/src/data/download_datasets.py", "Downloading datasets")
    
    # Preprocess data if requested
    if args.preprocess_data:
        for dataset in args.datasets:
            run_command(f"python c:/SDP_Project_P3/src/data/data_preprocessing.py --dataset {dataset}", 
                       f"Preprocessing {dataset} dataset")
    
    # Train and evaluate models
    if args.train_models:
        for model in args.models:
            cmd = f"python c:/SDP_Project_P3/src/main.py --model {model} --batch_size {args.batch_size} --epochs {args.epochs}"
            
            if args.explain:
                cmd += " --explain"
            
            run_command(cmd, f"Training and evaluating {model} model")
    
    # Perform statistical comparison
    if args.statistical_comparison:
        run_command("python c:/SDP_Project_P3/src/utils/statistical_comparison.py", "Performing statistical comparison")
    
    # Cross-dataset comparison
    if args.cross_dataset_comparison and len(args.datasets) > 1:
        datasets_str = " ".join(args.datasets)
        run_command(f"python c:/SDP_Project_P3/src/utils/statistical_comparison.py --cross_dataset --datasets {datasets_str}",
                   "Performing cross-dataset comparison")
    
    # Generate final report
    if args.generate_report:
        run_command("python c:/SDP_Project_P3/src/utils/generate_report.py", "Generating final report")
    
    # Calculate total runtime
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nProject run completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Results saved to c:/SDP_Project_P3/results")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Breast Cancer Detection Project Runner")
    
    # Data options
    parser.add_argument("--download_data", action="store_true", help="Download datasets")
    parser.add_argument("--preprocess_data", action="store_true", help="Preprocess datasets")
    parser.add_argument("--datasets", nargs="+", default=["CBIS-DDSM", "MIAS", "INbreast"],
                        help="Datasets to use")
    
    # Training options
    parser.add_argument("--train_models", action="store_true", help="Train models")
    parser.add_argument("--models", nargs="+", default=["cnn", "transformer", "hybrid_sequential", "hybrid_parallel"],
                        help="Models to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    
    # Evaluation options
    parser.add_argument("--explain", action="store_true", help="Generate explainable AI visualizations")
    parser.add_argument("--statistical_comparison", action="store_true", help="Perform statistical comparison")
    parser.add_argument("--cross_dataset_comparison", action="store_true", help="Perform cross-dataset comparison")
    
    # Report options
    parser.add_argument("--generate_report", action="store_true", help="Generate final report")
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no specific actions are requested, do everything
    if not any([args.download_data, args.preprocess_data, args.train_models, 
                args.statistical_comparison, args.cross_dataset_comparison, args.generate_report]):
        args.download_data = True
        args.preprocess_data = True
        args.train_models = True
        args.statistical_comparison = True
        args.cross_dataset_comparison = True
        args.generate_report = True
    
    main(args)