import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import mlflow
from datetime import datetime

# Import custom modules
from data.data_loader import get_data_loaders
from models.models import CNNModel, TransformerModel, HybridSequentialModel, HybridParallelModel
from models.train import Trainer
from utils.explainable_ai import ExplainableAI

def main(args):
    # Set up MLflow
    mlflow.set_tracking_uri("file:c:/SDP_Project_P3/mlruns")
    mlflow.set_experiment("Breast Cancer Detection")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Get data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"Data loaded: {len(train_loader.dataset)} training, {len(val_loader.dataset)} validation, {len(test_loader.dataset)} test samples")
    
    # Define models to train
    models = {}
    
    if args.model == "all" or args.model == "cnn":
        models["CNN"] = CNNModel(num_classes=3, pretrained=True)
    
    if args.model == "all" or args.model == "transformer":
        models["Transformer"] = TransformerModel(num_classes=3, pretrained=True)
    
    if args.model == "all" or args.model == "hybrid_sequential":
        models["HybridSequential"] = HybridSequentialModel(num_classes=3, pretrained=True)
    
    if args.model == "all" or args.model == "hybrid_parallel":
        models["HybridParallel"] = HybridParallelModel(num_classes=3, pretrained=True)
    
    # Train and evaluate each model
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name} model")
        print(f"{'='*50}")
        
        # Define optimizer and criterion
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            model_name=model_name
        )
        
        # Train model
        if not args.eval_only:
            trainer.train(num_epochs=args.epochs, early_stopping_patience=args.patience)
        
        # Evaluate model
        results[model_name] = trainer.evaluate(load_best=True)
        
        # Generate explainable AI visualizations if requested
        if args.explain:
            print(f"\nGenerating explainable AI visualizations for {model_name}...")
            
            # Create directory for explainable AI results
            explain_dir = f"c:/SDP_Project_P3/results/{model_name}_explanations"
            os.makedirs(explain_dir, exist_ok=True)
            
            # Initialize explainable AI
            explainer = ExplainableAI(model, device=device)
            
            # Get a few test samples
            test_samples = []
            for inputs, labels in test_loader:
                for i in range(min(5, len(inputs))):
                    # Save input image
                    img = inputs[i].permute(1, 2, 0).cpu().numpy()
                    img = (img * 255).astype('uint8')
                    img_path = f"{explain_dir}/sample_{len(test_samples)}.png"
                    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    
                    test_samples.append({
                        'image_path': img_path,
                        'label': labels[i].item() if len(labels[i].shape) == 0 else torch.argmax(labels[i]).item()
                    })
                
                if len(test_samples) >= 5:
                    break
            
            # Generate explanations for each sample
            for i, sample in enumerate(test_samples):
                # Determine target layer for Grad-CAM
                target_layer = None
                if model_name == "CNN":
                    target_layer = model.backbone.layer4[-1]
                elif model_name == "HybridSequential":
                    target_layer = model.cnn_backbone[-1]
                elif model_name == "HybridParallel":
                    target_layer = model.cnn_backbone[-1]
                
                # Generate and save visualizations
                save_path = f"{explain_dir}/explanation_sample_{i}.png"
                explainer.visualize_multiple_methods(
                    image_path=sample['image_path'],
                    target_layer=target_layer,
                    target_class=None,  # Use predicted class
                    save_path=save_path
                )
                
                print(f"Saved explanation for sample {i} to {save_path}")
    
    # Compare model results
    if len(results) > 1:
        print("\n" + "="*50)
        print("Model Comparison")
        print("="*50)
        
        # Create comparison table
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        print("\nMetrics Comparison:")
        header = "Model".ljust(20)
        for metric in metrics:
            header += f"{metric.capitalize().replace('_', ' ')}".ljust(15)
        print(header)
        print("-" * 100)
        
        for model_name, result in results.items():
            row = model_name.ljust(20)
            for metric in metrics:
                row += f"{result[metric]:.4f}".ljust(15)
            print(row)
        
        # Save comparison results
        import pandas as pd
        comparison_df = pd.DataFrame(columns=['Model'] + metrics)
        
        for model_name, result in results.items():
            row = {'Model': model_name}
            for metric in metrics:
                row[metric] = result[metric]
            comparison_df = comparison_df.append(row, ignore_index=True)
        
        comparison_df.to_csv("c:/SDP_Project_P3/results/model_comparison.csv", index=False)
        
        # Create comparison plots
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(15, 10))
        
        for i, metric in enumerate(metrics):
            plt.subplot(2, 3, i+1)
            sns.barplot(x='Model', y=metric, data=comparison_df)
            plt.title(f"{metric.capitalize().replace('_', ' ')} Comparison")
            plt.xticks(rotation=45)
            plt.tight_layout()
        
        plt.savefig("c:/SDP_Project_P3/results/metrics_comparison.png")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Breast Cancer Detection using Hybrid CNN and Transformer Models")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="c:/SDP_Project_P3/data/processed",
                        help="Directory containing processed data")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads for data loading")
    
    # Model parameters
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "cnn", "transformer", "hybrid_sequential", "hybrid_parallel"],
                        help="Model to train")
    
    # Other parameters
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate, don't train")
    parser.add_argument("--explain", action="store_true", help="Generate explainable AI visualizations")
    
    args = parser.parse_args()
    
    main(args)