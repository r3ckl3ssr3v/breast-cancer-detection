import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import mlflow
import mlflow.pytorch

class Trainer:
    """Trainer class for model training and evaluation."""
    
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 criterion=None, optimizer=None, scheduler=None, 
                 device=None, num_classes=3, model_name="model"):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data
            criterion: Loss function (default: CrossEntropyLoss)
            optimizer: Optimizer (default: Adam)
            scheduler: Learning rate scheduler (default: ReduceLROnPlateau)
            device: Device to use for training (default: cuda if available, else cpu)
            num_classes: Number of classes
            model_name: Name of the model for saving
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Set device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Set criterion
        self.criterion = criterion if criterion else nn.CrossEntropyLoss()
        
        # Set optimizer
        self.optimizer = optimizer if optimizer else optim.Adam(model.parameters(), lr=0.001)
        
        # Set scheduler
        self.scheduler = scheduler if scheduler else ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )
        
        # Initialize tracking variables
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.best_model_path = f"c:/SDP_Project_P3/models/{model_name}_best.pth"
        
        # Create directory for saving models if it doesn't exist
        os.makedirs("c:/SDP_Project_P3/models", exist_ok=True)
        os.makedirs("c:/SDP_Project_P3/results", exist_ok=True)
    
    def train_epoch(self):
        """Train the model for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for inputs, labels in tqdm(self.train_loader, desc="Training"):
            inputs = inputs.to(self.device)
            
            # Handle one-hot encoded labels
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                labels_for_loss = labels.to(self.device)
                labels_for_acc = torch.argmax(labels, dim=1).to(self.device)
            else:
                labels_for_loss = labels.to(self.device)
                labels_for_acc = labels.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels_for_loss)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_for_acc.cpu().numpy())
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Validation"):
                inputs = inputs.to(self.device)
                
                # Handle one-hot encoded labels
                if len(labels.shape) > 1 and labels.shape[1] > 1:
                    labels_for_loss = labels.to(self.device)
                    labels_for_acc = torch.argmax(labels, dim=1).to(self.device)
                else:
                    labels_for_loss = labels.to(self.device)
                    labels_for_acc = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels_for_loss)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_for_acc.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs=50, early_stopping_patience=10):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Number of epochs to wait for improvement before stopping
        """
        print(f"Training {self.model_name} on {self.device}")
        
        # Start MLflow run
        with mlflow.start_run(run_name=self.model_name):
            # Log model parameters
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("batch_size", self.train_loader.batch_size)
            mlflow.log_param("optimizer", self.optimizer.__class__.__name__)
            mlflow.log_param("learning_rate", self.optimizer.param_groups[0]['lr'])
            
            # Training loop
            start_time = time.time()
            no_improve_count = 0
            
            for epoch in range(num_epochs):
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                
                # Train and validate
                train_loss, train_acc = self.train_epoch()
                val_loss, val_acc = self.validate()
                
                # Update learning rate
                self.scheduler.step(val_loss)
                
                # Save statistics
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_accuracies.append(train_acc)
                self.val_accuracies.append(val_acc)
                
                # Log metrics to MLflow
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("train_accuracy", train_acc, step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)
                
                # Print statistics
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    print(f"Validation loss decreased from {self.best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
                    self.best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.best_model_path)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # Early stopping
                if no_improve_count >= early_stopping_patience:
                    print(f"No improvement for {early_stopping_patience} epochs. Early stopping.")
                    break
            
            # Calculate training time
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds")
            
            # Log the best model
            mlflow.pytorch.log_model(self.model, "model")
            
            # Plot and save training curves
            self.plot_training_curves()
    
    def evaluate(self, loader=None, load_best=True):
        """
        Evaluate the model on the given data loader.
        
        Args:
            loader: DataLoader to use for evaluation (default: test_loader)
            load_best: Whether to load the best model before evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Load the best model if requested
        if load_best and os.path.exists(self.best_model_path):
            self.model.load_state_dict(torch.load(self.best_model_path))
            print(f"Loaded best model from {self.best_model_path}")
        
        # Use test loader if no loader is specified
        if loader is None:
            loader = self.test_loader
        
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        running_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc="Evaluation"):
                inputs = inputs.to(self.device)
                
                # Handle one-hot encoded labels
                if len(labels.shape) > 1 and labels.shape[1] > 1:
                    labels_for_loss = labels.to(self.device)
                    labels_for_acc = torch.argmax(labels, dim=1).to(self.device)
                else:
                    labels_for_loss = labels.to(self.device)
                    labels_for_acc = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels_for_loss)
                
                # Get predictions and probabilities
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                # Accumulate statistics
                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels_for_acc.cpu().numpy())
        
        # Calculate metrics
        test_loss = running_loss / len(loader.dataset)
        test_acc = accuracy_score(all_labels, all_preds)
        
        # For multi-class, use macro averaging
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Calculate ROC AUC for multi-class
        all_probs = np.array(all_probs)
        all_labels_onehot = np.zeros((len(all_labels), self.num_classes))
        for i, label in enumerate(all_labels):
            all_labels_onehot[i, label] = 1
        
        try:
            roc_auc = roc_auc_score(all_labels_onehot, all_probs, multi_class='ovr')
        except ValueError:
            # Handle case where some classes might not be present
            roc_auc = 0
        
        # Compile results
        results = {
            'loss': test_loss,
            'accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'probabilities': all_probs,
            'true_labels': all_labels
        }
        
        # Print results
        print(f"\nEvaluation Results for {self.model_name}:")
        print(f"Loss: {test_loss:.4f}")
        print(f"Accuracy: {test_acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # Plot and save confusion matrix
        self.plot_confusion_matrix(cm)
        
        # Plot and save ROC curve
        self.plot_roc_curve(all_labels_onehot, all_probs)
        
        return results
    
    def plot_training_curves(self):
        """Plot and save training and validation curves."""
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"c:/SDP_Project_P3/results/{self.model_name}_training_curves.png")
        plt.close()
    
    def plot_confusion_matrix(self, cm):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Benign', 'Malignant'],
                    yticklabels=['Normal', 'Benign', 'Malignant'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.tight_layout()
        plt.savefig(f"c:/SDP_Project_P3/results/{self.model_name}_confusion_matrix.png")
        plt.close()
    
    def plot_roc_curve(self, y_true, y_score):
        """Plot and save ROC curve."""
        from sklearn.metrics import roc_curve, auc
        from itertools import cycle
        
        plt.figure(figsize=(10, 8))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Plot ROC curves
        plt.plot(fpr["micro"], tpr["micro"],
                label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
                color='deeppink', linestyle=':', linewidth=4)
        
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        class_names = ['Normal', 'Benign', 'Malignant']
        
        for i, color, name in zip(range(self.num_classes), colors, class_names):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'ROC curve of {name} (area = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f"c:/SDP_Project_P3/results/{self.model_name}_roc_curve.png")
        plt.close()