#!/usr/bin/env python3
"""
MNIST CNN Trainer with Hyperparameter Optimization
==================================================

This script implements a Convolutional Neural Network for MNIST digit classification
with comprehensive hyperparameter search and model evaluation capabilities.

Features:
- Simple but effective CNN architecture
- Grid search and random search for hyperparameter optimization
- Model persistence in .pkl format
- Comprehensive statistics and comparison reports
- Performance visualization

Author: Generated for PTDIA Assignment 6
Date: September 26, 2025
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.model_selection import ParameterGrid


@dataclass
class TrainingResult:
    """Container for training results and metadata."""
    hyperparams: Dict[str, Any]
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    train_loss: float
    val_loss: float
    training_time: float
    model_state_dict: Dict[str, Any]
    loss_history: List[float]
    accuracy_history: List[float]


class SimpleCNN(nn.Module):
    """
    Simple but effective CNN architecture for MNIST classification.
    
    Architecture:
    - Convolutional layers with configurable filters and kernel sizes
    - Batch normalization for training stability
    - Dropout for regularization
    - Fully connected layers for classification
    """
    
    def __init__(self, 
                 conv1_filters: int = 32,
                 conv2_filters: int = 64,
                 conv3_filters: int = 128,
                 dropout_rate: float = 0.25,
                 fc_hidden_size: int = 128,
                 num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, conv1_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(conv1_filters)
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(conv2_filters)
        self.conv3 = nn.Conv2d(conv2_filters, conv3_filters, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(conv3_filters)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout2d = nn.Dropout2d(dropout_rate)
        
        # Calculate the size of flattened features
        # After conv1->pool: 28x28 -> 14x14
        # After conv2->pool: 14x14 -> 7x7
        # After conv3->pool: 7x7 -> 3x3 (with padding adjustments)
        self.fc_input_size = conv3_filters * 3 * 3
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, num_classes)
    
    def forward(self, x):
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout2d(x)
        
        # Second conv block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout2d(x)
        
        # Third conv block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout2d(x)
        
        # Flatten and fully connected layers
        x = x.view(-1, self.fc_input_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


class MNISTTrainer:
    """Main trainer class for MNIST CNN with hyperparameter optimization."""
    
    def __init__(self, data_dir: str = './data', results_dir: str = './results'):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Data loading setup
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        print(f"Using device: {self.device}")
    
    def load_data(self, batch_size: int = 64, validation_split: float = 0.2):
        """Load and prepare MNIST dataset with train/validation/test splits."""
        print("Loading MNIST dataset...")
        
        # Download and load training data
        full_train_dataset = datasets.MNIST(
            root=self.data_dir, 
            train=True, 
            download=True, 
            transform=self.transform
        )
        
        # Load test data
        self.test_dataset = datasets.MNIST(
            root=self.data_dir, 
            train=False, 
            download=True, 
            transform=self.transform
        )
        
        # Split training data into train and validation
        val_size = int(len(full_train_dataset) * validation_split)
        train_size = len(full_train_dataset) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2
        )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2
        )
        
        print("Dataset loaded successfully!")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
    
    def train_model(self, hyperparams: Dict[str, Any]) -> TrainingResult:
        """Train a single model with given hyperparameters."""
        print(f"\nTraining with hyperparameters: {hyperparams}")
        start_time = time.time()
        
        # Create model
        model = SimpleCNN(
            conv1_filters=hyperparams['conv1_filters'],
            conv2_filters=hyperparams['conv2_filters'],
            conv3_filters=hyperparams['conv3_filters'],
            dropout_rate=hyperparams['dropout_rate'],
            fc_hidden_size=hyperparams['fc_hidden_size']
        ).to(self.device)
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(
            model.parameters(), 
            lr=hyperparams['learning_rate'],
            weight_decay=hyperparams['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=hyperparams['scheduler_step'], 
            gamma=hyperparams['scheduler_gamma']
        )
        
        # Training history
        loss_history = []
        accuracy_history = []
        
        # Training loop
        epochs = hyperparams['epochs']
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
            
            scheduler.step()
            
            # Calculate epoch metrics
            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = 100. * correct / total
            
            loss_history.append(epoch_loss)
            accuracy_history.append(epoch_acc)
            
            if epoch % 5 == 0:
                print(f'Epoch {epoch}/{epochs}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
        # Evaluate on validation and test sets
        val_loss, val_acc = self._evaluate(model, self.val_loader)
        test_loss, test_acc = self._evaluate(model, self.test_loader)
        train_loss, train_acc = self._evaluate(model, self.train_loader)
        
        training_time = time.time() - start_time
        
        print(f'Training completed in {training_time:.2f}s')
        print(f'Train Accuracy: {train_acc:.2f}%')
        print(f'Validation Accuracy: {val_acc:.2f}%')
        print(f'Test Accuracy: {test_acc:.2f}%')
        
        return TrainingResult(
            hyperparams=hyperparams,
            train_accuracy=train_acc,
            val_accuracy=val_acc,
            test_accuracy=test_acc,
            train_loss=train_loss,
            val_loss=val_loss,
            training_time=training_time,
            model_state_dict=model.state_dict(),
            loss_history=loss_history,
            accuracy_history=accuracy_history
        )
    
    def _evaluate(self, model: nn.Module, data_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on given data loader."""
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        test_loss /= total
        accuracy = 100. * correct / total
        
        return test_loss, accuracy


class HyperparameterSearcher:
    """Hyperparameter optimization using grid search and random search."""
    
    def __init__(self, trainer: MNISTTrainer):
        self.trainer = trainer
        self.results = []
    
    def define_search_space(self) -> Dict[str, List]:
        """Define the hyperparameter search space."""
        return {
            'conv1_filters': [32, 64],
            'conv2_filters': [64, 128],
            'conv3_filters': [64, 128, 256],
            'dropout_rate': [0.2, 0.25, 0.3],
            'fc_hidden_size': [128, 256],
            'learning_rate': [0.001, 0.0005, 0.002],
            'weight_decay': [1e-4, 1e-5],
            'epochs': [15, 20],
            'scheduler_step': [7, 10],
            'scheduler_gamma': [0.1, 0.2]
        }
    
    def grid_search(self, max_combinations: int = 20) -> List[TrainingResult]:
        """Perform grid search over hyperparameter space."""
        search_space = self.define_search_space()
        param_grid = list(ParameterGrid(search_space))
        
        # Limit the number of combinations to avoid excessive training time
        if len(param_grid) > max_combinations:
            print(f"Reducing grid search from {len(param_grid)} to {max_combinations} combinations")
            np.random.shuffle(param_grid)
            param_grid = param_grid[:max_combinations]
        
        print(f"Starting grid search with {len(param_grid)} combinations...")
        
        results = []
        for i, params in enumerate(param_grid):
            print(f"\n--- Training combination {i+1}/{len(param_grid)} ---")
            try:
                result = self.trainer.train_model(params)
                results.append(result)
            except Exception as e:
                print(f"Error training with params {params}: {str(e)}")
                continue
        
        self.results = results
        return results
    
    def random_search(self, n_trials: int = 10) -> List[TrainingResult]:
        """Perform random search over hyperparameter space."""
        search_space = self.define_search_space()
        
        print(f"Starting random search with {n_trials} trials...")
        
        results = []
        for i in range(n_trials):
            # Randomly sample parameters
            params = {}
            for param, values in search_space.items():
                params[param] = np.random.choice(values)
            
            print(f"\n--- Training trial {i+1}/{n_trials} ---")
            try:
                result = self.trainer.train_model(params)
                results.append(result)
            except Exception as e:
                print(f"Error training with params {params}: {str(e)}")
                continue
        
        self.results = results
        return results


class ModelPersistence:
    """Handle model saving and loading in .pkl format."""
    
    @staticmethod
    def save_model(model_state_dict: Dict, hyperparams: Dict, filepath: str):
        """Save model state dict and hyperparameters to .pkl file."""
        save_data = {
            'model_state_dict': model_state_dict,
            'hyperparams': hyperparams,
            'timestamp': datetime.now().isoformat(),
            'model_class': 'SimpleCNN'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> Tuple[SimpleCNN, Dict]:
        """Load model from .pkl file."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Recreate model with saved hyperparameters
        hyperparams = save_data['hyperparams']
        model = SimpleCNN(
            conv1_filters=hyperparams['conv1_filters'],
            conv2_filters=hyperparams['conv2_filters'],
            conv3_filters=hyperparams['conv3_filters'],
            dropout_rate=hyperparams['dropout_rate'],
            fc_hidden_size=hyperparams['fc_hidden_size']
        )
        
        model.load_state_dict(save_data['model_state_dict'])
        
        return model, hyperparams


class StatisticsReporter:
    """Generate comprehensive reports and visualizations."""
    
    def __init__(self, results_dir: str = './results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def generate_comparison_report(self, results: List[TrainingResult]) -> pd.DataFrame:
        """Generate detailed comparison report of all experiments."""
        if not results:
            print("No results to generate report from!")
            return pd.DataFrame()
        
        # Extract data for comparison
        comparison_data = []
        for i, result in enumerate(results):
            row = {
                'experiment_id': i + 1,
                'val_accuracy': result.val_accuracy,
                'test_accuracy': result.test_accuracy,
                'train_accuracy': result.train_accuracy,
                'val_loss': result.val_loss,
                'training_time': result.training_time,
                **result.hyperparams
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by validation accuracy (descending)
        df = df.sort_values('val_accuracy', ascending=False)
        
        # Save to CSV
        csv_path = os.path.join(self.results_dir, 'hyperparameter_comparison.csv')
        df.to_csv(csv_path, index=False)
        
        # Print summary statistics
        print("\n" + "="*80)
        print("HYPERPARAMETER SEARCH RESULTS SUMMARY")
        print("="*80)
        print(f"Total experiments: {len(results)}")
        print(f"Best validation accuracy: {df['val_accuracy'].max():.2f}%")
        print(f"Best test accuracy: {df['test_accuracy'].max():.2f}%")
        print(f"Average validation accuracy: {df['val_accuracy'].mean():.2f}%")
        print(f"Std validation accuracy: {df['val_accuracy'].std():.2f}%")
        print(f"Total training time: {df['training_time'].sum():.2f}s")
        
        print("\nTop 3 configurations by validation accuracy:")
        print("-" * 50)
        top_3 = df.head(3)
        for _, row in top_3.iterrows():
            print(f"Experiment {row['experiment_id']}: Val Acc = {row['val_accuracy']:.2f}%, "
                  f"Test Acc = {row['test_accuracy']:.2f}%")
            print(f"  Hyperparams: conv_filters=({row['conv1_filters']}, {row['conv2_filters']}, {row['conv3_filters']}), "
                  f"dropout={row['dropout_rate']}, lr={row['learning_rate']}")
            print()
        
        return df
    
    def create_visualizations(self, results: List[TrainingResult], df: pd.DataFrame):
        """Create comprehensive visualizations of the results."""
        if not results:
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        plt.figure(figsize=(20, 15))
        
        # 1. Accuracy comparison
        plt.subplot(2, 3, 1)
        x = range(len(results))
        plt.scatter(x, df['val_accuracy'], alpha=0.7, label='Validation', s=60)
        plt.scatter(x, df['test_accuracy'], alpha=0.7, label='Test', s=60)
        plt.xlabel('Experiment ID')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Comparison Across Experiments')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Training time vs accuracy
        plt.subplot(2, 3, 2)
        plt.scatter(df['training_time'], df['val_accuracy'], alpha=0.7, s=60)
        plt.xlabel('Training Time (s)')
        plt.ylabel('Validation Accuracy (%)')
        plt.title('Training Time vs Validation Accuracy')
        plt.grid(True, alpha=0.3)
        
        # 3. Learning rate impact
        plt.subplot(2, 3, 3)
        lr_groups = df.groupby('learning_rate')['val_accuracy']
        lr_means = lr_groups.mean()
        lr_stds = lr_groups.std()
        plt.bar(range(len(lr_means)), lr_means.values, 
                yerr=lr_stds.values, alpha=0.7, capsize=5)
        plt.xlabel('Learning Rate')
        plt.ylabel('Mean Validation Accuracy (%)')
        plt.title('Learning Rate Impact on Performance')
        plt.xticks(range(len(lr_means)), [f"{lr:.4f}" for lr in lr_means.index])
        plt.grid(True, alpha=0.3)
        
        # 4. Dropout rate impact
        plt.subplot(2, 3, 4)
        dropout_groups = df.groupby('dropout_rate')['val_accuracy']
        dropout_means = dropout_groups.mean()
        dropout_stds = dropout_groups.std()
        plt.bar(range(len(dropout_means)), dropout_means.values, 
                yerr=dropout_stds.values, alpha=0.7, capsize=5, color='orange')
        plt.xlabel('Dropout Rate')
        plt.ylabel('Mean Validation Accuracy (%)')
        plt.title('Dropout Rate Impact on Performance')
        plt.xticks(range(len(dropout_means)), [f"{dr:.2f}" for dr in dropout_means.index])
        plt.grid(True, alpha=0.3)
        
        # 5. Architecture impact (conv filters)
        plt.subplot(2, 3, 5)
        df['conv_arch'] = df['conv1_filters'].astype(str) + '-' + \
                         df['conv2_filters'].astype(str) + '-' + \
                         df['conv3_filters'].astype(str)
        arch_groups = df.groupby('conv_arch')['val_accuracy']
        arch_means = arch_groups.mean()
        plt.bar(range(len(arch_means)), arch_means.values, alpha=0.7, color='green')
        plt.xlabel('Conv Architecture (filters)')
        plt.ylabel('Mean Validation Accuracy (%)')
        plt.title('Architecture Impact on Performance')
        plt.xticks(range(len(arch_means)), arch_means.index, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 6. Training history of best model
        plt.subplot(2, 3, 6)
        best_idx = df['val_accuracy'].idxmax()
        best_result = results[best_idx]
        epochs = range(1, len(best_result.accuracy_history) + 1)
        plt.plot(epochs, best_result.accuracy_history, label='Training Accuracy', linewidth=2)
        plt.plot(epochs, [best_result.val_accuracy] * len(epochs), 
                 '--', label='Final Validation Accuracy', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training History of Best Model')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        viz_path = os.path.join(self.results_dir, 'hyperparameter_analysis.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualizations saved to {viz_path}")


def main():
    """Main execution function."""
    print("MNIST CNN Trainer with Hyperparameter Optimization")
    print("=" * 60)
    
    # Initialize trainer
    trainer = MNISTTrainer()
    trainer.load_data(batch_size=128)
    
    # Initialize hyperparameter searcher
    searcher = HyperparameterSearcher(trainer)
    
    # Perform hyperparameter search (choose between grid and random)
    print("\nChoose search method:")
    print("1. Grid Search (systematic, slower)")
    print("2. Random Search (faster, good coverage)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        results = searcher.grid_search(max_combinations=15)
    else:
        results = searcher.random_search(n_trials=10)
    
    if not results:
        print("No successful training runs completed!")
        return
    
    # Find best model
    best_result = max(results, key=lambda x: x.val_accuracy)
    
    # Save best model
    best_model_path = os.path.join(trainer.results_dir, 'best_mnist_cnn_model.pkl')
    ModelPersistence.save_model(
        best_result.model_state_dict, 
        best_result.hyperparams, 
        best_model_path
    )
    
    # Generate comprehensive report
    reporter = StatisticsReporter(trainer.results_dir)
    comparison_df = reporter.generate_comparison_report(results)
    reporter.create_visualizations(results, comparison_df)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Best model saved to: {best_model_path}")
    print(f"Results and reports saved to: {trainer.results_dir}")
    print(f"Best validation accuracy: {best_result.val_accuracy:.2f}%")
    print(f"Best test accuracy: {best_result.test_accuracy:.2f}%")


if __name__ == "__main__":
    main()