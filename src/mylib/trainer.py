"""Training utilities for feature selection."""
import torch
import torch.nn as nn
import numpy as np


class FeatureSelectionTrainer:
    """
    Trainer with proper lambda search and early stopping.
    Handles joint training of classification model and feature selector.
    """
    
    def __init__(self, model, selector, criterion, lambda_reg=0.1, device='cpu'):
        """
        Initialize trainer.
        
        Args:
            model: Classification model (nn.Module)
            selector: Feature selector (BaseFeatureSelector)
            criterion: Loss function
            lambda_reg: Regularization strength
            device: Device to run on
        """
        self.model = model.to(device)
        self.selector = selector.to(device)
        self.criterion = criterion
        self.device = device
        self.lambda_reg = lambda_reg
        
        self.optimizer_model = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.optimizer_selector = torch.optim.Adam(selector.parameters(), lr=0.01)
        
        self.best_state = None
        self.history = {
            'train_loss': [],
            'val_acc': [],
            'val_loss': [],
            'sel_count': [],
            'reg_loss': []
        }

    def train_epoch(self, X_train, y_train, X_val, y_val):
        """Train for one epoch."""
        self.model.train()
        self.selector.train()
        
        self.optimizer_model.zero_grad()
        self.optimizer_selector.zero_grad()
        
        X_selected = self.selector(X_train)
        predictions = self.model(X_selected)
        
        classification_loss = self.criterion(predictions, y_train)
        regularization_loss = self.selector.regularization_loss()
        total_loss = classification_loss + self.lambda_reg * regularization_loss
        
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.selector.parameters()),
            max_norm=1.0
        )
        
        self.optimizer_model.step()
        self.optimizer_selector.step()
        
        self.model.eval()
        self.selector.eval()
        
        with torch.no_grad():
            X_val_selected = self.selector(X_val)
            val_predictions = self.model(X_val_selected)
            val_loss = self.criterion(val_predictions, y_val)
            val_acc = (val_predictions.argmax(1) == y_val).float().mean().item() * 100
            sel_count = self.selector.get_selected_features().sum()
        
        return {
            'train_loss': total_loss.item(),
            'val_loss': val_loss.item(),
            'val_acc': val_acc,
            'sel_count': sel_count,
            'reg_loss': regularization_loss.item()
        }

    def fit(self, X_train, y_train, X_val, y_val, epochs=300, 
            patience=50, verbose=False):
        """
        Train the model with early stopping.
        
        Args:
            X_train: Training features [N_train, D]
            y_train: Training labels [N_train]
            X_val: Validation features [N_val, D]
            y_val: Validation labels [N_val]
            epochs: Maximum number of epochs
            patience: Early stopping patience
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        best_val_acc = 0
        wait = 0
        
        for epoch in range(epochs):
            metrics = self.train_epoch(X_train, y_train, X_val, y_val)
            
            for key, value in metrics.items():
                self.history[key].append(value)
            
            if metrics['val_acc'] > best_val_acc:
                best_val_acc = metrics['val_acc']
                wait = 0
                self.best_state = {
                    'model': self.model.state_dict(),
                    'selector': self.selector.state_dict(),
                    'epoch': epoch,
                    'val_acc': best_val_acc,
                    'sel_count': metrics['sel_count']
                }
            else:
                wait += 1
            
            if wait >= patience and epoch >= 100:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
            
            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}: "
                      f"val_acc={metrics['val_acc']:.2f}%, "
                      f"sel={metrics['sel_count']}, "
                      f"λ={self.lambda_reg:.4f}")
        
        if self.best_state:
            self.model.load_state_dict(self.best_state['model'])
            self.selector.load_state_dict(self.best_state['selector'])
        
        return self.history

    def evaluate(self, X_test, y_test):
        """
        Evaluate on test set.
        
        Args:
            X_test: Test features [N_test, D]
            y_test: Test labels [N_test]
            
        Returns:
            Dictionary with test metrics
        """
        self.model.eval()
        self.selector.eval()
        
        with torch.no_grad():
            X_selected = self.selector(X_test)
            predictions = self.model(X_selected)
            acc = (predictions.argmax(1) == y_test).float().mean().item() * 100
            sel_features = self.selector.get_selected_features()
            
        return {
            'test_acc': acc,
            'selected_count': sel_features.sum(),
            'selected_features': sel_features
        }

