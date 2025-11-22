#!/usr/bin/env python3
"""
02_train_baselines.py
=====================
Train baseline machine learning classifiers for MANET intrusion detection.

This script trains three baseline models (Logistic Regression, Decision Tree, 
XGBoost) and evaluates their performance on the test set.

Author: V.S.S. Karthik
Date: November 2024
"""

import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

def load_data():
    """Load train/test splits."""
    with open('data/train_test_split.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a single model."""
    print(f"\n{'='*70}")
    print(f"Training {model_name}...")
    print('='*70)
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate
    results = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test, average='weighted'),
        'recall': recall_score(y_test, y_pred_test, average='weighted'),
        'f1_score': f1_score(y_test, y_pred_test, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist(),
    }
    
    # Per-class metrics
    class_names = ['Normal', 'Flooding DoS', 'Blackhole']
    per_class = classification_report(y_test, y_pred_test, target_names=class_names, 
                                      output_dict=True)
    results['per_class'] = per_class
    
    # Print results
    print(f"\nTraining Accuracy: {results['train_accuracy']:.4f}")
    print(f"Testing Accuracy:  {results['test_accuracy']:.4f}")
    print(f"Precision:         {results['precision']:.4f}")
    print(f"Recall:            {results['recall']:.4f}")
    print(f"F1-Score:          {results['f1_score']:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = np.array(results['confusion_matrix'])
    print(f"{'':12} {'Normal':>10} {'Flooding':>10} {'Blackhole':>10}")
    for i, label in enumerate(class_names):
        print(f"{label:12} {cm[i,0]:>10} {cm[i,1]:>10} {cm[i,2]:>10}")
    
    print(f"\nPer-Class Performance:")
    for cls in class_names:
        metrics = per_class[cls]
        print(f"  {cls:15} - Precision: {metrics['precision']:.4f}, "
              f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1-score']:.4f}")
    
    return model, results

def main():
    """Main execution function."""
    print("="*70)
    print("MANET IDS Baseline Model Training")
    print("="*70)
    
    # Create output directories
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    # Load data
    print("\n1. Loading dataset...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples:  {len(X_test)}")
    
    # Scale features
    print("\n2. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("   Scaler saved to models/scaler.pkl")
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='lbfgs',
            multi_class='multinomial'
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
    }
    
    # Train all models
    print("\n3. Training models...")
    all_results = {}
    
    for model_name, model in models.items():
        trained_model, results = train_and_evaluate_model(
            model, X_train_scaled, X_test_scaled, y_train, y_test, model_name
        )
        
        # Save model
        model_filename = model_name.lower().replace(' ', '_')
        with open(f'models/{model_filename}.pkl', 'wb') as f:
            pickle.dump(trained_model, f)
        print(f"   Model saved to models/{model_filename}.pkl")
        
        all_results[model_name] = results
    
    # Save all results
    with open('results/baseline_performance.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY: Model Comparison")
    print("="*70)
    print(f"{'Model':20} {'Train Acc':>12} {'Test Acc':>12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    print("-"*70)
    
    for model_name, results in all_results.items():
        print(f"{model_name:20} "
              f"{results['train_accuracy']:>12.4f} "
              f"{results['test_accuracy']:>12.4f} "
              f"{results['precision']:>12.4f} "
              f"{results['recall']:>12.4f} "
              f"{results['f1_score']:>12.4f}")
    
    # Find best model
    best_model = max(all_results.items(), key=lambda x: x[1]['test_accuracy'])
    print("\n" + "="*70)
    print(f"Best Model: {best_model[0]} (Test Accuracy: {best_model[1]['test_accuracy']:.4f})")
    print("="*70)
    
    print("\nFiles saved:")
    print("  - models/scaler.pkl")
    print("  - models/logistic_regression.pkl")
    print("  - models/decision_tree.pkl")
    print("  - models/xgboost.pkl")
    print("  - results/baseline_performance.json")
    print("="*70)

if __name__ == '__main__':
    main()
