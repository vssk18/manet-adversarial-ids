#!/usr/bin/env python3
"""
03_adversarial_attacks.py
=========================
Generate standard adversarial attacks (FGSM and PGD) against MANET IDS models.

This script implements Fast Gradient Sign Method (FGSM) and Projected Gradient
Descent (PGD) attacks to evaluate model robustness.

Author: V.S.S. Karthik
Date: November 2024
"""

import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def load_models_and_data():
    """Load trained models and test data."""
    # Load data
    with open('data/train_test_split.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Load scaler
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load models
    models = {}
    for model_name in ['logistic_regression', 'decision_tree', 'xgboost']:
        with open(f'models/{model_name}.pkl', 'rb') as f:
            models[model_name] = pickle.load(f)
    
    X_test = scaler.transform(data['X_test'])
    y_test = data['y_test']
    
    return models, X_test, y_test, scaler

def compute_gradient(model, X, y):
    """
    Compute gradient of loss w.r.t. input for gradient-based attacks.
    Uses finite differences for non-differentiable models.
    """
    n_samples, n_features = X.shape
    gradients = np.zeros_like(X)
    epsilon_fd = 1e-4  # Finite difference epsilon
    
    # Get model predictions
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)
    else:
        # For models without predict_proba, use decision function
        pred = model.predict(X)
        probs = np.eye(3)[pred]  # One-hot encode
    
    # Compute gradients using finite differences
    for i in range(n_features):
        X_plus = X.copy()
        X_minus = X.copy()
        X_plus[:, i] += epsilon_fd
        X_minus[:, i] -= epsilon_fd
        
        if hasattr(model, 'predict_proba'):
            probs_plus = model.predict_proba(X_plus)
            probs_minus = model.predict_proba(X_minus)
        else:
            pred_plus = model.predict(X_plus)
            pred_minus = model.predict(X_minus)
            probs_plus = np.eye(3)[pred_plus]
            probs_minus = np.eye(3)[pred_minus]
        
        # Gradient of cross-entropy loss
        loss_plus = -np.sum(np.eye(3)[y] * np.log(probs_plus + 1e-10), axis=1)
        loss_minus = -np.sum(np.eye(3)[y] * np.log(probs_minus + 1e-10), axis=1)
        
        gradients[:, i] = (loss_plus - loss_minus) / (2 * epsilon_fd)
    
    return gradients

def fgsm_attack(model, X, y, epsilon):
    """
    Fast Gradient Sign Method (FGSM) attack.
    
    Args:
        model: Target classifier
        X: Input samples
        y: True labels
        epsilon: Perturbation budget
    
    Returns:
        X_adv: Adversarial samples
    """
    gradients = compute_gradient(model, X, y)
    perturbation = epsilon * np.sign(gradients)
    X_adv = X + perturbation
    
    return X_adv

def pgd_attack(model, X, y, epsilon, alpha=0.01, num_iter=40):
    """
    Projected Gradient Descent (PGD) attack.
    
    Args:
        model: Target classifier
        X: Input samples
        y: True labels
        epsilon: Perturbation budget
        alpha: Step size
        num_iter: Number of iterations
    
    Returns:
        X_adv: Adversarial samples
    """
    X_adv = X.copy()
    
    for _ in range(num_iter):
        gradients = compute_gradient(model, X_adv, y)
        X_adv = X_adv + alpha * np.sign(gradients)
        
        # Project back to epsilon ball
        perturbation = X_adv - X
        perturbation = np.clip(perturbation, -epsilon, epsilon)
        X_adv = X + perturbation
    
    return X_adv

def evaluate_attack(model, X_clean, X_adv, y, attack_name, epsilon):
    """Evaluate attack effectiveness."""
    acc_clean = accuracy_score(y, model.predict(X_clean))
    acc_adv = accuracy_score(y, model.predict(X_adv))
    
    success_rate = (acc_clean - acc_adv) / acc_clean * 100
    
    results = {
        'clean_accuracy': float(acc_clean),
        'adversarial_accuracy': float(acc_adv),
        'accuracy_drop': float(acc_clean - acc_adv),
        'attack_success_rate': float(success_rate),
        'epsilon': float(epsilon),
        'attack_name': attack_name
    }
    
    return results

def main():
    """Main execution function."""
    print("="*70)
    print("Standard Adversarial Attacks (FGSM & PGD)")
    print("="*70)
    
    # Create output directory
    Path('data/adversarial').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    # Load models and data
    print("\n1. Loading models and data...")
    models, X_test, y_test, scaler = load_models_and_data()
    print(f"   Test samples: {len(X_test)}")
    print(f"   Models loaded: {list(models.keys())}")
    
    # Test on subset for efficiency
    n_test = min(500, len(X_test))
    X_test_sample = X_test[:n_test]
    y_test_sample = y_test[:n_test]
    print(f"   Using {n_test} samples for attack evaluation")
    
    # Attack configurations
    epsilons = [0.3, 1.0, 3.0]
    attacks = {
        'FGSM': fgsm_attack,
        'PGD': pgd_attack
    }
    
    # Run attacks
    print("\n2. Generating adversarial attacks...")
    all_results = {}
    
    for model_name, model in models.items():
        print(f"\n   Target Model: {model_name.replace('_', ' ').title()}")
        model_results = {}
        
        for attack_name, attack_fn in attacks.items():
            for eps in epsilons:
                print(f"      {attack_name} (ε={eps})...", end=' ')
                
                # Generate adversarial examples
                if attack_name == 'FGSM':
                    X_adv = attack_fn(model, X_test_sample, y_test_sample, eps)
                else:  # PGD
                    X_adv = attack_fn(model, X_test_sample, y_test_sample, eps, 
                                     alpha=eps/10, num_iter=40)
                
                # Evaluate
                results = evaluate_attack(model, X_test_sample, X_adv, y_test_sample,
                                         attack_name, eps)
                
                print(f"Success: {results['attack_success_rate']:.1f}%")
                
                # Save adversarial examples
                adv_filename = f"data/adversarial/{model_name}_{attack_name.lower()}_eps{eps}.npy"
                np.save(adv_filename, X_adv)
                
                # Store results
                key = f"{attack_name}_eps_{eps}"
                model_results[key] = results
        
        all_results[model_name] = model_results
    
    # Save results
    with open('results/adversarial_attack_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: Attack Success Rates")
    print("="*70)
    
    for model_name, model_results in all_results.items():
        print(f"\n{model_name.replace('_', ' ').title()}:")
        print(f"  {'Attack':15} {'Epsilon':>10} {'Clean Acc':>12} {'Adv Acc':>12} {'Success %':>12}")
        print("  " + "-"*65)
        
        for key, results in model_results.items():
            print(f"  {results['attack_name']:15} "
                  f"{results['epsilon']:>10.1f} "
                  f"{results['clean_accuracy']:>12.4f} "
                  f"{results['adversarial_accuracy']:>12.4f} "
                  f"{results['attack_success_rate']:>12.1f}%")
    
    print("\n" + "="*70)
    print("Files saved:")
    print("  - data/adversarial/*.npy (adversarial samples)")
    print("  - results/adversarial_attack_results.json")
    print("="*70)

if __name__ == '__main__':
    main()
