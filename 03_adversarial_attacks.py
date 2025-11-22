"""
VERSION 2: Adversarial Attack Implementation
Standard FGSM and PGD attacks on MANET IDS models
NumPy-based implementation using numerical gradients
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import json

def compute_gradient(model, X, y, epsilon=1e-7):
    """
    Compute numerical gradient for sklearn models
    
    Args:
        model: sklearn model
        X: Input samples
        y: True labels
        epsilon: Small value for numerical differentiation
    
    Returns:
        gradients: Approximate gradients
    """
    n_samples, n_features = X.shape
    gradients = np.zeros_like(X)
    
    # Get baseline predictions
    probs = model.predict_proba(X)
    baseline_loss = -np.log(probs[np.arange(len(y)), y] + 1e-10)
    
    # Compute gradient for each feature
    for j in range(n_features):
        X_perturbed = X.copy()
        X_perturbed[:, j] += epsilon
        
        probs_perturbed = model.predict_proba(X_perturbed)
        perturbed_loss = -np.log(probs_perturbed[np.arange(len(y)), y] + 1e-10)
        
        gradients[:, j] = (perturbed_loss - baseline_loss) / epsilon
    
    return gradients

def fgsm_attack(model, X, y, epsilon=0.1):
    """
    Fast Gradient Sign Method (FGSM) attack
    
    Args:
        model: Target model
        X: Input samples (numpy array)
        y: True labels (numpy array)
        epsilon: Perturbation magnitude
    
    Returns:
        X_adv: Adversarial samples
    """
    # Compute gradients
    gradients = compute_gradient(model, X, y)
    
    # Generate adversarial examples
    X_adv = X + epsilon * np.sign(gradients)
    
    return X_adv

def pgd_attack(model, X, y, epsilon=0.1, alpha=0.01, num_iter=10):
    """
    Projected Gradient Descent (PGD) attack
    
    Args:
        model: Target model
        X: Input samples (numpy array)
        y: True labels (numpy array)
        epsilon: Maximum perturbation
        alpha: Step size
        num_iter: Number of iterations
    
    Returns:
        X_adv: Adversarial samples
    """
    # Initialize with random perturbation
    X_adv = X + np.random.uniform(-epsilon, epsilon, X.shape)
    X_adv = np.clip(X_adv, X - epsilon, X + epsilon)
    
    # Iterative attack
    for i in range(num_iter):
        # Compute gradients
        gradients = compute_gradient(model, X_adv, y)
        
        # Update adversarial examples
        X_adv = X_adv + alpha * np.sign(gradients)
        
        # Project back to epsilon ball
        X_adv = np.clip(X_adv, X - epsilon, X + epsilon)
    
    return X_adv

def evaluate_adversarial_attacks(model_name, epsilon=0.3):
    """Evaluate both FGSM and PGD attacks on a model"""
    
    print(f"\n{'='*60}")
    print(f"🎯 ATTACKING: {model_name.upper()}")
    print(f"{'='*60}")
    
    # Load model and data
    with open(f'models/{model_name}.pkl', 'rb') as f:
        sklearn_model = pickle.load(f)
    
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('data/train_test_split.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    y_test_np = y_test.values
    
    # Baseline accuracy
    y_pred_clean = sklearn_model.predict(X_test_scaled)
    clean_accuracy = accuracy_score(y_test_np, y_pred_clean)
    print(f"\n📊 Clean Accuracy: {clean_accuracy*100:.2f}%")
    
    # FGSM Attack
    print(f"\n🔴 FGSM Attack (ε={epsilon})...")
    print(f"   Computing gradients...")
    X_fgsm = fgsm_attack(sklearn_model, X_test_scaled, y_test_np, epsilon=epsilon)
    y_pred_fgsm = sklearn_model.predict(X_fgsm)
    fgsm_accuracy = accuracy_score(y_test_np, y_pred_fgsm)
    fgsm_success = 1 - fgsm_accuracy
    print(f"   Adversarial Accuracy: {fgsm_accuracy*100:.2f}%")
    print(f"   Attack Success Rate: {fgsm_success*100:.2f}%")
    
    # PGD Attack
    print(f"\n🔴 PGD Attack (ε={epsilon}, α=0.01, iter=10)...")
    print(f"   Running iterative optimization...")
    X_pgd = pgd_attack(sklearn_model, X_test_scaled, y_test_np, 
                       epsilon=epsilon, alpha=0.01, num_iter=10)
    y_pred_pgd = sklearn_model.predict(X_pgd)
    pgd_accuracy = accuracy_score(y_test_np, y_pred_pgd)
    pgd_success = 1 - pgd_accuracy
    print(f"   Adversarial Accuracy: {pgd_accuracy*100:.2f}%")
    print(f"   Attack Success Rate: {pgd_success*100:.2f}%")
    
    # Save adversarial samples for manifold analysis
    np.save(f'data/adversarial/{model_name}_fgsm.npy', X_fgsm)
    np.save(f'data/adversarial/{model_name}_pgd.npy', X_pgd)
    
    results = {
        'model': model_name,
        'epsilon': epsilon,
        'clean_accuracy': float(clean_accuracy),
        'fgsm': {
            'adversarial_accuracy': float(fgsm_accuracy),
            'attack_success_rate': float(fgsm_success)
        },
        'pgd': {
            'adversarial_accuracy': float(pgd_accuracy),
            'attack_success_rate': float(pgd_success)
        }
    }
    
    return results, X_fgsm, X_pgd, X_test_scaled

if __name__ == "__main__":
    import os
    os.makedirs('data/adversarial', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Test on all three models
    all_results = {}
    epsilon = 0.3
    
    for model_name in ['logistic_regression', 'decision_tree', 'xgboost']:
        results, X_fgsm, X_pgd, X_clean = evaluate_adversarial_attacks(model_name, epsilon)
        all_results[model_name] = results
    
    # Save comprehensive results
    with open('results/adversarial_attack_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✅ ADVERSARIAL ATTACKS COMPLETE")
    print(f"{'='*60}")
    
    # Summary
    print("\n📊 ATTACK SUCCESS RATES:")
    for model in all_results:
        fgsm_sr = all_results[model]['fgsm']['attack_success_rate']
        pgd_sr = all_results[model]['pgd']['attack_success_rate']
        print(f"   {model.replace('_', ' ').title()}:")
        print(f"      FGSM: {fgsm_sr*100:.1f}%  |  PGD: {pgd_sr*100:.1f}%")
