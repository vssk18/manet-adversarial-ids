"""
VERSION 2: Epsilon Sweep Analysis
Test multiple epsilon values to find off-manifold threshold
"""

import numpy as np
import pickle
from sklearn.neighbors import KDTree
from sklearn.metrics import accuracy_score
import json
import matplotlib.pyplot as plt

def load_models_and_data():
    """Load all necessary models and data"""
    
    with open('data/train_test_split.pkl', 'rb') as f:
        data = pickle.load(f)
    
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    models = {}
    for model_name in ['logistic_regression', 'decision_tree', 'xgboost']:
        with open(f'models/{model_name}.pkl', 'rb') as f:
            models[model_name] = pickle.load(f)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return models, X_train_scaled, X_test_scaled, y_train.values, y_test.values

def compute_gradient(model, X, y, epsilon=1e-7):
    """Compute numerical gradient"""
    n_samples, n_features = X.shape
    gradients = np.zeros_like(X)
    
    probs = model.predict_proba(X)
    baseline_loss = -np.log(probs[np.arange(len(y)), y] + 1e-10)
    
    for j in range(n_features):
        X_perturbed = X.copy()
        X_perturbed[:, j] += epsilon
        
        probs_perturbed = model.predict_proba(X_perturbed)
        perturbed_loss = -np.log(probs_perturbed[np.arange(len(y)), y] + 1e-10)
        
        gradients[:, j] = (perturbed_loss - baseline_loss) / epsilon
    
    return gradients

def fgsm_attack(model, X, y, epsilon=0.1):
    """Fast Gradient Sign Method"""
    gradients = compute_gradient(model, X, y)
    X_adv = X + epsilon * np.sign(gradients)
    return X_adv

def evaluate_epsilon(model, model_name, X_test, y_test, kdtree, baseline_distance, epsilon):
    """Evaluate a single epsilon value"""
    
    print(f"\n{'='*60}")
    print(f"Testing ε={epsilon:.2f} on {model_name}")
    print(f"{'='*60}")
    
    # Generate adversarial samples
    print("  Generating FGSM adversarials...")
    X_adv = fgsm_attack(model, X_test, y_test, epsilon=epsilon)
    
    # Measure attack success
    y_pred_clean = model.predict(X_test)
    y_pred_adv = model.predict(X_adv)
    
    clean_acc = accuracy_score(y_test, y_pred_clean)
    adv_acc = accuracy_score(y_test, y_pred_adv)
    success_rate = 1 - adv_acc
    
    print(f"  Clean Accuracy: {clean_acc*100:.2f}%")
    print(f"  Adversarial Accuracy: {adv_acc*100:.2f}%")
    print(f"  Attack Success: {success_rate*100:.2f}%")
    
    # Measure manifold distance
    distances, _ = kdtree.query(X_adv, k=1)
    mean_distance = np.mean(distances)
    ratio = mean_distance / baseline_distance
    
    print(f"  Mean 1-NN distance: {mean_distance:.6f}")
    print(f"  Distance ratio: {ratio:.2f}x")
    
    if ratio > 10:
        print(f"  ⚠️  OFF-MANIFOLD: {ratio:.1f}x beyond realistic traffic!")
    elif ratio > 2:
        print(f"  ⚠️  Moderately off-manifold")
    else:
        print(f"  ✅ On-manifold")
    
    return {
        'epsilon': float(epsilon),
        'clean_accuracy': float(clean_acc),
        'adversarial_accuracy': float(adv_acc),
        'success_rate': float(success_rate),
        'mean_distance': float(mean_distance),
        'distance_ratio': float(ratio),
        'status': 'off-manifold' if ratio > 10 else ('moderate' if ratio > 2 else 'on-manifold')
    }

def epsilon_sweep_analysis():
    """Test multiple epsilon values to find off-manifold threshold"""
    
    print("\n" + "="*60)
    print("🔬 EPSILON SWEEP ANALYSIS")
    print("="*60)
    
    # Load data and models
    print("\n📂 Loading models and data...")
    models, X_train, X_test, y_train, y_test = load_models_and_data()
    
    # Build KD-tree
    print("🌳 Building KD-tree...")
    kdtree = KDTree(X_train, leaf_size=30)
    
    # Get baseline distance
    distances_baseline, _ = kdtree.query(X_test, k=1)
    baseline_distance = np.mean(distances_baseline)
    print(f"✅ Baseline distance: {baseline_distance:.6f}")
    
    # Test multiple epsilon values
    epsilon_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    
    results = {}
    
    # Test on logistic regression (most interpretable)
    model_name = 'logistic_regression'
    model = models[model_name]
    
    print(f"\n{'='*60}")
    print(f"TESTING MODEL: {model_name.upper()}")
    print(f"{'='*60}")
    
    model_results = []
    
    for eps in epsilon_values:
        result = evaluate_epsilon(model, model_name, X_test, y_test, 
                                   kdtree, baseline_distance, eps)
        model_results.append(result)
    
    results[model_name] = model_results
    
    return results, epsilon_values

def plot_epsilon_sweep(results, epsilon_values):
    """Create visualization of epsilon sweep results"""
    
    print("\n📊 Creating visualizations...")
    
    model_name = 'logistic_regression'
    model_results = results[model_name]
    
    # Extract data
    success_rates = [r['success_rate'] * 100 for r in model_results]
    distance_ratios = [r['distance_ratio'] for r in model_results]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Attack Success Rate vs Epsilon
    ax1.plot(epsilon_values, success_rates, 'o-', linewidth=2, markersize=8, color='#e74c3c')
    ax1.set_xlabel('Epsilon (ε)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Attack Effectiveness vs Perturbation Size', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])
    
    # Plot 2: Distance Ratio vs Epsilon
    ax2.plot(epsilon_values, distance_ratios, 'o-', linewidth=2, markersize=8, color='#3498db')
    ax2.axhline(y=2, color='orange', linestyle='--', linewidth=2, label='Moderate threshold')
    ax2.axhline(y=10, color='red', linestyle='--', linewidth=2, label='Off-manifold threshold')
    ax2.set_xlabel('Epsilon (ε)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Distance Ratio (vs clean data)', fontsize=12, fontweight='bold')
    ax2.set_title('Manifold Distance vs Perturbation Size', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('results/figures/epsilon_sweep_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: results/figures/epsilon_sweep_analysis.png")
    
    plt.close()

def summarize_findings(results, epsilon_values):
    """Print summary of epsilon sweep findings"""
    
    print("\n" + "="*60)
    print("🎯 EPSILON SWEEP SUMMARY")
    print("="*60)
    
    model_name = 'logistic_regression'
    model_results = results[model_name]
    
    print(f"\n{'Epsilon':<10} {'Success Rate':<15} {'Distance Ratio':<18} {'Status':<15}")
    print("-" * 70)
    
    for i, eps in enumerate(epsilon_values):
        r = model_results[i]
        print(f"{eps:<10.2f} {r['success_rate']*100:>6.2f}%        "
              f"{r['distance_ratio']:>6.2f}x            {r['status']:<15}")
    
    # Find threshold
    threshold_idx = None
    for i, r in enumerate(model_results):
        if r['distance_ratio'] > 10:
            threshold_idx = i
            break
    
    if threshold_idx is not None:
        threshold_eps = epsilon_values[threshold_idx]
        print("\n" + "="*60)
        print("💡 KEY FINDING:")
        print("="*60)
        print(f"At ε ≥ {threshold_eps:.2f}, adversarial samples become OFF-MANIFOLD")
        print(f"({model_results[threshold_idx]['distance_ratio']:.1f}x beyond realistic MANET traffic)")
        print("\nThis represents PHYSICALLY IMPOSSIBLE network conditions that")
        print("would never occur in real-world MANET deployments.")
        print("\n➡️  IMPLICATION: Standard adversarial attacks with large ε are")
        print("   unrealistic for evaluating MANET IDS robustness!")
    else:
        print("\n" + "="*60)
        print("💡 KEY FINDING:")
        print("="*60)
        print("All tested epsilon values remain ON-MANIFOLD or moderately off.")
        print("For this dataset, even ε=3.0 produces semi-realistic perturbations.")
        print("\n➡️  IMPLICATION: Feature scaling affects manifold structure.")
        print("   Need to test in UNSCALED feature space for true physical constraints!")

if __name__ == "__main__":
    import os
    os.makedirs('results/figures', exist_ok=True)
    
    # Run epsilon sweep
    results, epsilon_values = epsilon_sweep_analysis()
    
    # Save results
    with open('results/epsilon_sweep_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n💾 Results saved to results/epsilon_sweep_results.json")
    
    # Create visualizations
    plot_epsilon_sweep(results, epsilon_values)
    
    # Print summary
    summarize_findings(results, epsilon_values)
    
    print("\n✅ EPSILON SWEEP COMPLETE")
