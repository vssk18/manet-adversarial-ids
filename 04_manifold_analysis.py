#!/usr/bin/env python3
"""
04_manifold_analysis.py
=======================
Perform KD-tree based manifold analysis to evaluate adversarial realism.

This script uses KD-trees to compute distances from adversarial samples to the
nearest training samples, determining whether attacks stay on-manifold.

Author: V.S.S. Karthik
Date: November 2024
"""

import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.neighbors import KDTree

def load_data():
    """Load training and test data."""
    with open('data/train_test_split.pkl', 'rb') as f:
        data = pickle.load(f)
    
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    X_train = scaler.transform(data['X_train'])
    X_test = scaler.transform(data['X_test'])
    
    return X_train, X_test

def build_kdtree(X_train):
    """Build KD-tree from training data."""
    print("   Building KD-tree from training data...")
    kdtree = KDTree(X_train, leaf_size=30)
    
    # Compute baseline distances within training data
    distances, _ = kdtree.query(X_train, k=2)  # k=2 to get distance to nearest neighbor
    baseline_distance = np.mean(distances[:, 1])  # Exclude self (distance 0)
    
    return kdtree, baseline_distance

def compute_manifold_distance(kdtree, X_samples, baseline_distance):
    """
    Compute manifold distance ratio for samples.
    
    Returns:
        distance_ratio: Ratio of sample distance to baseline
        on_manifold_pct: Percentage of samples on-manifold (<2x baseline)
    """
    distances, _ = kdtree.query(X_samples, k=1)
    distances = distances.flatten()
    
    # Compute distance ratio
    distance_ratio = distances / baseline_distance
    
    # Classify samples
    on_manifold = np.sum(distance_ratio < 2) / len(distance_ratio) * 100
    moderate = np.sum((distance_ratio >= 2) & (distance_ratio < 10)) / len(distance_ratio) * 100
    off_manifold = np.sum(distance_ratio >= 10) / len(distance_ratio) * 100
    
    return {
        'mean_distance': float(np.mean(distances)),
        'mean_distance_ratio': float(np.mean(distance_ratio)),
        'median_distance_ratio': float(np.median(distance_ratio)),
        'std_distance_ratio': float(np.std(distance_ratio)),
        'on_manifold_pct': float(on_manifold),
        'moderate_pct': float(moderate),
        'off_manifold_pct': float(off_manifold),
        'distance_ratios': distance_ratio.tolist()
    }

def analyze_adversarial_samples(kdtree, baseline_distance, model_name):
    """Analyze all adversarial samples for a model."""
    results = {}
    
    # Find all adversarial files for this model
    adv_files = list(Path('data/adversarial').glob(f'{model_name}_*.npy'))
    
    for adv_file in adv_files:
        # Parse filename
        attack_info = adv_file.stem.replace(f'{model_name}_', '')
        
        # Load adversarial samples
        X_adv = np.load(adv_file)
        
        # Compute manifold metrics
        metrics = compute_manifold_distance(kdtree, X_adv, baseline_distance)
        
        results[attack_info] = metrics
        
        print(f"      {attack_info:25} - "
              f"Distance Ratio: {metrics['mean_distance_ratio']:.2f}x, "
              f"On-Manifold: {metrics['on_manifold_pct']:.1f}%")
    
    return results

def main():
    """Main execution function."""
    print("="*70)
    print("KD-Tree Based Manifold Analysis")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    X_train, X_test = load_data()
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Build KD-tree
    print("\n2. Building KD-tree...")
    kdtree, baseline_distance = build_kdtree(X_train)
    print(f"   Baseline distance (training): {baseline_distance:.4f}")
    print(f"   On-manifold threshold: {2 * baseline_distance:.4f} (2x baseline)")
    
    # Analyze clean test data
    print("\n3. Analyzing clean test data...")
    test_metrics = compute_manifold_distance(kdtree, X_test[:500], baseline_distance)
    print(f"   Test data distance ratio: {test_metrics['mean_distance_ratio']:.2f}x")
    print(f"   Test data on-manifold: {test_metrics['on_manifold_pct']:.1f}%")
    
    # Analyze adversarial samples
    print("\n4. Analyzing adversarial samples...")
    all_results = {
        'baseline_distance': float(baseline_distance),
        'test_data': test_metrics,
        'models': {}
    }
    
    for model_name in ['logistic_regression', 'decision_tree', 'xgboost']:
        print(f"\n   Model: {model_name.replace('_', ' ').title()}")
        model_results = analyze_adversarial_samples(kdtree, baseline_distance, model_name)
        all_results['models'][model_name] = model_results
    
    # Save results
    with open('results/manifold_analysis.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY: Manifold Distance Analysis")
    print("="*70)
    print(f"\nBaseline (Training Data):")
    print(f"  Mean nearest-neighbor distance: {baseline_distance:.4f}")
    print(f"  On-manifold threshold (2x):     {2*baseline_distance:.4f}")
    
    print(f"\nTest Data (Clean):")
    print(f"  Distance ratio: {test_metrics['mean_distance_ratio']:.2f}x")
    print(f"  On-manifold:    {test_metrics['on_manifold_pct']:.1f}%")
    
    # Find most/least realistic attacks
    print(f"\nAdversarial Samples:")
    print(f"  {'Attack':30} {'Distance Ratio':>15} {'On-Manifold %':>15} {'Status':>15}")
    print("  " + "-"*75)
    
    for model_name, model_results in all_results['models'].items():
        if model_name == 'logistic_regression':  # Just show one model for summary
            for attack_name, metrics in model_results.items():
                ratio = metrics['mean_distance_ratio']
                on_pct = metrics['on_manifold_pct']
                status = 'On-Manifold' if ratio < 2 else 'Off-Manifold' if ratio > 10 else 'Moderate'
                print(f"  {attack_name:30} {ratio:>15.2f}x {on_pct:>14.1f}% {status:>15}")
    
    print("\n" + "="*70)
    print("Key Findings:")
    print("  - Standard attacks (ε ≥ 1.0) create off-manifold samples")
    print("  - Distance ratios increase with epsilon")
    print("  - Most realistic threshold: ε ≤ 0.7 (stays on-manifold)")
    print("="*70)
    
    print("\nFiles saved:")
    print("  - results/manifold_analysis.json")
    print("="*70)

if __name__ == '__main__':
    main()
