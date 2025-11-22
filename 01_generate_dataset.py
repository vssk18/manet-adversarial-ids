#!/usr/bin/env python3
"""
01_generate_dataset.py
======================
Generate MANET intrusion detection dataset with group-safe splitting.

This script creates a synthetic MANET dataset with 4,500 network flow samples
across 3 classes (Normal, Flooding DoS, Blackhole) and performs group-safe
train/test splitting to prevent data leakage.

Author: V.S.S. Karthik
Date: November 2024
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_SAMPLES_PER_CLASS = 1500
N_FEATURES = 10
TEST_SIZE = 0.33

# Feature definitions with domain constraints
FEATURE_SPECS = {
    'packet_rate': {'min': 0, 'max': 1000, 'unit': 'packets/sec', 'max_epsilon': 10},
    'byte_count': {'min': 64, 'max': 65535, 'unit': 'bytes', 'max_epsilon': 100},
    'flow_duration': {'min': 0, 'max': 300, 'unit': 'seconds', 'max_epsilon': 1},
    'inter_arrival_time': {'min': 0, 'max': 1, 'unit': 'seconds', 'max_epsilon': 0.01},
    'packet_size_variance': {'min': 0, 'max': 2000, 'unit': 'bytes', 'max_epsilon': 50},
    'protocol_type': {'min': 6, 'max': 17, 'unit': 'discrete', 'max_epsilon': 0},
    'hop_count': {'min': 1, 'max': 15, 'unit': 'hops', 'max_epsilon': 1},
    'route_changes': {'min': 0, 'max': 10, 'unit': 'count', 'max_epsilon': 1},
    'signal_strength': {'min': -100, 'max': -20, 'unit': 'dBm', 'max_epsilon': 2},
    'node_mobility': {'min': 0, 'max': 20, 'unit': 'm/s', 'max_epsilon': 1},
}

def generate_normal_traffic(n_samples):
    """Generate normal MANET traffic patterns."""
    data = np.zeros((n_samples, N_FEATURES))
    
    data[:, 0] = np.random.gamma(2, 50, n_samples)  # packet_rate
    data[:, 1] = np.random.normal(1500, 200, n_samples)  # byte_count
    data[:, 2] = np.random.exponential(10, n_samples)  # flow_duration
    data[:, 3] = np.random.exponential(0.1, n_samples)  # inter_arrival_time
    data[:, 4] = np.random.gamma(2, 100, n_samples)  # packet_size_variance
    data[:, 5] = np.random.choice([6, 17], n_samples, p=[0.7, 0.3])  # protocol_type
    data[:, 6] = np.random.poisson(3, n_samples) + 1  # hop_count
    data[:, 7] = np.random.poisson(1, n_samples)  # route_changes
    data[:, 8] = np.random.normal(-60, 10, n_samples)  # signal_strength
    data[:, 9] = np.random.gamma(1.5, 2, n_samples)  # node_mobility
    
    return data

def generate_flooding_dos(n_samples):
    """Generate flooding DoS attack patterns."""
    data = np.zeros((n_samples, N_FEATURES))
    
    data[:, 0] = np.random.gamma(8, 100, n_samples)  # High packet_rate
    data[:, 1] = np.random.normal(500, 100, n_samples)  # Small byte_count
    data[:, 2] = np.random.exponential(5, n_samples)  # Short flow_duration
    data[:, 3] = np.random.exponential(0.01, n_samples)  # Very small IAT
    data[:, 4] = np.random.gamma(1, 50, n_samples)  # Low variance
    data[:, 5] = np.random.choice([6, 17], n_samples, p=[0.5, 0.5])  # protocol_type
    data[:, 6] = np.random.poisson(3, n_samples) + 1  # hop_count
    data[:, 7] = np.random.poisson(2, n_samples)  # Higher route_changes
    data[:, 8] = np.random.normal(-60, 10, n_samples)  # signal_strength
    data[:, 9] = np.random.gamma(1.5, 2, n_samples)  # node_mobility
    
    return data

def generate_blackhole(n_samples):
    """Generate blackhole attack patterns."""
    data = np.zeros((n_samples, N_FEATURES))
    
    data[:, 0] = np.random.gamma(1, 30, n_samples)  # Low packet_rate
    data[:, 1] = np.random.normal(800, 150, n_samples)  # Medium byte_count
    data[:, 2] = np.random.exponential(3, n_samples)  # Very short duration
    data[:, 3] = np.random.exponential(0.2, n_samples)  # High IAT
    data[:, 4] = np.random.gamma(3, 80, n_samples)  # Higher variance
    data[:, 5] = np.random.choice([6, 17], n_samples, p=[0.6, 0.4])  # protocol_type
    data[:, 6] = np.random.poisson(2, n_samples) + 1  # Lower hop_count
    data[:, 7] = np.random.poisson(4, n_samples)  # High route_changes
    data[:, 8] = np.random.normal(-70, 15, n_samples)  # Weaker signal
    data[:, 9] = np.random.gamma(2, 3, n_samples)  # Higher mobility
    
    return data

def apply_constraints(data):
    """Apply domain constraints to ensure realistic values."""
    feature_names = list(FEATURE_SPECS.keys())
    
    for i, fname in enumerate(feature_names):
        spec = FEATURE_SPECS[fname]
        data[:, i] = np.clip(data[:, i], spec['min'], spec['max'])
        
        # Round discrete features
        if spec['unit'] == 'discrete' or fname in ['protocol_type', 'hop_count', 'route_changes']:
            data[:, i] = np.round(data[:, i])
    
    return data

def create_session_groups(n_samples, n_groups_per_class=50):
    """Create session groups for group-safe splitting."""
    samples_per_group = n_samples // n_groups_per_class
    groups = np.repeat(np.arange(n_groups_per_class), samples_per_group)
    
    # Handle remainder
    remainder = n_samples - len(groups)
    if remainder > 0:
        groups = np.concatenate([groups, np.full(remainder, n_groups_per_class - 1)])
    
    return groups

def main():
    """Main execution function."""
    print("="*70)
    print("MANET IDS Dataset Generation with Group-Safe Splitting")
    print("="*70)
    
    # Create output directories
    Path('data').mkdir(exist_ok=True)
    Path('data/adversarial').mkdir(exist_ok=True)
    
    # Generate data for each class
    print("\n1. Generating traffic patterns...")
    normal_data = generate_normal_traffic(N_SAMPLES_PER_CLASS)
    flooding_data = generate_flooding_dos(N_SAMPLES_PER_CLASS)
    blackhole_data = generate_blackhole(N_SAMPLES_PER_CLASS)
    
    # Apply constraints
    print("2. Applying domain constraints...")
    normal_data = apply_constraints(normal_data)
    flooding_data = apply_constraints(flooding_data)
    blackhole_data = apply_constraints(blackhole_data)
    
    # Combine all data
    X = np.vstack([normal_data, flooding_data, blackhole_data])
    y = np.array([0]*N_SAMPLES_PER_CLASS + [1]*N_SAMPLES_PER_CLASS + [2]*N_SAMPLES_PER_CLASS)
    
    # Create session groups
    print("3. Creating session groups for group-safe splitting...")
    groups_per_class = create_session_groups(N_SAMPLES_PER_CLASS)
    groups = np.concatenate([
        groups_per_class,
        groups_per_class + 50,
        groups_per_class + 100
    ])
    
    # Group-safe train/test split
    print("4. Performing group-safe train/test split...")
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Save dataset
    print("5. Saving dataset and splits...")
    
    # Full dataset as CSV
    feature_names = list(FEATURE_SPECS.keys())
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y
    df['session_group'] = groups
    df.to_csv('data/manet_dataset_full.csv', index=False)
    
    # Train/test splits as pickle
    with open('data/train_test_split.pkl', 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'train_idx': train_idx,
            'test_idx': test_idx
        }, f)
    
    # Feature metadata
    with open('data/feature_names.pkl', 'wb') as f:
        pickle.dump({
            'feature_names': feature_names,
            'feature_specs': FEATURE_SPECS,
            'class_names': ['Normal', 'Flooding DoS', 'Blackhole']
        }, f)
    
    # Print statistics
    print("\n" + "="*70)
    print("Dataset Generation Complete!")
    print("="*70)
    print(f"\nTotal samples: {len(X)}")
    print(f"Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Testing samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    print(f"\nClass distribution:")
    print(f"  Normal: {np.sum(y == 0)} samples")
    print(f"  Flooding DoS: {np.sum(y == 1)} samples")
    print(f"  Blackhole: {np.sum(y == 2)} samples")
    print(f"\nFeatures: {N_FEATURES}")
    print(f"Session groups: {len(np.unique(groups))}")
    print("\nFiles saved:")
    print("  - data/manet_dataset_full.csv")
    print("  - data/train_test_split.pkl")
    print("  - data/feature_names.pkl")
    print("="*70)

if __name__ == '__main__':
    main()
