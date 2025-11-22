"""
VERSION 2: MANET Adversarial IDS Dataset Generation
Group-safe splitting to prevent data leakage
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import pickle

# Set random seed for reproducibility
np.random.seed(42)

def generate_manet_traffic(n_samples, attack_type='normal'):
    """Generate realistic MANET network traffic features"""
    
    if attack_type == 'normal':
        # Normal MANET traffic characteristics
        data = {
            'packet_rate': np.random.normal(50, 15, n_samples),
            'byte_count': np.random.normal(1500, 400, n_samples),
            'flow_duration': np.random.exponential(5, n_samples),
            'inter_arrival_time': np.random.exponential(0.02, n_samples),
            'packet_size_variance': np.random.normal(200, 50, n_samples),
            'protocol_type': np.random.choice([6, 17], n_samples, p=[0.7, 0.3]),  # TCP/UDP
            'hop_count': np.random.poisson(3, n_samples),
            'route_changes': np.random.poisson(1, n_samples),
            'signal_strength': np.random.normal(-60, 10, n_samples),
            'node_mobility': np.random.uniform(0, 5, n_samples),
        }
    
    elif attack_type == 'flooding':
        # DoS flooding attack characteristics
        data = {
            'packet_rate': np.random.normal(500, 100, n_samples),  # Very high rate
            'byte_count': np.random.normal(1000, 200, n_samples),  # Smaller packets
            'flow_duration': np.random.exponential(2, n_samples),  # Shorter flows
            'inter_arrival_time': np.random.exponential(0.002, n_samples),  # Very fast
            'packet_size_variance': np.random.normal(50, 20, n_samples),  # Low variance
            'protocol_type': np.random.choice([17], n_samples),  # Mostly UDP
            'hop_count': np.random.poisson(2, n_samples),  # Direct paths
            'route_changes': np.random.poisson(0, n_samples),  # Stable routes
            'signal_strength': np.random.normal(-55, 5, n_samples),  # Strong signal
            'node_mobility': np.random.uniform(0, 2, n_samples),  # Low mobility
        }
    
    elif attack_type == 'blackhole':
        # Blackhole attack characteristics
        data = {
            'packet_rate': np.random.normal(30, 10, n_samples),  # Lower rate
            'byte_count': np.random.normal(1200, 300, n_samples),
            'flow_duration': np.random.exponential(1, n_samples),  # Very short
            'inter_arrival_time': np.random.exponential(0.05, n_samples),
            'packet_size_variance': np.random.normal(150, 40, n_samples),
            'protocol_type': np.random.choice([6, 17], n_samples, p=[0.5, 0.5]),
            'hop_count': np.random.poisson(5, n_samples),  # More hops
            'route_changes': np.random.poisson(4, n_samples),  # Frequent changes
            'signal_strength': np.random.normal(-70, 15, n_samples),  # Weak signal
            'node_mobility': np.random.uniform(3, 8, n_samples),  # High mobility
        }
    
    return pd.DataFrame(data)

def create_complete_dataset(samples_per_class=1500):
    """Create complete dataset with group IDs for safe splitting"""
    
    print("🔧 Generating MANET traffic dataset...")
    print(f"   Samples per class: {samples_per_class}")
    
    # Generate each class
    normal = generate_manet_traffic(samples_per_class, 'normal')
    normal['label'] = 0
    normal['attack_type'] = 'normal'
    
    flooding = generate_manet_traffic(samples_per_class, 'flooding')
    flooding['label'] = 1
    flooding['attack_type'] = 'flooding'
    
    blackhole = generate_manet_traffic(samples_per_class, 'blackhole')
    blackhole['label'] = 2
    blackhole['attack_type'] = 'blackhole'
    
    # Combine all classes
    df = pd.concat([normal, flooding, blackhole], ignore_index=True)
    
    # Create group IDs (sessions) - critical for preventing leakage
    # Each group represents a network session with 10-20 related flows
    n_groups = len(df) // 15  # ~15 flows per session
    group_ids = []
    for i in range(n_groups):
        group_size = np.random.randint(10, 21)
        group_ids.extend([i] * group_size)
    
    # Pad to match dataset size
    while len(group_ids) < len(df):
        group_ids.append(n_groups)
        n_groups += 1
    
    df['group_id'] = group_ids[:len(df)]
    
    print(f"✅ Dataset created: {len(df)} samples")
    print(f"   Classes: Normal={len(normal)}, Flooding={len(flooding)}, Blackhole={len(blackhole)}")
    print(f"   Groups (sessions): {df['group_id'].nunique()}")
    
    return df

def split_dataset_by_groups(df, test_size=0.3):
    """Split dataset by groups to prevent leakage"""
    
    print("\n🔀 Performing group-safe train/test split...")
    
    X = df.drop(['label', 'attack_type', 'group_id'], axis=1)
    y = df['label']
    groups = df['group_id']
    
    # Group-aware split
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"✅ Split complete:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test:  {len(X_test)} samples")
    print(f"   Train groups: {groups.iloc[train_idx].nunique()}")
    print(f"   Test groups:  {groups.iloc[test_idx].nunique()}")
    
    # Verify no group overlap
    train_groups = set(groups.iloc[train_idx].unique())
    test_groups = set(groups.iloc[test_idx].unique())
    overlap = train_groups.intersection(test_groups)
    
    if len(overlap) > 0:
        raise ValueError(f"❌ Group leakage detected! {len(overlap)} overlapping groups")
    else:
        print("✅ No group leakage - train/test groups are completely separate")
    
    return X_train, X_test, y_train, y_test

def save_dataset(df, X_train, X_test, y_train, y_test):
    """Save all dataset components"""
    
    print("\n💾 Saving dataset...")
    
    # Save full dataset
    df.to_csv('data/manet_dataset_full.csv', index=False)
    print("   ✅ Full dataset saved")
    
    # Save train/test splits
    with open('data/train_test_split.pkl', 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }, f)
    print("   ✅ Train/test splits saved")
    
    # Save feature names
    feature_names = X_train.columns.tolist()
    with open('data/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    print("   ✅ Feature names saved")
    
    print("\n✅ Dataset generation complete!")

if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    
    # Generate dataset
    df = create_complete_dataset(samples_per_class=1500)
    
    # Split by groups
    X_train, X_test, y_train, y_test = split_dataset_by_groups(df)
    
    # Save everything
    save_dataset(df, X_train, X_test, y_train, y_test)
    
    print("\n" + "="*60)
    print("DATASET READY FOR MODEL TRAINING")
    print("="*60)
