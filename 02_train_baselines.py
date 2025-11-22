"""
VERSION 2: Train Baseline Models (LR, DT, XGB)
Clean accuracy evaluation before adversarial attacks
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

def load_data():
    """Load the group-safe split dataset"""
    print("📂 Loading dataset...")
    
    with open('data/train_test_split.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test:  {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """Scale features - fit only on training data"""
    print("\n📊 Scaling features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("   ✅ Features scaled and scaler saved")
    
    return X_train_scaled, X_test_scaled, scaler

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train Logistic Regression baseline"""
    print("\n" + "="*60)
    print("🔵 LOGISTIC REGRESSION")
    print("="*60)
    
    model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Test Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Normal', 'Flooding', 'Blackhole']))
    
    # Save model
    with open('models/logistic_regression.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model, accuracy

def train_decision_tree(X_train, y_train, X_test, y_test):
    """Train Decision Tree baseline"""
    print("\n" + "="*60)
    print("🌳 DECISION TREE")
    print("="*60)
    
    model = DecisionTreeClassifier(max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Test Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Normal', 'Flooding', 'Blackhole']))
    
    # Save model
    with open('models/decision_tree.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model, accuracy

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost baseline"""
    print("\n" + "="*60)
    print("⚡ XGBOOST")
    print("="*60)
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Test Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Normal', 'Flooding', 'Blackhole']))
    
    # Save model
    with open('models/xgboost.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model, accuracy

def save_baseline_results(results):
    """Save baseline performance metrics"""
    print("\n💾 Saving baseline results...")
    
    with open('results/baseline_performance.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("   ✅ Results saved to results/baseline_performance.json")

if __name__ == "__main__":
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train all baseline models
    results = {}
    
    lr_model, lr_acc = train_logistic_regression(X_train_scaled, y_train, 
                                                   X_test_scaled, y_test)
    results['logistic_regression'] = {
        'accuracy': float(lr_acc),
        'model_type': 'Logistic Regression'
    }
    
    dt_model, dt_acc = train_decision_tree(X_train_scaled, y_train, 
                                            X_test_scaled, y_test)
    results['decision_tree'] = {
        'accuracy': float(dt_acc),
        'model_type': 'Decision Tree'
    }
    
    xgb_model, xgb_acc = train_xgboost(X_train_scaled, y_train, 
                                        X_test_scaled, y_test)
    results['xgboost'] = {
        'accuracy': float(xgb_acc),
        'model_type': 'XGBoost'
    }
    
    # Save results
    save_baseline_results(results)
    
    print("\n" + "="*60)
    print("✅ BASELINE TRAINING COMPLETE")
    print("="*60)
    print(f"Logistic Regression: {lr_acc*100:.2f}%")
    print(f"Decision Tree:       {dt_acc*100:.2f}%")
    print(f"XGBoost:             {xgb_acc*100:.2f}%")
    print("="*60)
