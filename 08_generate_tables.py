"""
Generate comprehensive results tables for paper
"""

import json
import pandas as pd
from tabulate import tabulate

def create_baseline_performance_table():
    """Table 1: Baseline Model Performance"""
    
    print("📊 Creating Table 1: Baseline Model Performance...")
    
    with open('results/baseline_performance.json', 'r') as f:
        results = json.load(f)
    
    data = []
    for model_name, metrics in results.items():
        clean_name = model_name.replace('_', ' ').title()
        accuracy = metrics['accuracy'] * 100
        data.append([clean_name, f"{accuracy:.2f}%"])
    
    df = pd.DataFrame(data, columns=['Model', 'Test Accuracy'])
    
    # Save as CSV
    df.to_csv('results/tables/table1_baseline_performance.csv', index=False)
    
    # Print LaTeX
    latex = df.to_latex(index=False, caption='Baseline Model Performance on Clean Test Data',
                        label='tab:baseline')
    
    with open('results/tables/table1_baseline_performance.tex', 'w') as f:
        f.write(latex)
    
    print("\n" + tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    print("\n✅ Saved: results/tables/table1_baseline_performance.csv")
    print("✅ Saved: results/tables/table1_baseline_performance.tex")

def create_standard_attacks_table():
    """Table 2: Standard Adversarial Attack Results"""
    
    print("\n📊 Creating Table 2: Standard Adversarial Attacks...")
    
    with open('results/adversarial_attack_results.json', 'r') as f:
        results = json.load(f)
    
    data = []
    for model_name, metrics in results.items():
        clean_name = model_name.replace('_', ' ').title()
        clean_acc = metrics['clean_accuracy'] * 100
        fgsm_acc = metrics['fgsm']['adversarial_accuracy'] * 100
        fgsm_sr = metrics['fgsm']['attack_success_rate'] * 100
        pgd_acc = metrics['pgd']['adversarial_accuracy'] * 100
        pgd_sr = metrics['pgd']['attack_success_rate'] * 100
        
        data.append([
            clean_name,
            f"{clean_acc:.2f}%",
            f"{fgsm_acc:.2f}%",
            f"{fgsm_sr:.2f}%",
            f"{pgd_acc:.2f}%",
            f"{pgd_sr:.2f}%"
        ])
    
    df = pd.DataFrame(data, columns=[
        'Model', 'Clean Acc', 'FGSM Acc', 'FGSM SR', 'PGD Acc', 'PGD SR'
    ])
    
    df.to_csv('results/tables/table2_standard_attacks.csv', index=False)
    
    latex = df.to_latex(index=False, 
                        caption='Standard Adversarial Attack Results (ε=0.3)',
                        label='tab:standard_attacks')
    
    with open('results/tables/table2_standard_attacks.tex', 'w') as f:
        f.write(latex)
    
    print("\n" + tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    print("\n✅ Saved: results/tables/table2_standard_attacks.csv")
    print("✅ Saved: results/tables/table2_standard_attacks.tex")

def create_epsilon_sweep_table():
    """Table 3: Epsilon Sweep Analysis"""
    
    print("\n📊 Creating Table 3: Epsilon Sweep Analysis...")
    
    with open('results/epsilon_sweep_results.json', 'r') as f:
        results = json.load(f)
    
    lr_results = results['logistic_regression']
    
    data = []
    for r in lr_results:
        eps = r['epsilon']
        success = r['success_rate'] * 100
        distance = r['distance_ratio']
        status = r['status'].replace('_', ' ').title()
        
        data.append([
            f"{eps:.2f}",
            f"{success:.2f}%",
            f"{distance:.2f}x",
            status
        ])
    
    df = pd.DataFrame(data, columns=[
        'Epsilon (ε)', 'Attack Success Rate', 'Distance Ratio', 'Manifold Status'
    ])
    
    df.to_csv('results/tables/table3_epsilon_sweep.csv', index=False)
    
    latex = df.to_latex(index=False,
                        caption='Epsilon Sweep Analysis: Attack Effectiveness vs. Manifold Distance',
                        label='tab:epsilon_sweep')
    
    with open('results/tables/table3_epsilon_sweep.tex', 'w') as f:
        f.write(latex)
    
    print("\n" + tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    print("\n✅ Saved: results/tables/table3_epsilon_sweep.csv")
    print("✅ Saved: results/tables/table3_epsilon_sweep.tex")

def create_feature_aware_table():
    """Table 4: Feature-Aware Attack Results"""
    
    print("\n📊 Creating Table 4: Feature-Aware Attacks...")
    
    with open('results/feature_aware_attack_results.json', 'r') as f:
        results = json.load(f)
    
    data = []
    for key, r in results.items():
        eps = r['epsilon_budget']
        success = r['success_rate'] * 100
        distance = r['distance_ratio']
        status = r['status'].replace('_', ' ').title()
        
        data.append([
            f"{eps:.3f}",
            f"{success:.2f}%",
            f"{distance:.2f}x",
            status
        ])
    
    df = pd.DataFrame(data, columns=[
        'Epsilon Budget', 'Attack Success Rate', 'Distance Ratio', 'Manifold Status'
    ])
    
    df.to_csv('results/tables/table4_feature_aware.csv', index=False)
    
    latex = df.to_latex(index=False,
                        caption='Feature-Aware Attack Results with Domain Constraints',
                        label='tab:feature_aware')
    
    with open('results/tables/table4_feature_aware.tex', 'w') as f:
        f.write(latex)
    
    print("\n" + tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    print("\n✅ Saved: results/tables/table4_feature_aware.csv")
    print("✅ Saved: results/tables/table4_feature_aware.tex")

def create_comparison_table():
    """Table 5: Method Comparison Summary"""
    
    print("\n📊 Creating Table 5: Method Comparison...")
    
    data = [
        ['Standard FGSM', '0.3', '21.4%', '1.21x', 'On-manifold', 'No'],
        ['Standard FGSM', '1.0', '95.2%', '2.09x', 'Moderate', 'No'],
        ['Standard FGSM', '3.0', '99.9%', '5.70x', 'Off-manifold', 'No'],
        ['Feature-Aware', '0.01', '1.9%', '0.99x', 'On-manifold', 'Yes'],
        ['Feature-Aware', '0.05', '1.9%', '0.99x', 'On-manifold', 'Yes'],
        ['Feature-Aware', '0.10', '2.2%', '0.99x', 'On-manifold', 'Yes'],
    ]
    
    df = pd.DataFrame(data, columns=[
        'Method', 'Epsilon', 'Success Rate', 'Distance', 'Status', 'Realistic'
    ])
    
    df.to_csv('results/tables/table5_comparison.csv', index=False)
    
    latex = df.to_latex(index=False,
                        caption='Comparison of Standard and Feature-Aware Adversarial Attacks',
                        label='tab:comparison')
    
    with open('results/tables/table5_comparison.tex', 'w') as f:
        f.write(latex)
    
    print("\n" + tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    print("\n✅ Saved: results/tables/table5_comparison.csv")
    print("✅ Saved: results/tables/table5_comparison.tex")

def create_feature_constraints_table():
    """Table 6: MANET Feature Constraints"""
    
    print("\n📊 Creating Table 6: Feature Constraints...")
    
    constraints = {
        'packet_rate': {'min': 0, 'max': 1000, 'epsilon': 10, 'unit': 'packets/sec'},
        'byte_count': {'min': 64, 'max': 65535, 'epsilon': 100, 'unit': 'bytes'},
        'flow_duration': {'min': 0, 'max': 300, 'epsilon': 1, 'unit': 'seconds'},
        'inter_arrival_time': {'min': 0, 'max': 1, 'epsilon': 0.01, 'unit': 'seconds'},
        'packet_size_variance': {'min': 0, 'max': 2000, 'epsilon': 50, 'unit': 'bytes'},
        'protocol_type': {'min': 6, 'max': 17, 'epsilon': 0, 'unit': 'discrete'},
        'hop_count': {'min': 1, 'max': 15, 'epsilon': 1, 'unit': 'hops'},
        'route_changes': {'min': 0, 'max': 10, 'epsilon': 1, 'unit': 'count'},
        'signal_strength': {'min': -100, 'max': -20, 'epsilon': 2, 'unit': 'dBm'},
        'node_mobility': {'min': 0, 'max': 20, 'epsilon': 1, 'unit': 'm/s'},
    }
    
    data = []
    for feature, cons in constraints.items():
        feature_name = feature.replace('_', ' ').title()
        data.append([
            feature_name,
            cons['min'],
            cons['max'],
            cons['epsilon'],
            cons['unit']
        ])
    
    df = pd.DataFrame(data, columns=[
        'Feature', 'Min', 'Max', 'Max Pert.', 'Unit'
    ])
    
    df.to_csv('results/tables/table6_constraints.csv', index=False)
    
    latex = df.to_latex(index=False,
                        caption='Domain Constraints for MANET Network Features',
                        label='tab:constraints')
    
    with open('results/tables/table6_constraints.tex', 'w') as f:
        f.write(latex)
    
    print("\n" + tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    print("\n✅ Saved: results/tables/table6_constraints.csv")
    print("✅ Saved: results/tables/table6_constraints.tex")

if __name__ == "__main__":
    import os
    os.makedirs('results/tables', exist_ok=True)
    
    print("="*60)
    print("GENERATING ALL RESULTS TABLES")
    print("="*60)
    
    create_baseline_performance_table()
    create_standard_attacks_table()
    create_epsilon_sweep_table()
    create_feature_aware_table()
    create_comparison_table()
    create_feature_constraints_table()
    
    print("\n" + "="*60)
    print("✅ ALL TABLES GENERATED SUCCESSFULLY")
    print("="*60)
    print("\nGenerated 6 tables in:")
    print("  - CSV format (for spreadsheets)")
    print("  - LaTeX format (for papers)")
