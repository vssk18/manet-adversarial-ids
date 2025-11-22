#!/usr/bin/env python3
"""
08_generate_tables.py
=====================
Generate LaTeX and CSV tables for publication.

This script creates formatted tables from all analysis results for inclusion
in the research paper.

Author: V.S.S. Karthik
Date: November 2024
"""

import json
import pandas as pd
from pathlib import Path

def load_results():
    """Load all result files."""
    results = {}
    
    result_files = [
        'baseline_performance.json',
        'adversarial_attack_results.json',
        'manifold_analysis.json',
        'epsilon_sweep_results.json',
        'feature_aware_attack_results.json'
    ]
    
    for filename in result_files:
        filepath = Path('results') / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                key = filename.replace('.json', '')
                results[key] = json.load(f)
    
    return results

def create_baseline_table(results):
    """Table 1: Baseline Model Performance."""
    baseline = results.get('baseline_performance', {})
    
    data = []
    for model_name, metrics in baseline.items():
        data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Test Accuracy': f"{metrics['test_accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}"
        })
    
    df = pd.DataFrame(data)
    return df

def create_attack_table(results):
    """Table 2: Standard Adversarial Attack Results."""
    attack_results = results.get('adversarial_attack_results', {})
    
    data = []
    # Use logistic regression results
    lr_results = attack_results.get('logistic_regression', {})
    
    for attack_key, metrics in lr_results.items():
        data.append({
            'Attack Method': metrics['attack_name'],
            'Epsilon': f"{metrics['epsilon']:.1f}",
            'Clean Acc.': f"{metrics['clean_accuracy']:.4f}",
            'Adv. Acc.': f"{metrics['adversarial_accuracy']:.4f}",
            'Success Rate': f"{metrics['attack_success_rate']:.1f}%"
        })
    
    df = pd.DataFrame(data)
    return df

def create_manifold_table(results):
    """Table 3: Manifold Distance Analysis."""
    manifold = results.get('manifold_analysis', {})
    
    data = []
    lr_manifold = manifold.get('models', {}).get('logistic_regression', {})
    
    for attack_name, metrics in lr_manifold.items():
        data.append({
            'Attack': attack_name.replace('_', ' ').upper(),
            'Mean Distance Ratio': f"{metrics['mean_distance_ratio']:.2f}x",
            'On-Manifold %': f"{metrics['on_manifold_pct']:.1f}%",
            'Off-Manifold %': f"{metrics['off_manifold_pct']:.1f}%"
        })
    
    df = pd.DataFrame(data)
    return df

def create_feature_aware_table(results):
    """Table 4: Feature-Aware Attack Results."""
    fa_results = results.get('feature_aware_attack_results', {})
    
    data = []
    for result in fa_results.get('results', []):
        data.append({
            'Epsilon Budget': f"{result['epsilon_budget']:.2f}",
            'Success Rate': f"{result['attack_success_rate']:.1f}%",
            'Distance Ratio': f"{result['mean_distance_ratio']:.2f}x",
            'On-Manifold %': f"{result['on_manifold_percentage']:.1f}%",
            'Compliance %': f"{result['constraint_compliance']:.1f}%"
        })
    
    df = pd.DataFrame(data)
    return df

def create_comparison_table(results):
    """Table 5: Standard vs. Feature-Aware Comparison."""
    fa_results = results.get('feature_aware_attack_results', {})
    comparison = fa_results.get('comparison', {})
    
    data = [
        {
            'Method': 'Standard FGSM',
            'Epsilon': '0.3',
            'Adv. Accuracy': f"{comparison['standard_fgsm']['adversarial_accuracy']:.4f}",
            'Distance Ratio': f"{comparison['standard_fgsm']['mean_distance_ratio']:.2f}x",
            'On-Manifold %': f"{comparison['standard_fgsm']['on_manifold_percentage']:.1f}%",
            'Realistic?': 'Marginal'
        },
        {
            'Method': 'Feature-Aware',
            'Epsilon': '0.3',
            'Adv. Accuracy': f"{comparison['feature_aware']['adversarial_accuracy']:.4f}",
            'Distance Ratio': f"{comparison['feature_aware']['mean_distance_ratio']:.2f}x",
            'On-Manifold %': f"{comparison['feature_aware']['on_manifold_percentage']:.1f}%",
            'Realistic?': 'Yes'
        }
    ]
    
    df = pd.DataFrame(data)
    return df

def df_to_latex(df, caption, label):
    """Convert DataFrame to LaTeX table."""
    latex = df.to_latex(index=False, escape=False)
    
    # Wrap in table environment
    full_latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
{latex}
\\end{{table}}
"""
    return full_latex

def main():
    """Main execution function."""
    print("="*70)
    print("Results Table Generation")
    print("="*70)
    
    # Create output directory
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("\n1. Loading results...")
    results = load_results()
    print(f"   Loaded {len(results)} result files")
    
    # Generate tables
    print("\n2. Generating tables...")
    
    tables = {
        'table1_baseline_performance': {
            'df': create_baseline_table(results),
            'caption': 'Baseline Model Performance on MANET IDS Dataset',
            'label': 'tab:baseline'
        },
        'table2_standard_attacks': {
            'df': create_attack_table(results),
            'caption': 'Standard Adversarial Attack Results (FGSM and PGD)',
            'label': 'tab:attacks'
        },
        'table3_manifold': {
            'df': create_manifold_table(results),
            'caption': 'KD-Tree Manifold Distance Analysis',
            'label': 'tab:manifold'
        },
        'table4_feature_aware': {
            'df': create_feature_aware_table(results),
            'caption': 'Feature-Aware Adversarial Attack Results',
            'label': 'tab:feature_aware'
        },
        'table5_comparison': {
            'df': create_comparison_table(results),
            'caption': 'Comparison: Standard FGSM vs. Feature-Aware Attacks',
            'label': 'tab:comparison'
        }
    }
    
    # Save tables
    for table_name, table_info in tables.items():
        df = table_info['df']
        
        # Save as CSV
        csv_path = f"results/tables/{table_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"   ✓ {csv_path}")
        
        # Save as LaTeX
        latex_content = df_to_latex(df, table_info['caption'], table_info['label'])
        tex_path = f"results/tables/{table_name}.tex"
        with open(tex_path, 'w') as f:
            f.write(latex_content)
        print(f"   ✓ {tex_path}")
    
    # Create summary document
    print("\n3. Creating summary document...")
    summary = []
    summary.append("# Results Tables Summary\n")
    summary.append("Generated tables for MANET Adversarial IDS research.\n\n")
    
    for i, (table_name, table_info) in enumerate(tables.items(), 1):
        summary.append(f"## Table {i}: {table_info['caption']}\n")
        summary.append(f"```\n{table_info['df'].to_string()}\n```\n\n")
    
    with open('results/tables/TABLES_SUMMARY.md', 'w') as f:
        f.writelines(summary)
    
    print("   ✓ results/tables/TABLES_SUMMARY.md")
    
    # Print sample
    print("\n" + "="*70)
    print("SAMPLE: Table 5 - Standard vs. Feature-Aware Comparison")
    print("="*70)
    print(tables['table5_comparison']['df'].to_string(index=False))
    
    print("\n" + "="*70)
    print("All tables saved to results/tables/")
    print("  - CSV format: For data analysis")
    print("  - LaTeX format: For paper inclusion")
    print("  - Summary: TABLES_SUMMARY.md")
    print("="*70)

if __name__ == '__main__':
    main()
