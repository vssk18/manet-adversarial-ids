"""
Create comprehensive comparison visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import json

def load_all_results():
    """Load all attack results"""
    
    with open('results/adversarial_attack_results.json', 'r') as f:
        standard_attacks = json.load(f)
    
    with open('results/epsilon_sweep_results.json', 'r') as f:
        epsilon_sweep = json.load(f)
    
    with open('results/feature_aware_attack_results.json', 'r') as f:
        feature_aware = json.load(f)
    
    return standard_attacks, epsilon_sweep, feature_aware

def create_comparison_figure():
    """Create comprehensive comparison figure"""
    
    print("📊 Creating comprehensive comparison figure...")
    
    standard_attacks, epsilon_sweep, feature_aware = load_all_results()
    
    # Extract data for logistic regression
    lr_results = epsilon_sweep['logistic_regression']
    
    # Epsilon values and metrics
    eps_values = [r['epsilon'] for r in lr_results]
    success_rates = [r['success_rate'] * 100 for r in lr_results]
    distance_ratios = [r['distance_ratio'] for r in lr_results]
    
    # Feature-aware results
    fa_eps = [0.01, 0.03, 0.05, 0.10]
    fa_success = []
    fa_distance = []
    
    for i, eps in enumerate(fa_eps):
        key = f'feature_aware_eps_{eps}'
        fa_success.append(feature_aware[key]['success_rate'] * 100)
        fa_distance.append(feature_aware[key]['distance_ratio'])
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Attack Success Rate vs Epsilon
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(eps_values, success_rates, 'o-', linewidth=2.5, markersize=10, 
             color='#e74c3c', label='Standard FGSM', alpha=0.8)
    ax1.plot(fa_eps, fa_success, 's-', linewidth=2.5, markersize=10,
             color='#27ae60', label='Feature-Aware', alpha=0.8)
    ax1.set_xlabel('Epsilon (ε)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Attack Success Rate (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Attack Effectiveness Comparison', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.set_ylim([0, 105])
    
    # Plot 2: Distance Ratio vs Epsilon
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(eps_values, distance_ratios, 'o-', linewidth=2.5, markersize=10,
             color='#3498db', label='Standard FGSM', alpha=0.8)
    ax2.plot(fa_eps, fa_distance, 's-', linewidth=2.5, markersize=10,
             color='#27ae60', label='Feature-Aware', alpha=0.8)
    ax2.axhline(y=2, color='orange', linestyle='--', linewidth=2, 
                label='Moderate threshold', alpha=0.7)
    ax2.axhline(y=10, color='red', linestyle='--', linewidth=2,
                label='Off-manifold threshold', alpha=0.7)
    ax2.set_xlabel('Epsilon (ε)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Distance Ratio (vs baseline)', fontsize=13, fontweight='bold')
    ax2.set_title('Manifold Distance Comparison', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, loc='upper left')
    ax2.set_ylim([0.5, 7])
    
    # Plot 3: Success Rate vs Distance Ratio (scatter)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(distance_ratios, success_rates, s=200, c=eps_values,
                cmap='Reds', alpha=0.7, edgecolors='black', linewidth=1.5,
                label='Standard FGSM')
    ax3.scatter(fa_distance, fa_success, s=200, marker='s',
                c='#27ae60', alpha=0.8, edgecolors='black', linewidth=1.5,
                label='Feature-Aware')
    ax3.axvline(x=2, color='orange', linestyle='--', linewidth=2, alpha=0.5)
    ax3.set_xlabel('Distance Ratio', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Attack Success Rate (%)', fontsize=13, fontweight='bold')
    ax3.set_title('Effectiveness vs Realism Trade-off', fontsize=15, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=11, loc='lower right')
    
    # Add colorbar for standard FGSM
    cbar = plt.colorbar(ax3.collections[0], ax=ax3)
    cbar.set_label('Epsilon (ε)', fontsize=11, fontweight='bold')
    
    # Plot 4: Summary table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    table_data = [
        ['Method', 'Epsilon', 'Success', 'Distance', 'Realistic?'],
        ['Standard FGSM', '0.3', '21.4%', '1.21x', '✅ Yes'],
        ['Standard FGSM', '1.0', '95.2%', '2.09x', '⚠️  Moderate'],
        ['Standard FGSM', '3.0', '99.9%', '5.70x', '❌ No'],
        ['Feature-Aware', '0.01', '1.9%', '0.99x', '✅ Yes'],
        ['Feature-Aware', '0.05', '1.9%', '0.99x', '✅ Yes'],
        ['Feature-Aware', '0.10', '2.2%', '0.99x', '✅ Yes'],
    ]
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.25, 0.15, 0.15, 0.15, 0.20])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows - alternate colors
    for i in range(1, len(table_data)):
        color = '#ecf0f1' if i % 2 == 0 else 'white'
        for j in range(5):
            table[(i, j)].set_facecolor(color)
    
    ax4.set_title('Attack Method Comparison Summary', 
                  fontsize=15, fontweight='bold', pad=20)
    
    plt.savefig('results/figures/comprehensive_attack_comparison.png', 
                dpi=300, bbox_inches='tight')
    print("   ✅ Saved: results/figures/comprehensive_attack_comparison.png")
    
    plt.close()

def create_key_findings_figure():
    """Create figure highlighting key findings"""
    
    print("📊 Creating key findings visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Trade-off visualization
    ax1 = axes[0]
    
    # Data points
    methods = ['Std ε=0.3', 'Std ε=1.0', 'Std ε=3.0', 'FA ε=0.05']
    success = [21.4, 95.2, 99.9, 1.9]
    realism = [1.21, 2.09, 5.70, 0.99]
    colors = ['#e74c3c', '#e67e22', '#c0392b', '#27ae60']
    
    scatter = ax1.scatter(realism, success, s=400, c=colors, alpha=0.8,
                         edgecolors='black', linewidth=2)
    
    for i, method in enumerate(methods):
        ax1.annotate(method, (realism[i], success[i]),
                    xytext=(10, -5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    ax1.axvline(x=2, color='orange', linestyle='--', linewidth=2, alpha=0.5,
                label='Realism threshold')
    ax1.set_xlabel('Distance Ratio (lower = more realistic)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('The Realism-Effectiveness Trade-off', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Key insight
    ax2 = axes[1]
    ax2.axis('off')
    
    insight_text = """
    KEY FINDINGS:
    
    1. Standard FGSM with large ε:
       • High attack success (>95%)
       • OFF-MANIFOLD samples
       • Physically unrealistic
    
    2. Feature-Aware attacks:
       • Lower attack success (~2%)
       • ON-MANIFOLD samples  
       • Physically realistic
       • TRUE adversarial threat
    
    3. Novel Contribution:
       Domain-constrained attacks for
       realistic IDS evaluation
    """
    
    ax2.text(0.1, 0.5, insight_text, fontsize=11, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8, pad=1))
    
    plt.tight_layout()
    plt.savefig('results/figures/key_findings.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: results/figures/key_findings.png")
    
    plt.close()

if __name__ == "__main__":
    import os
    os.makedirs('results/figures', exist_ok=True)
    
    create_comparison_figure()
    create_key_findings_figure()
    
    print("\n✅ All visualizations created!")
