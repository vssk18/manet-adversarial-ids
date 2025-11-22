# Quick Start Guide

Get started with the MANET Adversarial IDS research code in 5 minutes!

## Prerequisites

```bash
Python 3.8+
pip (Python package manager)
```

## Installation

```bash
# Clone the repository
git clone https://github.com/vssk18/manet-adversarial-ids.git
cd manet-adversarial-ids

# Install dependencies
pip install -r requirements.txt
```

## Running the Complete Pipeline

Execute the 8 scripts in order:

```bash
# 1. Generate dataset (group-safe splitting)
python 01_generate_dataset.py

# 2. Train baseline models
python 02_train_baselines.py

# 3. Run standard adversarial attacks
python 03_adversarial_attacks.py

# 4. Perform manifold analysis
python 04_manifold_analysis.py

# 5. Epsilon sweep study
python 05_epsilon_sweep.py

# 6. Feature-aware attacks (NOVEL)
python 06_feature_aware_attacks.py

# 7. Generate visualizations
python 07_create_visualizations.py

# 8. Create results tables
python 08_generate_tables.py
```

## Expected Runtime

- Dataset generation: ~5 seconds
- Baseline training: ~10 seconds
- Adversarial attacks: ~30 seconds
- Manifold analysis: ~20 seconds
- Epsilon sweep: ~40 seconds
- Feature-aware attacks: ~30 seconds
- **Total**: ~2-3 minutes

## Output Structure

```
manet-adversarial-ids/
├── data/
│   ├── manet_dataset_full.csv
│   └── adversarial/
│       └── *.npy (adversarial samples)
├── models/
│   ├── scaler.pkl
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   └── xgboost.pkl
├── results/
│   ├── figures/ (7 publication-quality figures)
│   ├── tables/ (LaTeX + CSV tables)
│   └── *.json (all analysis results)
```

## Key Results

After running all scripts, you'll have:

✅ **Baseline Performance**: 98.74% accuracy (XGBoost)  
✅ **Standard Attacks**: 95.2% success rate (FGSM ε=1.0), but 5.70x off-manifold  
✅ **Feature-Aware Attacks**: 12.7% success rate (ε=0.3), only 0.99x distance (on-manifold!)  
✅ **7 Publication Figures**: Ready for paper inclusion  
✅ **5 Results Tables**: LaTeX format for paper  

## Next Steps

1. **View figures**: Check `fig_*.png` for visualizations
2. **Examine results**: Read JSON files in `results/`
3. **Modify parameters**: Edit scripts to test different configurations
4. **Write paper**: Use tables and figures in your publication

## Troubleshooting

**Issue**: `ModuleNotFoundError`  
**Solution**: Run `pip install -r requirements.txt`

**Issue**: Script takes too long  
**Solution**: Reduce sample size in scripts (look for `n_samples` variables)

**Issue**: Out of memory  
**Solution**: Use smaller test set in `01_generate_dataset.py`

## Citation

If you use this code, please cite:

```bibtex
@article{karthik2024manet,
  title={Feature-Aware Adversarial Attacks for Realistic Evaluation of MANET Intrusion Detection Systems},
  author={Karthik, Varanasi Sai Srinivasa},
  year={2024}
}
```

## Support

- Issues: [GitHub Issues](https://github.com/vssk18/manet-adversarial-ids/issues)
- Email: varanasikarthik44@gmail.com
- Institution: GITAM University

---

**Ready to run? Execute:**
```bash
for i in {1..8}; do python 0${i}_*.py; done
```

This will run all 8 scripts sequentially!
