# 🚀 QUICK START GUIDE

Get started with the MANET Adversarial IDS project in 5 minutes!

---

## ⚡ Installation (1 minute)

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies**:
- numpy (numerical computing)
- pandas (data manipulation)
- scikit-learn (machine learning)
- xgboost (gradient boosting)
- matplotlib (visualization)
- tabulate (table formatting)

---

## 🎯 Run Complete Pipeline (3 minutes)

### Option 1: Run Everything

```bash
# Run all scripts in sequence
python 01_generate_dataset.py
python 02_train_baselines.py
python 03_adversarial_attacks.py
python 04_manifold_analysis.py
python 05_epsilon_sweep.py
python 06_feature_aware_attacks.py
python 07_create_visualizations.py
python 08_generate_tables.py
```

**Expected runtime**: ~3 minutes total

### Option 2: Run Individual Experiments

**Generate dataset only**:
```bash
python 01_generate_dataset.py
# Output: data/manet_dataset_full.csv
```

**Train models only**:
```bash
python 02_train_baselines.py
# Output: models/*.pkl
```

**Test feature-aware attacks** (novel contribution):
```bash
python 06_feature_aware_attacks.py
# Output: results/feature_aware_attack_results.json
```

---

## 📊 View Results (1 minute)

### Check Figures
```bash
# View generated figures
ls -lh results/figures/
# - epsilon_sweep_analysis.png
# - comprehensive_attack_comparison.png
# - key_findings.png
```

### Check Tables
```bash
# View results tables
ls -lh results/tables/
# - table1_baseline_performance.csv
# - table2_standard_attacks.csv
# - table3_epsilon_sweep.csv
# - table4_feature_aware.csv
# - table5_comparison.csv
# - table6_constraints.csv
```

### Check JSON Results
```bash
cat results/baseline_performance.json
cat results/feature_aware_attack_results.json
```

---

## 🔬 Key Results Summary

After running all scripts, you should see:

### Baseline Models
- Logistic Regression: 98.14% accuracy
- Decision Tree: 97.10% accuracy
- XGBoost: 98.74% accuracy

### Feature-Aware Attacks
- Attack success: ~2%
- Distance ratio: 0.99x (on-manifold!)
- Status: **Realistic adversarial threat**

### Standard Attacks
- ε=0.3: 21.4% success, 1.21x distance (on-manifold)
- ε=1.0: 95.2% success, 2.09x distance (moderate)
- ε=3.0: 99.9% success, 5.70x distance (off-manifold)

---

## 📁 Output Structure

After running all scripts:

```
manet-adversarial-ids/
├── data/
│   ├── manet_dataset_full.csv          (4,500 samples)
│   ├── train_test_split.pkl            (group-safe splits)
│   └── adversarial/                    (generated adversarials)
│       ├── logistic_regression_fgsm.npy
│       ├── logistic_regression_pgd.npy
│       └── logistic_regression_feature_aware.npy
├── models/
│   ├── scaler.pkl                      (feature scaler)
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   └── xgboost.pkl
└── results/
    ├── figures/                        (3 publication figures)
    ├── tables/                         (6 results tables)
    └── *.json                          (raw results)
```

---

## 🐛 Troubleshooting

### Import Errors
```bash
# If you get import errors, ensure all dependencies are installed
pip install -r requirements.txt --upgrade
```

### Memory Issues
```bash
# If dataset generation fails due to memory
# Edit 01_generate_dataset.py and reduce samples_per_class:
# samples_per_class=1500  →  samples_per_class=1000
```

### Numerical Gradient Warnings
```
# The warning about numerical gradients is expected
# This occurs during adversarial attack generation
# and does not affect results
```

---

## 📚 Next Steps

### For Research
1. Read `PAPER_SECTIONS.md` for paper structure
2. Review figures in `results/figures/`
3. Check tables in `results/tables/`
4. Write your paper!

### For Development
1. Modify feature constraints in `06_feature_aware_attacks.py`
2. Add new attack types
3. Test different classifiers
4. Extend to other datasets

### For Deployment
1. Read `README.md` for full documentation
2. Check `PROJECT_SUMMARY.md` for overview
3. Review code comments in each script

---

## 💡 Tips

**Reproducibility**: All scripts use `random_seed=42`

**Speed up**: Comment out PGD attacks in script 03 (they're slower)

**Custom data**: Modify `01_generate_dataset.py` to use your own MANET data

**Different models**: Add your own classifiers in `02_train_baselines.py`

---

## 🆘 Need Help?

**Documentation**:
- `README.md` - Full documentation
- `PROJECT_SUMMARY.md` - Complete overview
- `PAPER_SECTIONS.md` - Research paper draft

**Issues**:
- Check script comments for detailed explanations
- Review checkpoint files for development history
- Open an issue on GitHub

---

**Ready to start? Run the first script:**

```bash
python 01_generate_dataset.py
```

**Good luck! 🚀**
