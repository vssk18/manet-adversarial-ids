# Contributing to MANET Adversarial IDS

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

---

## 🎯 How to Contribute

### Reporting Issues

Found a bug or have a suggestion? Please:

1. **Check existing issues** to avoid duplicates
2. **Open a new issue** with:
   - Clear, descriptive title
   - Detailed description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - System information (OS, Python version)

### Suggesting Enhancements

Have an idea for improvement?

1. Open an issue with tag `enhancement`
2. Describe the feature and its benefits
3. Provide use cases or examples
4. Be open to discussion!

### Pull Requests

Want to contribute code?

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**:
   - Follow the code style (PEP 8 for Python)
   - Add comments for complex logic
   - Include docstrings for functions
4. **Test your changes**:
   - Ensure all scripts run without errors
   - Verify results are reproducible
5. **Commit with clear messages**:
   - Use present tense ("Add feature" not "Added feature")
   - Reference issues if applicable (#123)
6. **Push and create PR**:
   - Describe what changed and why
   - Link to related issues
   - Be responsive to feedback

---

## 💻 Development Setup

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/vssk18/manet-adversarial-ids.git
cd manet-adversarial-ids

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Tests

```bash
# Test dataset generation
python 01_generate_dataset.py

# Test model training
python 02_train_baselines.py

# Test attacks
python 06_feature_aware_attacks.py
```

---

## 📝 Code Style

### Python Code

- Follow **PEP 8** style guide
- Use **meaningful variable names**
- Add **docstrings** for functions and classes
- Keep functions **focused and small**
- Use **type hints** where helpful

**Example**:
```python
def feature_aware_attack(model, X, y, constraints, epsilon_budget=0.05):
    """
    Generate adversarial examples with domain constraints
    
    Args:
        model: Target classifier
        X: Input samples (numpy array)
        y: True labels (numpy array)
        constraints: Dict of feature constraints
        epsilon_budget: Attack budget (default: 0.05)
    
    Returns:
        X_adv: Adversarial samples respecting constraints
    """
    # Implementation...
```

### Documentation

- Use **Markdown** for documentation files
- Add **clear headers** and sections
- Include **code examples** where helpful
- Keep **README.md** up to date

---

## 🔬 Research Contributions

### New Attack Methods

Adding a new adversarial attack?

1. Create new script: `XX_your_attack_name.py`
2. Follow existing script structure
3. Save results to `results/` directory
4. Add visualization if applicable
5. Update README with new method
6. Include references in PAPER_SECTIONS.md

### New Datasets

Testing on different data?

1. Document dataset source and format
2. Update `01_generate_dataset.py` or create new script
3. Ensure group-safe splitting
4. Verify no data leakage
5. Compare results with original dataset

### New Models

Testing different classifiers?

1. Add to `02_train_baselines.py`
2. Ensure consistent evaluation metrics
3. Test with all attack methods
4. Update results tables
5. Analyze performance differences

---

## 📊 Adding Results

### Tables

- Save as **CSV** and **LaTeX** formats
- Place in `results/tables/`
- Update `08_generate_tables.py` if automated
- Reference in paper sections

### Figures

- Use **300 DPI** for publication quality
- Save as **PNG** format
- Place in `results/figures/`
- Include in visualization scripts
- Add descriptive filenames

---

## 🧪 Testing Guidelines

### Reproducibility

- Set **random seeds** (`random_state=42`)
- Document all **hyperparameters**
- Use **version control** for dependencies
- Provide **example outputs**

### Validation

- Verify **no data leakage**
- Check **statistical significance**
- Test on **multiple runs**
- Compare with **baseline methods**

---

## 📚 Documentation

### Code Comments

```python
# Good comment: explains WHY
# Use gradient-based perturbation to ensure attack effectiveness

# Avoid: explains WHAT (code already shows this)
# Set epsilon to 0.05
epsilon = 0.05
```

### README Updates

When adding features:

1. Update "Key Contributions" if novel
2. Add to "Repository Structure"
3. Update "Quick Start" if needed
4. Add to "Future Work" if incomplete

---

## 🤝 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of experience level, gender, gender identity, sexual orientation, disability, personal appearance, race, ethnicity, age, religion, or nationality.

### Expected Behavior

- Be respectful and constructive
- Accept feedback gracefully
- Focus on what's best for the project
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Personal or political attacks
- Publishing others' private information

### Enforcement

Instances of unacceptable behavior may be reported to varanasikarthik44@gmail.com. All complaints will be reviewed and investigated promptly and fairly.

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## ❓ Questions?

- **Open an issue** for questions about the code
- **Email** varanasikarthik44@gmail.com for other inquiries
- **Check** existing documentation first

---

## 🙏 Acknowledgments

Thank you for contributing to advancing research in adversarial machine learning and MANET security!

**Popular contribution areas**:
- Adding new attack methods
- Testing on real-world datasets
- Implementing defense mechanisms
- Improving documentation
- Fixing bugs
- Performance optimization

**Your contributions make a difference!** ⭐
