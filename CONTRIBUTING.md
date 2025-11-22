# Contributing to MANET Adversarial IDS

Thank you for your interest in contributing to this research project! We welcome contributions from the community.

## 📋 How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists in [GitHub Issues](https://github.com/vssk18/manet-adversarial-ids/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs. actual behavior
   - System information (OS, Python version)

### Suggesting Enhancements

We welcome ideas for:
- New attack methods
- Additional MANET features
- Performance optimizations
- Documentation improvements
- Visualization enhancements

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
   - Follow the existing code style
   - Add docstrings to new functions
   - Update documentation if needed
4. **Test your changes**
   ```bash
   ./run_all.sh  # Ensure everything still works
   ```
5. **Commit with clear messages**
   ```bash
   git commit -m "Add feature: brief description"
   ```
6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## 🎨 Code Style

### Python Code

- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings to all functions
- Keep functions focused and modular

**Example:**
```python
def feature_aware_attack(model, X, y, constraints, epsilon_budget):
    """
    Generate adversarial examples with domain constraints.
    
    Args:
        model: Target classifier
        X: Input samples (numpy array)
        y: True labels (numpy array)
        constraints: Dict of per-feature constraints
        epsilon_budget: Global epsilon budget (float)
    
    Returns:
        X_adv: Adversarial samples (numpy array)
    """
    # Implementation here
    pass
```

### Documentation

- Use clear, concise language
- Include code examples
- Update README.md if adding new features
- Add comments for complex logic

## 🧪 Testing

Before submitting a PR:

1. Run the complete pipeline:
   ```bash
   ./run_all.sh
   ```

2. Verify outputs:
   - Check data/ directory
   - Verify models/ saved correctly
   - Ensure figures generated properly
   - Confirm tables created

3. Test on clean environment:
   ```bash
   python3 -m venv test_env
   source test_env/bin/activate
   pip install -r requirements.txt
   ./run_all.sh
   ```

## 📚 Adding New Scripts

If adding a new analysis script:

1. **Follow naming convention**: `09_new_analysis.py`
2. **Add comprehensive docstring**:
   ```python
   """
   09_new_analysis.py
   ==================
   Brief description of what this script does.
   
   Author: Your Name
   Date: Month Year
   """
   ```
3. **Update run_all.sh** to include the new script
4. **Update README.md** with new functionality
5. **Add to QUICK_START.md** if relevant

## 🎯 Research Contributions

### Adding New Attack Methods

If implementing a new adversarial attack:

1. Add to appropriate script (e.g., `06_feature_aware_attacks.py`)
2. Include:
   - Clear algorithm description
   - Domain constraints (if applicable)
   - Evaluation metrics
   - Comparison with existing methods
3. Generate visualization showing effectiveness
4. Update results tables

### Adding New Features

For new MANET features:

1. Update `FEATURE_CONSTRAINTS` in `06_feature_aware_attacks.py`
2. Add to dataset generation in `01_generate_dataset.py`
3. Document the feature:
   - Physical meaning
   - Valid range
   - Units
   - Perturbation limits

### Adding New Visualizations

For new figures:

1. Add to `07_create_visualizations.py`
2. Use consistent style:
   - 300 DPI
   - Professional color scheme
   - Clear labels and legends
   - Informative captions
3. Save to `results/figures/`
4. Update README.md to showcase the figure

## 📄 Documentation Standards

### Docstring Format

Use Google-style docstrings:

```python
def example_function(param1, param2):
    """
    Brief description of function.
    
    Longer description if needed, explaining the algorithm,
    approach, or important details.
    
    Args:
        param1 (type): Description of param1
        param2 (type): Description of param2
    
    Returns:
        type: Description of return value
    
    Raises:
        ExceptionType: When this exception occurs
    
    Example:
        >>> result = example_function(arg1, arg2)
        >>> print(result)
        Expected output
    """
    pass
```

### README Updates

When adding features, update README sections:
- Features list
- Usage examples
- Results tables
- Citation information

## 🤝 Code Review Process

All PRs will be reviewed for:

1. **Code Quality**
   - Follows style guidelines
   - Well-documented
   - No unnecessary complexity

2. **Functionality**
   - Works as intended
   - Doesn't break existing code
   - Adds value to the project

3. **Documentation**
   - Clear explanations
   - Updated README/docs
   - Code comments where needed

4. **Testing**
   - Runs without errors
   - Produces expected outputs
   - Doesn't introduce bugs

## 🏆 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md (if created)
- Acknowledged in publications (for significant contributions)
- Credited in release notes

## 📞 Questions?

- **General questions**: Open a [GitHub Discussion](https://github.com/vssk18/manet-adversarial-ids/discussions)
- **Bug reports**: Open an [Issue](https://github.com/vssk18/manet-adversarial-ids/issues)
- **Direct contact**: varanasikarthik44@gmail.com

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping improve MANET Adversarial IDS research! 🙏
