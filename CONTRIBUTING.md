# Contributing to TAS-AI

Thank you for your interest in contributing to TAS-AI! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a branch for your changes
4. Make your changes
5. Test your changes
6. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/tasai.git
cd tasai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Code Style

We follow PEP 8 with the following additions:

- Line length: 100 characters
- Use type hints where practical
- Document public functions with docstrings

```python
def calculate_dispersion(H: float, K: float, L: float, 
                         J1: float = 5.0) -> float:
    """
    Calculate spin wave dispersion.
    
    Parameters
    ----------
    H, K, L : float
        Reciprocal space coordinates (r.l.u.)
    J1 : float, optional
        Exchange coupling (meV), default 5.0
        
    Returns
    -------
    float
        Spin wave energy (meV)
    """
    ...
```

## Testing

- Add tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=tasai --cov-report=html

# Run specific test file
pytest tests/test_sunny.py -v
```

## Pull Request Process

1. Update documentation if needed
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Request review from maintainers

## Types of Contributions

### Bug Reports

- Use GitHub Issues
- Include Python version, OS, and steps to reproduce
- Include error messages and tracebacks

### Feature Requests

- Open an issue to discuss before implementing
- Describe the use case and expected behavior

### Documentation

- Fix typos, improve clarity
- Add examples
- Translate to other languages

### Code

- Bug fixes
- New features (discuss first)
- Performance improvements
- Code cleanup/refactoring

## Physics Models

If adding new physics models:

1. Inherit from `SpinWaveModel` base class
2. Implement `dispersion()` and `intensity()` methods
3. Add unit tests comparing to known results
4. Document the Hamiltonian and parameters
5. Add example usage in `examples/`

## Questions?

- Open a GitHub Discussion
- Email: neutronscattering@nist.gov

## Code of Conduct

Be respectful and constructive. We're all here to advance neutron science!
