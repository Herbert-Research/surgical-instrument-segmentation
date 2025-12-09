# Contributing to Surgical Instrument Segmentation

Thank you for your interest in contributing to this research project. This document provides guidelines for development setup, testing, and code style conventions.

---

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Running Tests](#running-tests)
4. [Code Style](#code-style)
5. [Commit Message Convention](#commit-message-convention)
6. [Pull Request Process](#pull-request-process)
7. [Reporting Issues](#reporting-issues)

---

## Development Setup

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended for training)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/Herbert-Research/surgical-instrument-segmentation.git
cd surgical-instrument-segmentation

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Windows CMD:
.\.venv\Scripts\activate.bat
# On Linux/macOS:
source .venv/bin/activate

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (if available)
pre-commit install
```

### Alternative: Conda Environment

```bash
conda env create -f environment.yml
conda activate surgical-segmentation
pip install -e ".[dev]"
```

### Alternative: Docker

```bash
docker-compose up --build
```

---

## Project Structure

```
surgical-instrument-segmentation/
├── src/surgical_segmentation/    # Main package
│   ├── datasets/                 # Data loading and transforms
│   ├── evaluation/               # Metrics and analysis
│   ├── models/                   # Neural network architectures
│   ├── training/                 # Training loops and utilities
│   └── utils/                    # Helper functions
├── tests/                        # Unit and integration tests
├── scripts/                      # Utility scripts
├── docs/                         # Documentation
├── data/                         # Sample data (not in git)
├── datasets/                     # Full datasets (not in git)
└── outputs/                      # Training outputs
```

---

## Running Tests

### Run All Tests

```bash
# Basic test run
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=surgical_segmentation --cov-report=term-missing

# With coverage threshold enforcement
pytest tests/ -v --cov=surgical_segmentation --cov-fail-under=60
```

### Run Specific Test Files

```bash
# Test models only
pytest tests/test_models.py -v

# Test metrics only
pytest tests/test_metrics.py -v

# Test a specific test function
pytest tests/test_models.py::test_unet_output_shape -v
```

### Run Tests with Markers

```bash
# Run only fast tests (excludes integration tests)
pytest tests/ -v -m "not slow"

# Run integration tests only
pytest tests/ -v -m "integration"
```

---

## Code Style

This project enforces consistent code style using automated tools.

### Formatters and Linters

| Tool | Purpose | Configuration |
|------|---------|---------------|
| **Black** | Code formatting | Line length: 100 |
| **isort** | Import sorting | Black-compatible profile |
| **mypy** | Type checking | Strict mode for new code |
| **flake8** | Linting | PEP8 compliance |

### Running Formatters

```bash
# Format code with Black
black src/ tests/ scripts/

# Sort imports with isort
isort src/ tests/ scripts/

# Type check with mypy
mypy src/surgical_segmentation/

# Lint with flake8
flake8 src/ tests/
```

### Pre-commit Hooks

If pre-commit is configured, all checks run automatically before each commit:

```bash
# Run all hooks manually
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
```

### Code Style Guidelines

1. **Type Hints**: Use type hints for all public functions
   ```python
   def compute_iou(pred: np.ndarray, target: np.ndarray) -> float:
       ...
   ```

2. **Docstrings**: Use Google-style docstrings
   ```python
   def train_model(model: nn.Module, epochs: int) -> Tuple[nn.Module, List[float]]:
       """
       Train the segmentation model.

       Args:
           model: PyTorch model to train.
           epochs: Number of training epochs.

       Returns:
           Tuple of trained model and list of epoch losses.

       Raises:
           ValueError: If epochs is less than 1.
       """
   ```

3. **Constants**: Use UPPER_SNAKE_CASE for module-level constants
   ```python
   NUM_CLASSES = 2
   DEFAULT_LEARNING_RATE = 0.001
   ```

4. **Private Functions**: Prefix with underscore
   ```python
   def _normalize_mask(mask: np.ndarray) -> np.ndarray:
       ...
   ```

---

## Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/) for clear, standardized commit history.

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only changes |
| `style` | Code style (formatting, no logic changes) |
| `refactor` | Code refactoring (no feature or fix) |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks (deps, CI, etc.) |

### Examples

```bash
# Feature
feat(models): add attention mechanism to UNet decoder

# Bug fix
fix(dataset): correct mask remapping for CholecSeg8k class IDs

# Documentation
docs(readme): add installation instructions for Docker

# Tests
test(evaluation): add integration tests for metric calculations

# Refactor
refactor(trainer): extract epoch loop into separate function
```

### Bad Commit Messages (Avoid)

```bash
# Too vague
git commit -m "changes"
git commit -m "fix bug"
git commit -m "update"

# No context
git commit -m "wip"
git commit -m "stuff"
```

---

## Pull Request Process

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code following style guidelines
   - Add tests for new functionality
   - Update documentation if needed

3. **Run Checks Locally**
   ```bash
   # Format code
   black src/ tests/
   isort src/ tests/

   # Run tests
   pytest tests/ -v --cov=surgical_segmentation

   # Type check
   mypy src/surgical_segmentation/
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat(scope): description of changes"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **PR Requirements**
   - Clear description of changes
   - All CI checks passing
   - Test coverage maintained or improved
   - Documentation updated if applicable

---

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. **Environment Information**
   - Python version
   - PyTorch version
   - CUDA version (if applicable)
   - Operating system

2. **Steps to Reproduce**
   - Minimal code example
   - Dataset configuration
   - Exact commands run

3. **Expected vs Actual Behavior**
   - What you expected to happen
   - What actually happened
   - Error messages (full traceback)

### Feature Requests

When requesting features, please include:

1. **Use Case**: Why is this feature needed?
2. **Proposed Solution**: How should it work?
3. **Alternatives Considered**: What other approaches did you consider?

---

## Questions?

For questions about contributing, please open a GitHub issue with the `question` label or contact the maintainers directly.

---

*Thank you for contributing to medical AI research!*
