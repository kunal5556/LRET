# Contributing Guidelines

Welcome to LRET! This guide will help you contribute to the project.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Workflow](#development-workflow)
3. [Code Style](#code-style)
4. [Writing Tests](#writing-tests)
5. [Documentation](#documentation)
6. [Pull Request Process](#pull-request-process)
7. [Code Review](#code-review)
8. [Community Guidelines](#community-guidelines)

---

## Getting Started

### Prerequisites

**Required:**
- C++17 compiler (GCC ≥9, Clang ≥10, MSVC ≥2019)
- CMake ≥3.18
- Eigen3 ≥3.3
- Python ≥3.8
- Git

**Recommended:**
- OpenMP (for parallelization)
- Google Test (for C++ testing)
- pytest (for Python testing)
- clang-format (for code formatting)

### Setting Up Development Environment

**1. Fork and clone:**

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/<your-username>/LRET.git
cd LRET

# Add upstream remote
git remote add upstream https://github.com/kunal5556/LRET.git
```

**2. Create development branch:**

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

**3. Build in development mode:**

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
make -j$(nproc)
```

**4. Install Python package in editable mode:**

```bash
cd python
pip install -e .[dev]  # Install with development dependencies
```

---

## Development Workflow

### Branch Naming Convention

- **Feature branches:** `feature/short-description`
- **Bug fixes:** `fix/short-description`
- **Documentation:** `docs/short-description`
- **Performance:** `perf/short-description`
- **Refactoring:** `refactor/short-description`

**Examples:**
```
feature/add-custom-noise-channel
fix/memory-leak-in-truncation
docs/improve-api-reference
perf/optimize-gate-fusion
refactor/simplify-parallel-modes
```

---

### Commit Messages

Follow the **Conventional Commits** specification:

```
<type>(<scope>): <short summary>

<detailed description>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding/updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `chore`: Maintenance tasks

**Examples:**

```
feat(simulator): Add support for custom noise channels

Implement infrastructure for user-defined noise channels with
arbitrary Kraus operators. Includes validation of CPTP conditions.

Closes #42
```

```
fix(truncation): Prevent negative ranks after SVD

Handle edge case where SVD produces near-zero singular values
that cause negative ranks when truncated.

Fixes #87
```

```
docs(readme): Update installation instructions for Windows

Add detailed steps for building with Visual Studio 2022 and
MinGW-w64. Include troubleshooting section.
```

---

### Keeping Your Branch Updated

```bash
# Fetch latest changes from upstream
git fetch upstream

# Rebase your branch on upstream/main
git checkout feature/your-feature
git rebase upstream/main

# If conflicts arise, resolve them and continue
git add <resolved-files>
git rebase --continue

# Force-push to your fork (only if already pushed)
git push origin feature/your-feature --force-with-lease
```

---

## Code Style

### C++ Style Guide

**Formatting:**

Use **clang-format** with the provided configuration:

```bash
# Format all C++ files
find src include -name '*.cpp' -o -name '*.h' | xargs clang-format -i

# Check formatting without modifying
clang-format --dry-run --Werror src/simulator.cpp
```

**Configuration (.clang-format):**

```yaml
BasedOnStyle: Google
IndentWidth: 4
ColumnLimit: 100
AllowShortFunctionsOnASingleLine: Empty
AllowShortIfStatementsOnASingleLine: Never
PointerAlignment: Left
```

---

**Naming Conventions:**

```cpp
// Classes: PascalCase
class LRETSimulator {
public:
    // Public methods: snake_case
    void apply_gate(const Gate& gate);
    int get_rank() const;
    
private:
    // Private members: snake_case with trailing underscore
    MatrixXcd L_;
    int n_qubits_;
    double noise_level_;
    
    // Private methods: snake_case
    void truncate_rank();
};

// Functions: snake_case
MatrixXcd compute_choi_matrix(const MatrixXcd& unitary);

// Constants: UPPER_CASE
constexpr int MAX_QUBITS = 30;
constexpr double DEFAULT_THRESHOLD = 1e-6;

// Enums: PascalCase for type, UPPER_CASE for values
enum class GateType {
    SINGLE_QUBIT,
    TWO_QUBIT,
    THREE_QUBIT
};

// Namespaces: snake_case
namespace qlret {
namespace internal {
    // Internal implementation
}
}
```

---

**Header Guards:**

```cpp
// Use #pragma once (preferred)
#pragma once

#include <vector>
#include <complex>

namespace qlret {
// ...
}
```

---

**Documentation Comments:**

```cpp
/**
 * @brief Apply a quantum gate to the state.
 * 
 * Applies the specified gate to the current state using the Choi matrix
 * representation. Updates the low-rank factor L accordingly.
 * 
 * @param gate The gate to apply (contains unitary and target qubits)
 * 
 * @throws std::invalid_argument if gate targets invalid qubits
 * 
 * @note This method may increase the rank of the state
 * 
 * @example
 * ```cpp
 * LRETSimulator sim(4, 0.0);
 * sim.apply_gate(hadamard_gate(0));
 * ```
 */
void apply_gate(const Gate& gate);
```

---

### Python Style Guide

**Follow PEP 8:**

Use **black** for formatting:

```bash
# Format Python files
black python/qlret python/tests

# Check without modifying
black --check python/qlret
```

**Configuration (pyproject.toml):**

```toml
[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310']
```

---

**Naming Conventions:**

```python
# Classes: PascalCase
class QuantumSimulator:
    """Quantum simulator using LRET."""
    
    def __init__(self, n_qubits, noise_level=0.0):
        # Public attributes: snake_case
        self.n_qubits = n_qubits
        self.noise_level = noise_level
        
        # Private attributes: _leading_underscore
        self._sim = LRETSimulator(n_qubits, noise_level)
    
    # Public methods: snake_case
    def apply_gate(self, gate):
        """Apply a gate to the state."""
        self._sim.apply_gate(gate)
    
    # Private methods: _leading_underscore
    def _validate_qubit(self, qubit):
        if not 0 <= qubit < self.n_qubits:
            raise ValueError(f"Invalid qubit: {qubit}")

# Functions: snake_case
def create_bell_state(simulator):
    """Create a Bell state."""
    simulator.h(0)
    simulator.cnot(0, 1)

# Constants: UPPER_CASE
MAX_QUBITS = 30
DEFAULT_SHOTS = 1000
```

---

**Type Hints:**

```python
from typing import List, Dict, Optional, Union
import numpy as np

def measure_all(
    self,
    shots: int = 1000,
    return_counts: bool = True
) -> Dict[str, int]:
    """Measure all qubits.
    
    Args:
        shots: Number of measurement shots
        return_counts: If True, return histogram; if False, return list
    
    Returns:
        Dictionary mapping outcomes (e.g., "0101") to counts
    
    Raises:
        ValueError: If shots <= 0
    """
    if shots <= 0:
        raise ValueError("shots must be positive")
    
    return self._sim.measure_all(shots)
```

---

**Docstrings:**

Use **Google style**:

```python
def apply_noise_channel(self, qubit: int, noise_type: str, parameter: float) -> None:
    """Apply a noise channel to a qubit.
    
    Applies the specified noise channel to the given qubit. Supported channels
    include depolarizing, amplitude damping, and phase damping.
    
    Args:
        qubit: Index of the qubit (0-indexed)
        noise_type: Type of noise ("depolarizing", "amplitude_damping", etc.)
        parameter: Noise parameter (interpretation depends on noise_type)
    
    Raises:
        ValueError: If qubit is out of range or noise_type is invalid
        TypeError: If parameter is not a float
    
    Example:
        >>> sim = QuantumSimulator(n_qubits=4)
        >>> sim.apply_noise_channel(0, "depolarizing", 0.01)
    """
    self._validate_qubit(qubit)
    # Implementation...
```

---

## Writing Tests

### Test Coverage Requirements

- **New features:** Must include tests (minimum 80% coverage)
- **Bug fixes:** Must include regression test
- **Refactoring:** Existing tests must pass

### C++ Tests

**Create test file:** `tests/test_your_feature.cpp`

```cpp
#include <gtest/gtest.h>
#include "your_feature.h"

class YourFeatureTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code
    }
    
    void TearDown() override {
        // Cleanup code
    }
};

TEST_F(YourFeatureTest, BasicFunctionality) {
    // Arrange
    YourClass obj;
    
    // Act
    int result = obj.your_method();
    
    // Assert
    EXPECT_EQ(result, expected_value);
}

TEST_F(YourFeatureTest, EdgeCases) {
    YourClass obj;
    
    // Test edge case
    EXPECT_THROW(obj.invalid_operation(), std::invalid_argument);
}
```

**Run tests:**

```bash
cd build
ctest --output-on-failure
./test_your_feature --gtest_verbose
```

---

### Python Tests

**Create test file:** `python/tests/test_your_feature.py`

```python
import pytest
from qlret import YourClass

@pytest.fixture
def your_object():
    """Fixture for test object."""
    return YourClass()

def test_basic_functionality(your_object):
    """Test basic functionality."""
    # Arrange
    expected = 42
    
    # Act
    result = your_object.your_method()
    
    # Assert
    assert result == expected

def test_edge_cases(your_object):
    """Test edge cases."""
    with pytest.raises(ValueError):
        your_object.invalid_operation()

@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_parametrized(your_object, input, expected):
    """Test with multiple inputs."""
    assert your_object.double(input) == expected
```

**Run tests:**

```bash
cd python/tests
pytest -v --cov=qlret
```

---

## Documentation

### Documentation Requirements

- **New features:** Must be documented in relevant guides
- **API changes:** Must update API reference
- **User-facing changes:** Must update user guide

### Adding Documentation

**User guides:** `docs/user-guide/`

```markdown
# Your Feature Title

Brief introduction explaining the feature.

## Basic Usage

```python
from qlret import QuantumSimulator

sim = QuantumSimulator(n_qubits=4)
sim.your_new_feature()
```

## Advanced Usage

Detailed explanation with examples...

## See Also

- [Related Guide](link-to-related-guide.md)
```

**Developer guides:** `docs/developer-guide/`

```markdown
# Implementing Your Feature

## Architecture

Explanation of how the feature fits into the system...

## Implementation Details

```cpp
class YourFeature {
    // Implementation details
};
```

## Testing

How to test the feature...
```

**API reference:** `docs/api-reference/`

Auto-generated from docstrings using Doxygen (C++) and Sphinx (Python).

---

## Pull Request Process

### Before Submitting

**Checklist:**

- [ ] Code follows style guidelines
- [ ] All tests pass (C++ and Python)
- [ ] New tests added for new features/fixes
- [ ] Documentation updated
- [ ] Commit messages follow conventions
- [ ] Branch is up-to-date with `main`
- [ ] No merge conflicts

**Run full test suite:**

```bash
# C++ tests
cd build
cmake .. -DBUILD_TESTS=ON
make -j$(nproc)
ctest --output-on-failure

# Python tests
cd python/tests
pytest -v --cov=qlret

# Check formatting
clang-format --dry-run --Werror src/*.cpp include/*.h
black --check python/qlret python/tests
```

---

### Creating Pull Request

**1. Push to your fork:**

```bash
git push origin feature/your-feature
```

**2. Open PR on GitHub:**

- Go to https://github.com/kunal5556/LRET/pulls
- Click "New Pull Request"
- Select your fork and branch
- Fill out PR template

**PR Template:**

```markdown
## Description

Brief description of changes.

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Code refactoring

## Related Issues

Closes #<issue-number>

## Testing

Describe testing performed:
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing performed

## Checklist

- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Commented hard-to-understand areas
- [ ] Documentation updated
- [ ] Tests added
- [ ] All tests pass
- [ ] No breaking changes (or documented)
```

---

## Code Review

### Review Process

1. **Automated checks:** CI runs tests and linters
2. **Maintainer review:** At least one maintainer reviews
3. **Address feedback:** Make requested changes
4. **Approval:** Maintainer approves PR
5. **Merge:** Maintainer merges into `main`

### Responding to Feedback

**Be respectful and collaborative:**

```markdown
> I think this function could be simplified.

Good point! I've refactored it to be more concise. See commit abc123.
```

**If you disagree, explain why:**

```markdown
> This seems overly complex.

I understand the concern. The complexity is necessary to handle edge case X
(see test_edge_case_x). Happy to discuss alternatives!
```

---

### Review Criteria

**Code quality:**
- [ ] Follows style guide
- [ ] No code duplication
- [ ] Clear variable/function names
- [ ] Appropriate comments

**Correctness:**
- [ ] Logic is correct
- [ ] Edge cases handled
- [ ] No obvious bugs
- [ ] Tests are comprehensive

**Performance:**
- [ ] No unnecessary allocations
- [ ] Efficient algorithms used
- [ ] Parallelization where appropriate

**Documentation:**
- [ ] Public API documented
- [ ] Complex logic explained
- [ ] Examples provided

---

## Community Guidelines

### Code of Conduct

**Our Pledge:**

We pledge to make participation in LRET a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity, level of experience, nationality, personal appearance, race, religion, or sexual identity.

**Our Standards:**

**Positive behaviors:**
- Using welcoming language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what's best for the community
- Showing empathy towards others

**Unacceptable behaviors:**
- Harassment or discriminatory language
- Personal attacks
- Trolling or insulting comments
- Public or private harassment
- Publishing others' private information

---

### Getting Help

**Channels:**

- **GitHub Issues:** Bug reports, feature requests
- **GitHub Discussions:** Questions, ideas, general discussion
- **Email:** kunal5556@example.com (maintainers)

**Issue Template:**

```markdown
## Description

Clear description of the issue.

## Steps to Reproduce

1. Step one
2. Step two
3. Step three

## Expected Behavior

What should happen.

## Actual Behavior

What actually happens.

## Environment

- OS: Ubuntu 22.04
- Compiler: GCC 11.3
- LRET version: 1.0.0
- Python version: 3.10

## Additional Context

Logs, screenshots, etc.
```

---

### Recognition

**Contributors will be:**
- Listed in `CONTRIBUTORS.md`
- Mentioned in release notes
- Credited in relevant documentation

---

## Quick Reference

### Common Commands

```bash
# Build
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
make -j$(nproc)

# Test
ctest --output-on-failure
cd python/tests && pytest -v

# Format
clang-format -i src/*.cpp include/*.h
black python/qlret python/tests

# Clean
rm -rf build

# Update branch
git fetch upstream
git rebase upstream/main
```

---

## See Also

- **[Development Setup](01-building-from-source.md)** - Detailed build instructions
- **[Testing Framework](05-testing.md)** - Testing guide
- **[Code Structure](02-code-structure.md)** - Repository organization
- **[Project Roadmap](../../ROADMAP.md)** - Future plans

---

## Thank You!

Thank you for contributing to LRET! Your efforts help make quantum simulation more accessible to everyone. 🎉

If you have questions about contributing, feel free to open a GitHub Discussion or contact the maintainers.

**Happy coding!** 🚀
