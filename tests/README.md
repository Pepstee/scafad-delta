# Tests

This directory contains comprehensive testing suite for the SCAFAD Delta system.

## Overview

The `tests` folder provides unit tests, integration tests, and performance tests to ensure system reliability, correctness, and performance.

## Contents

- **__init__.py** - Package initialization file
- **README.md** - This file
- **unit/** - Unit tests for individual components
  - **__init__.py** - Unit test package initialization
  - **test_layer1_core.py** - Core Layer 1 functionality tests
  - **test_layer1_validation.py** - Validation system tests
- **integration/** - Integration tests for component interactions
  - **__init__.py** - Integration test package initialization
  - **test_end_to_end_workflow.py** - End-to-end workflow tests
- **performance/** - Performance and benchmark tests
  - **__init__.py** - Performance test package initialization
  - **test_performance_benchmarks.py** - Performance benchmark tests
- **preservation/** - Data preservation specific tests
  - **__init__.py** - Preservation test package initialization
  - **test_anomaly_preservation.py** - Anomaly preservation tests
- **privacy/** - Privacy and compliance tests
  - **__init__.py** - Privacy test package initialization
  - **test_privacy_compliance.py** - Privacy compliance tests

## Test Categories

### Unit Tests
- Individual component functionality
- Isolated testing with mocked dependencies
- Fast execution for development feedback

### Integration Tests
- Component interaction testing
- End-to-end workflow validation
- Real dependency testing

### Performance Tests
- Benchmarking and performance regression
- Load testing and stress testing
- Resource utilization analysis

### Specialized Tests
- Preservation quality validation
- Privacy compliance verification
- Schema evolution testing

## Running Tests

### All Tests
```bash
python -m pytest tests/
```

### Specific Test Categories
```bash
# Unit tests only
python -m pytest tests/unit/

# Integration tests only
python -m pytest tests/integration/

# Performance tests only
python -m pytest tests/performance/
```

### Specific Test Files
```bash
python -m pytest tests/unit/test_layer1_core.py
```

## Test Configuration

Tests use configuration from:
- `pytest.ini` or `pyproject.toml`
- Environment variables
- Test-specific configuration files
- Mock data and fixtures

## Test Data

Test data is managed through:
- Fixtures defined in test files
- Mock data generators
- Sample data files
- Database snapshots

## Coverage

Test coverage targets:
- Core functionality: 95%+
- Integration paths: 90%+
- Error handling: 85%+
- Performance critical paths: 100%

## Related Resources

- See `docs/` directory for system documentation
- See `examples/` directory for usage examples
- See `evaluation/` directory for performance analysis tools
