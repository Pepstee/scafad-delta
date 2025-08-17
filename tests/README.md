# SCAFAD Delta Test Suite

This directory contains a comprehensive testing suite for the SCAFAD Delta system, designed to ensure system reliability, correctness, and performance across all components.

## 🚀 Quick Start

### 1. Install Test Dependencies

```bash
# Install all test dependencies
pip install -r requirements-test.txt

# Or install core dependencies only
pip install pytest pytest-cov pytest-benchmark
```

### 2. Run Tests

```bash
# Run all tests
python tests/run_tests.py --all

# Run specific test categories
python tests/run_tests.py --categories unit privacy validation

# Run with coverage
python tests/run_tests.py --all --coverage

# Discover available tests
python tests/run_tests.py --discover
```

### 3. Run with pytest directly

```bash
# Run all tests
python -m pytest tests/

# Run specific categories
python -m pytest tests/unit/ tests/privacy/

# Run with coverage
python -m pytest tests/ --cov=core --cov=subsystems --cov=utils
```

## 📁 Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── run_tests.py             # Test runner script
├── requirements-test.txt     # Test dependencies
├── README.md                # This file
├── unit/                    # Unit tests for individual components
│   ├── test_layer1_core.py
│   ├── test_layer1_validation.py
│   ├── test_layer1_privacy.py
│   ├── test_layer1_preservation.py
│   ├── test_layer1_schema.py
│   ├── test_layer1_sanitization.py
│   └── test_layer1_hashing.py
├── integration/             # Integration tests
│   └── test_end_to_end_workflow.py
├── performance/             # Performance and benchmark tests
│   └── test_performance_benchmarks.py
├── preservation/            # Data preservation tests
│   └── test_anomaly_preservation.py
└── privacy/                 # Privacy compliance tests
    └── test_privacy_compliance.py
```

## 🧪 Test Categories

### Unit Tests (`tests/unit/`)
- **Core Module Tests**: Test individual functions and classes
- **Validation Tests**: Input validation and schema compliance
- **Privacy Tests**: GDPR/CCPA/HIPAA compliance functionality
- **Preservation Tests**: Anomaly detection and preservation
- **Schema Tests**: Schema evolution and migration
- **Sanitization Tests**: Data cleaning and normalization
- **Hashing Tests**: Cryptographic hashing and registry

### Integration Tests (`tests/integration/`)
- **End-to-End Workflows**: Complete processing pipelines
- **Component Interactions**: Cross-module functionality
- **System Integration**: Full system behavior validation

### Performance Tests (`tests/performance/`)
- **Benchmark Tests**: Performance measurement and comparison
- **Load Tests**: High-volume data processing
- **Stress Tests**: System limits and edge cases
- **Memory Tests**: Resource utilization analysis

### Specialized Tests
- **Preservation Tests**: Anomaly detection accuracy
- **Privacy Tests**: Compliance framework validation
- **Schema Tests**: Evolution and migration workflows

## 🎯 Test Runner Features

The `run_tests.py` script provides:

- **Multiple Execution Modes**: Fast, standard, thorough, performance, debug
- **Category Selection**: Run specific test types
- **Coverage Reporting**: Code coverage analysis
- **Performance Benchmarking**: Automated performance testing
- **Result Saving**: JSON output for CI/CD integration
- **Parallel Execution**: Multi-worker test execution

### Execution Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `fast` | Quick execution, minimal coverage | Development feedback |
| `standard` | Balanced execution, good coverage | Regular testing |
| `thorough` | Comprehensive execution, max coverage | Release testing |
| `performance` | Performance-focused execution | Benchmarking |
| `debug` | Verbose output, detailed errors | Troubleshooting |

### Examples

```bash
# Quick development testing
python tests/run_tests.py --categories unit --mode fast

# Comprehensive testing with coverage
python tests/run_tests.py --all --mode thorough --coverage

# Performance benchmarking
python tests/run_tests.py --performance

# Debug specific failures
python tests/run_tests.py --categories validation --mode debug

# Parallel execution
python tests/run_tests.py --all --parallel 4

# Save results for CI/CD
python tests/run_tests.py --all --save-results --output-file results.json
```

## 🔧 Configuration

### pytest.ini
The test suite uses a comprehensive pytest configuration with:
- Test discovery patterns
- Execution options
- Coverage settings
- Performance test configuration
- Logging setup
- Timeout settings

### conftest.py
Shared test fixtures provide:
- Sample test data
- Mock objects and services
- Performance test datasets
- Edge case data samples
- Test environment setup

## 📊 Coverage and Quality

### Coverage Targets
- **Core Functionality**: 95%+
- **Integration Paths**: 90%+
- **Error Handling**: 85%+
- **Performance Critical Paths**: 100%

### Quality Metrics
- **Test Execution Time**: Optimized for CI/CD pipelines
- **Memory Usage**: Monitored for resource efficiency
- **Error Detection**: Comprehensive failure scenarios
- **Edge Case Coverage**: Boundary condition testing

## 🚦 Running Tests in CI/CD

### GitHub Actions Example
```yaml
- name: Run Tests
  run: |
    pip install -r tests/requirements-test.txt
    python tests/run_tests.py --all --coverage --save-results

- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

### Jenkins Example
```groovy
stage('Test') {
    steps {
        sh 'pip install -r tests/requirements-test.txt'
        sh 'python tests/run_tests.py --all --coverage'
        publishHTML([
            allowMissing: false,
            alwaysLinkToLastBuild: true,
            keepAll: true,
            reportDir: 'htmlcov',
            reportFiles: 'index.html',
            reportName: 'Coverage Report'
        ])
    }
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure project root is in Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Missing Dependencies**
   ```bash
   # Install test requirements
   pip install -r tests/requirements-test.txt
   ```

3. **Test Discovery Issues**
   ```bash
   # Check test discovery
   python tests/run_tests.py --discover
   ```

4. **Performance Test Failures**
   ```bash
   # Run with debug mode
   python tests/run_tests.py --performance --mode debug
   ```

### Debug Mode
Use debug mode for detailed error information:
```bash
python tests/run_tests.py --categories unit --mode debug
```

## 📈 Performance Testing

### Benchmark Tests
Performance tests measure:
- **Processing Speed**: Records per second
- **Memory Usage**: Peak and average consumption
- **CPU Utilization**: Processing efficiency
- **Scalability**: Performance with data volume

### Running Benchmarks
```bash
# Run all performance tests
python tests/run_tests.py --performance

# Run specific benchmarks
python -m pytest tests/performance/ --benchmark-only

# Compare with previous runs
python -m pytest tests/performance/ --benchmark-autosave
```

## 🔒 Privacy and Security Testing

### Compliance Testing
Privacy tests validate:
- **GDPR Compliance**: Data protection requirements
- **CCPA Compliance**: California privacy laws
- **HIPAA Compliance**: Healthcare data protection
- **Data Redaction**: PII removal and masking

### Security Testing
Security tests verify:
- **Hash Verification**: Cryptographic integrity
- **Access Control**: Permission validation
- **Data Encryption**: Secure data handling
- **Audit Trails**: Compliance logging

## 📝 Writing New Tests

### Test Structure
```python
import pytest
from unittest.mock import Mock, patch

class TestNewFeature:
    """Test the new feature functionality."""
    
    def test_basic_functionality(self):
        """Test basic feature operation."""
        # Arrange
        feature = NewFeature()
        
        # Act
        result = feature.process("test_data")
        
        # Assert
        assert result.is_successful is True
        assert "processed" in result.data
    
    def test_error_handling(self):
        """Test error handling scenarios."""
        # Arrange
        feature = NewFeature()
        
        # Act & Assert
        with pytest.raises(ValueError):
            feature.process(None)
```

### Fixture Usage
```python
def test_with_fixtures(sample_telemetry_data, mock_service):
    """Test using shared fixtures."""
    # Use sample data and mocked services
    result = process_telemetry(sample_telemetry_data, mock_service)
    assert result is not None
```

### Performance Testing
```python
def test_performance(benchmark, large_dataset):
    """Test processing performance."""
    def process_data():
        return process_large_dataset(large_dataset)
    
    result = benchmark(process_data)
    assert result.stats.mean < 1.0  # Should complete in under 1 second
```

## 🤝 Contributing

### Adding Tests
1. **Follow Naming**: Use `test_*.py` for test files
2. **Use Fixtures**: Leverage shared test fixtures
3. **Add Coverage**: Ensure new code is well-tested
4. **Performance**: Include performance tests for critical paths
5. **Documentation**: Add docstrings to test classes and methods

### Test Standards
- **Descriptive Names**: Clear test method names
- **Arrange-Act-Assert**: Structured test layout
- **Edge Cases**: Test boundary conditions
- **Error Scenarios**: Validate error handling
- **Performance**: Measure execution time

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Coverage](https://pytest-cov.readthedocs.io/)
- [pytest-benchmark](https://pytest-benchmark.readthedocs.io/)
- [Testing Best Practices](https://realpython.com/python-testing/)

## 📞 Support

For test suite issues:
1. Check the troubleshooting section
2. Run with debug mode for detailed errors
3. Review test discovery output
4. Verify dependency installation

For system-specific issues:
1. Check core module documentation
2. Review integration test examples
3. Examine fixture configurations
4. Validate test data formats
