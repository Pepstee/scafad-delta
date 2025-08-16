# Scafad Layer1 Test Suite

This directory contains comprehensive tests for the Scafad Layer1 system, ensuring code quality, functionality, and performance across all components.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── unit/                       # Unit tests for individual modules
│   ├── __init__.py
│   ├── test_layer1_core.py    # Core processor tests
│   └── test_layer1_validation.py # Validation module tests
├── integration/                # End-to-end integration tests
│   ├── __init__.py
│   └── test_end_to_end_workflow.py # Complete workflow tests
├── performance/                # Performance benchmarking tests
│   ├── __init__.py
│   └── test_performance_benchmarks.py # Performance metrics
├── privacy/                    # Privacy compliance tests
│   ├── __init__.py
│   └── test_privacy_compliance.py # Privacy regulation tests
├── preservation/               # Anomaly preservation tests
│   ├── __init__.py
│   └── test_anomaly_preservation.py # Anomaly handling tests
└── README.md                   # This file
```

## Test Categories

### 1. Unit Tests (`unit/`)
- **Purpose**: Test individual components in isolation
- **Coverage**: Core modules, utility functions, and classes
- **Dependencies**: Minimal external dependencies, focused testing
- **Examples**: 
  - `test_layer1_core.py`: Tests the main orchestrator
  - `test_layer1_validation.py`: Tests validation logic

### 2. Integration Tests (`integration/`)
- **Purpose**: Test component interactions and end-to-end workflows
- **Coverage**: Complete data processing pipelines
- **Dependencies**: Full system integration, realistic data scenarios
- **Examples**:
  - `test_end_to_end_workflow.py`: Complete processing workflows

### 3. Performance Tests (`performance/`)
- **Purpose**: Measure system performance characteristics
- **Coverage**: Latency, throughput, memory usage, scalability
- **Dependencies**: Performance monitoring tools, large datasets
- **Examples**:
  - `test_performance_benchmarks.py`: Performance metrics and benchmarks

### 4. Privacy Tests (`privacy/`)
- **Purpose**: Validate privacy compliance and PII handling
- **Coverage**: GDPR, CCPA, LGPD compliance, data anonymization
- **Dependencies**: Privacy policy engine, compliance frameworks
- **Examples**:
  - `test_privacy_compliance.py`: Privacy regulation compliance

### 5. Preservation Tests (`preservation/`)
- **Purpose**: Test anomaly detection and preservation mechanisms
- **Coverage**: Anomaly detection, context preservation, data integrity
- **Dependencies**: Anomaly detection algorithms, preservation engine
- **Examples**:
  - `test_anomaly_preservation.py`: Anomaly handling and preservation

## Running Tests

### Prerequisites
```bash
# Install required dependencies
pip install -r requirements.txt

# Install test-specific dependencies
pip install psutil pytest pytest-cov
```

### Basic Test Execution
```bash
# Run all tests
python -m pytest tests/

# Run specific test category
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/performance/
python -m pytest tests/privacy/
python -m pytest tests/preservation/

# Run specific test file
python -m pytest tests/unit/test_layer1_core.py

# Run specific test class
python -m pytest tests/unit/test_layer1_core.py::TestLayer1Processor

# Run specific test method
python -m pytest tests/unit/test_layer1_core.py::TestLayer1Processor::test_processor_initialization
```

### Advanced Test Execution
```bash
# Run with coverage reporting
python -m pytest tests/ --cov=core --cov=subsystems --cov-report=html

# Run with verbose output
python -m pytest tests/ -v

# Run with parallel execution
python -m pytest tests/ -n auto

# Run only fast tests (exclude performance tests)
python -m pytest tests/ -m "not slow"

# Run tests matching pattern
python -m pytest tests/ -k "privacy"
```

### Test Configuration
```bash
# Set test environment
export TEST_ENV=development
export TEST_DB_URL=sqlite:///test.db

# Run with custom configuration
python -m pytest tests/ --config=tests/test_config.yaml
```

## Test Data and Fixtures

### Sample Data
Tests use realistic sample data to ensure comprehensive coverage:

```python
# Example test data structure
sample_data = {
    "id": "test_123",
    "content": "Sample content for testing",
    "metadata": {
        "source": "test_system",
        "timestamp": "2024-01-01T00:00:00Z",
        "user_id": "user_456"
    },
    "anomalies": [
        {
            "type": "outlier",
            "field": "value",
            "score": 0.95,
            "description": "Test anomaly"
        }
    ]
}
```

### Test Fixtures
Common test fixtures are defined in test classes:

```python
def setUp(self):
    """Set up test fixtures."""
    self.processor = Layer1Processor()
    self.validator = DataValidator()
    self.sample_data = {...}
```

## Test Assertions and Validation

### Common Assertions
```python
# Basic assertions
self.assertIsNotNone(result)
self.assertEqual(result['id'], expected_id)
self.assertIn('processed_at', result)

# Performance assertions
self.assertLess(latency, threshold)
self.assertGreater(throughput, min_throughput)
self.assertLess(memory_usage, max_memory)

# Privacy assertions
self.assertNotIn("email@example.com", str(result))
self.assertIn("hashed_fields", result)

# Anomaly assertions
self.assertIn("anomalies", result)
self.assertEqual(len(result["anomalies"]), expected_count)
```

### Custom Assertions
```python
def assert_data_integrity(self, original_data, processed_data):
    """Assert that data integrity is maintained."""
    self.assertEqual(processed_data['id'], original_data['id'])
    self.assertIn('processing_metadata', processed_data)

def assert_privacy_compliance(self, data, privacy_level):
    """Assert privacy compliance for given level."""
    if privacy_level == "strict":
        self.assertNotIn("email", str(data))
        self.assertIn("anonymized", str(data))
```

## Performance Testing

### Performance Metrics
- **Latency**: Processing time per record
- **Throughput**: Records processed per second
- **Memory Usage**: Memory consumption during processing
- **Scalability**: Performance with increasing data sizes
- **Concurrency**: Multi-threaded processing performance

### Performance Thresholds
```python
# Performance thresholds
self.latency_threshold = 0.1      # 100ms
self.throughput_threshold = 1000  # records/second
self.memory_threshold = 100 * 1024 * 1024  # 100MB
```

### Performance Monitoring
```python
import time
import psutil

# Measure execution time
start_time = time.perf_counter()
result = self.processor.process(data)
end_time = time.perf_counter()
latency = end_time - start_time

# Measure memory usage
process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss
```

## Privacy Testing

### Compliance Frameworks
- **GDPR**: General Data Protection Regulation
- **CCPA**: California Consumer Privacy Act
- **LGPD**: Brazilian Data Protection Law
- **PIPEDA**: Canadian Privacy Law

### Privacy Test Scenarios
```python
# PII detection tests
def test_pii_detection(self):
    detected_pii = self.privacy_filter.detect_pii(sensitive_data)
    self.assertIn("emails", detected_pii)
    self.assertIn("credit_cards", detected_pii)

# Anonymization tests
def test_data_anonymization(self):
    anonymized_data = self.privacy_filter.anonymize_data(sensitive_data)
    self.assertNotIn("john.doe@example.com", str(anonymized_data))

# Consent management tests
def test_consent_management(self):
    consent_id = self.policy_engine.record_consent(...)
    is_valid = self.policy_engine.validate_consent(consent_id, ...)
    self.assertTrue(is_valid)
```

## Anomaly Preservation Testing

### Anomaly Types
- **Statistical Outliers**: Values outside normal ranges
- **Pattern Anomalies**: Deviations from expected patterns
- **Behavioral Anomalies**: Unusual user behavior
- **Context Anomalies**: Unexpected data relationships

### Preservation Mechanisms
```python
# Anomaly detection
outlier_result = self.preserver.detect_statistical_anomalies(data, threshold=2.0)

# Context preservation
context_result = self.preserver.extract_anomaly_context(data, anomaly_index=0)

# Metadata preservation
metadata_result = self.preserver.preserve_anomaly_metadata(anomaly_data, additional_context)
```

## Test Reporting and Analysis

### Coverage Reports
```bash
# Generate HTML coverage report
python -m pytest tests/ --cov=core --cov-report=html

# Generate XML coverage report for CI/CD
python -m pytest tests/ --cov=core --cov-report=xml
```

### Test Results
```bash
# Run tests with detailed output
python -m pytest tests/ -v --tb=short

# Generate JUnit XML report
python -m pytest tests/ --junitxml=test-results.xml

# Run with custom markers
python -m pytest tests/ -m "privacy or performance"
```

## Continuous Integration

### CI/CD Integration
Tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions configuration
- name: Run Tests
  run: |
    python -m pytest tests/ --cov=core --cov-report=xml
    python -m pytest tests/performance/ --durations=10
    python -m pytest tests/privacy/ --tb=short
```

### Test Automation
```bash
# Automated test execution
./scripts/run_tests.sh

# Test result analysis
./scripts/analyze_results.py

# Performance regression detection
./scripts/check_performance.py
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use sys.path in tests
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
```

#### Performance Test Failures
```bash
# Check system resources
htop
free -h
df -h

# Run performance tests with longer timeouts
python -m pytest tests/performance/ --timeout=300
```

#### Privacy Test Failures
```bash
# Verify privacy configuration
python -c "from core.layer1_privacy import PrivacyFilter; print(PrivacyFilter().get_default_settings())"

# Check compliance frameworks
python -c "from subsystems.privacy_policy_engine import PrivacyPolicyEngine; print(PrivacyPolicyEngine().get_supported_regulations())"
```

### Debug Mode
```bash
# Run tests with debug output
python -m pytest tests/ -s --pdb

# Run specific test with debugger
python -m pytest tests/unit/test_layer1_core.py::TestLayer1Processor::test_processor_initialization --pdb
```

## Contributing

### Adding New Tests
1. **Follow Naming Convention**: `test_<feature_name>.py`
2. **Use Descriptive Names**: Clear test method names
3. **Include Documentation**: Docstrings for test classes and methods
4. **Add to Appropriate Category**: Unit, integration, performance, privacy, or preservation
5. **Update This README**: Document new test categories or patterns

### Test Standards
- **Isolation**: Tests should not depend on each other
- **Deterministic**: Tests should produce consistent results
- **Fast**: Unit tests should complete quickly
- **Comprehensive**: Cover edge cases and error conditions
- **Maintainable**: Easy to understand and modify

### Test Data Management
- **Realistic Data**: Use realistic but safe test data
- **No Sensitive Information**: Avoid real PII in tests
- **Consistent Format**: Maintain consistent data structures
- **Version Control**: Include test data in version control

## Support and Resources

### Documentation
- **API Reference**: Complete API documentation
- **Architecture Guide**: System architecture overview
- **Privacy Compliance**: Privacy regulation compliance guide
- **Schema Evolution**: Schema versioning and migration guide

### Support Channels
- **Development Team**: Technical implementation support
- **Privacy Team**: Privacy compliance assistance
- **Performance Team**: Performance optimization guidance
- **Testing Team**: Test framework and methodology support

### Additional Resources
- **Test Framework**: pytest documentation and best practices
- **Performance Testing**: Performance testing methodologies
- **Privacy Testing**: Privacy testing frameworks and tools
- **Anomaly Detection**: Anomaly detection algorithms and techniques

---

*This test suite is maintained by the Scafad Development Team and updated regularly to ensure comprehensive coverage and quality assurance.*
