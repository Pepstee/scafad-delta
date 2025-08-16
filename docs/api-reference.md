# SCAFAD Layer 1: API Reference

## Overview

This document provides comprehensive API reference documentation for SCAFAD Layer 1's behavioral intake zone. It covers all public interfaces, data models, and configuration options.

## Core API Reference

### Layer1_BehavioralIntakeZone

The main orchestrator class that coordinates all behavioral intake and data conditioning processes.

#### Constructor

```python
def __init__(self, config: Optional[Layer1Config] = None)
```

**Parameters:**
- `config` (Optional[Layer1Config]): Configuration object. If None, default configuration is used.

**Example:**
```python
from core.layer1_core import Layer1_BehavioralIntakeZone
from configs.layer1_config import Layer1Config

# With custom configuration
config = Layer1Config(
    processing_mode=ProcessingMode.REAL_TIME,
    performance_profile=PerformanceProfile.ULTRA_LOW_LATENCY
)
layer1 = Layer1_BehavioralIntakeZone(config)

# With default configuration
layer1 = Layer1_BehavioralIntakeZone()
```

#### Main Processing Method

```python
async def process_telemetry_batch(
    self, 
    telemetry_records: List[TelemetryRecord],
    processing_context: Optional[Dict[str, Any]] = None
) -> ProcessedBatch
```

**Parameters:**
- `telemetry_records` (List[TelemetryRecord]): List of telemetry records to process
- `processing_context` (Optional[Dict[str, Any]]): Optional context for batch processing

**Returns:**
- `ProcessedBatch`: Processing result containing cleaned records and metadata

**Example:**
```python
# Process a batch of telemetry records
telemetry_batch = [
    TelemetryRecord(
        event_id="event_001",
        timestamp=datetime.now(timezone.utc),
        function_id="function_001",
        telemetry_data={"cpu_usage": 75.5, "memory_usage": 256}
    )
]

result = await layer1.process_telemetry_batch(telemetry_batch)
print(f"Processed {len(result.cleaned_records)} records")
print(f"Preservation rate: {result.preservation_report.preservation_rate:.2%}")
```

#### Performance Monitoring

```python
def get_performance_metrics(self) -> Dict[str, Any]
```

**Returns:**
- `Dict[str, Any]`: Current performance metrics

**Example:**
```python
metrics = layer1.get_performance_metrics()
print(f"Total records processed: {metrics['total_records_processed']}")
print(f"Average latency: {metrics['average_latency_ms']:.2f} ms")
```

## Data Models

### TelemetryRecord

Represents a single telemetry record from Layer 0.

```python
@dataclass
class TelemetryRecord:
    event_id: str
    timestamp: datetime
    function_id: str
    session_id: Optional[str] = None
    telemetry_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Fields:**
- `event_id` (str): Unique identifier for the event
- `timestamp` (datetime): Event timestamp
- `function_id` (str): Identifier for the function that generated the telemetry
- `session_id` (Optional[str]): Optional session identifier
- `telemetry_data` (Dict[str, Any]): Telemetry data payload
- `metadata` (Dict[str, Any]): Additional metadata

**Example:**
```python
from datetime import datetime, timezone
from core.layer1_core import TelemetryRecord

record = TelemetryRecord(
    event_id="event_123",
    timestamp=datetime.now(timezone.utc),
    function_id="lambda_function_001",
    session_id="session_456",
    telemetry_data={
        "cpu_usage": 85.2,
        "memory_usage": 512,
        "execution_time_ms": 150,
        "error_count": 0
    },
    metadata={
        "source": "aws_lambda",
        "region": "us-east-1",
        "environment": "production"
    }
)
```

### ProcessedBatch

Represents the result of processing a batch of telemetry records.

```python
@dataclass
class ProcessedBatch:
    batch_id: str
    original_count: int
    cleaned_records: List[ProcessedRecord]
    processing_summary: ProcessingSummary
    privacy_audit_trail: PrivacyAuditTrail
    preservation_report: PreservationReport
    quality_metrics: QualityMetrics
    audit_trail: ProcessingAudit
    metadata: Optional[Dict[str, Any]] = None
```

**Fields:**
- `batch_id` (str): Unique identifier for the batch
- `original_count` (int): Number of original records
- `cleaned_records` (List[ProcessedRecord]): Processed and cleaned records
- `processing_summary` (ProcessingSummary): Summary of processing results
- `privacy_audit_trail` (PrivacyAuditTrail): Privacy compliance audit trail
- `preservation_report` (PreservationReport): Anomaly preservation report
- `quality_metrics` (QualityMetrics): Data quality metrics
- `audit_trail` (ProcessingAudit): Complete processing audit trail
- `metadata` (Optional[Dict[str, Any]]): Additional metadata

### ProcessedRecord

Represents a single processed telemetry record.

```python
@dataclass
class ProcessedRecord:
    original_record: TelemetryRecord
    cleaned_data: Dict[str, Any]
    schema_version: str
    processing_flags: List[str]
    quality_score: float
    preservation_indicators: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
```

## Configuration API

### Layer1Config

Main configuration class for Layer 1.

```python
@dataclass
class Layer1Config:
    processing_mode: ProcessingMode = ProcessingMode.REAL_TIME
    performance_profile: PerformanceProfile = PerformanceProfile.BALANCED
    privacy_compliance_level: PrivacyComplianceLevel = PrivacyComplianceLevel.STANDARD
    anomaly_preservation_mode: AnomalyPreservationMode = AnomalyPreservationMode.BALANCED
    schema_evolution_strategy: SchemaEvolutionStrategy = SchemaEvolutionStrategy.MIGRATE
    enable_debug_mode: bool = False
    max_processing_latency_ms: float = 2.0
    max_batch_size: int = 1000
    hash_algorithms: List[str] = field(default_factory=lambda: ["sha256", "sha512"])
    enable_audit_logging: bool = True
    enable_performance_monitoring: bool = True
```

**Configuration Options:**

#### ProcessingMode
- `REAL_TIME`: Sub-2ms latency, minimal buffering
- `BATCH_OPTIMIZED`: Optimized for throughput
- `QUALITY_FOCUSED`: Maximum quality, higher latency
- `RESEARCH`: Full preservation, no time constraints

#### PerformanceProfile
- `ULTRA_LOW_LATENCY`: <1ms target
- `BALANCED`: 2ms target, optimized
- `HIGH_THROUGHPUT`: 5ms target, max throughput
- `QUALITY_OPTIMIZED`: 10ms target, best quality

#### PrivacyComplianceLevel
- `MINIMAL`: Basic PII redaction
- `STANDARD`: GDPR/CCPA compliance
- `STRICT`: HIPAA/SOX compliance
- `MAXIMUM`: Maximum privacy protection

#### AnomalyPreservationMode
- `CONSERVATIVE`: Preserve only critical anomaly features
- `BALANCED`: Balance preservation with performance
- `AGGRESSIVE`: Maximum anomaly feature preservation
- `RESEARCH`: Full preservation for research

#### SchemaEvolutionStrategy
- `STRICT`: Reject incompatible schemas
- `MIGRATE`: Attempt automatic migration
- `FLEXIBLE`: Accept with warnings
- `AUTO_LEARN`: Learn and adapt automatically

**Example:**
```python
from configs.layer1_config import Layer1Config, ProcessingMode, PerformanceProfile

config = Layer1Config(
    processing_mode=ProcessingMode.REAL_TIME,
    performance_profile=PerformanceProfile.ULTRA_LOW_LATENCY,
    privacy_compliance_level=PrivacyComplianceLevel.STRICT,
    anomaly_preservation_mode=AnomalyPreservationMode.AGGRESSIVE,
    max_processing_latency_ms=1.5,
    max_batch_size=500,
    enable_debug_mode=True
)
```

## Subsystem APIs

### Schema Registry

Manages schema versioning and compatibility.

```python
from subsystems.schema_registry import SchemaRegistry

# Initialize schema registry
registry = SchemaRegistry()

# Register new schema version
registry.register_schema_version("v2.2", schema_definition)

# Check compatibility
compatibility = registry.check_compatibility("v2.1", "v2.2")

# Get migration path
migration_path = registry.get_migration_path("v2.1", "v2.2")
```

### Privacy Policy Engine

Manages dynamic privacy policy application.

```python
from subsystems.privacy_policy_engine import PrivacyPolicyEngine

# Initialize privacy policy engine
policy_engine = PrivacyPolicyEngine()

# Update privacy policy
policy_engine.update_privacy_policy("gdpr", new_gdpr_policy)

# Apply privacy filters
filtered_data = policy_engine.apply_privacy_filters(telemetry_data)

# Generate compliance report
compliance_report = policy_engine.generate_compliance_report()
```

### Semantic Analyzer

Analyzes behavioral semantics and preserves anomaly signatures.

```python
from subsystems.semantic_analyzer import SemanticAnalyzer

# Initialize semantic analyzer
analyzer = SemanticAnalyzer()

# Analyze behavioral features
features = analyzer.extract_behavioral_features(telemetry_data)

# Assess preservation risk
risk_assessment = analyzer.assess_preservation_risk(original_data, processed_data)

# Generate preservation recommendations
recommendations = analyzer.generate_preservation_recommendations(features)
```

### Quality Monitor

Monitors data quality and processing effectiveness.

```python
from subsystems.quality_monitor import QualityAssuranceMonitor

# Initialize quality monitor
monitor = QualityAssuranceMonitor()

# Assess batch quality
quality_metrics = monitor.assess_batch_quality(processed_batch)

# Monitor processing performance
performance_metrics = monitor.get_performance_metrics()

# Generate quality report
quality_report = monitor.generate_quality_report()
```

### Audit Trail Generator

Generates comprehensive processing audit trails.

```python
from subsystems.audit_trail_generator import AuditTrailGenerator

# Initialize audit trail generator
audit_generator = AuditTrailGenerator()

# Generate processing audit
audit_trail = audit_generator.generate_processing_audit(processing_result)

# Export audit data
audit_export = audit_generator.export_audit_data(audit_trail, format="json")

# Generate compliance report
compliance_report = audit_generator.generate_compliance_report(audit_trail)
```

## Utility APIs

### Hash Library

Provides cryptographic hash functions and utilities.

```python
from utils.hash_library import CryptographicHasher

# Initialize hasher
hasher = CryptographicHasher(["sha256", "sha512"])

# Hash data
hash_result = hasher.hash_data(data, algorithm="sha256")

# Verify hash
is_valid = hasher.verify_hash(data, hash_result.hash_value, algorithm="sha256")

# Get performance statistics
stats = hasher.get_performance_statistics()
```

### Redaction Manager

Manages PII redaction policies and execution.

```python
from utils.redaction_manager import RedactionPolicyManager

# Initialize redaction manager
redaction_manager = RedactionPolicyManager()

# Apply redaction policy
redacted_data = redaction_manager.apply_redaction_policy(data, policy_name)

# Update redaction rules
redaction_manager.update_redaction_rules(policy_name, new_rules)

# Get redaction effectiveness
effectiveness = redaction_manager.get_redaction_effectiveness()
```

### Field Mapper

Manages schema field mapping and transformation.

```python
from utils.field_mapper import FieldMappingEngine

# Initialize field mapper
field_mapper = FieldMappingEngine()

# Create field mapping
mapping = field_mapper.create_field_mapping(source_schema, target_schema)

# Apply field mapping
mapped_data = field_mapper.apply_field_mapping(data, mapping)

# Validate field mapping
validation_result = field_mapper.validate_field_mapping(mapping)
```

### Compression Optimizer

Optimizes payload sizes through intelligent compression.

```python
from utils.compression_optimizer import CompressionOptimizer

# Initialize compression optimizer
compression_optimizer = CompressionOptimizer()

# Optimize payload
compression_result = compression_optimizer.optimize_payload(data)

# Optimize for anomaly preservation
preservation_result = compression_optimizer.optimize_for_anomaly_preservation(data)

# Get performance statistics
stats = compression_optimizer.get_performance_statistics()
```

### Validators

Provides input validation utilities.

```python
from utils.validators import TelemetryRecordValidator

# Initialize validator
validator = TelemetryRecordValidator()

# Validate telemetry record
validation_result = validator.validate_telemetry_record(record)

# Validate batch
batch_validation = validator.validate_telemetry_batch(records)

# Get validation errors
errors = validator.get_validation_errors(validation_result)
```

## Error Handling

### Error Types

```python
class Layer1Error(Exception):
    """Base exception for Layer 1 errors"""
    pass

class ValidationError(Layer1Error):
    """Validation-related errors"""
    pass

class SchemaError(Layer1Error):
    """Schema-related errors"""
    pass

class PrivacyError(Layer1Error):
    """Privacy compliance errors"""
    pass

class PreservationError(Layer1Error):
    """Anomaly preservation errors"""
    pass

class ProcessingError(Layer1Error):
    """General processing errors"""
    pass
```

### Error Handling Example

```python
try:
    result = await layer1.process_telemetry_batch(telemetry_records)
except ValidationError as e:
    print(f"Validation error: {e}")
    # Handle validation errors
except PrivacyError as e:
    print(f"Privacy error: {e}")
    # Handle privacy errors
except PreservationError as e:
    print(f"Preservation error: {e}")
    # Handle preservation errors
except ProcessingError as e:
    print(f"Processing error: {e}")
    # Handle general processing errors
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle unexpected errors
```

## Performance Tuning

### Configuration Optimization

```python
# For ultra-low latency
config = Layer1Config(
    processing_mode=ProcessingMode.REAL_TIME,
    performance_profile=PerformanceProfile.ULTRA_LOW_LATENCY,
    max_processing_latency_ms=1.0,
    max_batch_size=100
)

# For high throughput
config = Layer1Config(
    processing_mode=ProcessingMode.BATCH_OPTIMIZED,
    performance_profile=PerformanceProfile.HIGH_THROUGHPUT,
    max_processing_latency_ms=5.0,
    max_batch_size=5000
)

# For maximum quality
config = Layer1Config(
    processing_mode=ProcessingMode.QUALITY_FOCUSED,
    performance_profile=PerformanceProfile.QUALITY_OPTIMIZED,
    anomaly_preservation_mode=AnomalyPreservationMode.AGGRESSIVE,
    max_processing_latency_ms=10.0
)
```

### Performance Monitoring

```python
# Get real-time performance metrics
metrics = layer1.get_performance_metrics()

# Monitor specific phases
phase_metrics = layer1.get_phase_performance_metrics()

# Get memory usage
memory_usage = layer1.get_memory_usage()

# Get throughput statistics
throughput_stats = layer1.get_throughput_statistics()
```

## Logging and Debugging

### Logging Configuration

```python
import logging

# Configure logging level
logging.getLogger("SCAFAD.Layer1").setLevel(logging.DEBUG)

# Enable debug mode in configuration
config = Layer1Config(enable_debug_mode=True)
```

### Debug Information

```python
# Get debug information
debug_info = layer1.get_debug_information()

# Get processing state
processing_state = layer1.get_processing_state()

# Get component status
component_status = layer1.get_component_status()
```

## Testing and Validation

### Unit Testing

```python
import pytest
from core.layer1_core import Layer1_BehavioralIntakeZone

def test_layer1_initialization():
    layer1 = Layer1_BehavioralIntakeZone()
    assert layer1 is not None
    assert layer1.config is not None

def test_telemetry_processing():
    layer1 = Layer1_BehavioralIntakeZone()
    test_record = TelemetryRecord(...)
    result = await layer1.process_telemetry_batch([test_record])
    assert result is not None
    assert len(result.cleaned_records) == 1
```

### Integration Testing

```python
def test_full_pipeline():
    layer1 = Layer1_BehavioralIntakeZone()
    
    # Test data
    test_records = [create_test_record() for _ in range(100)]
    
    # Process batch
    result = await layer1.process_telemetry_batch(test_records)
    
    # Validate results
    assert result.original_count == 100
    assert result.processing_summary.success_count == 100
    assert result.preservation_report.preservation_rate >= 0.95
```

## Best Practices

### Configuration Management

1. **Use Environment Variables**: Store sensitive configuration in environment variables
2. **Validate Configuration**: Always validate configuration before use
3. **Default Values**: Provide sensible defaults for all configuration options
4. **Configuration Validation**: Use configuration schemas for validation

### Error Handling

1. **Specific Exceptions**: Catch specific exception types, not generic Exception
2. **Error Logging**: Log all errors with appropriate context
3. **Graceful Degradation**: Implement fallback mechanisms for non-critical failures
4. **User Feedback**: Provide meaningful error messages to users

### Performance Optimization

1. **Batch Processing**: Use appropriate batch sizes for your use case
2. **Async Processing**: Use async/await for I/O operations
3. **Memory Management**: Monitor memory usage and implement cleanup
4. **Caching**: Use caching for frequently accessed data

### Security and Privacy

1. **Data Validation**: Always validate input data
2. **Privacy Compliance**: Ensure compliance with relevant regulations
3. **Audit Logging**: Maintain comprehensive audit trails
4. **Access Control**: Implement appropriate access controls

## Conclusion

This API reference provides comprehensive documentation for SCAFAD Layer 1's behavioral intake zone. The modular design and comprehensive configuration options allow for flexible deployment and optimization for various use cases.

For additional information, refer to:
- [Architecture Documentation](architecture.md)
- [Privacy Compliance Guide](privacy-compliance.md)
- [Schema Evolution Guide](schema-evolution.md)
- [Performance Optimization Guide](performance.md)
