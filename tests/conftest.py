"""
Pytest configuration and shared fixtures for SCAFAD Delta test suite.

This file provides:
- Shared test fixtures
- Test data generators
- Mock configurations
- Test environment setup
"""

import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
from datetime import datetime, timezone

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Core imports for testing
from core.layer1_core import (
    Layer1Processor, ProcessingPipeline, ProcessingStage,
    TelemetryRecord, ProcessingResult, ProcessingConfig
)
from core.layer1_validation import InputValidationGateway, ValidationResult
from core.layer1_schema import SchemaEvolutionEngine, SchemaMetadata
from core.layer1_sanitization import SanitizationProcessor
from core.layer1_privacy import PrivacyComplianceFilter
from core.layer1_hashing import DeferredHashingManager
from core.layer1_preservation import AnomalyPreservationGuard

# Subsystem imports
from subsystems.schema_registry import SchemaRegistry
from subsystems.privacy_policy_engine import PrivacyPolicyEngine
from subsystems.semantic_analyzer import SemanticAnalyzer
from subsystems.quality_monitor import QualityAssuranceMonitor
from subsystems.audit_trail_generator import AuditTrailGenerator

# Utility imports
from utils.hash_library import HashFunction, CryptographicHasher
from utils.redaction_manager import RedactionPolicyManager
from utils.field_mapper import FieldMappingEngine
from utils.compression_optimizer import CompressionOptimizer
from utils.validators import TelemetryRecordValidator


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary test data directory."""
    temp_dir = tempfile.mkdtemp(prefix="scafad_test_")
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def sample_telemetry_data():
    """Sample telemetry data for testing."""
    return {
        "id": "test_123",
        "timestamp": "2024-01-01T00:00:00Z",
        "source": "test_device",
        "data": {
            "cpu_usage": 75.5,
            "memory_usage": 60.2,
            "network_activity": 1024,
            "anomaly_score": 0.85
        },
        "metadata": {
            "version": "1.0",
            "environment": "test",
            "user_id": "user_456"
        }
    }


@pytest.fixture(scope="session")
def sample_anomaly_data():
    """Sample anomaly data for preservation testing."""
    return {
        "id": "anomaly_789",
        "timestamp": "2024-01-01T00:00:00Z",
        "anomaly_type": "performance_spike",
        "severity": "high",
        "indicators": {
            "cpu_spike": 95.8,
            "memory_leak": True,
            "response_time": 2.5
        },
        "context": {
            "process": "test_service",
            "user_session": "session_123"
        }
    }


@pytest.fixture(scope="session")
def sample_privacy_sensitive_data():
    """Sample data with privacy-sensitive information."""
    return {
        "id": "privacy_test_001",
        "timestamp": "2024-01-01T00:00:00Z",
        "user": {
            "email": "user@example.com",
            "phone": "+1234567890",
            "ip_address": "192.168.1.100",
            "location": "New York, NY"
        },
        "behavioral_data": {
            "login_time": "08:30:00",
            "pages_visited": ["/dashboard", "/profile", "/settings"],
            "session_duration": 1800
        }
    }


@pytest.fixture(scope="session")
def mock_schema_registry():
    """Mock schema registry for testing."""
    registry = Mock(spec=SchemaRegistry)
    registry.get_schema.return_value = {
        "version": "1.0",
        "fields": ["id", "timestamp", "data", "metadata"],
        "required": ["id", "timestamp"]
    }
    registry.validate_schema.return_value = True
    return registry


@pytest.fixture(scope="session")
def mock_privacy_engine():
    """Mock privacy policy engine for testing."""
    engine = Mock(spec=PrivacyPolicyEngine)
    engine.apply_policy.return_value = {
        "redacted_fields": ["email", "phone"],
        "compliance_status": "compliant",
        "audit_trail": "audit_123"
    }
    return engine


@pytest.fixture(scope="session")
def mock_semantic_analyzer():
    """Mock semantic analyzer for testing."""
    analyzer = Mock(spec=SemanticAnalyzer)
    analyzer.extract_features.return_value = {
        "behavioral_pattern": "normal",
        "anomaly_likelihood": 0.15,
        "feature_vector": [0.1, 0.2, 0.3, 0.4]
    }
    return analyzer


@pytest.fixture(scope="session")
def mock_quality_monitor():
    """Mock quality assurance monitor for testing."""
    monitor = Mock(spec=QualityAssuranceMonitor)
    monitor.assess_quality.return_value = {
        "overall_score": 0.92,
        "data_quality": 0.95,
        "preservation_score": 0.89,
        "privacy_score": 0.98
    }
    return monitor


@pytest.fixture(scope="session")
def mock_audit_generator():
    """Mock audit trail generator for testing."""
    generator = Mock(spec=AuditTrailGenerator)
    generator.generate_audit.return_value = {
        "audit_id": "audit_456",
        "timestamp": "2024-01-01T00:00:00Z",
        "actions": ["validation", "sanitization", "privacy_filtering"],
        "status": "success"
    }
    return generator


@pytest.fixture(scope="function")
def layer1_processor():
    """Create a Layer1Processor instance for testing."""
    return Layer1Processor()


@pytest.fixture(scope="function")
def processing_pipeline():
    """Create a ProcessingPipeline instance for testing."""
    return ProcessingPipeline()


@pytest.fixture(scope="function")
def validation_gateway():
    """Create an InputValidationGateway instance for testing."""
    return InputValidationGateway()


@pytest.fixture(scope="function")
def schema_engine():
    """Create a SchemaEvolutionEngine instance for testing."""
    return SchemaEvolutionEngine()


@pytest.fixture(scope="function")
def sanitization_processor():
    """Create a SanitizationProcessor instance for testing."""
    return SanitizationProcessor()


@pytest.fixture(scope="function")
def privacy_filter():
    """Create a PrivacyComplianceFilter instance for testing."""
    return PrivacyComplianceFilter()


@pytest.fixture(scope="function")
def hashing_manager():
    """Create a DeferredHashingManager instance for testing."""
    return DeferredHashingManager()


@pytest.fixture(scope="function")
def preservation_guard():
    """Create an AnomalyPreservationGuard instance for testing."""
    return AnomalyPreservationGuard()


@pytest.fixture(scope="function")
def test_config():
    """Test configuration for processing."""
    return {
        "preserve_anomalies": True,
        "privacy_level": "high",
        "preservation_mode": "balanced",
        "enable_schema_evolution": True,
        "deferred_hashing": True,
        "audit_trail": True
    }


@pytest.fixture(scope="function")
def mock_logging():
    """Mock logging for testing."""
    with patch('logging.getLogger') as mock_logger:
        mock_logger.return_value = Mock()
        yield mock_logger


@pytest.fixture(scope="function")
def mock_time():
    """Mock time functions for consistent testing."""
    with patch('time.time') as mock_time_func:
        mock_time_func.return_value = 1704067200.0  # 2024-01-01 00:00:00
        yield mock_time_func


@pytest.fixture(scope="function")
def mock_uuid():
    """Mock UUID generation for consistent testing."""
    with patch('uuid.uuid4') as mock_uuid_func:
        mock_uuid_func.return_value = "test-uuid-1234-5678-90ab-cdef12345678"
        yield mock_uuid_func


@pytest.fixture(scope="function")
def mock_datetime():
    """Mock datetime for consistent testing."""
    with patch('datetime.datetime') as mock_dt:
        mock_dt.now.return_value = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        mock_dt.utcnow.return_value = datetime(2024, 1, 1, 0, 0, 0)
        yield mock_dt


# Performance testing fixtures
@pytest.fixture(scope="session")
def large_dataset():
    """Generate large dataset for performance testing."""
    return [
        {
            "id": f"record_{i}",
            "timestamp": "2024-01-01T00:00:00Z",
            "data": {"value": i, "anomaly_score": i % 100 / 100.0},
            "metadata": {"batch": i // 1000}
        }
        for i in range(10000)
    ]


@pytest.fixture(scope="session")
def stress_test_data():
    """Generate stress test data."""
    return [
        {
            "id": f"stress_{i}",
            "timestamp": "2024-01-01T00:00:00Z",
            "data": {"load": i % 1000, "stress_factor": i / 10000.0},
            "metadata": {"test_type": "stress", "iteration": i}
        }
        for i in range(50000)
    ]


# Error testing fixtures
@pytest.fixture(scope="session")
def malformed_data_samples():
    """Various malformed data samples for error testing."""
    return [
        None,  # None data
        "",    # Empty string
        [],    # Empty list
        {},    # Empty dict
        {"invalid": "data", "missing_required": True},  # Missing required fields
        {"id": 123, "timestamp": "invalid_timestamp"},  # Invalid types
        {"id": "test", "data": {"nested": {"too": {"deep": True}}}},  # Too deep nesting
        {"id": "test" * 1000, "data": "x" * 1000000},  # Extremely large data
    ]


@pytest.fixture(scope="session")
def edge_case_data():
    """Edge case data for boundary testing."""
    return [
        {"id": "", "timestamp": "2024-01-01T00:00:00Z"},  # Empty ID
        {"id": "test", "timestamp": "1970-01-01T00:00:00Z"},  # Unix epoch
        {"id": "test", "timestamp": "9999-12-31T23:59:59Z"},  # Far future
        {"id": "test", "data": {"unicode": "🚀🌟💻"},  # Unicode characters
        {"id": "test", "data": {"special_chars": "!@#$%^&*()"},  # Special characters
        {"id": "test", "data": {"null_value": None}, "metadata": {}},  # Null values
    ]


# Cleanup fixture
@pytest.fixture(scope="session", autouse=True)
def cleanup_test_environment():
    """Clean up test environment after all tests."""
    yield
    # Any cleanup code can go here
