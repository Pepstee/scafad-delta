"""
End-to-end integration tests for Scafad Layer1 system.

Tests complete data processing workflows from input to output,
including all processing stages and subsystem interactions.
"""

import unittest
import sys
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock

# Add the project root to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.layer1_core import Layer1Processor
from core.layer1_validation import DataValidator
from core.layer1_schema import SchemaManager
from core.layer1_sanitization import DataSanitizer
from core.layer1_privacy import PrivacyFilter
from core.layer1_hashing import HashManager
from core.layer1_preservation import AnomalyPreserver


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete end-to-end data processing workflows."""
    
    def setUp(self):
        """Set up test fixtures for end-to-end testing."""
        self.processor = Layer1Processor()
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample test data
        self.sample_data = {
            "id": "test_123",
            "content": "This is sample content with sensitive information like email@example.com",
            "metadata": {
                "source": "test_system",
                "timestamp": "2024-01-01T00:00:00Z",
                "user_id": "user_456",
                "session_id": "session_789"
            },
            "anomalies": [
                {"type": "outlier", "field": "value", "score": 0.95}
            ]
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_data_processing_pipeline(self):
        """Test complete data processing pipeline with all stages."""
        # Configure processor with all components
        config = {
            "preserve_anomalies": True,
            "privacy_level": "high",
            "hash_sensitive_fields": True,
            "sanitize_content": True
        }
        
        # Process data through complete pipeline
        result = self.processor.process(self.sample_data, config=config)
        
        # Verify processing completed
        self.assertIsNotNone(result)
        self.assertIn('processed_at', result)
        self.assertIn('processing_metadata', result)
        
        # Verify data integrity
        self.assertEqual(result['id'], self.sample_data['id'])
        self.assertIn('anomalies', result)
    
    def test_data_validation_integration(self):
        """Test data validation integration within the pipeline."""
        # Create invalid data
        invalid_data = {
            "content": "Missing required ID field",
            "metadata": {"source": "test"}
        }
        
        # Process should fail validation
        with self.assertRaises(Exception):
            self.processor.process(invalid_data)
    
    def test_schema_evolution_integration(self):
        """Test schema evolution integration during processing."""
        # Data with new fields not in current schema
        evolving_data = {
            "id": "test_123",
            "content": "Sample content",
            "metadata": {"source": "test"},
            "new_field": "new_value",  # New field
            "nested_new": {"deep_field": "deep_value"}  # Nested new field
        }
        
        # Process should handle schema evolution
        result = self.processor.process(evolving_data)
        self.assertIsNotNone(result)
        self.assertIn('new_field', result)
        self.assertIn('nested_new', result)
    
    def test_privacy_filtering_integration(self):
        """Test privacy filtering integration."""
        sensitive_data = {
            "id": "test_123",
            "content": "Contains email@example.com and phone 555-1234",
            "metadata": {
                "source": "test",
                "user_email": "user@example.com",
                "phone_number": "+1-555-123-4567"
            }
        }
        
        config = {"privacy_level": "strict", "redact_pii": True}
        result = self.processor.process(sensitive_data, config=config)
        
        # Verify PII was redacted
        self.assertNotIn("email@example.com", str(result))
        self.assertNotIn("555-1234", str(result))
        self.assertNotIn("user@example.com", str(result))
        self.assertNotIn("555-123-4567", str(result))
    
    def test_anomaly_preservation_integration(self):
        """Test anomaly preservation integration."""
        data_with_anomalies = {
            "id": "test_123",
            "content": "Normal content",
            "metadata": {"source": "test"},
            "anomalies": [
                {"type": "outlier", "field": "value", "score": 0.98},
                {"type": "pattern_break", "field": "sequence", "score": 0.87}
            ]
        }
        
        config = {"preserve_anomalies": True, "anomaly_threshold": 0.8}
        result = self.processor.process(data_with_anomalies, config=config)
        
        # Verify anomalies were preserved
        self.assertIn('anomalies', result)
        self.assertEqual(len(result['anomalies']), 2)
        self.assertEqual(result['anomalies'][0]['score'], 0.98)
    
    def test_hashing_integration(self):
        """Test deferred hashing integration."""
        data_with_sensitive_fields = {
            "id": "test_123",
            "content": "Sample content",
            "metadata": {
                "source": "test",
                "user_id": "user_456",
                "session_token": "secret_token_12345"
            }
        }
        
        config = {"hash_sensitive_fields": True, "hash_algorithm": "sha256"}
        result = self.processor.process(data_with_sensitive_fields, config=config)
        
        # Verify sensitive fields were hashed
        self.assertIn('hashed_fields', result)
        self.assertIn('user_id', result['hashed_fields'])
        self.assertIn('session_token', result['hashed_fields'])
        
        # Verify original values are not present
        self.assertNotIn('user_456', str(result))
        self.assertNotIn('secret_token_12345', str(result))
    
    def test_sanitization_integration(self):
        """Test data sanitization integration."""
        dirty_data = {
            "id": "test_123",
            "content": "Content with <script>alert('xss')</script> and SQL injection '; DROP TABLE users; --",
            "metadata": {"source": "test"}
        }
        
        config = {"sanitize_content": True, "remove_html": True, "escape_sql": True}
        result = self.processor.process(dirty_data, config=config)
        
        # Verify malicious content was sanitized
        self.assertNotIn("<script>", result['content'])
        self.assertNotIn("DROP TABLE", result['content'])
    
    def test_error_handling_integration(self):
        """Test error handling integration across the pipeline."""
        # Data that will cause errors in different stages
        problematic_data = {
            "id": "test_123",
            "content": None,  # Will cause validation error
            "metadata": {"source": "test"}
        }
        
        # Should handle errors gracefully
        with self.assertRaises(Exception):
            self.processor.process(problematic_data)
    
    def test_performance_integration(self):
        """Test performance characteristics of the integrated system."""
        import time
        
        # Large dataset for performance testing
        large_data = {
            "id": "perf_test",
            "content": "x" * 10000,  # 10KB content
            "metadata": {"source": "performance_test"},
            "large_array": list(range(1000))  # 1000 elements
        }
        
        start_time = time.time()
        result = self.processor.process(large_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Verify processing completed successfully
        self.assertIsNotNone(result)
        self.assertLess(processing_time, 5.0)  # Should complete within 5 seconds
        
        # Store performance metrics
        result['performance_metrics'] = {
            "processing_time": processing_time,
            "input_size": len(str(large_data)),
            "output_size": len(str(result))
        }


class TestSubsystemIntegration(unittest.TestCase):
    """Test integration between different subsystems."""
    
    def setUp(self):
        """Set up test fixtures for subsystem testing."""
        self.processor = Layer1Processor()
    
    def test_schema_registry_integration(self):
        """Test schema registry integration."""
        # Test schema versioning and evolution
        data_v1 = {"id": "test", "content": "old format"}
        data_v2 = {"id": "test", "content": "new format", "new_field": "value"}
        
        # Process both versions
        result_v1 = self.processor.process(data_v1)
        result_v2 = self.processor.process(data_v2)
        
        # Both should process successfully
        self.assertIsNotNone(result_v1)
        self.assertIsNotNone(result_v2)
    
    def test_privacy_policy_engine_integration(self):
        """Test privacy policy engine integration."""
        # Test dynamic policy application
        data = {
            "id": "test",
            "content": "Contains PII",
            "metadata": {"user_email": "test@example.com"}
        }
        
        # Different privacy levels
        config_low = {"privacy_level": "low"}
        config_high = {"privacy_level": "high"}
        
        result_low = self.processor.process(data, config=config_low)
        result_high = self.processor.process(data, config=config_high)
        
        # High privacy should be more restrictive
        self.assertIsNotNone(result_low)
        self.assertIsNotNone(result_high)
    
    def test_quality_monitor_integration(self):
        """Test quality monitor integration."""
        # Test data quality metrics
        data = {
            "id": "test",
            "content": "Quality test content",
            "metadata": {"source": "test"}
        }
        
        result = self.processor.process(data)
        
        # Should include quality metrics
        self.assertIn('quality_metrics', result)
        self.assertIsInstance(result['quality_metrics'], dict)


if __name__ == '__main__':
    unittest.main()
