"""
Anomaly preservation tests for Scafad Layer1 system.

Tests anomaly detection, preservation mechanisms,
and data integrity during processing.
"""

import unittest
import sys
import os
import json
import math
from unittest.mock import Mock, patch, MagicMock

# Add the project root to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.layer1_preservation import AnomalyPreserver
from core.layer1_core import Layer1Processor


class TestAnomalyPreservation(unittest.TestCase):
    """Test anomaly preservation functionality."""
    
    def setUp(self):
        """Set up test fixtures for anomaly preservation testing."""
        self.preserver = AnomalyPreserver()
        self.processor = Layer1Processor()
        
        # Sample data with various anomaly types
        self.normal_data = {
            "id": "normal_001",
            "value": 100,
            "timestamp": "2024-01-01T00:00:00Z",
            "category": "standard"
        }
        
        self.anomalous_data = {
            "id": "anomaly_001",
            "value": 9999,  # Unusually high value
            "timestamp": "2024-01-01T00:00:00Z",
            "category": "outlier",
            "anomalies": [
                {
                    "type": "statistical_outlier",
                    "field": "value",
                    "score": 0.95,
                    "description": "Value significantly above normal range",
                    "detected_at": "2024-01-01T00:00:00Z"
                }
            ]
        }
        
        self.complex_anomalies = {
            "id": "complex_001",
            "transaction_amount": 50000,
            "user_age": 25,
            "location": "unusual_location",
            "anomalies": [
                {
                    "type": "amount_outlier",
                    "field": "transaction_amount",
                    "score": 0.98,
                    "description": "Unusually high transaction amount"
                },
                {
                    "type": "location_anomaly",
                    "field": "location",
                    "score": 0.87,
                    "description": "Unusual location for user"
                },
                {
                    "type": "behavioral_pattern",
                    "field": "user_age",
                    "score": 0.76,
                    "description": "Age-related behavioral anomaly"
                }
            ]
        }
    
    def test_anomaly_detection(self):
        """Test anomaly detection capabilities."""
        print("\n=== Anomaly Detection Test ===")
        
        # Test statistical outlier detection
        outlier_result = self.preserver.detect_statistical_anomalies(
            data=[100, 105, 98, 102, 9999, 103, 101],
            threshold=2.0  # 2 standard deviations
        )
        
        self.assertIsInstance(outlier_result, dict)
        self.assertIn("anomalies", outlier_result)
        self.assertIn("outlier_indices", outlier_result)
        
        # Should detect the outlier value 9999
        self.assertIn(4, outlier_result["outlier_indices"])
        
        # Test pattern anomaly detection
        pattern_data = [
            {"sequence": [1, 2, 3, 4, 5], "expected": "increasing"},
            {"sequence": [5, 4, 3, 2, 1], "expected": "decreasing"},
            {"sequence": [1, 2, 3, 7, 4], "expected": "increasing"}  # Anomaly
        ]
        
        pattern_result = self.preserver.detect_pattern_anomalies(pattern_data)
        self.assertIsInstance(pattern_result, dict)
        self.assertIn("anomalies", pattern_result)
        
        # Test behavioral anomaly detection
        behavioral_data = {
            "user_id": "user_123",
            "normal_behavior": {
                "login_time": "09:00",
                "location": "office",
                "activity_level": "medium"
            },
            "current_behavior": {
                "login_time": "03:00",
                "location": "unusual_location",
                "activity_level": "high"
            }
        }
        
        behavioral_result = self.preserver.detect_behavioral_anomalies(behavioral_data)
        self.assertIsInstance(behavioral_result, dict)
        self.assertIn("anomaly_score", behavioral_result)
        self.assertGreater(behavioral_result["anomaly_score"], 0.5)
    
    def test_anomaly_preservation(self):
        """Test anomaly preservation during data processing."""
        print("\n=== Anomaly Preservation Test ===")
        
        # Configure processor to preserve anomalies
        config = {
            "preserve_anomalies": True,
            "anomaly_threshold": 0.7,
            "preservation_mode": "full"
        }
        
        # Process data with anomalies
        result = self.processor.process(self.anomalous_data, config=config)
        
        # Verify anomalies were preserved
        self.assertIn("anomalies", result)
        self.assertEqual(len(result["anomalies"]), 1)
        
        anomaly = result["anomalies"][0]
        self.assertEqual(anomaly["type"], "statistical_outlier")
        self.assertEqual(anomaly["field"], "value")
        self.assertEqual(anomaly["score"], 0.95)
        
        # Verify original data integrity
        self.assertEqual(result["id"], self.anomalous_data["id"])
        self.assertEqual(result["value"], self.anomalous_data["value"])
        
        # Test anomaly filtering by threshold
        config_filtered = {
            "preserve_anomalies": True,
            "anomaly_threshold": 0.9,  # Higher threshold
            "preservation_mode": "filtered"
        }
        
        result_filtered = self.processor.process(self.complex_anomalies, config_filtered)
        
        # Should only preserve high-scoring anomalies
        preserved_anomalies = result_filtered["anomalies"]
        self.assertLess(len(preserved_anomalies), len(self.complex_anomalies["anomalies"]))
        
        for anomaly in preserved_anomalies:
            self.assertGreaterEqual(anomaly["score"], 0.9)
    
    def test_anomaly_classification(self):
        """Test anomaly classification and categorization."""
        print("\n=== Anomaly Classification Test ===")
        
        # Test anomaly type classification
        classification_result = self.preserver.classify_anomalies(
            self.complex_anomalies["anomalies"]
        )
        
        self.assertIsInstance(classification_result, dict)
        self.assertIn("by_type", classification_result)
        self.assertIn("by_severity", classification_result)
        self.assertIn("by_field", classification_result)
        
        # Verify type classification
        type_classification = classification_result["by_type"]
        self.assertIn("amount_outlier", type_classification)
        self.assertIn("location_anomaly", type_classification)
        self.assertIn("behavioral_pattern", type_classification)
        
        # Verify severity classification
        severity_classification = classification_result["by_severity"]
        self.assertIn("high", severity_classification)  # score >= 0.9
        self.assertIn("medium", severity_classification)  # 0.7 <= score < 0.9
        self.assertIn("low", severity_classification)  # score < 0.7
        
        # Test anomaly scoring
        scoring_result = self.preserver.score_anomaly_severity(
            self.complex_anomalies["anomalies"]
        )
        
        self.assertIsInstance(scoring_result, dict)
        self.assertIn("overall_score", scoring_result)
        self.assertIn("field_scores", scoring_result)
        
        # Overall score should be weighted average of individual scores
        expected_overall = (0.98 + 0.87 + 0.76) / 3
        self.assertAlmostEqual(scoring_result["overall_score"], expected_overall, places=2)
    
    def test_anomaly_context_preservation(self):
        """Test preservation of anomaly context and metadata."""
        print("\n=== Anomaly Context Preservation Test ===")
        
        # Test context extraction
        context_result = self.preserver.extract_anomaly_context(
            self.anomalous_data,
            anomaly_index=0
        )
        
        self.assertIsInstance(context_result, dict)
        self.assertIn("preceding_data", context_result)
        self.assertIn("succeeding_data", context_result)
        self.assertIn("temporal_context", context_result)
        self.assertIn("spatial_context", context_result)
        
        # Test metadata preservation
        metadata_result = self.preserver.preserve_anomaly_metadata(
            self.anomalous_data["anomalies"][0],
            additional_context={
                "processing_stage": "validation",
                "data_source": "user_input",
                "confidence": 0.95
            }
        )
        
        self.assertIn("original_metadata", metadata_result)
        self.assertIn("processing_metadata", metadata_result)
        self.assertIn("preservation_timestamp", metadata_result)
        
        # Test context reconstruction
        reconstructed_context = self.preserver.reconstruct_anomaly_context(
            anomaly_data=self.anomalous_data["anomalies"][0],
            context_data=context_result
        )
        
        self.assertIsInstance(reconstructed_context, dict)
        self.assertIn("full_context", reconstructed_context)
        self.assertIn("anomaly_indicators", reconstructed_context)
    
    def test_anomaly_aggregation(self):
        """Test anomaly aggregation and summarization."""
        print("\n=== Anomaly Aggregation Test ===")
        
        # Test anomaly aggregation
        aggregation_result = self.preserver.aggregate_anomalies([
            self.anomalous_data["anomalies"][0],
            self.complex_anomalies["anomalies"][0],
            self.complex_anomalies["anomalies"][1]
        ])
        
        self.assertIsInstance(aggregation_result, dict)
        self.assertIn("total_count", aggregation_result)
        self.assertIn("by_type", aggregation_result)
        self.assertIn("by_severity", aggregation_result)
        self.assertIn("trends", aggregation_result)
        
        # Verify aggregation counts
        self.assertEqual(aggregation_result["total_count"], 3)
        self.assertEqual(len(aggregation_result["by_type"]), 3)
        
        # Test anomaly summarization
        summary_result = self.preserver.summarize_anomalies(
            anomalies=[
                self.anomalous_data["anomalies"][0],
                self.complex_anomalies["anomalies"][0],
                self.complex_anomalies["anomalies"][1],
                self.complex_anomalies["anomalies"][2]
            ]
        )
        
        self.assertIsInstance(summary_result, dict)
        self.assertIn("summary", summary_result)
        self.assertIn("key_insights", summary_result)
        self.assertIn("recommendations", summary_result)
        
        # Test trend analysis
        trend_result = self.preserver.analyze_anomaly_trends([
            {"timestamp": "2024-01-01T00:00:00Z", "anomaly_count": 5},
            {"timestamp": "2024-01-02T00:00:00Z", "anomaly_count": 8},
            {"timestamp": "2024-01-03T00:00:00Z", "anomaly_count": 12}
        ])
        
        self.assertIn("trend_direction", trend_result)
        self.assertIn("trend_strength", trend_result)
        self.assertIn("forecast", trend_result)
    
    def test_anomaly_validation(self):
        """Test anomaly validation and verification."""
        print("\n=== Anomaly Validation Test ===")
        
        # Test anomaly validation
        validation_result = self.preserver.validate_anomaly(
            self.anomalous_data["anomalies"][0],
            original_data=self.anomalous_data
        )
        
        self.assertIsInstance(validation_result, dict)
        self.assertIn("is_valid", validation_result)
        self.assertIn("confidence_score", validation_result)
        self.assertIn("validation_errors", validation_result)
        
        # Should be valid
        self.assertTrue(validation_result["is_valid"])
        self.assertGreater(validation_result["confidence_score"], 0.8)
        
        # Test false positive detection
        false_positive_data = {
            "id": "false_positive_001",
            "value": 150,  # Within normal range
            "anomalies": [
                {
                    "type": "statistical_outlier",
                    "field": "value",
                    "score": 0.95,
                    "description": "False positive detection"
                }
            ]
        }
        
        fp_validation = self.preserver.validate_anomaly(
            false_positive_data["anomalies"][0],
            original_data=false_positive_data
        )
        
        # Should detect false positive
        self.assertFalse(fp_validation["is_valid"])
        self.assertLess(fp_validation["confidence_score"], 0.5)
    
    def test_anomaly_recovery(self):
        """Test anomaly recovery and data restoration."""
        print("\n=== Anomaly Recovery Test ===")
        
        # Test anomaly recovery
        recovery_result = self.preserver.recover_from_anomaly(
            self.anomalous_data,
            recovery_strategy="statistical_correction"
        )
        
        self.assertIsInstance(recovery_result, dict)
        self.assertIn("recovered_data", recovery_result)
        self.assertIn("recovery_method", recovery_result)
        self.assertIn("confidence", recovery_result)
        
        # Verify recovery
        recovered_data = recovery_result["recovered_data"]
        self.assertIn("value", recovered_data)
        self.assertNotEqual(recovered_data["value"], 9999)  # Should be corrected
        
        # Test multiple recovery strategies
        strategies = ["statistical_correction", "interpolation", "default_value"]
        recovery_results = {}
        
        for strategy in strategies:
            result = self.preserver.recover_from_anomaly(
                self.anomalous_data,
                recovery_strategy=strategy
            )
            recovery_results[strategy] = result
        
        # All strategies should produce different results
        values = [r["recovered_data"]["value"] for r in recovery_results.values()]
        self.assertEqual(len(set(values)), len(strategies))
    
    def test_anomaly_reporting(self):
        """Test anomaly reporting and alerting."""
        print("\n=== Anomaly Reporting Test ===")
        
        # Test anomaly report generation
        report_result = self.preserver.generate_anomaly_report([
            self.anomalous_data["anomalies"][0],
            self.complex_anomalies["anomalies"][0],
            self.complex_anomalies["anomalies"][1]
        ])
        
        self.assertIsInstance(report_result, dict)
        self.assertIn("report_id", report_result)
        self.assertIn("summary", report_result)
        self.assertIn("detailed_analysis", report_result)
        self.assertIn("recommendations", report_result)
        
        # Test alert generation
        alert_result = self.preserver.generate_anomaly_alert(
            anomaly=self.anomalous_data["anomalies"][0],
            severity="high",
            recipients=["admin@example.com", "security@example.com"]
        )
        
        self.assertIsInstance(alert_result, dict)
        self.assertIn("alert_id", alert_result)
        self.assertIn("severity", alert_result)
        self.assertIn("recipients", alert_result)
        self.assertIn("message", alert_result)
        
        # Test report export
        export_result = self.preserver.export_anomaly_report(
            report_id=report_result["report_id"],
            format="json"
        )
        
        self.assertIsInstance(export_result, dict)
        self.assertIn("export_data", export_result)
        self.assertIn("format", export_result)
        self.assertIn("timestamp", export_result)
    
    def test_anomaly_learning(self):
        """Test anomaly learning and adaptation."""
        print("\n=== Anomaly Learning Test ===")
        
        # Test anomaly pattern learning
        learning_result = self.preserver.learn_anomaly_patterns([
            self.anomalous_data["anomalies"][0],
            self.complex_anomalies["anomalies"][0],
            self.complex_anomalies["anomalies"][1]
        ])
        
        self.assertIsInstance(learning_result, dict)
        self.assertIn("patterns_learned", learning_result)
        self.assertIn("model_updated", learning_result)
        self.assertIn("confidence_improvement", learning_result)
        
        # Test adaptive threshold adjustment
        threshold_result = self.preserver.adjust_detection_thresholds(
            historical_anomalies=[
                {"score": 0.8, "was_anomaly": True},
                {"score": 0.6, "was_anomaly": False},
                {"score": 0.9, "was_anomaly": True}
            ],
            target_precision=0.9
        )
        
        self.assertIsInstance(threshold_result, dict)
        self.assertIn("new_threshold", threshold_result)
        self.assertIn("expected_precision", threshold_result)
        self.assertIn("expected_recall", threshold_result)
        
        # Test model retraining
        retraining_result = self.preserver.retrain_anomaly_model(
            training_data=[
                {"features": [1, 2, 3], "is_anomaly": False},
                {"features": [10, 20, 30], "is_anomaly": True},
                {"features": [2, 4, 6], "is_anomaly": False}
            ]
        )
        
        self.assertIsInstance(retraining_result, dict)
        self.assertIn("model_performance", retraining_result)
        self.assertIn("accuracy", retraining_result)
        self.assertIn("training_time", retraining_result)


class TestAnomalyPreservationIntegration(unittest.TestCase):
    """Test integration of anomaly preservation with other components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = Layer1Processor()
        self.preserver = AnomalyPreserver()
    
    def test_end_to_end_anomaly_preservation(self):
        """Test complete anomaly preservation workflow."""
        print("\n=== End-to-End Anomaly Preservation Test ===")
        
        # Data with multiple anomaly types
        complex_data = {
            "id": "integration_test_001",
            "user_behavior": {
                "login_time": "03:00",
                "location": "unusual_location",
                "activity_level": "extreme"
            },
            "financial_data": {
                "transaction_amount": 99999,
                "currency": "USD",
                "merchant_category": "unusual"
            },
            "anomalies": [
                {
                    "type": "behavioral_anomaly",
                    "field": "user_behavior",
                    "score": 0.92,
                    "description": "Unusual login time and location"
                },
                {
                    "type": "financial_anomaly",
                    "field": "financial_data",
                    "score": 0.98,
                    "description": "Extremely high transaction amount"
                }
            ]
        }
        
        # Process with anomaly preservation
        config = {
            "preserve_anomalies": True,
            "anomaly_threshold": 0.8,
            "preservation_mode": "full",
            "anomaly_analysis": True,
            "context_preservation": True
        }
        
        result = self.processor.process(complex_data, config=config)
        
        # Verify complete preservation
        self.assertIn("anomalies", result)
        self.assertEqual(len(result["anomalies"]), 2)
        
        # Verify anomaly metadata
        for anomaly in result["anomalies"]:
            self.assertIn("preservation_metadata", anomaly)
            self.assertIn("context_data", anomaly)
            self.assertIn("processing_timestamp", anomaly)
        
        # Verify data integrity
        self.assertEqual(result["id"], complex_data["id"])
        self.assertEqual(result["user_behavior"], complex_data["user_behavior"])
        self.assertEqual(result["financial_data"], complex_data["financial_data"])
    
    def test_anomaly_preservation_with_privacy(self):
        """Test anomaly preservation with privacy controls."""
        print("\n=== Anomaly Preservation with Privacy Test ===")
        
        # Sensitive data with anomalies
        sensitive_anomalous_data = {
            "id": "privacy_test_001",
            "user_email": "user@example.com",
            "credit_card": "4111-1111-1111-1111",
            "unusual_activity": {
                "login_time": "02:00",
                "location": "foreign_country",
                "device": "unknown_device"
            },
            "anomalies": [
                {
                    "type": "security_anomaly",
                    "field": "unusual_activity",
                    "score": 0.95,
                    "description": "Suspicious login pattern"
                }
            ]
        }
        
        # Process with both anomaly preservation and privacy
        config = {
            "preserve_anomalies": True,
            "privacy_level": "strict",
            "anomaly_threshold": 0.8,
            "redact_pii": True,
            "preserve_anomaly_context": True
        }
        
        result = self.processor.process(sensitive_anomalous_data, config=config)
        
        # Verify privacy protection
        self.assertNotIn("user@example.com", str(result))
        self.assertNotIn("4111-1111-1111-1111", str(result))
        
        # Verify anomaly preservation
        self.assertIn("anomalies", result)
        self.assertEqual(len(result["anomalies"]), 1)
        
        # Verify context preservation without PII
        anomaly = result["anomalies"][0]
        self.assertIn("context_data", anomaly)
        self.assertNotIn("user@example.com", str(anomaly["context_data"]))
    
    def test_anomaly_preservation_with_validation(self):
        """Test anomaly preservation with data validation."""
        print("\n=== Anomaly Preservation with Validation Test ===")
        
        # Data with validation issues and anomalies
        problematic_data = {
            "id": "validation_test_001",
            "required_field": None,  # Validation issue
            "numeric_field": "not_a_number",  # Type issue
            "unusual_value": 999999,  # Anomaly
            "anomalies": [
                {
                    "type": "value_anomaly",
                    "field": "unusual_value",
                    "score": 0.97,
                    "description": "Extremely high value"
                }
            ]
        }
        
        # Process with validation and anomaly preservation
        config = {
            "preserve_anomalies": True,
            "strict_validation": False,  # Allow processing despite validation issues
            "anomaly_threshold": 0.8,
            "preservation_mode": "full"
        }
        
        result = self.processor.process(problematic_data, config=config)
        
        # Verify anomaly preservation despite validation issues
        self.assertIn("anomalies", result)
        self.assertEqual(len(result["anomalies"]), 1)
        
        # Verify anomaly context includes validation issues
        anomaly = result["anomalies"][0]
        self.assertIn("context_data", anomaly)
        
        # Should include validation context
        context = anomaly["context_data"]
        self.assertIn("validation_issues", context)
        self.assertIn("data_quality_metrics", context)


if __name__ == '__main__':
    # Run anomaly preservation tests
    unittest.main(verbosity=2)
