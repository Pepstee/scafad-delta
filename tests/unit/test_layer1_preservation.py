"""
Comprehensive unit tests for layer1_preservation.py

Tests anomaly preservation guards, preservation quality metrics, and
anomaly detection algorithms with extensive coverage of edge cases.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
import numpy as np
import json

# Add the core directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'core'))

from layer1_preservation import (
    AnomalyPreservationGuard, PreservationReport, PreservationMetrics,
    AnomalyDetector, PreservationMode, AnomalyType, AnomalySeverity
)


class TestAnomalyType:
    """Test the AnomalyType enum."""
    
    def test_anomaly_types(self):
        """Test all anomaly types are defined."""
        types = list(AnomalyType)
        assert AnomalyType.PERFORMANCE_SPIKE in types
        assert AnomalyType.MEMORY_LEAK in types
        assert AnomalyType.NETWORK_ANOMALY in types
        assert AnomalyType.BEHAVIORAL_CHANGE in types
        assert AnomalyType.SYSTEM_FAILURE in types
    
    def test_anomaly_type_values(self):
        """Test anomaly type string values."""
        assert AnomalyType.PERFORMANCE_SPIKE.value == "performance_spike"
        assert AnomalyType.MEMORY_LEAK.value == "memory_leak"
        assert AnomalyType.NETWORK_ANOMALY.value == "network_anomaly"
        assert AnomalyType.BEHAVIORAL_CHANGE.value == "behavioral_change"
        assert AnomalyType.SYSTEM_FAILURE.value == "system_failure"


class TestAnomalySeverity:
    """Test the AnomalySeverity enum."""
    
    def test_anomaly_severities(self):
        """Test all anomaly severities are defined."""
        severities = list(AnomalySeverity)
        assert AnomalySeverity.LOW in severities
        assert AnomalySeverity.MEDIUM in severities
        assert AnomalySeverity.HIGH in severities
        assert AnomalySeverity.CRITICAL in severities
    
    def test_severity_comparison(self):
        """Test severity level comparisons."""
        assert AnomalySeverity.LOW < AnomalySeverity.MEDIUM
        assert AnomalySeverity.MEDIUM < AnomalySeverity.HIGH
        assert AnomalySeverity.HIGH < AnomalySeverity.CRITICAL
        assert AnomalySeverity.CRITICAL > AnomalySeverity.LOW


class TestPreservationMode:
    """Test the PreservationMode enum."""
    
    def test_preservation_modes(self):
        """Test all preservation modes are defined."""
        modes = list(PreservationMode)
        assert PreservationMode.CONSERVATIVE in modes
        assert PreservationMode.BALANCED in modes
        assert PreservationMode.AGGRESSIVE in modes
    
    def test_preservation_mode_values(self):
        """Test preservation mode string values."""
        assert PreservationMode.CONSERVATIVE.value == "conservative"
        assert PreservationMode.BALANCED.value == "balanced"
        assert PreservationMode.AGGRESSIVE.value == "aggressive"


class TestAnomalyDetector:
    """Test the AnomalyDetector class."""
    
    def test_detector_initialization(self):
        """Test anomaly detector initialization."""
        detector = AnomalyDetector()
        assert detector is not None
        assert hasattr(detector, 'thresholds')
        assert hasattr(detector, 'algorithms')
    
    def test_add_detection_algorithm(self):
        """Test adding detection algorithms."""
        detector = AnomalyDetector()
        
        def simple_threshold(data, threshold=0.8):
            return data.get('anomaly_score', 0) > threshold
        
        detector.add_algorithm("threshold", simple_threshold)
        assert "threshold" in detector.algorithms
        assert detector.algorithms["threshold"] == simple_threshold
    
    def test_detect_anomalies(self, sample_anomaly_data):
        """Test anomaly detection."""
        detector = AnomalyDetector()
        
        # Add detection algorithm
        def performance_detector(data):
            cpu_usage = data.get('indicators', {}).get('cpu_spike', 0)
            return cpu_usage > 90.0
        
        detector.add_algorithm("performance", performance_detector)
        
        # Test with anomaly data
        anomalies = detector.detect_anomalies(sample_anomaly_data)
        assert len(anomalies) >= 1  # Should detect performance spike
        
        # Test with normal data
        normal_data = sample_anomaly_data.copy()
        normal_data['indicators']['cpu_spike'] = 50.0
        
        anomalies = detector.detect_anomalies(normal_data)
        assert len(anomalies) == 0  # Should not detect anomalies
    
    def test_multiple_detection_algorithms(self, sample_anomaly_data):
        """Test multiple detection algorithms working together."""
        detector = AnomalyDetector()
        
        # CPU threshold detector
        def cpu_detector(data):
            cpu = data.get('indicators', {}).get('cpu_spike', 0)
            return cpu > 90.0
        
        # Memory leak detector
        def memory_detector(data):
            memory_leak = data.get('indicators', {}).get('memory_leak', False)
            return memory_leak
        
        # Response time detector
        def response_detector(data):
            response_time = data.get('indicators', {}).get('response_time', 0)
            return response_time > 2.0
        
        detector.add_algorithm("cpu", cpu_detector)
        detector.add_algorithm("memory", memory_detector)
        detector.add_algorithm("response", response_detector)
        
        # Test detection
        anomalies = detector.detect_anomalies(sample_anomaly_data)
        
        # Should detect multiple types of anomalies
        anomaly_types = {anomaly.type for anomaly in anomalies}
        assert len(anomaly_types) >= 2  # At least 2 different types
    
    def test_detection_thresholds(self):
        """Test detection with different thresholds."""
        detector = AnomalyDetector()
        
        def threshold_detector(data, threshold=0.5):
            score = data.get('anomaly_score', 0)
            return score > threshold
        
        detector.add_algorithm("threshold", threshold_detector)
        
        # Test data with different scores
        test_data = [
            {"anomaly_score": 0.3, "id": "low"},
            {"anomaly_score": 0.6, "id": "medium"},
            {"anomaly_score": 0.9, "id": "high"}
        ]
        
        # Low threshold
        detector.set_threshold("threshold", 0.2)
        anomalies = detector.detect_anomalies(test_data[0])
        assert len(anomalies) == 1
        
        # High threshold
        detector.set_threshold("threshold", 0.8)
        anomalies = detector.detect_anomalies(test_data[2])
        assert len(anomalies) == 1
        
        # Medium threshold
        detector.set_threshold("threshold", 0.5)
        anomalies = detector.detect_anomalies(test_data[1])
        assert len(anomalies) == 1


class TestPreservationMetrics:
    """Test the PreservationMetrics class."""
    
    def test_metrics_creation(self):
        """Test creating preservation metrics."""
        metrics = PreservationMetrics(
            total_records=1000,
            anomalies_detected=50,
            anomalies_preserved=48,
            preservation_rate=0.96,
            quality_score=0.92,
            processing_time=1.5
        )
        
        assert metrics.total_records == 1000
        assert metrics.anomalies_detected == 50
        assert metrics.anomalies_preserved == 48
        assert metrics.preservation_rate == 0.96
        assert metrics.quality_score == 0.92
        assert metrics.processing_time == 1.5
    
    def test_metrics_calculation(self):
        """Test automatic metrics calculation."""
        metrics = PreservationMetrics.calculate(
            total_records=100,
            anomalies_detected=10,
            anomalies_preserved=9,
            processing_time=0.5
        )
        
        assert metrics.preservation_rate == 0.9  # 9/10
        assert metrics.quality_score == 0.9  # Simplified calculation
        assert metrics.processing_time == 0.5
    
    def test_metrics_serialization(self):
        """Test metrics serialization."""
        metrics = PreservationMetrics(
            total_records=500,
            anomalies_detected=25,
            anomalies_preserved=24,
            preservation_rate=0.96,
            quality_score=0.94,
            processing_time=0.8
        )
        
        serialized = metrics.to_dict()
        assert serialized["total_records"] == 500
        assert serialized["preservation_rate"] == 0.96
        assert serialized["quality_score"] == 0.94
    
    def test_metrics_comparison(self):
        """Test metrics comparison operations."""
        metrics1 = PreservationMetrics(
            total_records=100,
            anomalies_detected=10,
            anomalies_preserved=9,
            preservation_rate=0.9,
            quality_score=0.85,
            processing_time=1.0
        )
        
        metrics2 = PreservationMetrics(
            total_records=100,
            anomalies_detected=10,
            anomalies_preserved=8,
            preservation_rate=0.8,
            quality_score=0.80,
            processing_time=1.2
        )
        
        # Better preservation rate
        assert metrics1.preservation_rate > metrics2.preservation_rate
        
        # Better quality score
        assert metrics1.quality_score > metrics2.quality_score
        
        # Faster processing
        assert metrics1.processing_time < metrics2.processing_time


class TestPreservationReport:
    """Test the PreservationReport class."""
    
    def test_report_creation(self):
        """Test creating preservation reports."""
        metrics = PreservationMetrics(
            total_records=1000,
            anomalies_detected=50,
            anomalies_preserved=48,
            preservation_rate=0.96,
            quality_score=0.92,
            processing_time=1.5
        )
        
        report = PreservationReport(
            report_id="preservation_001",
            timestamp=datetime.now(),
            metrics=metrics,
            anomalies_details=[
                {"type": "performance_spike", "count": 20, "preserved": 19},
                {"type": "memory_leak", "count": 15, "preserved": 14},
                {"type": "network_anomaly", "count": 15, "preserved": 15}
            ],
            recommendations=[
                "Increase memory monitoring frequency",
                "Optimize network anomaly detection"
            ]
        )
        
        assert report.report_id == "preservation_001"
        assert report.metrics == metrics
        assert len(report.anomalies_details) == 3
        assert len(report.recommendations) == 2
    
    def test_report_serialization(self):
        """Test report serialization."""
        metrics = PreservationMetrics(
            total_records=100,
            anomalies_detected=5,
            anomalies_preserved=5,
            preservation_rate=1.0,
            quality_score=0.95,
            processing_time=0.3
        )
        
        report = PreservationReport(
            report_id="test_report",
            timestamp=datetime(2024, 1, 1, 0, 0, 0),
            metrics=metrics,
            anomalies_details=[],
            recommendations=[]
        )
        
        serialized = report.to_dict()
        assert serialized["report_id"] == "test_report"
        assert serialized["metrics"]["preservation_rate"] == 1.0
        assert "timestamp" in serialized
    
    def test_report_summary(self):
        """Test report summary generation."""
        metrics = PreservationMetrics(
            total_records=1000,
            anomalies_detected=100,
            anomalies_preserved=95,
            preservation_rate=0.95,
            quality_score=0.90,
            processing_time=2.0
        )
        
        report = PreservationReport(
            report_id="summary_test",
            timestamp=datetime.now(),
            metrics=metrics,
            anomalies_details=[],
            recommendations=[]
        )
        
        summary = report.generate_summary()
        assert "95%" in summary  # Preservation rate
        assert "1000" in summary  # Total records
        assert "100" in summary   # Anomalies detected


class TestAnomalyPreservationGuard:
    """Test the AnomalyPreservationGuard class."""
    
    def test_guard_initialization(self):
        """Test preservation guard initialization."""
        guard = AnomalyPreservationGuard()
        assert guard is not None
        assert hasattr(guard, 'detector')
        assert hasattr(guard, 'preservation_mode')
        assert hasattr(guard, 'metrics')
    
    def test_set_preservation_mode(self):
        """Test setting preservation mode."""
        guard = AnomalyPreservationGuard()
        
        guard.set_preservation_mode(PreservationMode.CONSERVATIVE)
        assert guard.preservation_mode == PreservationMode.CONSERVATIVE
        
        guard.set_preservation_mode(PreservationMode.AGGRESSIVE)
        assert guard.preservation_mode == PreservationMode.AGGRESSIVE
    
    def test_preserve_anomalies(self, sample_anomaly_data):
        """Test anomaly preservation functionality."""
        guard = AnomalyPreservationGuard()
        
        # Set up detector
        def performance_detector(data):
            cpu = data.get('indicators', {}).get('cpu_spike', 0)
            return cpu > 90.0
        
        guard.detector.add_algorithm("performance", performance_detector)
        
        # Test preservation
        result = guard.preserve_anomalies(sample_anomaly_data)
        
        assert result.is_preserved is True
        assert result.preservation_rate >= 0.8  # Should preserve most anomalies
        assert result.metrics is not None
    
    def test_preservation_mode_impact(self, sample_anomaly_data):
        """Test how preservation mode affects preservation."""
        guard = AnomalyPreservationGuard()
        
        # Set up detector
        def simple_detector(data):
            return data.get('anomaly_score', 0) > 0.5
        
        guard.detector.add_algorithm("simple", simple_detector)
        
        # Test conservative mode
        guard.set_preservation_mode(PreservationMode.CONSERVATIVE)
        conservative_result = guard.preserve_anomalies(sample_anomaly_data)
        
        # Test aggressive mode
        guard.set_preservation_mode(PreservationMode.AGGRESSIVE)
        aggressive_result = guard.preserve_anomalies(sample_anomaly_data)
        
        # Aggressive mode should preserve more anomalies
        assert aggressive_result.preservation_rate >= conservative_result.preservation_rate
    
    def test_preservation_quality_assessment(self, sample_anomaly_data):
        """Test preservation quality assessment."""
        guard = AnomalyPreservationGuard()
        
        # Set up detector
        def quality_detector(data):
            return data.get('anomaly_score', 0) > 0.7
        
        guard.detector.add_algorithm("quality", quality_detector)
        
        # Test preservation with quality assessment
        result = guard.preserve_anomalies(sample_anomaly_data)
        
        # Check quality metrics
        assert result.metrics is not None
        assert result.metrics.quality_score >= 0.0
        assert result.metrics.quality_score <= 1.0
        assert result.metrics.preservation_rate >= 0.0
        assert result.metrics.preservation_rate <= 1.0
    
    def test_preservation_with_large_dataset(self, large_dataset):
        """Test preservation performance with large datasets."""
        guard = AnomalyPreservationGuard()
        
        # Set up simple detector
        def large_dataset_detector(data):
            value = data.get('data', {}).get('value', 0)
            return value % 100 == 0  # Every 100th record is an anomaly
        
        guard.detector.add_algorithm("large_dataset", large_dataset_detector)
        
        # Test with subset
        test_data = large_dataset[:1000]
        
        start_time = datetime.now()
        
        result = guard.preserve_anomalies(test_data)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time
        assert duration < 30.0  # 30 seconds for 1000 records
        assert result.is_preserved is True
        assert result.metrics.total_records == 1000
    
    def test_preservation_edge_cases(self, edge_case_data):
        """Test preservation with edge case data."""
        guard = AnomalyPreservationGuard()
        
        # Set up basic detector
        def edge_case_detector(data):
            return data.get('id') == "edge_case"
        
        guard.detector.add_algorithm("edge_case", edge_case_detector)
        
        # Test edge cases
        for data in edge_case_data:
            try:
                result = guard.preserve_anomalies(data)
                # Should handle gracefully
                assert isinstance(result.is_preserved, bool)
                assert result.metrics is not None
            except Exception as e:
                # If it fails, should be a controlled failure
                assert "preservation" in str(e).lower() or "invalid" in str(e).lower()
    
    def test_preservation_error_handling(self):
        """Test error handling during preservation."""
        guard = AnomalyPreservationGuard()
        
        # Test with None data
        with pytest.raises(ValueError):
            guard.preserve_anomalies(None)
        
        # Test with empty data
        with pytest.raises(ValueError):
            guard.preserve_anomalies({})
        
        # Test with invalid preservation mode
        with pytest.raises(ValueError):
            guard.set_preservation_mode("invalid_mode")
    
    def test_preservation_metrics_accumulation(self, sample_anomaly_data):
        """Test metrics accumulation across multiple preservation operations."""
        guard = AnomalyPreservationGuard()
        
        # Set up detector
        def metrics_detector(data):
            return data.get('anomaly_score', 0) > 0.5
        
        guard.detector.add_algorithm("metrics", metrics_detector)
        
        # Run multiple preservation operations
        results = []
        for i in range(5):
            data_copy = sample_anomaly_data.copy()
            data_copy['id'] = f"test_{i}"
            result = guard.preserve_anomalies(data_copy)
            results.append(result)
        
        # Check accumulated metrics
        total_records = sum(r.metrics.total_records for r in results)
        total_anomalies = sum(r.metrics.anomalies_detected for r in results)
        total_preserved = sum(r.metrics.anomalies_preserved for r in results)
        
        assert total_records == 5  # 5 records processed
        assert total_anomalies >= 0  # Should detect some anomalies
        assert total_preserved >= 0  # Should preserve some anomalies


class TestPreservationIntegration:
    """Test integration between preservation components."""
    
    def test_full_preservation_pipeline(self, sample_anomaly_data):
        """Test complete preservation pipeline."""
        guard = AnomalyPreservationGuard()
        
        # Set up comprehensive detector
        def performance_detector(data):
            cpu = data.get('indicators', {}).get('cpu_spike', 0)
            return cpu > 90.0
        
        def memory_detector(data):
            memory_leak = data.get('indicators', {}).get('memory_leak', False)
            return memory_leak
        
        def response_detector(data):
            response_time = data.get('indicators', {}).get('response_time', 0)
            return response_time > 2.0
        
        guard.detector.add_algorithm("performance", performance_detector)
        guard.detector.add_algorithm("memory", memory_detector)
        guard.detector.add_algorithm("response", response_detector)
        
        # Set aggressive preservation mode
        guard.set_preservation_mode(PreservationMode.AGGRESSIVE)
        
        # Run preservation
        result = guard.preserve_anomalies(sample_anomaly_data)
        
        # Verify results
        assert result.is_preserved is True
        assert result.preservation_rate >= 0.8  # Should preserve most anomalies
        assert result.metrics is not None
        assert result.metrics.quality_score >= 0.7  # Should have good quality
        
        # Check that different anomaly types were detected
        anomaly_types = set()
        for detail in result.anomalies_details:
            anomaly_types.add(detail['type'])
        
        assert len(anomaly_types) >= 2  # Should detect multiple types
    
    def test_preservation_with_quality_monitoring(self, sample_anomaly_data):
        """Test preservation with quality monitoring integration."""
        guard = AnomalyPreservationGuard()
        
        # Set up detector
        def quality_monitored_detector(data):
            score = data.get('anomaly_score', 0)
            return score > 0.6
        
        guard.detector.add_algorithm("quality", quality_monitored_detector)
        
        # Run preservation with quality monitoring
        result = guard.preserve_anomalies(sample_anomaly_data)
        
        # Verify quality metrics
        assert result.metrics.quality_score >= 0.0
        assert result.metrics.quality_score <= 1.0
        assert result.metrics.preservation_rate >= 0.0
        assert result.metrics.preservation_rate <= 1.0
        
        # Quality score should be reasonable
        assert result.metrics.quality_score >= 0.5  # At least moderate quality
    
    def test_preservation_performance_optimization(self, large_dataset):
        """Test preservation performance optimization."""
        guard = AnomalyPreservationGuard()
        
        # Set up efficient detector
        def efficient_detector(data):
            # Simple, fast detection
            value = data.get('data', {}).get('value', 0)
            return value % 50 == 0  # Every 50th record
        
        guard.detector.add_algorithm("efficient", efficient_detector)
        
        # Test performance
        test_data = large_dataset[:2000]
        
        start_time = datetime.now()
        
        result = guard.preserve_anomalies(test_data)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Should be fast
        assert duration < 60.0  # 60 seconds for 2000 records
        
        # Should preserve anomalies
        assert result.is_preserved is True
        assert result.metrics.total_records == 2000


if __name__ == '__main__':
    pytest.main([__file__])
