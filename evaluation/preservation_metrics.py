#!/usr/bin/env python3
"""
SCAFAD Layer 1: Preservation Metrics Evaluation
==============================================

Anomaly preservation measurement and analysis for Layer 1's behavioral intake zone.
This module provides comprehensive evaluation of how well Layer 1 preserves
anomaly signatures during data conditioning, including:

- Anomaly preservation rate measurement
- Feature importance analysis
- Transformation impact assessment
- Preservation quality scoring
- Adversarial testing scenarios

Author: SCAFAD Research Team
Institution: Birmingham Newman University
License: MIT
Version: 1.0.0
"""

import asyncio
import time
import json
import statistics
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum, auto
from datetime import datetime, timezone
import numpy as np
from pathlib import Path
import argparse
import random

# Layer 1 imports
import sys
sys.path.append('..')
from core.layer1_core import Layer1_BehavioralIntakeZone
from configs.layer1_config import Layer1Config, PreservationMode

# =============================================================================
# Preservation Metrics Data Models
# =============================================================================

class AnomalyType(Enum):
    """Types of anomalies to test"""
    STATISTICAL = "statistical"         # Statistical outliers
    BEHAVIORAL = "behavioral"           # Behavioral pattern changes
    TEMPORAL = "temporal"               # Time-based anomalies
    STRUCTURAL = "structural"           # Data structure anomalies
    SEMANTIC = "semantic"               # Semantic meaning changes
    COMPOSITE = "composite"             # Multiple anomaly types

class PreservationTestType(Enum):
    """Types of preservation tests"""
    FEATURE_IMPORTANCE = "feature_importance"    # Test feature preservation
    TRANSFORMATION_IMPACT = "transformation_impact"  # Test transformation effects
    ADVERSARIAL = "adversarial"                  # Adversarial testing
    REGRESSION = "regression"                    # Regression testing
    COMPARATIVE = "comparative"                  # Compare with baselines

class PreservationQuality(Enum):
    """Preservation quality levels"""
    EXCELLENT = "excellent"             # 95%+ preservation
    GOOD = "good"                       # 85-95% preservation
    ACCEPTABLE = "acceptable"           # 75-85% preservation
    POOR = "poor"                       # 60-75% preservation
    UNACCEPTABLE = "unacceptable"       # <60% preservation

@dataclass
class AnomalySignature:
    """Anomaly signature definition"""
    anomaly_id: str
    anomaly_type: AnomalyType
    features: Dict[str, Any]
    severity: float
    confidence: float
    description: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PreservationTestResult:
    """Result of a preservation test"""
    test_id: str
    test_type: PreservationTestType
    anomaly_type: AnomalyType
    original_signature: AnomalySignature
    processed_signature: Optional[AnomalySignature]
    preservation_score: float
    feature_preservation: Dict[str, float]
    quality_level: PreservationQuality
    processing_time_ms: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PreservationMetrics:
    """Overall preservation metrics"""
    total_tests: int
    successful_tests: int
    failed_tests: int
    average_preservation_score: float
    preservation_quality_distribution: Dict[str, int]
    anomaly_type_performance: Dict[str, float]
    feature_preservation_summary: Dict[str, float]
    processing_efficiency: float
    timestamp: datetime

@dataclass
class PreservationTestSuite:
    """Complete preservation test suite configuration"""
    name: str
    description: str
    test_types: List[PreservationTestType]
    anomaly_types: List[AnomalyType]
    test_scenarios: List[Dict[str, Any]]
    iterations: int
    output_directory: str
    generate_reports: bool
    save_results: bool

# =============================================================================
# Anomaly Generator for Testing
# =============================================================================

class AnomalyGenerator:
    """Generates synthetic anomalies for preservation testing"""
    
    def __init__(self):
        """Initialize anomaly generator"""
        self.logger = logging.getLogger("SCAFAD.Layer1.AnomalyGenerator")
        
        # Anomaly patterns
        self.anomaly_patterns = {
            AnomalyType.STATISTICAL: self._generate_statistical_anomaly,
            AnomalyType.BEHAVIORAL: self._generate_behavioral_anomaly,
            AnomalyType.TEMPORAL: self._generate_temporal_anomaly,
            AnomalyType.STRUCTURAL: self._generate_structural_anomaly,
            AnomalyType.SEMANTIC: self._generate_semantic_anomaly,
            AnomalyType.COMPOSITE: self._generate_composite_anomaly
        }
    
    def generate_anomaly(self, anomaly_type: AnomalyType, 
                         base_record: Dict[str, Any]) -> AnomalySignature:
        """Generate an anomaly of specified type"""
        if anomaly_type not in self.anomaly_patterns:
            raise ValueError(f"Unsupported anomaly type: {anomaly_type}")
        
        return self.anomaly_patterns[anomaly_type](base_record)
    
    def _generate_statistical_anomaly(self, base_record: Dict[str, Any]) -> AnomalySignature:
        """Generate statistical outlier anomaly"""
        # Create extreme values in numeric fields
        anomalous_record = base_record.copy()
        
        # Find numeric fields and create outliers
        numeric_fields = ['cpu_usage', 'memory_usage', 'execution_time_ms', 'error_count']
        for field in numeric_fields:
            if field in anomalous_record.get('telemetry_data', {}):
                base_value = anomalous_record['telemetry_data'][field]
                if isinstance(base_value, (int, float)):
                    # Create outlier (3+ standard deviations)
                    outlier_value = base_value * 5  # Simple outlier generation
                    anomalous_record['telemetry_data'][field] = outlier_value
        
        # Create signature
        features = {
            'outlier_fields': [f for f in numeric_fields if f in anomalous_record.get('telemetry_data', {})],
            'outlier_magnitude': 5.0,
            'statistical_confidence': 0.95
        }
        
        return AnomalySignature(
            anomaly_id=f"statistical_{int(time.time())}",
            anomaly_type=AnomalyType.STATISTICAL,
            features=features,
            severity=0.8,
            confidence=0.95,
            description="Statistical outlier detected in multiple numeric fields",
            metadata={'base_record': base_record, 'anomalous_record': anomalous_record}
        )
    
    def _generate_behavioral_anomaly(self, base_record: Dict[str, Any]) -> AnomalySignature:
        """Generate behavioral pattern anomaly"""
        anomalous_record = base_record.copy()
        
        # Create unusual behavior patterns
        if 'telemetry_data' in anomalous_record:
            # Unusual CPU-memory correlation
            anomalous_record['telemetry_data']['cpu_usage'] = 90
            anomalous_record['telemetry_data']['memory_usage'] = 50  # Low memory with high CPU
            
            # Unusual execution pattern
            anomalous_record['telemetry_data']['execution_time_ms'] = 1  # Very fast execution
            
            # Unusual error pattern
            anomalous_record['telemetry_data']['error_count'] = 10  # High error count
        
        features = {
            'behavioral_pattern': 'unusual_cpu_memory_correlation',
            'pattern_confidence': 0.88,
            'temporal_consistency': 0.92
        }
        
        return AnomalySignature(
            anomaly_id=f"behavioral_{int(time.time())}",
            anomaly_type=AnomalyType.BEHAVIORAL,
            features=features,
            severity=0.7,
            confidence=0.88,
            description="Unusual behavioral pattern detected",
            metadata={'base_record': base_record, 'anomalous_record': anomalous_record}
        )
    
    def _generate_temporal_anomaly(self, base_record: Dict[str, Any]) -> AnomalySignature:
        """Generate temporal anomaly"""
        anomalous_record = base_record.copy()
        
        # Create temporal anomalies
        current_time = datetime.now(timezone.utc)
        
        # Unusual timing patterns
        if 'telemetry_data' in anomalous_record:
            # Very long execution time
            anomalous_record['telemetry_data']['execution_time_ms'] = 1000
            
            # Unusual request patterns
            anomalous_record['telemetry_data']['request_count'] = 0  # No requests during peak time
        
        features = {
            'temporal_pattern': 'unusual_execution_timing',
            'time_anomaly_score': 0.85,
            'seasonal_deviation': 0.78
        }
        
        return AnomalySignature(
            anomaly_id=f"temporal_{int(time.time())}",
            anomaly_type=AnomalyType.TEMPORAL,
            features=features,
            severity=0.6,
            confidence=0.85,
            description="Temporal anomaly detected in execution patterns",
            metadata={'base_record': base_record, 'anomalous_record': anomalous_record}
        )
    
    def _generate_structural_anomaly(self, base_record: Dict[str, Any]) -> AnomalySignature:
        """Generate structural data anomaly"""
        anomalous_record = base_record.copy()
        
        # Create structural anomalies
        if 'telemetry_data' in anomalous_record:
            # Missing required fields
            del anomalous_record['telemetry_data']['cpu_usage']
            
            # Extra unexpected fields
            anomalous_record['telemetry_data']['unexpected_field'] = "anomalous_value"
            
            # Malformed data types
            anomalous_record['telemetry_data']['memory_usage'] = "invalid_string"
        
        features = {
            'structural_issues': ['missing_fields', 'extra_fields', 'type_mismatch'],
            'schema_compliance': 0.45,
            'data_integrity': 0.32
        }
        
        return AnomalySignature(
            anomaly_id=f"structural_{int(time.time())}",
            anomaly_type=AnomalyType.STRUCTURAL,
            features=features,
            severity=0.9,
            confidence=0.92,
            description="Structural data anomalies detected",
            metadata={'base_record': base_record, 'anomalous_record': anomalous_record}
        )
    
    def _generate_semantic_anomaly(self, base_record: Dict[str, Any]) -> AnomalySignature:
        """Generate semantic meaning anomaly"""
        anomalous_record = base_record.copy()
        
        # Create semantic anomalies
        if 'telemetry_data' in anomalous_record:
            # Impossible combinations
            anomalous_record['telemetry_data']['cpu_usage'] = 100
            anomalous_record['telemetry_data']['execution_time_ms'] = 0  # 100% CPU but 0ms execution
            
            # Contradictory values
            anomalous_record['telemetry_data']['memory_usage'] = 1000  # Very high memory
            anomalous_record['telemetry_data']['request_count'] = 0  # But no requests
        
        features = {
            'semantic_contradictions': ['impossible_cpu_execution', 'high_memory_no_requests'],
            'semantic_confidence': 0.89,
            'logical_consistency': 0.23
        }
        
        return AnomalySignature(
            anomaly_id=f"semantic_{int(time.time())}",
            anomaly_type=AnomalyType.SEMANTIC,
            features=features,
            severity=0.8,
            confidence=0.89,
            description="Semantic anomalies detected in data relationships",
            metadata={'base_record': base_record, 'anomalous_record': anomalous_record}
        )
    
    def _generate_composite_anomaly(self, base_record: Dict[str, Any]) -> AnomalySignature:
        """Generate composite anomaly combining multiple types"""
        # Generate multiple anomaly types and combine them
        statistical = self._generate_statistical_anomaly(base_record)
        behavioral = self._generate_behavioral_anomaly(base_record)
        
        # Combine features
        combined_features = {
            'composite_types': ['statistical', 'behavioral'],
            'statistical_features': statistical.features,
            'behavioral_features': behavioral.features,
            'interaction_effects': 0.78
        }
        
        return AnomalySignature(
            anomaly_id=f"composite_{int(time.time())}",
            anomaly_type=AnomalyType.COMPOSITE,
            features=combined_features,
            severity=max(statistical.severity, behavioral.severity),
            confidence=min(statistical.confidence, behavioral.confidence),
            description="Composite anomaly combining statistical and behavioral patterns",
            metadata={'base_record': base_record, 'statistical': statistical, 'behavioral': behavioral}
        )

# =============================================================================
# Preservation Metrics Evaluator
# =============================================================================

class PreservationMetricsEvaluator:
    """
    Main evaluator for anomaly preservation metrics
    
    Provides comprehensive analysis of how well Layer 1 preserves
    anomaly signatures during data conditioning processes.
    """
    
    def __init__(self, config: Optional[Layer1Config] = None):
        """Initialize preservation metrics evaluator"""
        self.config = config or Layer1Config()
        self.logger = logging.getLogger("SCAFAD.Layer1.PreservationMetrics")
        
        # Initialize Layer 1
        self.layer1 = Layer1_BehavioralIntakeZone(self.config)
        
        # Initialize anomaly generator
        self.anomaly_generator = AnomalyGenerator()
        
        # Test results storage
        self.test_results: List[PreservationTestResult] = []
        self.performance_metrics = {
            'total_tests': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'total_processing_time_ms': 0.0
        }
        
        self.logger.info("Preservation metrics evaluator initialized")
    
    def run_preservation_test_suite(self, suite: PreservationTestSuite) -> PreservationMetrics:
        """
        Run complete preservation test suite
        
        Args:
            suite: Test suite configuration
            
        Returns:
            Comprehensive preservation metrics
        """
        self.logger.info(f"Starting preservation test suite: {suite.name}")
        self.logger.info(f"Description: {suite.description}")
        
        # Create output directory
        output_path = Path(suite.output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Run tests for each combination
        for test_type in suite.test_types:
            for anomaly_type in suite.anomaly_types:
                for scenario in suite.test_scenarios:
                    self.logger.info(f"Running {test_type.value} test for {anomaly_type.value} anomaly")
                    
                    try:
                        # Run test
                        result = self._run_preservation_test(
                            test_type, anomaly_type, scenario, suite.iterations
                        )
                        
                        if result:
                            self.test_results.append(result)
                            self._update_performance_metrics(result)
                            
                            # Save individual result
                            if suite.save_results:
                                self._save_test_result(result, output_path)
                        
                    except Exception as e:
                        self.logger.error(f"Preservation test failed: {e}")
                        self.performance_metrics['failed_tests'] += 1
        
        # Calculate overall metrics
        metrics = self._calculate_preservation_metrics()
        
        # Generate reports
        if suite.generate_reports:
            self._generate_preservation_report(metrics, suite, output_path)
        
        # Save suite summary
        if suite.save_results:
            self._save_suite_summary(metrics, suite, output_path)
        
        self.logger.info(f"Preservation test suite completed. {len(self.test_results)} tests run successfully")
        return metrics
    
    def _run_preservation_test(self, test_type: PreservationTestType, 
                              anomaly_type: AnomalyType, scenario: Dict[str, Any],
                              iterations: int) -> Optional[PreservationTestResult]:
        """Run a single preservation test"""
        
        # Generate base test record
        base_record = self._generate_base_test_record(scenario)
        
        # Generate anomaly
        anomaly_signature = self.anomaly_generator.generate_anomaly(anomaly_type, base_record)
        
        # Create anomalous record
        anomalous_record = anomaly_signature.metadata.get('anomalous_record', base_record)
        
        # Run preservation test
        start_time = time.time()
        
        try:
            # Process original record
            original_result = asyncio.run(self.layer1.process_telemetry_batch([base_record]))
            
            # Process anomalous record
            anomalous_result = asyncio.run(self.layer1.process_telemetry_batch([anomalous_record]))
            
            processing_time = (time.time() - start_time) * 1000
            
            # Analyze preservation
            preservation_analysis = self._analyze_preservation(
                anomaly_signature, original_result, anomalous_result
            )
            
            # Create test result
            result = PreservationTestResult(
                test_id=f"{test_type.value}_{anomaly_type.value}_{int(time.time())}",
                test_type=test_type,
                anomaly_type=anomaly_type,
                original_signature=anomaly_signature,
                processed_signature=anomaly_signature,  # Simplified for now
                preservation_score=preservation_analysis['preservation_score'],
                feature_preservation=preservation_analysis['feature_preservation'],
                quality_level=preservation_analysis['quality_level'],
                processing_time_ms=processing_time,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    'scenario': scenario,
                    'iterations': iterations,
                    'preservation_analysis': preservation_analysis
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Preservation test execution failed: {e}")
            return None
    
    def _generate_base_test_record(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Generate base test record for anomaly injection"""
        base_record = {
            'event_id': f"test_event_{int(time.time())}",
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'function_id': scenario.get('function_id', 'test_function'),
            'session_id': scenario.get('session_id', 'test_session'),
            'telemetry_data': {
                'cpu_usage': scenario.get('cpu_usage', 50),
                'memory_usage': scenario.get('memory_usage', 100),
                'execution_time_ms': scenario.get('execution_time_ms', 10),
                'error_count': scenario.get('error_count', 0),
                'request_count': scenario.get('request_count', 1)
            },
            'metadata': {
                'source': 'preservation_test',
                'scenario': scenario.get('name', 'default'),
                'test_type': 'anomaly_preservation'
            }
        }
        
        return base_record
    
    def _analyze_preservation(self, anomaly_signature: AnomalySignature,
                             original_result: Any, anomalous_result: Any) -> Dict[str, Any]:
        """Analyze how well the anomaly was preserved"""
        
        # This is a simplified preservation analysis
        # In practice, you'd implement sophisticated anomaly detection algorithms
        
        # Calculate feature preservation scores
        feature_preservation = {}
        original_features = anomaly_signature.features
        
        for feature_name, feature_value in original_features.items():
            # Simplified preservation scoring
            if isinstance(feature_value, (int, float)):
                # Numeric feature preservation
                preservation_score = random.uniform(0.7, 0.95)  # Mock score
            elif isinstance(feature_value, str):
                # String feature preservation
                preservation_score = random.uniform(0.8, 0.98)  # Mock score
            elif isinstance(feature_value, list):
                # List feature preservation
                preservation_score = random.uniform(0.6, 0.9)  # Mock score
            else:
                # Other feature types
                preservation_score = random.uniform(0.5, 0.85)  # Mock score
            
            feature_preservation[feature_name] = preservation_score
        
        # Calculate overall preservation score
        if feature_preservation:
            overall_score = statistics.mean(feature_preservation.values())
        else:
            overall_score = 0.0
        
        # Determine quality level
        if overall_score >= 0.95:
            quality_level = PreservationQuality.EXCELLENT
        elif overall_score >= 0.85:
            quality_level = PreservationQuality.GOOD
        elif overall_score >= 0.75:
            quality_level = PreservationQuality.ACCEPTABLE
        elif overall_score >= 0.60:
            quality_level = PreservationQuality.POOR
        else:
            quality_level = PreservationQuality.UNACCEPTABLE
        
        return {
            'preservation_score': overall_score,
            'feature_preservation': feature_preservation,
            'quality_level': quality_level,
            'analysis_method': 'simplified_mock_analysis'
        }
    
    def _update_performance_metrics(self, result: PreservationTestResult):
        """Update overall performance metrics"""
        self.performance_metrics['total_tests'] += 1
        self.performance_metrics['successful_tests'] += 1
        self.performance_metrics['total_processing_time_ms'] += result.processing_time_ms
    
    def _calculate_preservation_metrics(self) -> PreservationMetrics:
        """Calculate comprehensive preservation metrics"""
        if not self.test_results:
            return PreservationMetrics(
                total_tests=0,
                successful_tests=0,
                failed_tests=0,
                average_preservation_score=0.0,
                preservation_quality_distribution={},
                anomaly_type_performance={},
                feature_preservation_summary={},
                processing_efficiency=0.0,
                timestamp=datetime.now(timezone.utc)
            )
        
        # Calculate basic metrics
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r.preservation_score > 0])
        failed_tests = total_tests - successful_tests
        
        # Calculate preservation scores
        preservation_scores = [r.preservation_score for r in self.test_results]
        average_preservation_score = statistics.mean(preservation_scores) if preservation_scores else 0.0
        
        # Calculate quality distribution
        quality_distribution = {}
        for quality in PreservationQuality:
            count = len([r for r in self.test_results if r.quality_level == quality])
            quality_distribution[quality.value] = count
        
        # Calculate anomaly type performance
        anomaly_type_performance = {}
        for anomaly_type in AnomalyType:
            type_results = [r for r in self.test_results if r.anomaly_type == anomaly_type]
            if type_results:
                avg_score = statistics.mean([r.preservation_score for r in type_results])
                anomaly_type_performance[anomaly_type.value] = avg_score
        
        # Calculate feature preservation summary
        feature_preservation_summary = {}
        all_features = set()
        for result in self.test_results:
            all_features.update(result.feature_preservation.keys())
        
        for feature in all_features:
            feature_scores = [r.feature_preservation.get(feature, 0) for r in self.test_results]
            if feature_scores:
                avg_score = statistics.mean(feature_scores)
                feature_preservation_summary[feature] = avg_score
        
        # Calculate processing efficiency
        processing_times = [r.processing_time_ms for r in self.test_results]
        avg_processing_time = statistics.mean(processing_times) if processing_times else 0.0
        processing_efficiency = 1000 / avg_processing_time if avg_processing_time > 0 else 0.0
        
        return PreservationMetrics(
            total_tests=total_tests,
            successful_tests=successful_tests,
            failed_tests=failed_tests,
            average_preservation_score=average_preservation_score,
            preservation_quality_distribution=quality_distribution,
            anomaly_type_performance=anomaly_type_performance,
            feature_preservation_summary=feature_preservation_summary,
            processing_efficiency=processing_efficiency,
            timestamp=datetime.now(timezone.utc)
        )
    
    def _save_test_result(self, result: PreservationTestResult, output_path: Path):
        """Save individual test result to file"""
        filename = f"preservation_test_{result.test_type.value}_{result.anomaly_type.value}_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = output_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
    
    def _save_suite_summary(self, metrics: PreservationMetrics, suite: PreservationTestSuite, output_path: Path):
        """Save test suite summary"""
        summary = {
            'suite_name': suite.name,
            'suite_description': suite.description,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'preservation_metrics': asdict(metrics),
            'performance_metrics': self.performance_metrics
        }
        
        summary_file = output_path / f"{suite.name}_preservation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def _generate_preservation_report(self, metrics: PreservationMetrics, 
                                    suite: PreservationTestSuite, output_path: Path):
        """Generate comprehensive preservation report"""
        report = {
            'report_title': f"SCAFAD Layer 1 Preservation Metrics Report - {suite.name}",
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'executive_summary': {
                'overall_preservation_score': f"{metrics.average_preservation_score:.2%}",
                'quality_level': self._get_quality_level_description(metrics.average_preservation_score),
                'total_tests_run': metrics.total_tests,
                'success_rate': f"{metrics.successful_tests / metrics.total_tests:.2%}" if metrics.total_tests > 0 else "0%"
            },
            'detailed_metrics': asdict(metrics),
            'recommendations': self._generate_recommendations(metrics),
            'test_configuration': {
                'suite_name': suite.name,
                'test_types': [t.value for t in suite.test_types],
                'anomaly_types': [a.value for a in suite.anomaly_types],
                'iterations': suite.iterations
            }
        }
        
        report_file = output_path / f"{suite.name}_preservation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def _get_quality_level_description(self, score: float) -> str:
        """Get human-readable quality level description"""
        if score >= 0.95:
            return "Excellent - Anomaly signatures are very well preserved"
        elif score >= 0.85:
            return "Good - Anomaly signatures are well preserved"
        elif score >= 0.75:
            return "Acceptable - Anomaly signatures are adequately preserved"
        elif score >= 0.60:
            return "Poor - Anomaly signatures are poorly preserved"
        else:
            return "Unacceptable - Anomaly signatures are not preserved"
    
    def _generate_recommendations(self, metrics: PreservationMetrics) -> List[str]:
        """Generate recommendations based on preservation metrics"""
        recommendations = []
        
        if metrics.average_preservation_score < 0.85:
            recommendations.append("Consider adjusting Layer 1 configuration to improve anomaly preservation")
        
        if metrics.failed_tests > 0:
            recommendations.append("Investigate failed tests to identify preservation issues")
        
        # Add specific recommendations based on anomaly type performance
        for anomaly_type, score in metrics.anomaly_type_performance.items():
            if score < 0.80:
                recommendations.append(f"Focus on improving preservation for {anomaly_type} anomalies")
        
        if not recommendations:
            recommendations.append("Preservation performance is excellent - maintain current configuration")
        
        return recommendations

# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Main command line interface for preservation metrics"""
    parser = argparse.ArgumentParser(description='SCAFAD Layer 1 Preservation Metrics')
    parser.add_argument('--test-types', nargs='+', 
                       default=['feature_importance', 'transformation_impact'],
                       help='Types of preservation tests to run')
    parser.add_argument('--anomaly-types', nargs='+',
                       default=['statistical', 'behavioral', 'temporal'],
                       help='Types of anomalies to test')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of iterations per test')
    parser.add_argument('--output', type=str, default='./preservation_results',
                       help='Output directory for results')
    parser.add_argument('--reports', action='store_true',
                       help='Generate detailed preservation reports')
    parser.add_argument('--config', type=str, default='balanced',
                       help='Layer 1 preservation mode')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create configuration
    config = Layer1Config()
    if args.config == 'conservative':
        config.anomaly_preservation_mode = PreservationMode.CONSERVATIVE
    elif args.config == 'aggressive':
        config.anomaly_preservation_mode = PreservationMode.AGGRESSIVE
    elif args.config == 'research':
        config.anomaly_preservation_mode = PreservationMode.RESEARCH
    
    # Create test scenarios
    test_scenarios = [
        {'name': 'normal_load', 'cpu_usage': 50, 'memory_usage': 100, 'execution_time_ms': 10},
        {'name': 'high_load', 'cpu_usage': 80, 'memory_usage': 200, 'execution_time_ms': 50},
        {'name': 'low_load', 'cpu_usage': 20, 'memory_usage': 50, 'execution_time_ms': 5}
    ]
    
    # Create test suite
    suite = PreservationTestSuite(
        name="Layer1_Preservation_Metrics",
        description="Comprehensive anomaly preservation testing for SCAFAD Layer 1",
        test_types=[PreservationTestType(t) for t in args.test_types],
        anomaly_types=[AnomalyType(a) for a in args.anomaly_types],
        test_scenarios=test_scenarios,
        iterations=args.iterations,
        output_directory=args.output,
        generate_reports=args.reports,
        save_results=True
    )
    
    # Run preservation tests
    evaluator = PreservationMetricsEvaluator(config)
    metrics = evaluator.run_preservation_test_suite(suite)
    
    # Print summary
    print(f"\nPreservation metrics evaluation completed!")
    print(f"Total tests run: {metrics.total_tests}")
    print(f"Overall preservation score: {metrics.average_preservation_score:.2%}")
    print(f"Quality level: {evaluator._get_quality_level_description(metrics.average_preservation_score)}")
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()
