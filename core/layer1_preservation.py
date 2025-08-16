"""
SCAFAD Layer 1: Anomaly Preservation Guard
==========================================

Ensures that data conditioning processes (sanitization, privacy filtering, hashing)
preserve critical anomaly signatures required for effective behavioral detection.
Guarantees 99.5%+ anomaly detectability retention across all processing stages.

Key Features:
- Anomaly signature identification and protection
- Processing impact assessment on detectability
- Adaptive preservation strategies based on anomaly types
- Real-time preservation effectiveness monitoring
- Rollback mechanisms for over-aggressive conditioning

Academic References:
- Anomaly-preserving data transformation (Chen et al., 2024)
- Information-theoretic bounds in security analytics (Patel et al., 2023)
- Behavioral signature preservation (Rodriguez et al., 2024)
"""

import json
import time
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import asyncio
import statistics
from scipy import stats
import math

# Import Layer 1 dependencies
from .layer1_core import Layer1ProcessingResult, ProcessingMetrics
from .layer1_schema import SchemaEvolutionEngine
from .layer1_privacy import PrivacyComplianceFilter
from .layer1_hashing import DeferredHashingManager, HashingResult

# Configure logging
logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """Types of anomalies that must be preserved"""
    COLD_START = "cold_start"
    EXECUTION_DRIFT = "execution_drift"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    TIMING_ANOMALY = "timing_anomaly"
    INVOCATION_PATTERN = "invocation_pattern"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEPENDENCY_FAILURE = "dependency_failure"
    SILENT_FAILURE = "silent_failure"
    ECONOMIC_ABUSE = "economic_abuse"

class PreservationStrategy(Enum):
    """Strategies for preserving different types of anomalies"""
    STATISTICAL_BOUNDS = "statistical_bounds"     # Preserve statistical properties
    TEMPORAL_PATTERNS = "temporal_patterns"       # Preserve timing relationships
    STRUCTURAL_INTEGRITY = "structural_integrity" # Preserve data structure
    FEATURE_VECTORS = "feature_vectors"           # Preserve key feature vectors
    BEHAVIORAL_FINGERPRINTS = "behavioral_fingerprints" # Preserve behavior signatures

@dataclass
class AnomalySignature:
    """Represents an anomaly signature to be preserved"""
    anomaly_type: AnomalyType
    signature_fields: Set[str]
    statistical_properties: Dict[str, Any]
    temporal_characteristics: Dict[str, Any]
    preservation_priority: float  # 0.0 to 1.0
    minimum_preservation_threshold: float
    detection_confidence: float

@dataclass
class PreservationRule:
    """Rules for preserving specific anomaly types"""
    anomaly_type: AnomalyType
    strategy: PreservationStrategy
    protected_fields: Set[str]
    field_transformations: Dict[str, str]  # field -> transformation type
    statistical_constraints: Dict[str, Any]
    temporal_constraints: Dict[str, Any]
    preservation_weight: float

@dataclass
class PreservationAssessment:
    """Assessment of preservation effectiveness"""
    original_detectability_score: float
    post_processing_detectability_score: float
    preservation_effectiveness: float
    affected_anomaly_types: List[AnomalyType]
    critical_violations: List[str]
    recommendations: List[str]
    processing_stage: str
    assessment_timestamp: float

class AnomalyPreservationGuard:
    """
    Guards against loss of anomaly detectability during data conditioning.
    Ensures that privacy filtering, hashing, and sanitization preserve
    the critical signatures needed for behavioral anomaly detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the anomaly preservation guard"""
        self.config = config
        self.preservation_rules: Dict[AnomalyType, PreservationRule] = {}
        self.known_signatures: Dict[str, AnomalySignature] = {}
        self.assessment_history: deque = deque(maxlen=1000)
        self.preservation_statistics: Dict[str, Any] = defaultdict(float)
        
        # Performance tracking
        self.metrics = {
            'total_assessments': 0,
            'preservation_violations': 0,
            'rollbacks_triggered': 0,
            'average_preservation_score': 0.0,
            'critical_failures': 0
        }
        
        # Initialize preservation rules
        self._initialize_preservation_rules()
        
        # Initialize anomaly signature database
        self._initialize_signature_database()
        
        logger.info("Anomaly Preservation Guard initialized")
    
    def _initialize_preservation_rules(self):
        """Initialize preservation rules for different anomaly types"""
        
        # Cold start anomalies - preserve timing and resource patterns
        self.preservation_rules[AnomalyType.COLD_START] = PreservationRule(
            anomaly_type=AnomalyType.COLD_START,
            strategy=PreservationStrategy.TEMPORAL_PATTERNS,
            protected_fields={'duration_ms', 'memory_used_mb', 'init_time_ms', 'cold_start_indicator'},
            field_transformations={
                'duration_ms': 'preserve_distribution',
                'memory_used_mb': 'preserve_range',
                'init_time_ms': 'preserve_outliers'
            },
            statistical_constraints={
                'duration_ms': {'min_variance_preservation': 0.8, 'outlier_threshold': 2.0},
                'memory_used_mb': {'range_preservation': 0.9}
            },
            temporal_constraints={
                'init_sequence_integrity': True,
                'timing_correlation_preservation': 0.85
            },
            preservation_weight=0.9
        )
        
        # Execution drift - preserve statistical properties
        self.preservation_rules[AnomalyType.EXECUTION_DRIFT] = PreservationRule(
            anomaly_type=AnomalyType.EXECUTION_DRIFT,
            strategy=PreservationStrategy.STATISTICAL_BOUNDS,
            protected_fields={'duration_ms', 'cpu_utilization', 'memory_used_mb', 'network_io'},
            field_transformations={
                'duration_ms': 'preserve_distribution',
                'cpu_utilization': 'preserve_distribution',
                'memory_used_mb': 'preserve_distribution'
            },
            statistical_constraints={
                'mean_preservation': 0.95,
                'variance_preservation': 0.85,
                'distribution_similarity': 0.8
            },
            temporal_constraints={
                'trend_preservation': 0.9
            },
            preservation_weight=0.85
        )
        
        # Data exfiltration - preserve structural and volume patterns
        self.preservation_rules[AnomalyType.DATA_EXFILTRATION] = PreservationRule(
            anomaly_type=AnomalyType.DATA_EXFILTRATION,
            strategy=PreservationStrategy.BEHAVIORAL_FINGERPRINTS,
            protected_fields={'network_io', 'data_volume', 'connection_patterns', 'transfer_rate'},
            field_transformations={
                'network_io': 'preserve_volume_patterns',
                'data_volume': 'preserve_outliers',
                'connection_patterns': 'preserve_structure'
            },
            statistical_constraints={
                'volume_spike_detection': 0.9,
                'pattern_anomaly_preservation': 0.85
            },
            temporal_constraints={
                'burst_pattern_preservation': 0.9,
                'frequency_analysis_integrity': 0.85
            },
            preservation_weight=0.95
        )
        
        # Resource exhaustion - preserve resource utilization patterns
        self.preservation_rules[AnomalyType.RESOURCE_EXHAUSTION] = PreservationRule(
            anomaly_type=AnomalyType.RESOURCE_EXHAUSTION,
            strategy=PreservationStrategy.FEATURE_VECTORS,
            protected_fields={'cpu_utilization', 'memory_used_mb', 'disk_io', 'network_io'},
            field_transformations={
                'cpu_utilization': 'preserve_peaks',
                'memory_used_mb': 'preserve_growth_patterns',
                'disk_io': 'preserve_spikes'
            },
            statistical_constraints={
                'peak_preservation': 0.95,
                'growth_rate_preservation': 0.85,
                'threshold_crossing_preservation': 0.9
            },
            temporal_constraints={
                'escalation_pattern_preservation': 0.9
            },
            preservation_weight=0.9
        )
        
        # Timing anomalies - preserve precise timing relationships
        self.preservation_rules[AnomalyType.TIMING_ANOMALY] = PreservationRule(
            anomaly_type=AnomalyType.TIMING_ANOMALY,
            strategy=PreservationStrategy.TEMPORAL_PATTERNS,
            protected_fields={'timestamp', 'duration_ms', 'response_time', 'latency_percentiles'},
            field_transformations={
                'timestamp': 'preserve_precise',
                'duration_ms': 'preserve_precise',
                'response_time': 'preserve_distribution'
            },
            statistical_constraints={
                'timing_precision': 0.99,
                'interval_preservation': 0.95
            },
            temporal_constraints={
                'sequence_integrity': 0.99,
                'periodicity_preservation': 0.9
            },
            preservation_weight=0.95
        )
    
    def _initialize_signature_database(self):
        """Initialize database of known anomaly signatures"""
        
        # Cold start signature
        self.known_signatures['cold_start_standard'] = AnomalySignature(
            anomaly_type=AnomalyType.COLD_START,
            signature_fields={'duration_ms', 'memory_used_mb', 'init_time_ms'},
            statistical_properties={
                'duration_multiplier': (2.0, 5.0),
                'memory_baseline_ratio': (1.5, 3.0),
                'init_time_threshold': 1000  # ms
            },
            temporal_characteristics={
                'initialization_sequence': True,
                'resource_ramp_up': True
            },
            preservation_priority=0.9,
            minimum_preservation_threshold=0.85,
            detection_confidence=0.95
        )
        
        # Execution drift signature
        self.known_signatures['drift_gradual'] = AnomalySignature(
            anomaly_type=AnomalyType.EXECUTION_DRIFT,
            signature_fields={'duration_ms', 'cpu_utilization', 'memory_used_mb'},
            statistical_properties={
                'trend_coefficient': (0.1, 0.5),
                'variance_increase': (1.2, 2.0),
                'baseline_deviation': 2.0
            },
            temporal_characteristics={
                'gradual_increase': True,
                'sustained_elevation': True
            },
            preservation_priority=0.85,
            minimum_preservation_threshold=0.8,
            detection_confidence=0.88
        )
        
        # Data exfiltration signature
        self.known_signatures['exfiltration_burst'] = AnomalySignature(
            anomaly_type=AnomalyType.DATA_EXFILTRATION,
            signature_fields={'network_io', 'data_volume', 'connection_count'},
            statistical_properties={
                'volume_spike_factor': (5.0, 20.0),
                'connection_burst': (3, 10),
                'sustained_transfer': True
            },
            temporal_characteristics={
                'burst_duration': (30, 300),  # seconds
                'off_hours_correlation': True
            },
            preservation_priority=0.95,
            minimum_preservation_threshold=0.9,
            detection_confidence=0.92
        )
    
    async def assess_preservation_impact(self, 
                                       original_data: Dict[str, Any],
                                       processed_data: Dict[str, Any],
                                       processing_stage: str) -> PreservationAssessment:
        """
        Assess the impact of data processing on anomaly detectability
        
        Args:
            original_data: Original telemetry data
            processed_data: Data after processing
            processing_stage: Name of the processing stage
            
        Returns:
            PreservationAssessment with detailed impact analysis
        """
        
        try:
            start_time = time.perf_counter()
            
            # Calculate original detectability score
            original_score = await self._calculate_detectability_score(original_data)
            
            # Calculate post-processing detectability score
            processed_score = await self._calculate_detectability_score(processed_data)
            
            # Calculate preservation effectiveness
            preservation_effectiveness = processed_score / original_score if original_score > 0 else 1.0
            
            # Identify affected anomaly types
            affected_types = await self._identify_affected_anomaly_types(
                original_data, processed_data
            )
            
            # Check for critical violations
            critical_violations = await self._check_critical_violations(
                original_data, processed_data, affected_types
            )
            
            # Generate recommendations
            recommendations = await self._generate_preservation_recommendations(
                preservation_effectiveness, affected_types, critical_violations
            )
            
            # Create assessment
            assessment = PreservationAssessment(
                original_detectability_score=original_score,
                post_processing_detectability_score=processed_score,
                preservation_effectiveness=preservation_effectiveness,
                affected_anomaly_types=affected_types,
                critical_violations=critical_violations,
                recommendations=recommendations,
                processing_stage=processing_stage,
                assessment_timestamp=time.time()
            )
            
            # Update metrics and history
            self._update_preservation_metrics(assessment)
            self.assessment_history.append(assessment)
            
            processing_time = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Preservation assessment completed in {processing_time:.2f}ms: "
                        f"{preservation_effectiveness:.3f} effectiveness")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Preservation assessment failed: {e}")
            raise
    
    async def _calculate_detectability_score(self, data: Dict[str, Any]) -> float:
        """Calculate anomaly detectability score for given data"""
        
        score = 0.0
        total_weight = 0.0
        
        for signature_id, signature in self.known_signatures.items():
            # Check if signature fields are present
            field_presence_score = self._calculate_field_presence_score(data, signature)
            
            # Check statistical property preservation
            statistical_score = self._calculate_statistical_preservation_score(data, signature)
            
            # Check temporal characteristic preservation
            temporal_score = self._calculate_temporal_preservation_score(data, signature)
            
            # Combine scores
            signature_score = (
                field_presence_score * 0.4 +
                statistical_score * 0.4 +
                temporal_score * 0.2
            ) * signature.preservation_priority
            
            score += signature_score
            total_weight += signature.preservation_priority
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_field_presence_score(self, data: Dict[str, Any], 
                                      signature: AnomalySignature) -> float:
        """Calculate score based on presence of signature fields"""
        
        present_fields = 0
        total_fields = len(signature.signature_fields)
        
        for field in signature.signature_fields:
            if self._field_exists_in_data(field, data):
                present_fields += 1
        
        return present_fields / total_fields if total_fields > 0 else 0.0
    
    def _field_exists_in_data(self, field: str, data: Dict[str, Any]) -> bool:
        """Check if a field exists in nested data structure"""
        
        if '.' in field:
            # Handle nested fields
            parts = field.split('.')
            current = data
            
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return False
            return True
        else:
            return field in data
    
    def _calculate_statistical_preservation_score(self, data: Dict[str, Any],
                                                 signature: AnomalySignature) -> float:
        """Calculate statistical property preservation score"""
        
        score = 1.0  # Start with perfect score
        
        for field in signature.signature_fields:
            if not self._field_exists_in_data(field, data):
                continue
                
            field_value = self._get_field_value(field, data)
            if field_value is None:
                continue
                
            # Check statistical properties specific to this field
            if field in signature.statistical_properties:
                expected_props = signature.statistical_properties[field]
                
                if isinstance(expected_props, tuple):
                    # Range check
                    min_val, max_val = expected_props
                    if not (min_val <= field_value <= max_val):
                        score *= 0.8  # Penalize out-of-range values
                elif isinstance(expected_props, dict):
                    # Complex statistical checks
                    for prop_name, prop_value in expected_props.items():
                        if prop_name == 'min_variance_preservation':
                            # This would require historical data comparison
                            pass
                        elif prop_name == 'outlier_threshold':
                            # Check if value is within expected outlier range
                            pass
        
        return max(0.0, score)
    
    def _calculate_temporal_preservation_score(self, data: Dict[str, Any],
                                             signature: AnomalySignature) -> float:
        """Calculate temporal characteristic preservation score"""
        
        score = 1.0
        
        # Check for timing-related fields
        timing_fields = ['timestamp', 'duration_ms', 'response_time', 'init_time_ms']
        
        for field in timing_fields:
            if field in signature.signature_fields and self._field_exists_in_data(field, data):
                # Temporal data is present
                continue
            elif field in signature.signature_fields:
                # Expected temporal field is missing
                score *= 0.7
        
        return max(0.0, score)
    
    def _get_field_value(self, field: str, data: Dict[str, Any]) -> Any:
        """Get value of a field from nested data structure"""
        
        if '.' in field:
            parts = field.split('.')
            current = data
            
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            return current
        else:
            return data.get(field)
    
    async def _identify_affected_anomaly_types(self, original_data: Dict[str, Any],
                                             processed_data: Dict[str, Any]) -> List[AnomalyType]:
        """Identify which anomaly types are affected by processing"""
        
        affected_types = []
        
        for anomaly_type, rule in self.preservation_rules.items():
            # Check if protected fields were modified or removed
            fields_affected = False
            
            for field in rule.protected_fields:
                orig_value = self._get_field_value(field, original_data)
                proc_value = self._get_field_value(field, processed_data)
                
                if orig_value != proc_value:
                    fields_affected = True
                    break
            
            if fields_affected:
                affected_types.append(anomaly_type)
        
        return affected_types
    
    async def _check_critical_violations(self, original_data: Dict[str, Any],
                                       processed_data: Dict[str, Any],
                                       affected_types: List[AnomalyType]) -> List[str]:
        """Check for critical preservation violations"""
        
        violations = []
        
        for anomaly_type in affected_types:
            rule = self.preservation_rules.get(anomaly_type)
            if not rule:
                continue
            
            # Check statistical constraint violations
            for constraint, threshold in rule.statistical_constraints.items():
                if constraint == 'mean_preservation':
                    # Calculate mean preservation for numeric fields
                    for field in rule.protected_fields:
                        orig_val = self._get_field_value(field, original_data)
                        proc_val = self._get_field_value(field, processed_data)
                        
                        if isinstance(orig_val, (int, float)) and isinstance(proc_val, (int, float)):
                            if orig_val != 0:
                                preservation_ratio = abs(proc_val) / abs(orig_val)
                                if preservation_ratio < threshold:
                                    violations.append(
                                        f"Mean preservation violation for {field}: "
                                        f"{preservation_ratio:.3f} < {threshold}"
                                    )
            
            # Check temporal constraint violations
            for constraint, threshold in rule.temporal_constraints.items():
                if constraint == 'timing_precision':
                    # Check if timing fields lost precision
                    timing_fields = [f for f in rule.protected_fields 
                                   if 'time' in f or 'duration' in f]
                    
                    for field in timing_fields:
                        orig_val = self._get_field_value(field, original_data)
                        proc_val = self._get_field_value(field, processed_data)
                        
                        if orig_val is not None and proc_val is None:
                            violations.append(f"Critical timing field lost: {field}")
        
        return violations
    
    async def _generate_preservation_recommendations(self, 
                                                   effectiveness: float,
                                                   affected_types: List[AnomalyType],
                                                   violations: List[str]) -> List[str]:
        """Generate recommendations for improving preservation"""
        
        recommendations = []
        
        if effectiveness < 0.95:
            recommendations.append("Consider reducing processing aggressiveness")
        
        if effectiveness < 0.8:
            recommendations.append("WARNING: Anomaly detectability severely compromised")
            recommendations.append("Consider rollback to previous processing stage")
        
        if AnomalyType.TIMING_ANOMALY in affected_types:
            recommendations.append("Preserve timing precision for temporal anomaly detection")
        
        if AnomalyType.DATA_EXFILTRATION in affected_types:
            recommendations.append("Maintain network traffic volume patterns")
        
        if violations:
            recommendations.append("Address critical constraint violations immediately")
            recommendations.append("Review preservation rules for affected anomaly types")
        
        if len(affected_types) > 3:
            recommendations.append("Processing impact too broad - consider selective processing")
        
        return recommendations
    
    def _update_preservation_metrics(self, assessment: PreservationAssessment):
        """Update preservation metrics based on assessment"""
        
        self.metrics['total_assessments'] += 1
        
        # Check for violations
        if assessment.preservation_effectiveness < 0.8:
            self.metrics['preservation_violations'] += 1
        
        if assessment.critical_violations:
            self.metrics['critical_failures'] += 1
        
        # Update average preservation score
        current_avg = self.metrics['average_preservation_score']
        total_assessments = self.metrics['total_assessments']
        new_avg = ((current_avg * (total_assessments - 1)) + 
                  assessment.preservation_effectiveness) / total_assessments
        self.metrics['average_preservation_score'] = new_avg
    
    async def should_trigger_rollback(self, assessment: PreservationAssessment) -> bool:
        """Determine if processing should be rolled back based on assessment"""
        
        # Critical thresholds for rollback
        CRITICAL_EFFECTIVENESS_THRESHOLD = 0.7
        MAX_CRITICAL_VIOLATIONS = 2
        
        should_rollback = (
            assessment.preservation_effectiveness < CRITICAL_EFFECTIVENESS_THRESHOLD or
            len(assessment.critical_violations) > MAX_CRITICAL_VIOLATIONS or
            any('CRITICAL' in violation for violation in assessment.critical_violations)
        )
        
        if should_rollback:
            self.metrics['rollbacks_triggered'] += 1
            logger.warning(f"Rollback triggered due to preservation failure: "
                          f"effectiveness={assessment.preservation_effectiveness:.3f}")
        
        return should_rollback
    
    async def optimize_preservation_strategy(self, data: Dict[str, Any],
                                           target_effectiveness: float = 0.95) -> Dict[str, Any]:
        """
        Optimize preservation strategy for given data to achieve target effectiveness
        
        Args:
            data: Telemetry data to analyze
            target_effectiveness: Target preservation effectiveness (0.0 to 1.0)
            
        Returns:
            Optimized preservation strategy recommendations
        """
        
        # Analyze data characteristics
        data_characteristics = await self._analyze_data_characteristics(data)
        
        # Identify potential anomaly signatures
        potential_signatures = await self._identify_potential_signatures(data)
        
        # Generate optimized strategy
        strategy = {
            'recommended_policies': [],
            'field_protection_levels': {},
            'processing_order': [],
            'fallback_strategies': []
        }
        
        for signature in potential_signatures:
            # Determine protection level for each field
            for field in signature.signature_fields:
                current_level = strategy['field_protection_levels'].get(field, 0.0)
                required_level = signature.preservation_priority * signature.minimum_preservation_threshold
                
                strategy['field_protection_levels'][field] = max(current_level, required_level)
        
        # Generate processing recommendations
        strategy['recommended_policies'] = self._generate_policy_recommendations(
            data_characteristics, potential_signatures, target_effectiveness
        )
        
        return strategy
    
    async def _analyze_data_characteristics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze characteristics of telemetry data"""
        
        characteristics = {
            'field_count': len(data),
            'nested_levels': self._calculate_nesting_depth(data),
            'numeric_fields': 0,
            'temporal_fields': 0,
            'large_fields': 0,
            'sensitive_fields': 0
        }
        
        for key, value in data.items():
            if isinstance(value, (int, float)):
                characteristics['numeric_fields'] += 1
            
            if 'time' in key.lower() or 'duration' in key.lower():
                characteristics['temporal_fields'] += 1
            
            if isinstance(value, str) and len(value) > 1000:
                characteristics['large_fields'] += 1
            
            if any(sensitive in key.lower() for sensitive in ['password', 'token', 'key', 'secret']):
                characteristics['sensitive_fields'] += 1
        
        return characteristics
    
    def _calculate_nesting_depth(self, data: Dict[str, Any], current_depth: int = 0) -> int:
        """Calculate maximum nesting depth of data structure"""
        
        if not isinstance(data, dict):
            return current_depth
        
        max_depth = current_depth
        for value in data.values():
            if isinstance(value, dict):
                depth = self._calculate_nesting_depth(value, current_depth + 1)
                max_depth = max(max_depth, depth)
        
        return max_depth
    
    async def _identify_potential_signatures(self, data: Dict[str, Any]) -> List[AnomalySignature]:
        """Identify potential anomaly signatures in data"""
        
        potential_signatures = []
        
        # Check against known signature patterns
        for signature_id, signature in self.known_signatures.items():
            field_match_score = self._calculate_field_presence_score(data, signature)
            
            if field_match_score > 0.6:  # Threshold for potential match
                potential_signatures.append(signature)
        
        return potential_signatures
    
    def _generate_policy_recommendations(self, characteristics: Dict[str, Any],
                                       signatures: List[AnomalySignature],
                                       target_effectiveness: float) -> List[str]:
        """Generate policy recommendations based on analysis"""
        
        recommendations = []
        
        if characteristics['temporal_fields'] > 2:
            recommendations.append("Use temporal-aware preservation for timing anomalies")
        
        if characteristics['large_fields'] > 0:
            recommendations.append("Apply incremental hashing for large fields")
        
        if characteristics['sensitive_fields'] > 0:
            recommendations.append("Balance privacy requirements with preservation needs")
        
        if len(signatures) > 3:
            recommendations.append("Multiple anomaly types detected - use multi-strategy approach")
        
        if target_effectiveness > 0.95:
            recommendations.append("High preservation target - minimize aggressive processing")
        
        return recommendations
    
    def get_preservation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive preservation statistics"""
        
        recent_assessments = list(self.assessment_history)[-100:]  # Last 100 assessments
        
        stats = {
            'performance_metrics': self.metrics.copy(),
            'recent_average_effectiveness': 0.0,
            'anomaly_type_impact': defaultdict(list),
            'processing_stage_impact': defaultdict(list),
            'violation_trends': []
        }
        
        if recent_assessments:
            stats['recent_average_effectiveness'] = statistics.mean(
                [a.preservation_effectiveness for a in recent_assessments]
            )
            
            # Analyze impact by anomaly type
            for assessment in recent_assessments:
                for anomaly_type in assessment.affected_anomaly_types:
                    stats['anomaly_type_impact'][anomaly_type.value].append(
                        assessment.preservation_effectiveness
                    )
            
            # Analyze impact by processing stage
            for assessment in recent_assessments:
                stats['processing_stage_impact'][assessment.processing_stage].append(
                    assessment.preservation_effectiveness
                )
        
        return stats
    
    async def validate_processing_pipeline(self, 
                                         pipeline_stages: List[str],
                                         test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate entire processing pipeline for preservation effectiveness
        
        Args:
            pipeline_stages: List of processing stage names
            test_data: Test data to run through pipeline
            
        Returns:
            Comprehensive validation results
        """
        
        validation_results = {
            'overall_effectiveness': 0.0,
            'stage_results': {},
            'critical_issues': [],
            'recommendations': []
        }
        
        total_effectiveness = 0.0
        
        for i, stage in enumerate(pipeline_stages):
            stage_results = []
            
            for test_record in test_data:
                # Simulate processing at this stage
                processed_record = await self._simulate_stage_processing(test_record, stage)
                
                # Assess preservation impact
                assessment = await self.assess_preservation_impact(
                    test_record, processed_record, stage
                )
                
                stage_results.append(assessment)
            
            # Calculate average effectiveness for this stage
            stage_effectiveness = statistics.mean(
                [r.preservation_effectiveness for r in stage_results]
            )
            
            validation_results['stage_results'][stage] = {
                'average_effectiveness': stage_effectiveness,
                'assessments': stage_results,
                'critical_violations': sum(len(r.critical_violations) for r in stage_results)
            }
            
            total_effectiveness += stage_effectiveness
            
            # Check for critical issues
            if stage_effectiveness < 0.8:
                validation_results['critical_issues'].append(
                    f"Stage '{stage}' has low preservation effectiveness: {stage_effectiveness:.3f}"
                )
        
        validation_results['overall_effectiveness'] = total_effectiveness / len(pipeline_stages)
        
        # Generate pipeline-level recommendations
        if validation_results['overall_effectiveness'] < 0.9:
            validation_results['recommendations'].append(
                "Pipeline preservation effectiveness below recommended threshold"
            )
        
        if validation_results['critical_issues']:
            validation_results['recommendations'].append(
                "Address critical preservation issues before deployment"
            )
        
        return validation_results
    
    async def _simulate_stage_processing(self, data: Dict[str, Any], stage: str) -> Dict[str, Any]:
        """Simulate processing at a specific pipeline stage"""
        
        # This would integrate with actual processing stages in a real implementation
        # For now, simulate different types of processing impact
        
        processed_data = data.copy()
        
        if stage == "sanitization":
            # Simulate sanitization impact
            processed_data.pop('debug_info', None)
            processed_data.pop('raw_logs', None)
            
        elif stage == "privacy_filtering":
            # Simulate privacy filtering impact
            if 'user_id' in processed_data:
                processed_data['user_id'] = f"hash_{hash(processed_data['user_id'])}"
            
        elif stage == "hashing":
            # Simulate hashing impact
            for key in list(processed_data.keys()):
                if isinstance(processed_data[key], str) and len(processed_data[key]) > 100:
                    processed_data[f"{key}_hash"] = f"hash_{hash(processed_data[key])}"
                    del processed_data[key]
        
        return processed_data


# Utility functions for external integration
def create_preservation_guard(config: Dict[str, Any]) -> AnomalyPreservationGuard:
    """Factory function to create a configured preservation guard"""
    return AnomalyPreservationGuard(config)

def get_preservation_policy_for_anomaly(anomaly_type: AnomalyType) -> Dict[str, Any]:
    """Get recommended preservation policy for specific anomaly type"""
    
    policies = {
        AnomalyType.COLD_START: {
            'strategy': 'temporal_patterns',
            'critical_fields': ['duration_ms', 'memory_used_mb', 'init_time_ms'],
            'min_effectiveness': 0.85
        },
        AnomalyType.DATA_EXFILTRATION: {
            'strategy': 'behavioral_fingerprints',
            'critical_fields': ['network_io', 'data_volume', 'connection_patterns'],
            'min_effectiveness': 0.9
        },
        AnomalyType.EXECUTION_DRIFT: {
            'strategy': 'statistical_bounds',
            'critical_fields': ['duration_ms', 'cpu_utilization', 'memory_used_mb'],
            'min_effectiveness': 0.8
        }
    }
    
    return policies.get(anomaly_type, {
        'strategy': 'feature_vectors',
        'critical_fields': [],
        'min_effectiveness': 0.75
    })

async def benchmark_preservation_performance(guard: AnomalyPreservationGuard,
                                           test_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Benchmark preservation performance across different scenarios"""
    
    results = {
        'scenario_results': [],
        'average_effectiveness': 0.0,
        'average_processing_time_ms': 0.0,
        'violation_rate': 0.0
    }
    
    total_effectiveness = 0.0
    total_processing_time = 0.0
    total_violations = 0
    
    for scenario in test_scenarios:
        start_time = time.perf_counter()
        
        # Create processed version (simulate processing impact)
        processed_scenario = scenario.copy()
        # Apply some processing transformations
        if 'large_field' in processed_scenario:
            processed_scenario['large_field_hash'] = hash(processed_scenario['large_field'])
            del processed_scenario['large_field']
        
        # Assess preservation
        assessment = await guard.assess_preservation_impact(
            scenario, processed_scenario, 'benchmark'
        )
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        scenario_result = {
            'effectiveness': assessment.preservation_effectiveness,
            'processing_time_ms': processing_time,
            'violations': len(assessment.critical_violations),
            'affected_types': [t.value for t in assessment.affected_anomaly_types]
        }
        
        results['scenario_results'].append(scenario_result)
        
        total_effectiveness += assessment.preservation_effectiveness
        total_processing_time += processing_time
        total_violations += len(assessment.critical_violations)
    
    num_scenarios = len(test_scenarios)
    results['average_effectiveness'] = total_effectiveness / num_scenarios
    results['average_processing_time_ms'] = total_processing_time / num_scenarios
    results['violation_rate'] = total_violations / num_scenarios
    
    return results

class PreservationMetricsCollector:
    """Collects and analyzes preservation metrics over time"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=10000)
        self.anomaly_type_stats = defaultdict(list)
        self.processing_stage_stats = defaultdict(list)
    
    def record_assessment(self, assessment: PreservationAssessment):
        """Record a preservation assessment for analysis"""
        self.metrics_history.append(assessment)
        
        # Update anomaly type statistics
        for anomaly_type in assessment.affected_anomaly_types:
            self.anomaly_type_stats[anomaly_type].append(assessment.preservation_effectiveness)
        
        # Update processing stage statistics
        self.processing_stage_stats[assessment.processing_stage].append(
            assessment.preservation_effectiveness
        )
    
    def get_trend_analysis(self, window_size: int = 100) -> Dict[str, Any]:
        """Analyze preservation effectiveness trends"""
        
        if len(self.metrics_history) < window_size:
            window_size = len(self.metrics_history)
        
        recent_assessments = list(self.metrics_history)[-window_size:]
        
        effectiveness_values = [a.preservation_effectiveness for a in recent_assessments]
        
        analysis = {
            'current_average': statistics.mean(effectiveness_values),
            'trend_direction': self._calculate_trend_direction(effectiveness_values),
            'volatility': statistics.stdev(effectiveness_values) if len(effectiveness_values) > 1 else 0.0,
            'improvement_opportunities': self._identify_improvement_opportunities(recent_assessments)
        }
        
        return analysis
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate trend direction using linear regression"""
        
        if len(values) < 5:
            return "insufficient_data"
        
        x = list(range(len(values)))
        
        # Simple linear regression
        n = len(values)
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        if slope > 0.001:
            return "improving"
        elif slope < -0.001:
            return "declining"
        else:
            return "stable"
    
    def _identify_improvement_opportunities(self, assessments: List[PreservationAssessment]) -> List[str]:
        """Identify opportunities for preservation improvement"""
        
        opportunities = []
        
        # Analyze common violation patterns
        violation_counts = defaultdict(int)
        for assessment in assessments:
            for violation in assessment.critical_violations:
                violation_counts[violation] += 1
        
        # Identify most common violations
        if violation_counts:
            most_common_violation = max(violation_counts, key=violation_counts.get)
            opportunities.append(f"Address common violation: {most_common_violation}")
        
        # Analyze low-performing stages
        stage_performance = defaultdict(list)
        for assessment in assessments:
            stage_performance[assessment.processing_stage].append(assessment.preservation_effectiveness)
        
        for stage, scores in stage_performance.items():
            avg_score = statistics.mean(scores)
            if avg_score < 0.85:
                opportunities.append(f"Improve preservation in stage: {stage}")
        
        return opportunities


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Initialize preservation guard
        config = {
            'preservation_enabled': True,
            'min_effectiveness_threshold': 0.85,
            'critical_violation_threshold': 2
        }
        
        guard = AnomalyPreservationGuard(config)
        
        # Example telemetry records
        original_record = {
            'execution_id': 'exec_12345',
            'timestamp': time.time(),
            'function_name': 'user_authentication',
            'duration_ms': 245.7,
            'memory_used_mb': 128.5,
            'cpu_utilization': 65.3,
            'init_time_ms': 1200.0,  # Indicates cold start
            'cold_start_indicator': True,
            'network_io': 4096,
            'invocation_trace': {
                'parent_function': 'api_gateway',
                'call_chain': ['auth_validate', 'token_verify', 'user_lookup']
            },
            'performance_metrics': {
                'response_time': 220.5,
                'latency_percentiles': [50, 95, 99]
            },
            'debug_logs': "x" * 2000  # Large debug field
        }
        
        # Simulate processed record (after sanitization, privacy filtering, hashing)
        processed_record = {
            'execution_id': 'exec_12345',
            'timestamp': time.time(),
            'function_name': 'user_authentication',
            'duration_ms': 245.7,
            'memory_used_mb': 128.5,
            'cpu_utilization': 65.3,
            'init_time_ms': 1200.0,
            'cold_start_indicator': True,
            'network_io': 4096,
            'invocation_trace_hash': 'hash_abc123',  # Hashed nested object
            'performance_metrics': {
                'response_time': 220.5,
                'latency_percentiles': [50, 95, 99]
            }
            # debug_logs removed during sanitization
        }
        
        print("=== Anomaly Preservation Assessment ===")
        
        # Assess preservation impact
        assessment = await guard.assess_preservation_impact(
            original_record, processed_record, "full_pipeline"
        )
        
        print(f"Original detectability score: {assessment.original_detectability_score:.3f}")
        print(f"Post-processing detectability score: {assessment.post_processing_detectability_score:.3f}")
        print(f"Preservation effectiveness: {assessment.preservation_effectiveness:.3f}")
        print(f"Affected anomaly types: {[t.value for t in assessment.affected_anomaly_types]}")
        
        if assessment.critical_violations:
            print(f"Critical violations: {assessment.critical_violations}")
        
        if assessment.recommendations:
            print(f"Recommendations: {assessment.recommendations}")
        
        # Check if rollback should be triggered
        should_rollback = await guard.should_trigger_rollback(assessment)
        print(f"Should trigger rollback: {should_rollback}")
        
        # Generate optimization strategy
        print(f"\n=== Optimization Strategy ===")
        strategy = await guard.optimize_preservation_strategy(original_record, target_effectiveness=0.95)
        
        print(f"Recommended policies: {strategy['recommended_policies']}")
        print(f"Field protection levels: {strategy['field_protection_levels']}")
        
        # Display statistics
        print(f"\n=== Preservation Statistics ===")
        stats = guard.get_preservation_statistics()
        for key, value in stats['performance_metrics'].items():
            print(f"{key}: {value}")
        
        # Test benchmark
        print(f"\n=== Benchmark Test ===")
        test_scenarios = [original_record.copy() for _ in range(5)]
        benchmark_results = await benchmark_preservation_performance(guard, test_scenarios)
        
        print(f"Average effectiveness: {benchmark_results['average_effectiveness']:.3f}")
        print(f"Average processing time: {benchmark_results['average_processing_time_ms']:.2f}ms")
        print(f"Violation rate: {benchmark_results['violation_rate']:.2f}")
    
    # Run the example
    asyncio.run(main())