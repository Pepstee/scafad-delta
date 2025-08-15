#!/usr/bin/env python3
"""
SCAFAD Layer 1: Behavioral Intake Zone - Core Orchestrator
=========================================================

Layer 1 serves as the behavioral intake and data conditioning layer that:
- Sanitizes and normalizes telemetry data while preserving anomaly semantics
- Manages schema evolution and backward compatibility
- Applies privacy compliance filters (GDPR/CCPA/HIPAA)
- Optimizes payloads through deferred hashing
- Ensures 99.5%+ anomaly preservation rate

Author: SCAFAD Research Team
Institution: Birmingham Newman University
License: MIT
Version: 1.0.0
"""

import asyncio
import time
import json
import hashlib
import logging
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum, auto
from datetime import datetime, timezone
import traceback

# Layer 1 Core Imports
from layer1_validation import InputValidationGateway, ValidationResult, ValidationLevel
from layer1_schema import SchemaEvolutionEngine, SchemaMetadata, MigrationResult
from layer1_sanitization import SanitizationProcessor, SanitizationResult
from layer1_privacy import PrivacyComplianceFilter, PrivacyAuditTrail, RedactionResult
from layer1_hashing import DeferredHashingManager, HashedRecord, HashRegistry
from layer1_preservation import AnomalyPreservationGuard, PreservationReport

# Supporting Subsystems
from subsystems.schema_registry import SchemaRegistry, SchemaVersion
from subsystems.privacy_policy_engine import PrivacyPolicyEngine, PrivacyPolicy
from subsystems.semantic_analyzer import SemanticAnalyzer, BehavioralFeatures
from subsystems.quality_monitor import QualityAssuranceMonitor, QualityMetrics
from subsystems.audit_trail_generator import AuditTrailGenerator, ProcessingAudit

# Utility Services
from utils.hash_library import HashFunction, CryptographicHasher
from utils.redaction_manager import RedactionPolicyManager, RedactionPolicy
from utils.field_mapper import FieldMappingEngine, FieldMapping
from utils.compression_optimizer import CompressionOptimizer, CompressionResult
from utils.validators import validate_telemetry_record, TelemetryRecordValidator

# Configuration and Data Models
from config.layer1_config import Layer1Config, ProcessingMode, PerformanceProfile


# =============================================================================
# Core Data Models and Enums
# =============================================================================

class PrivacyLevel(Enum):
    """Privacy protection levels"""
    MINIMAL = "minimal"       # Basic PII redaction
    MODERATE = "moderate"     # Standard privacy filters
    HIGH = "high"            # Aggressive privacy protection
    MAXIMUM = "maximum"      # Maximum privacy with minimal data retention

class PreservationMode(Enum):
    """Anomaly preservation modes"""
    CONSERVATIVE = "conservative"  # Preserve only critical anomaly features
    BALANCED = "balanced"         # Balance preservation with performance
    AGGRESSIVE = "aggressive"     # Maximum anomaly feature preservation

class ProcessingPhase(Enum):
    """Layer 1 processing phases"""
    VALIDATION = auto()
    SCHEMA_EVOLUTION = auto()
    SANITIZATION = auto()
    PRIVACY_FILTERING = auto()
    DEFERRED_HASHING = auto()
    PRESERVATION_VALIDATION = auto()
    QUALITY_ASSURANCE = auto()
    AUDIT_GENERATION = auto()

class ProcessingStatus(Enum):
    """Processing status codes"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    VALIDATION_FAILED = "validation_failed"
    SCHEMA_MIGRATION_FAILED = "schema_migration_failed"
    PRIVACY_VIOLATION = "privacy_violation"
    PRESERVATION_FAILED = "preservation_failed"
    SYSTEM_ERROR = "system_error"


@dataclass
class TelemetryRecord:
    """
    Standardized telemetry record structure from Layer 0
    """
    record_id: str
    timestamp: float
    function_name: str
    execution_phase: str
    anomaly_type: str
    telemetry_data: Dict[str, Any]
    provenance_chain: Optional[Dict[str, Any]] = None
    context_metadata: Optional[Dict[str, Any]] = None
    schema_version: str = "v2.1"
    
    def __post_init__(self):
        if not self.record_id:
            self.record_id = str(uuid.uuid4())


@dataclass
class CleanTelemetryRecord:
    """
    Processed telemetry record ready for Layer 2
    """
    record_id: str
    original_record_id: str
    timestamp: float
    function_name: str
    execution_phase: str
    anomaly_type: str
    cleaned_data: Dict[str, Any]
    schema_metadata: SchemaMetadata
    privacy_compliance: Dict[str, Any]
    hash_mappings: Optional[Dict[str, str]] = None
    preservation_score: float = 0.0
    quality_score: float = 0.0
    processing_latency_ms: float = 0.0


@dataclass
class ProcessedBatch:
    """
    Complete processed batch output for Layer 2
    """
    batch_id: str
    cleaned_records: List[CleanTelemetryRecord]
    schema_metadata: SchemaMetadata
    privacy_audit_trail: PrivacyAuditTrail
    preservation_report: PreservationReport
    quality_metrics: QualityMetrics
    processing_summary: Dict[str, Any]
    audit_trail: ProcessingAudit
    total_processing_time_ms: float = 0.0


@dataclass
class Layer1Config:
    """
    Layer 1 configuration with performance and compliance settings
    """
    # Schema Configuration
    schema_version: str = "v2.1"
    enable_schema_migration: bool = True
    backward_compatibility_mode: bool = True
    
    # Privacy Configuration
    privacy_level: PrivacyLevel = PrivacyLevel.MODERATE
    enable_gdpr_compliance: bool = True
    enable_ccpa_compliance: bool = True
    enable_hipaa_compliance: bool = False
    
    # Processing Configuration
    processing_mode: ProcessingMode = ProcessingMode.PRODUCTION
    performance_profile: PerformanceProfile = PerformanceProfile.BALANCED
    anomaly_preservation_mode: PreservationMode = PreservationMode.AGGRESSIVE
    
    # Performance Targets
    max_processing_latency_ms: int = 2
    target_throughput_records_per_sec: int = 10000
    max_memory_overhead_mb: int = 32
    min_anomaly_preservation_rate: float = 0.995
    
    # Hash Configuration
    hash_algorithms: List[str] = field(default_factory=lambda: ["sha256", "blake2b"])
    enable_deferred_hashing: bool = True
    hash_threshold_bytes: int = 1024
    
    # Quality Assurance
    enable_quality_monitoring: bool = True
    quality_threshold: float = 0.95
    enable_real_time_validation: bool = True
    
    # Audit and Compliance
    generate_audit_trails: bool = True
    audit_detail_level: str = "comprehensive"
    compliance_validation_mode: str = "strict"
    
    # Development and Testing
    enable_debug_mode: bool = False
    enable_performance_profiling: bool = False
    test_mode: bool = False


# =============================================================================
# Layer 1 Main Orchestrator
# =============================================================================

class Layer1_BehavioralIntakeZone:
    """
    Main Layer 1 orchestrator - coordinates all behavioral intake and data conditioning
    
    This is the central coordinator that manages the 6-phase processing pipeline:
    1. Input Validation Gateway
    2. Schema Evolution Engine  
    3. Sanitization Processor
    4. Privacy Compliance Filter
    5. Deferred Hashing Manager
    6. Anomaly Preservation Guard
    
    Performance Targets:
    - Processing latency: <2ms per record
    - Throughput: 10,000+ records/sec
    - Anomaly preservation: 99.5%+
    - Privacy compliance: 100%
    """
    
    def __init__(self, config: Layer1Config = None):
        """Initialize Layer 1 with configuration and all processing components"""
        self.config = config or Layer1Config()
        self._setup_logging()
        
        # Initialize core processing components
        self._initialize_core_components()
        
        # Initialize supporting subsystems
        self._initialize_subsystems()
        
        # Initialize utility services
        self._initialize_utilities()
        
        # Performance and monitoring
        self._initialize_monitoring()
        
        # Processing state
        self.processing_state = {
            'total_records_processed': 0,
            'successful_records': 0,
            'failed_records': 0,
            'total_processing_time_ms': 0.0,
            'average_latency_ms': 0.0,
            'anomaly_preservation_rate': 0.0,
            'privacy_compliance_rate': 1.0
        }
        
        self.logger.info(f"Layer 1 Behavioral Intake Zone initialized with config: {self.config.processing_mode}")
    
    def _setup_logging(self):
        """Setup comprehensive logging for Layer 1"""
        self.logger = logging.getLogger("SCAFAD.Layer1")
        if self.config.enable_debug_mode:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
    
    def _initialize_core_components(self):
        """Initialize the 6 core processing components"""
        self.logger.info("Initializing Layer 1 core processing components...")
        
        # Phase 1: Input Validation Gateway
        self.validator = InputValidationGateway(self.config)
        
        # Phase 2: Schema Evolution Engine
        self.schema_engine = SchemaEvolutionEngine(self.config)
        
        # Phase 3: Sanitization Processor
        self.sanitizer = SanitizationProcessor(self.config)
        
        # Phase 4: Privacy Compliance Filter
        self.privacy_filter = PrivacyComplianceFilter(self.config)
        
        # Phase 5: Deferred Hashing Manager
        self.hash_manager = DeferredHashingManager(self.config)
        
        # Phase 6: Anomaly Preservation Guard
        self.preservation_guard = AnomalyPreservationGuard(self.config)
        
        self.logger.info("Core processing components initialized successfully")
    
    def _initialize_subsystems(self):
        """Initialize supporting subsystems"""
        self.logger.info("Initializing Layer 1 supporting subsystems...")
        
        # Schema Registry & Versioning
        self.schema_registry = SchemaRegistry(self.config)
        
        # Privacy Policy Engine
        self.privacy_policy_engine = PrivacyPolicyEngine(self.config)
        
        # Semantic Analyzer
        self.semantic_analyzer = SemanticAnalyzer(self.config)
        
        # Quality Assurance Monitor
        self.quality_monitor = QualityAssuranceMonitor(self.config)
        
        # Audit Trail Generator
        self.audit_generator = AuditTrailGenerator(self.config)
        
        self.logger.info("Supporting subsystems initialized successfully")
    
    def _initialize_utilities(self):
        """Initialize utility services"""
        self.logger.info("Initializing Layer 1 utility services...")
        
        # Hash Function Library
        self.hash_library = CryptographicHasher(self.config.hash_algorithms)
        
        # Redaction Policy Manager
        self.redaction_manager = RedactionPolicyManager(self.config)
        
        # Field Mapping Engine
        self.field_mapper = FieldMappingEngine(self.config)
        
        # Compression Optimizer
        self.compression_optimizer = CompressionOptimizer(self.config)
        
        # Telemetry Record Validator
        self.record_validator = TelemetryRecordValidator(self.config)
        
        self.logger.info("Utility services initialized successfully")
    
    def _initialize_monitoring(self):
        """Initialize performance monitoring and metrics collection"""
        self.performance_metrics = {
            'phase_latencies': {phase.name: [] for phase in ProcessingPhase},
            'throughput_samples': [],
            'memory_usage_samples': [],
            'anomaly_preservation_samples': [],
            'error_counts': {},
            'start_time': time.time()
        }
    
    # =========================================================================
    # Main Processing Pipeline
    # =========================================================================
    
    async def process_telemetry_batch(self, 
                                    telemetry_records: List[TelemetryRecord],
                                    processing_context: Optional[Dict[str, Any]] = None) -> ProcessedBatch:
        """
        Main processing pipeline: Layer 0 telemetry → sanitized, schema-compliant data → Layer 2
        
        Args:
            telemetry_records: Raw telemetry records from Layer 0
            processing_context: Optional context for batch processing
            
        Returns:
            ProcessedBatch: Complete batch ready for Layer 2 consumption
        """
        batch_start_time = time.time()
        batch_id = str(uuid.uuid4())
        
        self.logger.info(f"Starting Layer 1 processing for batch {batch_id} with {len(telemetry_records)} records")
        
        try:
            # Initialize batch processing state
            processing_summary = {
                'batch_id': batch_id,
                'total_records': len(telemetry_records),
                'start_time': batch_start_time,
                'phases_completed': [],
                'errors': []
            }
            
            # Phase 1: Input Validation Gateway
            validated_records = await self._phase_1_validation(telemetry_records, processing_summary)
            
            # Phase 2: Schema Evolution Engine
            migrated_records = await self._phase_2_schema_evolution(validated_records, processing_summary)
            
            # Phase 3: Sanitization Processor
            sanitized_records = await self._phase_3_sanitization(migrated_records, processing_summary)
            
            # Phase 4: Privacy Compliance Filter
            privacy_filtered_records, privacy_audit = await self._phase_4_privacy_filtering(
                sanitized_records, processing_summary
            )
            
            # Phase 5: Deferred Hashing Manager
            hashed_records = await self._phase_5_deferred_hashing(
                privacy_filtered_records, processing_summary
            )
            
            # Phase 6: Anomaly Preservation Guard
            final_records, preservation_report = await self._phase_6_preservation_validation(
                hashed_records, telemetry_records, processing_summary
            )
            
            # Phase 7: Quality Assurance and Metrics
            quality_metrics = await self._phase_7_quality_assurance(final_records, processing_summary)
            
            # Phase 8: Audit Trail Generation
            audit_trail = await self._phase_8_audit_generation(processing_summary)
            
            # Assemble final processed batch
            total_processing_time = (time.time() - batch_start_time) * 1000  # Convert to ms
            
            processed_batch = ProcessedBatch(
                batch_id=batch_id,
                cleaned_records=final_records,
                schema_metadata=self.schema_engine.get_current_metadata(),
                privacy_audit_trail=privacy_audit,
                preservation_report=preservation_report,
                quality_metrics=quality_metrics,
                processing_summary=processing_summary,
                audit_trail=audit_trail,
                total_processing_time_ms=total_processing_time
            )
            
            # Update processing statistics
            self._update_processing_statistics(processed_batch)
            
            self.logger.info(
                f"Layer 1 batch {batch_id} processed successfully: "
                f"{len(final_records)} records in {total_processing_time:.2f}ms "
                f"(avg {total_processing_time/len(final_records):.2f}ms/record)"
            )
            
            return processed_batch
            
        except Exception as e:
            self.logger.error(f"Layer 1 batch {batch_id} processing failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # Return partial results with error information
            return await self._handle_processing_error(
                batch_id, telemetry_records, processing_summary, e, batch_start_time
            )
    
    # =========================================================================
    # Processing Phase Implementations
    # =========================================================================
    
    async def _phase_1_validation(self, 
                                records: List[TelemetryRecord],
                                processing_summary: Dict[str, Any]) -> List[TelemetryRecord]:
        """Phase 1: Input Validation Gateway"""
        phase_start = time.time()
        
        self.logger.debug("Phase 1: Input Validation Gateway")
        
        validated_records = []
        validation_errors = []
        
        for record in records:
            try:
                # Validate record structure and content
                validation_result = await self.validator.validate_telemetry_record(record)
                
                if validation_result.is_valid:
                    # Apply input sanitization
                    sanitized_record = await self.validator.sanitize_malformed_fields(record)
                    validated_records.append(sanitized_record)
                else:
                    validation_errors.append({
                        'record_id': record.record_id,
                        'errors': validation_result.errors
                    })
                    self.logger.warning(f"Record {record.record_id} failed validation: {validation_result.errors}")
                    
            except Exception as e:
                validation_errors.append({
                    'record_id': record.record_id,
                    'error': str(e)
                })
                self.logger.error(f"Validation error for record {record.record_id}: {str(e)}")
        
        phase_time = (time.time() - phase_start) * 1000
        self.performance_metrics['phase_latencies']['VALIDATION'].append(phase_time)
        
        processing_summary['phases_completed'].append({
            'phase': 'VALIDATION',
            'duration_ms': phase_time,
            'records_processed': len(validated_records),
            'errors': validation_errors
        })
        
        self.logger.debug(f"Phase 1 completed: {len(validated_records)} records validated in {phase_time:.2f}ms")
        return validated_records
    
    async def _phase_2_schema_evolution(self, 
                                      records: List[TelemetryRecord],
                                      processing_summary: Dict[str, Any]) -> List[TelemetryRecord]:
        """Phase 2: Schema Evolution Engine"""
        phase_start = time.time()
        
        self.logger.debug("Phase 2: Schema Evolution Engine")
        
        migrated_records = []
        migration_errors = []
        
        for record in records:
            try:
                # Check if migration is needed
                if record.schema_version != self.config.schema_version:
                    migration_result = await self.schema_engine.migrate_record_to_current_schema(record)
                    
                    if migration_result.success:
                        migrated_records.append(migration_result.migrated_record)
                    else:
                        migration_errors.append({
                            'record_id': record.record_id,
                            'from_version': record.schema_version,
                            'to_version': self.config.schema_version,
                            'error': migration_result.error_message
                        })
                else:
                    # No migration needed
                    migrated_records.append(record)
                    
            except Exception as e:
                migration_errors.append({
                    'record_id': record.record_id,
                    'error': str(e)
                })
                self.logger.error(f"Schema migration error for record {record.record_id}: {str(e)}")
        
        phase_time = (time.time() - phase_start) * 1000
        self.performance_metrics['phase_latencies']['SCHEMA_EVOLUTION'].append(phase_time)
        
        processing_summary['phases_completed'].append({
            'phase': 'SCHEMA_EVOLUTION',
            'duration_ms': phase_time,
            'records_processed': len(migrated_records),
            'errors': migration_errors
        })
        
        self.logger.debug(f"Phase 2 completed: {len(migrated_records)} records migrated in {phase_time:.2f}ms")
        return migrated_records
    
    async def _phase_3_sanitization(self, 
                                  records: List[TelemetryRecord],
                                  processing_summary: Dict[str, Any]) -> List[TelemetryRecord]:
        """Phase 3: Sanitization Processor"""
        phase_start = time.time()
        
        self.logger.debug("Phase 3: Sanitization Processor")
        
        sanitized_records = []
        sanitization_errors = []
        
        for record in records:
            try:
                # Apply comprehensive sanitization while preserving anomaly semantics
                sanitization_result = await self.sanitizer.sanitize_telemetry_record(record)
                
                if sanitization_result.success:
                    # Verify anomaly preservation during sanitization
                    preservation_check = await self.preservation_guard.analyze_anomaly_risk_before_transform(record)
                    
                    if preservation_check.anomaly_preservation_score >= self.config.min_anomaly_preservation_rate:
                        sanitized_records.append(sanitization_result.sanitized_record)
                    else:
                        # Apply semantic-preserving sanitization
                        preserved_record = await self.preservation_guard.apply_semantic_preserving_transforms(record)
                        sanitized_records.append(preserved_record)
                else:
                    sanitization_errors.append({
                        'record_id': record.record_id,
                        'error': sanitization_result.error_message
                    })
                    
            except Exception as e:
                sanitization_errors.append({
                    'record_id': record.record_id,
                    'error': str(e)
                })
                self.logger.error(f"Sanitization error for record {record.record_id}: {str(e)}")
        
        phase_time = (time.time() - phase_start) * 1000
        self.performance_metrics['phase_latencies']['SANITIZATION'].append(phase_time)
        
        processing_summary['phases_completed'].append({
            'phase': 'SANITIZATION',
            'duration_ms': phase_time,
            'records_processed': len(sanitized_records),
            'errors': sanitization_errors
        })
        
        self.logger.debug(f"Phase 3 completed: {len(sanitized_records)} records sanitized in {phase_time:.2f}ms")
        return sanitized_records
    
    async def _phase_4_privacy_filtering(self, 
                                       records: List[TelemetryRecord],
                                       processing_summary: Dict[str, Any]) -> Tuple[List[TelemetryRecord], PrivacyAuditTrail]:
        """Phase 4: Privacy Compliance Filter"""
        phase_start = time.time()
        
        self.logger.debug("Phase 4: Privacy Compliance Filter")
        
        filtered_records = []
        privacy_errors = []
        privacy_actions = []
        
        for record in records:
            try:
                # Apply privacy compliance based on configured level
                if self.config.enable_gdpr_compliance:
                    gdpr_result = await self.privacy_filter.apply_gdpr_filters(record)
                    record = gdpr_result.filtered_record
                    privacy_actions.extend(gdpr_result.actions_taken)
                
                if self.config.enable_ccpa_compliance:
                    ccpa_result = await self.privacy_filter.apply_ccpa_filters(record)
                    record = ccpa_result.filtered_record
                    privacy_actions.extend(ccpa_result.actions_taken)
                
                if self.config.enable_hipaa_compliance:
                    hipaa_result = await self.privacy_filter.apply_hipaa_filters(record)
                    record = hipaa_result.filtered_record
                    privacy_actions.extend(hipaa_result.actions_taken)
                
                # Apply PII redaction
                redaction_result = await self.privacy_filter.redact_pii_fields(record)
                if redaction_result.success:
                    filtered_records.append(redaction_result.redacted_record)
                    privacy_actions.extend(redaction_result.redaction_actions)
                else:
                    privacy_errors.append({
                        'record_id': record.record_id,
                        'error': redaction_result.error_message
                    })
                    
            except Exception as e:
                privacy_errors.append({
                    'record_id': record.record_id,
                    'error': str(e)
                })
                self.logger.error(f"Privacy filtering error for record {record.record_id}: {str(e)}")
        
        # Generate privacy audit trail
        privacy_audit_trail = PrivacyAuditTrail(
            batch_id=processing_summary['batch_id'],
            records_processed=len(filtered_records),
            privacy_level=self.config.privacy_level,
            actions_taken=privacy_actions,
            compliance_status={
                'gdpr': self.config.enable_gdpr_compliance,
                'ccpa': self.config.enable_ccpa_compliance,
                'hipaa': self.config.enable_hipaa_compliance
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        phase_time = (time.time() - phase_start) * 1000
        self.performance_metrics['phase_latencies']['PRIVACY_FILTERING'].append(phase_time)
        
        processing_summary['phases_completed'].append({
            'phase': 'PRIVACY_FILTERING',
            'duration_ms': phase_time,
            'records_processed': len(filtered_records),
            'errors': privacy_errors,
            'privacy_actions': len(privacy_actions)
        })
        
        self.logger.debug(f"Phase 4 completed: {len(filtered_records)} records privacy-filtered in {phase_time:.2f}ms")
        return filtered_records, privacy_audit_trail
    
    async def _phase_5_deferred_hashing(self, 
                                      records: List[TelemetryRecord],
                                      processing_summary: Dict[str, Any]) -> List[TelemetryRecord]:
        """Phase 5: Deferred Hashing Manager"""
        phase_start = time.time()
        
        self.logger.debug("Phase 5: Deferred Hashing Manager")
        
        hashed_records = []
        hashing_errors = []
        
        if self.config.enable_deferred_hashing:
            for record in records:
                try:
                    # Identify fields suitable for hashing
                    hashable_fields = await self.hash_manager.identify_hashable_fields(record)
                    
                    if hashable_fields:
                        # Apply deferred hashing to optimize payload size
                        hashing_result = await self.hash_manager.apply_deferred_hashing(record)
                        
                        if hashing_result.success:
                            hashed_records.append(hashing_result.hashed_record.to_telemetry_record())
                        else:
                            hashing_errors.append({
                                'record_id': record.record_id,
                                'error': hashing_result.error_message
                            })
                            # Fall back to original record
                            hashed_records.append(record)
                    else:
                        # No hashing needed
                        hashed_records.append(record)
                        
                except Exception as e:
                    hashing_errors.append({
                        'record_id': record.record_id,
                        'error': str(e)
                    })
                    self.logger.error(f"Hashing error for record {record.record_id}: {str(e)}")
                    # Fall back to original record
                    hashed_records.append(record)
        else:
            # Deferred hashing disabled, pass through
            hashed_records = records
        
        phase_time = (time.time() - phase_start) * 1000
        self.performance_metrics['phase_latencies']['DEFERRED_HASHING'].append(phase_time)
        
        processing_summary['phases_completed'].append({
            'phase': 'DEFERRED_HASHING',
            'duration_ms': phase_time,
            'records_processed': len(hashed_records),
            'errors': hashing_errors,
            'hashing_enabled': self.config.enable_deferred_hashing
        })
        
        self.logger.debug(f"Phase 5 completed: {len(hashed_records)} records hashed in {phase_time:.2f}ms")
        return hashed_records
    
    async def _phase_6_preservation_validation(self, 
                                             final_records: List[TelemetryRecord],
                                             original_records: List[TelemetryRecord],
                                             processing_summary: Dict[str, Any]) -> Tuple[List[CleanTelemetryRecord], PreservationReport]:
        """Phase 6: Anomaly Preservation Guard"""
        phase_start = time.time()
        
        self.logger.debug("Phase 6: Anomaly Preservation Guard")
        
        clean_records = []
        preservation_errors = []
        preservation_scores = []
        
        # Create mapping of original to final records
        record_mapping = {r.record_id: r for r in original_records}
        
        for final_record in final_records:
            try:
                original_record = record_mapping.get(final_record.record_id)
                if not original_record:
                    preservation_errors.append({
                        'record_id': final_record.record_id,
                        'error': 'Original record not found for preservation validation'
                    })
                    continue
                
                # Validate post-processing anomaly detectability
                preservation_result = await self.preservation_guard.validate_post_processing_detectability(
                    original_record, final_record
                )
                
                preservation_scores.append(preservation_result.preservation_score)
                
                # Create clean telemetry record
                clean_record = CleanTelemetryRecord(
                    record_id=str(uuid.uuid4()),
                    original_record_id=final_record.record_id,
                    timestamp=final_record.timestamp,
                    function_name=final_record.function_name,
                    execution_phase=final_record.execution_phase,
                    anomaly_type=final_record.anomaly_type,
                    cleaned_data=final_record.telemetry_data,
                    schema_metadata=self.schema_engine.get_current_metadata(),
                    privacy_compliance={
                        'privacy_level': self.config.privacy_level.value,
                        'compliant': True
                    },
                    preservation_score=preservation_result.preservation_score,
                    quality_score=0.0,  # Will be set in quality assurance phase
                    processing_latency_ms=0.0  # Will be calculated per record
                )
                
                clean_records.append(clean_record)
                
            except Exception as e:
                preservation_errors.append({
                    'record_id': final_record.record_id,
                    'error': str(e)
                })
                self.logger.error(f"Preservation validation error for record {final_record.record_id}: {str(e)}")
        
        # Generate preservation report
        avg_preservation_score = sum(preservation_scores) / len(preservation_scores) if preservation_scores else 0.0
        
        preservation_report = PreservationReport(
            batch_id=processing_summary['batch_id'],
            records_analyzed=len(clean_records),
            preservation_rate=avg_preservation_score,
            preservation_mode=self.config.anomaly_preservation_mode,
            critical_features_preserved=avg_preservation_score >= self.config.min_anomaly_preservation_rate,
            preservation_details={
                'min_score': min(preservation_scores) if preservation_scores else 0.0,
                'max_score': max(preservation_scores) if preservation_scores else 0.0,
                'std_deviation': self._calculate_std_deviation(preservation_scores)
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        phase_time = (time.time() - phase_start) * 1000
        self.performance_metrics['phase_latencies']['PRESERVATION_VALIDATION'].append(phase_time)
        
        processing_summary['phases_completed'].append({
            'phase': 'PRESERVATION_VALIDATION',
            'duration_ms': phase_time,
            'records_processed': len(clean_records),
            'errors': preservation_errors,
            'avg_preservation_score': avg_preservation_score
        })
        
        self.logger.debug(
            f"Phase 6 completed: {len(clean_records)} records validated with "
            f"{avg_preservation_score:.1%} preservation rate in {phase_time:.2f}ms"
        )
        return clean_records, preservation_report
    
    async def _phase_7_quality_assurance(self, 
                                       records: List[CleanTelemetryRecord],
                                       processing_summary: Dict[str, Any]) -> QualityMetrics:
        """Phase 7: Quality Assurance and Metrics"""
        phase_start = time.time()
        
        self.logger.debug("Phase 7: Quality Assurance Monitor")
        
        # Assess batch quality
        quality_assessment = await self.quality_monitor.assess_batch_quality(records)
        
        # Update quality scores in records
        for i, record in enumerate(records):
            record.quality_score = quality_assessment.individual_scores[i] if i < len(quality_assessment.individual_scores) else 0.0
            record.processing_latency_ms = processing_summary.get('total_duration_ms', 0.0) / len(records)
        
        phase_time = (time.time() - phase_start) * 1000
        self.performance_metrics['phase_latencies']['QUALITY_ASSURANCE'].append(phase_time)
        
        processing_summary['phases_completed'].append({
            'phase': 'QUALITY_ASSURANCE',
            'duration_ms': phase_time,
            'records_assessed': len(records),
            'overall_quality_score': quality_assessment.overall_score
        })
        
        self.logger.debug(f"Phase 7 completed: Quality assessment in {phase_time:.2f}ms")
        return quality_assessment
    
    async def _phase_8_audit_generation(self, 
                                      processing_summary: Dict[str, Any]) -> ProcessingAudit:
        """Phase 8: Audit Trail Generation"""
        phase_start = time.time()
        
        self.logger.debug("Phase 8: Audit Trail Generation")
        
        # Generate comprehensive audit trail
        audit_trail = await self.audit_generator.generate_processing_audit(
            processing_summary,
            self.config,
            self.performance_metrics
        )
        
        phase_time = (time.time() - phase_start) * 1000
        
        processing_summary['phases_completed'].append({
            'phase': 'AUDIT_GENERATION',
            'duration_ms': phase_time,
            'audit_generated': True
        })
        
        self.logger.debug(f"Phase 8 completed: Audit trail generated in {phase_time:.2f}ms")
        return audit_trail
    
    # =========================================================================
    # Error Handling and Recovery
    # =========================================================================
    
    async def _handle_processing_error(self,
                                     batch_id: str,
                                     original_records: List[TelemetryRecord],
                                     processing_summary: Dict[str, Any],
                                     error: Exception,
                                     batch_start_time: float) -> ProcessedBatch:
        """Handle processing errors and return partial results"""
        
        self.logger.error(f"Handling processing error for batch {batch_id}: {str(error)}")
        
        # Create minimal clean records from original data
        fallback_records = []
        for record in original_records:
            fallback_record = CleanTelemetryRecord(
                record_id=str(uuid.uuid4()),
                original_record_id=record.record_id,
                timestamp=record.timestamp,
                function_name=record.function_name,
                execution_phase=record.execution_phase,
                anomaly_type=record.anomaly_type,
                cleaned_data=record.telemetry_data,  # Minimal processing
                schema_metadata=SchemaMetadata(
                    version=record.schema_version,
                    compatibility_level="unknown",
                    migration_applied=False
                ),
                privacy_compliance={'error': 'Processing failed'},
                preservation_score=1.0,  # Assume preserved since unprocessed
                quality_score=0.0,
                processing_latency_ms=0.0
            )
            fallback_records.append(fallback_record)
        
        # Create error audit trail
        error_audit_trail = ProcessingAudit(
            batch_id=batch_id,
            processing_status=ProcessingStatus.SYSTEM_ERROR,
            error_details=str(error),
            phases_completed=processing_summary.get('phases_completed', []),
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        total_processing_time = (time.time() - batch_start_time) * 1000
        
        return ProcessedBatch(
            batch_id=batch_id,
            cleaned_records=fallback_records,
            schema_metadata=SchemaMetadata(version="unknown", compatibility_level="error", migration_applied=False),
            privacy_audit_trail=PrivacyAuditTrail(
                batch_id=batch_id,
                records_processed=0,
                privacy_level=self.config.privacy_level,
                actions_taken=[],
                compliance_status={'error': str(error)},
                timestamp=datetime.now(timezone.utc).isoformat()
            ),
            preservation_report=PreservationReport(
                batch_id=batch_id,
                records_analyzed=0,
                preservation_rate=0.0,
                preservation_mode=self.config.anomaly_preservation_mode,
                critical_features_preserved=False,
                preservation_details={'error': str(error)},
                timestamp=datetime.now(timezone.utc).isoformat()
            ),
            quality_metrics=QualityMetrics(
                overall_score=0.0,
                individual_scores=[],
                error_message=str(error)
            ),
            processing_summary=processing_summary,
            audit_trail=error_audit_trail,
            total_processing_time_ms=total_processing_time
        )
    
    # =========================================================================
    # Performance Monitoring and Statistics
    # =========================================================================
    
    def _update_processing_statistics(self, processed_batch: ProcessedBatch):
        """Update internal processing statistics"""
        
        self.processing_state['total_records_processed'] += len(processed_batch.cleaned_records)
        self.processing_state['successful_records'] += len(processed_batch.cleaned_records)
        self.processing_state['total_processing_time_ms'] += processed_batch.total_processing_time_ms
        
        # Calculate running averages
        if self.processing_state['total_records_processed'] > 0:
            self.processing_state['average_latency_ms'] = (
                self.processing_state['total_processing_time_ms'] / 
                self.processing_state['total_records_processed']
            )
        
        # Update anomaly preservation rate
        if processed_batch.preservation_report.records_analyzed > 0:
            self.processing_state['anomaly_preservation_rate'] = processed_batch.preservation_report.preservation_rate
        
        # Log performance metrics
        if self.config.enable_performance_profiling:
            self.logger.info(
                f"Performance Update - Total Records: {self.processing_state['total_records_processed']}, "
                f"Avg Latency: {self.processing_state['average_latency_ms']:.2f}ms, "
                f"Preservation Rate: {self.processing_state['anomaly_preservation_rate']:.1%}"
            )
    
    def _calculate_std_deviation(self, values: List[float]) -> float:
        """Calculate standard deviation of a list of values"""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    # =========================================================================
    # Public Interface Methods
    # =========================================================================
    
    async def process_single_record(self, record: TelemetryRecord) -> CleanTelemetryRecord:
        """Process a single telemetry record (convenience method)"""
        
        batch_result = await self.process_telemetry_batch([record])
        
        if batch_result.cleaned_records:
            return batch_result.cleaned_records[0]
        else:
            raise RuntimeError(f"Failed to process record {record.record_id}")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        
        uptime = time.time() - self.performance_metrics['start_time']
        
        return {
            **self.processing_state,
            'uptime_seconds': uptime,
            'throughput_records_per_second': (
                self.processing_state['total_records_processed'] / uptime 
                if uptime > 0 else 0
            ),
            'phase_performance': {
                phase.name: {
                    'count': len(latencies),
                    'avg_ms': sum(latencies) / len(latencies) if latencies else 0,
                    'min_ms': min(latencies) if latencies else 0,
                    'max_ms': max(latencies) if latencies else 0
                }
                for phase, latencies in self.performance_metrics['phase_latencies'].items()
            }
        }
    
    def reset_statistics(self):
        """Reset processing statistics (for testing/benchmarking)"""
        
        self.processing_state = {
            'total_records_processed': 0,
            'successful_records': 0,
            'failed_records': 0,
            'total_processing_time_ms': 0.0,
            'average_latency_ms': 0.0,
            'anomaly_preservation_rate': 0.0,
            'privacy_compliance_rate': 1.0
        }
        
        self.performance_metrics = {
            'phase_latencies': {phase.name: [] for phase in ProcessingPhase},
            'throughput_samples': [],
            'memory_usage_samples': [],
            'anomaly_preservation_samples': [],
            'error_counts': {},
            'start_time': time.time()
        }
        
        self.logger.info("Layer 1 processing statistics reset")
    
    async def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration and return validation report"""
        
        validation_report = {
            'configuration_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Validate performance targets
        if self.config.max_processing_latency_ms < 1:
            validation_report['warnings'].append(
                "Very aggressive latency target (<1ms) may impact processing quality"
            )
        
        # Validate privacy configuration
        if (self.config.privacy_level == PrivacyLevel.MAXIMUM and 
            self.config.min_anomaly_preservation_rate > 0.9):
            validation_report['warnings'].append(
                "Maximum privacy with high preservation rate may be challenging to achieve"
            )
        
        # Validate schema configuration
        if not self.config.enable_schema_migration and self.config.backward_compatibility_mode:
            validation_report['errors'].append(
                "Cannot enable backward compatibility without schema migration"
            )
            validation_report['configuration_valid'] = False
        
        # Performance recommendations
        if self.config.enable_debug_mode and self.config.processing_mode == ProcessingMode.PRODUCTION:
            validation_report['recommendations'].append(
                "Consider disabling debug mode in production for better performance"
            )
        
        return validation_report
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of Layer 1 components"""
        
        health_status = {
            'overall_status': 'healthy',
            'components': {},
            'performance': {},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        try:
            # Test core components
            health_status['components']['validator'] = await self._health_check_component(self.validator)
            health_status['components']['schema_engine'] = await self._health_check_component(self.schema_engine)
            health_status['components']['sanitizer'] = await self._health_check_component(self.sanitizer)
            health_status['components']['privacy_filter'] = await self._health_check_component(self.privacy_filter)
            health_status['components']['hash_manager'] = await self._health_check_component(self.hash_manager)
            health_status['components']['preservation_guard'] = await self._health_check_component(self.preservation_guard)
            
            # Check performance metrics
            stats = self.get_processing_statistics()
            health_status['performance'] = {
                'average_latency_ms': stats['average_latency_ms'],
                'throughput_rps': stats['throughput_records_per_second'],
                'preservation_rate': stats['anomaly_preservation_rate'],
                'total_processed': stats['total_records_processed']
            }
            
            # Determine overall status
            component_statuses = [status['status'] for status in health_status['components'].values()]
            if any(status == 'error' for status in component_statuses):
                health_status['overall_status'] = 'error'
            elif any(status == 'warning' for status in component_statuses):
                health_status['overall_status'] = 'warning'
            
        except Exception as e:
            health_status['overall_status'] = 'error'
            health_status['error'] = str(e)
        
        return health_status
    
    async def _health_check_component(self, component) -> Dict[str, str]:
        """Health check for individual component"""
        
        try:
            if hasattr(component, 'health_check'):
                return await component.health_check()
            else:
                return {'status': 'healthy', 'message': 'Component operational'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}


# =============================================================================
# Layer 1 Factory and Utility Functions
# =============================================================================

def create_layer1_instance(config_dict: Optional[Dict[str, Any]] = None) -> Layer1_BehavioralIntakeZone:
    """
    Factory function to create Layer 1 instance with configuration
    
    Args:
        config_dict: Optional configuration dictionary
        
    Returns:
        Configured Layer 1 instance
    """
    
    if config_dict:
        config = Layer1Config(**config_dict)
    else:
        config = Layer1Config()
    
    return Layer1_BehavioralIntakeZone(config)


async def validate_layer1_deployment() -> Dict[str, Any]:
    """
    Validate Layer 1 deployment and dependencies
    
    Returns:
        Deployment validation report
    """
    
    validation_report = {
        'deployment_valid': True,
        'dependencies': {},
        'configuration': {},
        'performance': {},
        'errors': []
    }
    
    try:
        # Test Layer 1 instantiation
        layer1 = create_layer1_instance()
        
        # Validate configuration
        config_validation = await layer1.validate_configuration()
        validation_report['configuration'] = config_validation
        
        # Health check
        health_status = await layer1.health_check()
        validation_report['health'] = health_status
        
        # Test processing with minimal record
        test_record = TelemetryRecord(
            record_id="test-validation",
            timestamp=time.time(),
            function_name="test_function",
            execution_phase="execution",
            anomaly_type="benign",
            telemetry_data={"test": "data"}
        )
        
        start_time = time.time()
        result = await layer1.process_single_record(test_record)
        processing_time = (time.time() - start_time) * 1000
        
        validation_report['performance'] = {
            'test_processing_time_ms': processing_time,
            'preservation_score': result.preservation_score,
            'quality_score': result.quality_score
        }
        
        # Check if performance meets targets
        if processing_time > layer1.config.max_processing_latency_ms:
            validation_report['errors'].append(
                f"Processing latency {processing_time:.2f}ms exceeds target {layer1.config.max_processing_latency_ms}ms"
            )
            validation_report['deployment_valid'] = False
        
    except Exception as e:
        validation_report['deployment_valid'] = False
        validation_report['errors'].append(f"Deployment validation failed: {str(e)}")
    
    return validation_report


# =============================================================================
# Main Execution and Testing
# =============================================================================

async def main():
    """Main function for testing and demonstration"""
    
    print("SCAFAD Layer 1: Behavioral Intake Zone")
    print("=" * 50)
    
    # Create Layer 1 instance
    config = Layer1Config(
        privacy_level=PrivacyLevel.MODERATE,
        enable_debug_mode=True,
        test_mode=True
    )
    
    layer1 = Layer1_BehavioralIntakeZone(config)
    
    # Validate deployment
    print("Validating deployment...")
    deployment_status = await validate_layer1_deployment()
    print(f"Deployment Valid: {deployment_status['deployment_valid']}")
    
    if deployment_status['errors']:
        print("Errors:")
        for error in deployment_status['errors']:
            print(f"  - {error}")
    
    # Test processing
    print("\nTesting Layer 1 processing...")
    
    test_records = [
        TelemetryRecord(
            record_id=f"test-{i}",
            timestamp=time.time(),
            function_name=f"test_function_{i}",
            execution_phase="execution",
            anomaly_type="benign" if i % 2 == 0 else "suspicious",
            telemetry_data={
                "memory_usage_mb": 128 + i * 10,
                "execution_time_ms": 100 + i * 5,
                "cpu_utilization": 45.0 + i,
                "test_field": f"test_value_{i}"
            }
        )
        for i in range(5)
    ]
    
    start_time = time.time()
    processed_batch = await layer1.process_telemetry_batch(test_records)
    processing_time = time.time() - start_time
    
    print(f"Processed {len(processed_batch.cleaned_records)} records in {processing_time*1000:.2f}ms")
    print(f"Average latency: {processing_time*1000/len(test_records):.2f}ms per record")
    print(f"Preservation rate: {processed_batch.preservation_report.preservation_rate:.1%}")
    print(f"Quality score: {processed_batch.quality_metrics.overall_score:.2f}")
    
    # Display statistics
    print("\nProcessing Statistics:")
    stats = layer1.get_processing_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())