#!/usr/bin/env python3
"""
SCAFAD Layer 1: Configuration Management
=======================================

Configuration classes and enums for Layer 1 Behavioral Intake Zone.
Supports environment-specific settings, academic research requirements,
and production scalability.

Author: SCAFAD Research Team
Institution: Birmingham Newman University
License: MIT
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum, auto


# =============================================================================
# Configuration Enums
# =============================================================================

class ProcessingMode(Enum):
    """Layer 1 processing modes"""
    DEVELOPMENT = "development"     # Debug mode, comprehensive logging
    TESTING = "testing"            # Test mode, synthetic data support
    PRODUCTION = "production"      # Production mode, optimized performance
    RESEARCH = "research"          # Academic research mode, full telemetry

class PerformanceProfile(Enum):
    """Performance optimization profiles"""
    MINIMAL = "minimal"            # Minimal resource usage
    BALANCED = "balanced"          # Balance performance and resources
    PERFORMANCE = "performance"    # Maximum performance, higher resources
    ACADEMIC = "academic"          # Research-optimized, full metrics

class PrivacyLevel(Enum):
    """Privacy protection levels"""
    MINIMAL = "minimal"           # Basic PII redaction
    MODERATE = "moderate"         # Standard privacy filters
    HIGH = "high"                # Aggressive privacy protection
    MAXIMUM = "maximum"          # Maximum privacy with minimal data retention

class PreservationMode(Enum):
    """Anomaly preservation modes"""
    CONSERVATIVE = "conservative"  # Preserve only critical anomaly features
    BALANCED = "balanced"         # Balance preservation with performance
    AGGRESSIVE = "aggressive"     # Maximum anomaly feature preservation

class AuditLevel(Enum):
    """Audit trail detail levels"""
    MINIMAL = "minimal"           # Basic audit information
    STANDARD = "standard"         # Standard audit trails
    COMPREHENSIVE = "comprehensive"  # Full audit information
    FORENSIC = "forensic"         # Maximum audit detail for investigations

class ComplianceMode(Enum):
    """Regulatory compliance modes"""
    PERMISSIVE = "permissive"     # Minimal compliance checking
    STANDARD = "standard"         # Standard compliance validation
    STRICT = "strict"            # Strict compliance enforcement
    PARANOID = "paranoid"        # Maximum compliance validation


# =============================================================================
# Layer 1 Configuration Class
# =============================================================================

@dataclass
class Layer1Config:
    """
    Comprehensive Layer 1 configuration with performance and compliance settings
    """
    
    # ==== Core Configuration ====
    schema_version: str = "v2.1"
    processing_mode: ProcessingMode = ProcessingMode.PRODUCTION
    performance_profile: PerformanceProfile = PerformanceProfile.BALANCED
    
    # ==== Schema Configuration ====
    enable_schema_migration: bool = True
    backward_compatibility_mode: bool = True
    schema_validation_level: str = "strict"
    auto_schema_evolution: bool = False
    
    # ==== Privacy Configuration ====
    privacy_level: PrivacyLevel = PrivacyLevel.MODERATE
    enable_gdpr_compliance: bool = True
    enable_ccpa_compliance: bool = True
    enable_hipaa_compliance: bool = False
    pii_detection_threshold: float = 0.8
    
    # ==== Processing Configuration ====
    anomaly_preservation_mode: PreservationMode = PreservationMode.AGGRESSIVE
    enable_parallel_processing: bool = True
    enable_batch_optimization: bool = True
    
    # ==== Performance Targets ====
    max_processing_latency_ms: int = 2
    target_throughput_records_per_sec: int = 10000
    max_memory_overhead_mb: int = 32
    min_anomaly_preservation_rate: float = 0.995
    
    # ==== Hash Configuration ====
    hash_algorithms: List[str] = field(default_factory=lambda: ["sha256", "blake2b"])
    enable_deferred_hashing: bool = True
    hash_threshold_bytes: int = 1024
    hash_cache_size: int = 1000
    
    # ==== Quality Assurance ====
    enable_quality_monitoring: bool = True
    quality_threshold: float = 0.95
    enable_real_time_validation: bool = True
    quality_check_interval_ms: int = 100
    
    # ==== Audit and Compliance ====
    generate_audit_trails: bool = True
    audit_detail_level: AuditLevel = AuditLevel.COMPREHENSIVE
    compliance_validation_mode: ComplianceMode = ComplianceMode.STRICT
    audit_retention_days: int = 90
    
    # ==== Enhanced Features (Version 2.0) ====
    enable_circuit_breaker: bool = True
    enable_caching: bool = True
    enable_resource_management: bool = True
    enable_metrics: bool = True
    
    # ==== Circuit Breaker Settings ====
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: float = 60.0
    circuit_half_open_requests: int = 3
    
    # ==== Cache Settings ====
    cache_max_size: int = 1000
    cache_ttl_seconds: float = 300.0
    cache_eviction_policy: str = "lru"
    
    # ==== Batch Settings ====
    batch_min_size: int = 10
    batch_max_size: int = 100
    batch_max_wait_ms: float = 10.0
    enable_dynamic_batching: bool = True
    
    # ==== Resource Management ====
    max_memory_mb: int = 32
    memory_cleanup_threshold: float = 0.8
    memory_critical_threshold: float = 0.95
    gc_interval_seconds: int = 60
    
    # ==== Metrics and Monitoring ====
    enable_prometheus_metrics: bool = True
    metrics_port: int = 8090
    health_check_interval_ms: int = 5000
    performance_sampling_rate: float = 1.0
    
    # ==== Development and Testing ====
    enable_debug_mode: bool = False
    enable_performance_profiling: bool = False
    test_mode: bool = False
    log_level: str = "INFO"
    
    # ==== Security Settings ====
    enable_input_sanitization: bool = True
    enable_injection_protection: bool = True
    max_field_size_bytes: int = 1048576  # 1MB
    max_nested_depth: int = 10
    
    # ==== Integration Settings ====
    layer0_interface_version: str = "v1.0"
    layer2_interface_version: str = "v1.0"
    enable_layer_validation: bool = True
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        self._validate_configuration()
        self._apply_profile_optimizations()
    
    def _validate_configuration(self):
        """Validate configuration for consistency and feasibility"""
        
        # Performance validation
        if self.max_processing_latency_ms < 1:
            raise ValueError("Processing latency target too aggressive (<1ms)")
        
        if self.target_throughput_records_per_sec < 1:
            raise ValueError("Throughput target must be positive")
        
        # Privacy and preservation validation
        if (self.privacy_level == PrivacyLevel.MAXIMUM and 
            self.min_anomaly_preservation_rate > 0.9):
            import warnings
            warnings.warn(
                "Maximum privacy with high preservation rate may be challenging",
                UserWarning
            )
        
        # Schema validation
        if not self.enable_schema_migration and self.backward_compatibility_mode:
            raise ValueError(
                "Cannot enable backward compatibility without schema migration"
            )
        
        # Hash algorithm validation
        if self.enable_deferred_hashing and not self.hash_algorithms:
            raise ValueError("Deferred hashing enabled but no algorithms specified")
        
        # Batch validation
        if self.batch_min_size > self.batch_max_size:
            raise ValueError("Batch min size cannot exceed max size")
    
    def _apply_profile_optimizations(self):
        """Apply optimizations based on performance profile"""
        
        if self.performance_profile == PerformanceProfile.MINIMAL:
            # Optimize for minimal resource usage
            self.max_memory_overhead_mb = min(self.max_memory_overhead_mb, 16)
            self.cache_max_size = min(self.cache_max_size, 100)
            self.enable_parallel_processing = False
            self.enable_metrics = False
            
        elif self.performance_profile == PerformanceProfile.PERFORMANCE:
            # Optimize for maximum performance
            self.enable_parallel_processing = True
            self.enable_batch_optimization = True
            self.enable_caching = True
            self.max_memory_overhead_mb = max(self.max_memory_overhead_mb, 64)
            self.cache_max_size = max(self.cache_max_size, 2000)
            
        elif self.performance_profile == PerformanceProfile.ACADEMIC:
            # Optimize for research and comprehensive metrics
            self.enable_metrics = True
            self.enable_performance_profiling = True
            self.audit_detail_level = AuditLevel.FORENSIC
            self.performance_sampling_rate = 1.0
            self.enable_debug_mode = True
            
        elif self.performance_profile == PerformanceProfile.BALANCED:
            # Already configured with balanced defaults
            pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, list):
                result[key] = value.copy()
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Layer1Config':
        """Create configuration from dictionary"""
        # Convert enum strings back to enums
        if 'processing_mode' in config_dict:
            config_dict['processing_mode'] = ProcessingMode(config_dict['processing_mode'])
        if 'performance_profile' in config_dict:
            config_dict['performance_profile'] = PerformanceProfile(config_dict['performance_profile'])
        if 'privacy_level' in config_dict:
            config_dict['privacy_level'] = PrivacyLevel(config_dict['privacy_level'])
        if 'anomaly_preservation_mode' in config_dict:
            config_dict['anomaly_preservation_mode'] = PreservationMode(config_dict['anomaly_preservation_mode'])
        if 'audit_detail_level' in config_dict:
            config_dict['audit_detail_level'] = AuditLevel(config_dict['audit_detail_level'])
        if 'compliance_validation_mode' in config_dict:
            config_dict['compliance_validation_mode'] = ComplianceMode(config_dict['compliance_validation_mode'])
        
        return cls(**config_dict)


# =============================================================================
# Configuration Factory Functions
# =============================================================================

def create_development_config() -> Layer1Config:
    """Create configuration optimized for development"""
    return Layer1Config(
        processing_mode=ProcessingMode.DEVELOPMENT,
        performance_profile=PerformanceProfile.MINIMAL,
        enable_debug_mode=True,
        log_level="DEBUG",
        test_mode=True,
        max_processing_latency_ms=10,  # Relaxed for development
        target_throughput_records_per_sec=1000,
        enable_performance_profiling=True
    )

def create_testing_config() -> Layer1Config:
    """Create configuration optimized for testing"""
    return Layer1Config(
        processing_mode=ProcessingMode.TESTING,
        performance_profile=PerformanceProfile.BALANCED,
        test_mode=True,
        enable_debug_mode=True,
        audit_detail_level=AuditLevel.COMPREHENSIVE,
        enable_metrics=True,
        max_processing_latency_ms=5
    )

def create_production_config() -> Layer1Config:
    """Create configuration optimized for production"""
    return Layer1Config(
        processing_mode=ProcessingMode.PRODUCTION,
        performance_profile=PerformanceProfile.PERFORMANCE,
        privacy_level=PrivacyLevel.HIGH,
        compliance_validation_mode=ComplianceMode.STRICT,
        enable_debug_mode=False,
        log_level="WARNING",
        max_processing_latency_ms=2,
        target_throughput_records_per_sec=10000
    )

def create_research_config() -> Layer1Config:
    """Create configuration optimized for academic research"""
    return Layer1Config(
        processing_mode=ProcessingMode.RESEARCH,
        performance_profile=PerformanceProfile.ACADEMIC,
        audit_detail_level=AuditLevel.FORENSIC,
        enable_performance_profiling=True,
        enable_metrics=True,
        performance_sampling_rate=1.0,
        enable_debug_mode=True,
        max_processing_latency_ms=5,  # Relaxed for research
        privacy_level=PrivacyLevel.MODERATE  # More data for research
    )

def create_minimal_config() -> Layer1Config:
    """Create minimal configuration for resource-constrained environments"""
    return Layer1Config(
        processing_mode=ProcessingMode.PRODUCTION,
        performance_profile=PerformanceProfile.MINIMAL,
        enable_parallel_processing=False,
        enable_caching=False,
        enable_metrics=False,
        max_memory_overhead_mb=8,
        cache_max_size=50,
        audit_detail_level=AuditLevel.MINIMAL
    )


# =============================================================================
# Environment-Based Configuration Loading
# =============================================================================

def load_config_from_environment() -> Layer1Config:
    """Load configuration based on environment variables"""
    import os
    
    # Detect environment
    env = os.getenv('SCAFAD_ENVIRONMENT', 'production').lower()
    
    if env == 'development':
        return create_development_config()
    elif env == 'testing':
        return create_testing_config()
    elif env == 'research':
        return create_research_config()
    elif env == 'minimal':
        return create_minimal_config()
    else:
        return create_production_config()

def load_config_from_file(file_path: str) -> Layer1Config:
    """Load configuration from JSON file"""
    import json
    
    with open(file_path, 'r') as f:
        config_dict = json.load(f)
    
    return Layer1Config.from_dict(config_dict)


# =============================================================================
# Configuration Validation Utilities
# =============================================================================

def validate_config_compatibility(config: Layer1Config) -> Dict[str, Any]:
    """Validate configuration for compatibility and performance"""
    
    report = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'recommendations': []
    }
    
    # Performance validation
    if config.max_processing_latency_ms < 2 and config.enable_metrics:
        report['warnings'].append(
            "Aggressive latency target with metrics enabled may impact performance"
        )
    
    # Privacy vs preservation validation
    if (config.privacy_level in [PrivacyLevel.HIGH, PrivacyLevel.MAXIMUM] and
        config.min_anomaly_preservation_rate > 0.95):
        report['warnings'].append(
            "High privacy with high preservation rate may be conflicting"
        )
    
    # Resource validation
    if config.max_memory_overhead_mb < 16:
        report['recommendations'].append(
            "Consider increasing memory allocation for better performance"
        )
    
    # Feature compatibility
    if config.enable_parallel_processing and config.performance_profile == PerformanceProfile.MINIMAL:
        report['warnings'].append(
            "Parallel processing may conflict with minimal performance profile"
        )
    
    return report


# =============================================================================
# Export Key Configuration Components
# =============================================================================

__all__ = [
    'Layer1Config',
    'ProcessingMode',
    'PerformanceProfile', 
    'PrivacyLevel',
    'PreservationMode',
    'AuditLevel',
    'ComplianceMode',
    'create_development_config',
    'create_testing_config',
    'create_production_config',
    'create_research_config',
    'create_minimal_config',
    'load_config_from_environment',
    'load_config_from_file',
    'validate_config_compatibility'
]