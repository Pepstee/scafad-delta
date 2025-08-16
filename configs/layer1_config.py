#!/usr/bin/env python3
"""
SCAFAD Layer 1: Configuration and Data Models
============================================

Configuration management for Layer 1's behavioral intake zone, including:
- Processing modes and performance profiles
- Privacy compliance settings
- Schema evolution parameters
- Anomaly preservation configurations
- Performance tuning options

Author: SCAFAD Research Team
Institution: Birmingham Newman University
License: MIT
Version: 1.0.0
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum, auto
from datetime import datetime, timezone
import logging

# =============================================================================
# Configuration Enums
# =============================================================================

class ProcessingMode(Enum):
    """Processing operation modes"""
    REAL_TIME = "real_time"           # Sub-2ms latency, minimal buffering
    BATCH_OPTIMIZED = "batch_optimized"  # Optimized for throughput
    QUALITY_FOCUSED = "quality_focused"  # Maximum quality, higher latency
    RESEARCH = "research"             # Full preservation, no time constraints

class PerformanceProfile(Enum):
    """Performance optimization profiles"""
    ULTRA_LOW_LATENCY = "ultra_low_latency"  # <1ms target
    BALANCED = "balanced"                     # 2ms target, optimized
    HIGH_THROUGHPUT = "high_throughput"      # 5ms target, max throughput
    QUALITY_OPTIMIZED = "quality_optimized"   # 10ms target, best quality

class PrivacyComplianceLevel(Enum):
    """Privacy compliance levels"""
    MINIMAL = "minimal"               # Basic PII redaction
    STANDARD = "standard"             # GDPR/CCPA compliance
    STRICT = "strict"                 # HIPAA/SOX compliance
    MAXIMUM = "maximum"               # Maximum privacy protection

class AnomalyPreservationMode(Enum):
    """Anomaly preservation strategies"""
    CONSERVATIVE = "conservative"     # Preserve only critical features
    BALANCED = "balanced"            # Balance preservation with performance
    AGGRESSIVE = "aggressive"        # Maximum feature preservation
    RESEARCH = "research"            # Full preservation for research

class SchemaEvolutionStrategy(Enum):
    """Schema evolution handling strategies"""
    STRICT = "strict"                # Reject incompatible schemas
    MIGRATE = "migrate"              # Attempt automatic migration
    FLEXIBLE = "flexible"            # Accept with warnings
    AUTO_LEARN = "auto_learn"        # Learn and adapt automatically

# =============================================================================
# Core Configuration Classes
# =============================================================================

@dataclass
class HashAlgorithmConfig:
    """Configuration for cryptographic hash algorithms"""
    primary_algorithm: str = "sha256"
    secondary_algorithms: List[str] = field(default_factory=lambda: ["sha512", "blake2b"])
    hash_salt_length: int = 32
    hash_iterations: int = 10000
    enable_adaptive_hashing: bool = True
    max_hash_size_bytes: int = 64

@dataclass
class PrivacyConfig:
    """Privacy compliance configuration"""
    compliance_level: PrivacyComplianceLevel = PrivacyComplianceLevel.STANDARD
    enable_gdpr_compliance: bool = True
    enable_ccpa_compliance: bool = True
    enable_hipaa_compliance: bool = False
    enable_sox_compliance: bool = False
    pii_detection_threshold: float = 0.85
    redaction_policy: str = "standard"
    data_retention_days: int = 90
    enable_consent_tracking: bool = True
    enable_data_minimization: bool = True

@dataclass
class SchemaConfig:
    """Schema evolution configuration"""
    evolution_strategy: SchemaEvolutionStrategy = SchemaEvolutionStrategy.MIGRATE
    current_schema_version: str = "v2.1"
    supported_versions: List[str] = field(default_factory=lambda: ["v2.0", "v2.1"])
    enable_backward_compatibility: bool = True
    enable_forward_compatibility: bool = False
    max_schema_migration_attempts: int = 3
    schema_validation_strictness: str = "moderate"
    enable_schema_learning: bool = False

@dataclass
class PreservationConfig:
    """Anomaly preservation configuration"""
    preservation_mode: AnomalyPreservationMode = AnomalyPreservationMode.BALANCED
    target_preservation_rate: float = 0.995
    critical_features: List[str] = field(default_factory=lambda: [
        "execution_phase", "anomaly_type", "function_name", "timestamp"
    ])
    enable_semantic_analysis: bool = True
    enable_feature_importance: bool = True
    preservation_validation_threshold: float = 0.95
    enable_adaptive_preservation: bool = True

@dataclass
class PerformanceConfig:
    """Performance tuning configuration"""
    target_latency_ms: float = 2.0
    max_batch_size: int = 1000
    enable_parallel_processing: bool = True
    max_worker_threads: int = 4
    enable_memory_optimization: bool = True
    max_memory_usage_mb: int = 512
    enable_compression: bool = True
    compression_level: int = 6
    enable_caching: bool = True
    cache_ttl_seconds: int = 300

@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration"""
    enable_performance_monitoring: bool = True
    enable_quality_monitoring: bool = True
    enable_error_tracking: bool = True
    metrics_collection_interval: float = 1.0
    enable_alerting: bool = False
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "latency_ms": 5.0,
        "error_rate": 0.01,
        "preservation_rate": 0.95
    })
    enable_audit_logging: bool = True
    audit_log_level: str = "INFO"

@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_input_validation: bool = True
    enable_sql_injection_protection: bool = True
    enable_xss_protection: bool = True
    enable_path_traversal_protection: bool = True
    max_input_size_bytes: int = 1048576  # 1MB
    enable_rate_limiting: bool = True
    max_requests_per_second: int = 1000
    enable_encryption: bool = True
    encryption_algorithm: str = "AES-256-GCM"

# =============================================================================
# Main Configuration Class
# =============================================================================

@dataclass
class Layer1Config:
    """
    Main configuration class for SCAFAD Layer 1
    
    This class consolidates all configuration options for the behavioral intake zone,
    providing a single point of configuration management.
    """
    
    # Core processing configuration
    processing_mode: ProcessingMode = ProcessingMode.REAL_TIME
    performance_profile: PerformanceProfile = PerformanceProfile.BALANCED
    
    # Component-specific configurations
    hash_config: HashAlgorithmConfig = field(default_factory=HashAlgorithmConfig)
    privacy_config: PrivacyConfig = field(default_factory=PrivacyConfig)
    schema_config: SchemaConfig = field(default_factory=SchemaConfig)
    preservation_config: PreservationConfig = field(default_factory=PreservationConfig)
    performance_config: PerformanceConfig = field(default_factory=PerformanceConfig)
    monitoring_config: MonitoringConfig = field(default_factory=MonitoringConfig)
    security_config: SecurityConfig = field(default_factory=SecurityConfig)
    
    # System configuration
    environment: str = "development"
    log_level: str = "INFO"
    enable_debug_mode: bool = False
    config_file_path: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        self._validate_configuration()
        self._setup_logging()
    
    def _validate_configuration(self):
        """Validate configuration parameters"""
        if self.performance_config.target_latency_ms < 0:
            raise ValueError("Target latency must be positive")
        
        if self.preservation_config.target_preservation_rate < 0 or self.preservation_config.target_preservation_rate > 1:
            raise ValueError("Preservation rate must be between 0 and 1")
        
        if self.performance_config.max_batch_size < 1:
            raise ValueError("Max batch size must be at least 1")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert configuration to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Layer1Config':
        """Create configuration from dictionary"""
        # Handle enum conversions
        if 'processing_mode' in config_dict:
            config_dict['processing_mode'] = ProcessingMode(config_dict['processing_mode'])
        if 'performance_profile' in config_dict:
            config_dict['performance_profile'] = PerformanceProfile(config_dict['performance_profile'])
        if 'privacy_config' in config_dict and 'compliance_level' in config_dict['privacy_config']:
            config_dict['privacy_config']['compliance_level'] = PrivacyComplianceLevel(
                config_dict['privacy_config']['compliance_level']
            )
        if 'preservation_config' in config_dict and 'preservation_mode' in config_dict['preservation_config']:
            config_dict['preservation_config']['preservation_mode'] = AnomalyPreservationMode(
                config_dict['preservation_config']['preservation_mode']
            )
        if 'schema_config' in config_dict and 'evolution_strategy' in config_dict['schema_config']:
            config_dict['schema_config']['evolution_strategy'] = SchemaEvolutionStrategy(
                config_dict['schema_config']['evolution_strategy']
            )
        
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_string: str) -> 'Layer1Config':
        """Create configuration from JSON string"""
        config_dict = json.loads(json_string)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_file(cls, file_path: str) -> 'Layer1Config':
        """Create configuration from file"""
        with open(file_path, 'r') as f:
            json_string = f.read()
        return cls.from_json(json_string)
    
    def save_to_file(self, file_path: str):
        """Save configuration to file"""
        with open(file_path, 'w') as f:
            f.write(self.to_json())
    
    def get_effective_config(self) -> Dict[str, Any]:
        """Get effective configuration with environment overrides"""
        config = self.to_dict()
        
        # Apply environment-specific overrides
        if self.environment == "production":
            config['log_level'] = "WARNING"
            config['enable_debug_mode'] = False
            config['monitoring_config']['enable_alerting'] = True
        
        elif self.environment == "testing":
            config['log_level'] = "DEBUG"
            config['enable_debug_mode'] = True
            config['performance_config']['target_latency_ms'] = 10.0
        
        return config

# =============================================================================
# Default Configuration Factory
# =============================================================================

def create_default_config() -> Layer1Config:
    """Create default configuration for development"""
    return Layer1Config()

def create_production_config() -> Layer1Config:
    """Create production-optimized configuration"""
    config = Layer1Config()
    config.environment = "production"
    config.log_level = "WARNING"
    config.monitoring_config.enable_alerting = True
    config.performance_config.target_latency_ms = 1.5
    config.security_config.enable_rate_limiting = True
    return config

def create_testing_config() -> Layer1Config:
    """Create testing configuration"""
    config = Layer1Config()
    config.environment = "testing"
    config.log_level = "DEBUG"
    config.enable_debug_mode = True
    config.performance_config.target_latency_ms = 10.0
    config.monitoring_config.metrics_collection_interval = 0.1
    return config

# =============================================================================
# Configuration Validation
# =============================================================================

def validate_configuration(config: Layer1Config) -> Tuple[bool, List[str]]:
    """Validate configuration and return validation results"""
    errors = []
    
    # Performance validation
    if config.performance_config.target_latency_ms < 0.1:
        errors.append("Target latency too low (<0.1ms)")
    
    if config.performance_config.max_batch_size > 10000:
        errors.append("Max batch size too high (>10,000)")
    
    # Privacy validation
    if config.privacy_config.data_retention_days < 1:
        errors.append("Data retention must be at least 1 day")
    
    # Schema validation
    if not config.schema_config.supported_versions:
        errors.append("Must support at least one schema version")
    
    # Preservation validation
    if config.preservation_config.target_preservation_rate < 0.5:
        errors.append("Preservation rate too low (<50%)")
    
    return len(errors) == 0, errors

# =============================================================================
# Environment Configuration
# =============================================================================

def load_environment_config() -> Layer1Config:
    """Load configuration from environment variables"""
    config = create_default_config()
    
    # Override with environment variables
    if os.getenv('SCAFAD_ENVIRONMENT'):
        config.environment = os.getenv('SCAFAD_ENVIRONMENT')
    
    if os.getenv('SCAFAD_LOG_LEVEL'):
        config.log_level = os.getenv('SCAFAD_LOG_LEVEL')
    
    if os.getenv('SCAFAD_TARGET_LATENCY'):
        config.performance_config.target_latency_ms = float(os.getenv('SCAFAD_TARGET_LATENCY'))
    
    if os.getenv('SCAFAD_MAX_BATCH_SIZE'):
        config.performance_config.max_batch_size = int(os.getenv('SCAFAD_MAX_BATCH_SIZE'))
    
    return config

if __name__ == "__main__":
    # Example usage
    config = create_default_config()
    print("Default Configuration:")
    print(config.to_json())
    
    # Save to file
    config.save_to_file("configs/default_layer1_config.json")
    print("\nConfiguration saved to configs/default_layer1_config.json")
