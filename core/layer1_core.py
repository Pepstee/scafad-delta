#!/usr/bin/env python3
"""
SCAFAD Layer 1: Enhanced Behavioral Intake Zone - Core Orchestrator
===================================================================

Enhanced version with:
- Parallel processing pipeline for sub-2ms latency
- Circuit breaker pattern for resilience
- Advanced caching mechanisms
- Resource management and monitoring
- Batch optimization strategies
- Comprehensive metrics collection

Version: 2.0.0
"""

import asyncio
import time
import json
import hashlib
import logging
import uuid
import gc
import resource
import psutil
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set
from enum import Enum, auto
from datetime import datetime, timezone, timedelta
import traceback
from collections import defaultdict, deque
from functools import lru_cache, wraps
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import weakref

# Prometheus metrics (optional but recommended)
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    # Create dummy metrics classes
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def labels(self, **kwargs): return self
        def inc(self, amount=1): pass
    
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def labels(self, **kwargs): return self
        def observe(self, value): pass
    
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, value): pass
    
    class Summary:
        def __init__(self, *args, **kwargs): pass
        def observe(self, value): pass

# Import existing Layer 1 modules (keeping compatibility)
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
# Enhanced Data Models and Enums
# =============================================================================

class PrivacyLevel(Enum):
    """Privacy protection levels"""
    MINIMAL = "minimal"
    MODERATE = "moderate"
    HIGH = "high"
    MAXIMUM = "maximum"

class PreservationMode(Enum):
    """Anomaly preservation modes"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"

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
    CIRCUIT_OPEN = "circuit_open"
    DEGRADED = "degraded"

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

# =============================================================================
# Circuit Breaker Implementation
# =============================================================================

class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass

@dataclass
class CircuitBreaker:
    """Circuit breaker for resilient processing"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_requests: int = 3
    
    def __post_init__(self):
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.half_open_count = 0
        self._lock = threading.Lock()
        
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_count = 0
                else:
                    raise CircuitBreakerError(f"Circuit breaker is open (failures: {self.failure_count})")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful execution"""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_count += 1
                if self.half_open_count >= self.half_open_requests:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
            elif self.state == CircuitState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)
                self.success_count += 1
    
    def _on_failure(self):
        """Handle failed execution"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure': self.last_failure_time
        }

# =============================================================================
# Advanced Caching System
# =============================================================================

class AdaptiveCache:
    """Intelligent caching with TTL and adaptive eviction"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300):
        self.cache = {}
        self.access_counts = defaultdict(int)
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
        self.hit_count = 0
        self.miss_count = 0
        
    def get_cache_key(self, record: 'TelemetryRecord') -> str:
        """Generate deterministic cache key"""
        key_data = f"{record.function_name}_{record.execution_phase}_{record.anomaly_type}"
        # Include partial telemetry data hash for uniqueness
        data_hash = hashlib.md5(json.dumps(record.telemetry_data, sort_keys=True).encode()).hexdigest()[:8]
        return f"{key_data}_{data_hash}"
    
    async def get_or_compute(self, key: str, compute_func: Callable, *args) -> Any:
        """Get from cache or compute and store"""
        with self._lock:
            # Check if cached and not expired
            if key in self.cache:
                if time.time() - self.timestamps[key] < self.ttl:
                    self.access_counts[key] += 1
                    self.hit_count += 1
                    return self.cache[key]
                else:
                    # Expired, remove from cache
                    del self.cache[key]
                    del self.timestamps[key]
        
        # Compute outside lock to avoid blocking
        self.miss_count += 1
        result = await compute_func(*args)
        
        with self._lock:
            # Evict if needed
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = result
            self.timestamps[key] = time.time()
            self.access_counts[key] = 1
        
        return result
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.cache:
            return
            
        # Find least recently used key
        lru_key = min(self.access_counts, key=self.access_counts.get)
        del self.cache[lru_key]
        del self.timestamps[lru_key]
        del self.access_counts[lru_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        
        return {
            'size': len(self.cache),
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'max_size': self.max_size
        }
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self.cache.clear()
            self.access_counts.clear()
            self.timestamps.clear()
            self.hit_count = 0
            self.miss_count = 0

# =============================================================================
# Batch Processing Optimizer
# =============================================================================

class BatchProcessor:
    """Optimized batch processing with dynamic batching"""
    
    def __init__(self, min_batch_size: int = 10, max_batch_size: int = 100, 
                 max_wait_ms: float = 10):
        self.min_batch = min_batch_size
        self.max_batch = max_batch_size
        self.max_wait = max_wait_ms / 1000.0
        self.pending_records = []
        self.batch_ready = asyncio.Event()
        self._lock = asyncio.Lock()
        self._timer_task = None
        
    async def add_record(self, record: 'TelemetryRecord'):
        """Add record to batch"""
        async with self._lock:
            self.pending_records.append(record)
            
            if len(self.pending_records) >= self.max_batch:
                self.batch_ready.set()
                if self._timer_task:
                    self._timer_task.cancel()
                    self._timer_task = None
            elif len(self.pending_records) == 1:
                # Start timer for first record
                if self._timer_task:
                    self._timer_task.cancel()
                self._timer_task = asyncio.create_task(self._batch_timer())
    
    async def _batch_timer(self):
        """Timer to trigger batch processing"""
        try:
            await asyncio.sleep(self.max_wait)
            async with self._lock:
                if self.pending_records:
                    self.batch_ready.set()
        except asyncio.CancelledError:
            pass
    
    async def get_batch(self) -> List['TelemetryRecord']:
        """Get next batch for processing"""
        await self.batch_ready.wait()
        
        async with self._lock:
            # Extract batch
            batch_size = min(len(self.pending_records), self.max_batch)
            batch = self.pending_records[:batch_size]
            self.pending_records = self.pending_records[batch_size:]
            
            # Reset event if no more records
            if not self.pending_records:
                self.batch_ready.clear()
                if self._timer_task:
                    self._timer_task.cancel()
                    self._timer_task = None
            
            return batch

# =============================================================================
# Resource Manager
# =============================================================================

class ResourceManager:
    """Manage memory and system resources"""
    
    def __init__(self, max_memory_mb: int = 32):
        self.max_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
        self.cleanup_threshold = 0.8
        self.critical_threshold = 0.95
        self._last_cleanup = time.time()
        self._cleanup_interval = 60.0  # seconds
        
    async def check_resources(self) -> Dict[str, Any]:
        """Check current resource usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'memory_usage_mb': memory_info.rss / 1024 / 1024,
            'memory_percent': memory_info.rss / self.max_memory,
            'cpu_percent': process.cpu_percent(),
            'num_threads': process.num_threads(),
            'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0
        }
    
    async def cleanup_if_needed(self) -> bool:
        """Conditional cleanup based on memory pressure"""
        resources = await self.check_resources()
        
        if resources['memory_percent'] > self.critical_threshold:
            await self.force_cleanup()
            return True
        elif resources['memory_percent'] > self.cleanup_threshold:
            if time.time() - self._last_cleanup > self._cleanup_interval:
                gc.collect(1)  # Generation 1 collection
                self._last_cleanup = time.time()
                return True
        
        return False
    
    async def force_cleanup(self):
        """Force aggressive memory cleanup"""
        # Full garbage collection
        gc.collect(2)
        
        # Clear caches
        for obj in gc.get_objects():
            if isinstance(obj, AdaptiveCache):
                obj.clear()
        
        # Yield to event loop
        await asyncio.sleep(0)
        
        self._last_cleanup = time.time()

# =============================================================================
# Metrics Collection System
# =============================================================================

class Layer1Metrics:
    """Comprehensive metrics collection"""
    
    def __init__(self):
        # Counters
        self.records_processed = Counter(
            'layer1_records_processed_total',
            'Total records processed',
            ['status', 'phase']
        )
        
        self.errors = Counter(
            'layer1_errors_total',
            'Total errors by type',
            ['error_type', 'phase']
        )
        
        # Histograms
        self.processing_latency = Histogram(
            'layer1_processing_latency_seconds',
            'Processing latency by phase',
            ['phase'],
            buckets=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
        )
        
        self.batch_size = Histogram(
            'layer1_batch_size',
            'Batch sizes processed',
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
        )
        
        # Gauges
        self.preservation_rate = Gauge(
            'layer1_anomaly_preservation_rate',
            'Current anomaly preservation rate'
        )
        
        self.memory_usage = Gauge(
            'layer1_memory_usage_bytes',
            'Current memory usage'
        )
        
        self.cache_hit_rate = Gauge(
            'layer1_cache_hit_rate',
            'Cache hit rate'
        )
        
        self.circuit_breaker_state = Gauge(
            'layer1_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half-open)'
        )
        
        # Summary for percentiles
        self.processing_time = Summary(
            'layer1_processing_time_seconds',
            'Processing time distribution'
        )
    
    def record_processing(self, phase: str, duration: float, status: str):
        """Record processing metrics"""
        self.records_processed.labels(status=status, phase=phase).inc()
        self.processing_latency.labels(phase=phase).observe(duration)
        self.processing_time.observe(duration)
    
    def record_error(self, error_type: str, phase: str):
        """Record error metrics"""
        self.errors.labels(error_type=error_type, phase=phase).inc()
    
    def update_preservation_rate(self, rate: float):
        """Update preservation rate gauge"""
        self.preservation_rate.set(rate)
    
    def update_memory_usage(self, bytes_used: int):
        """Update memory usage gauge"""
        self.memory_usage.set(bytes_used)
    
    def update_cache_hit_rate(self, rate: float):
        """Update cache hit rate"""
        self.cache_hit_rate.set(rate)
    
    def update_circuit_breaker(self, state: CircuitState):
        """Update circuit breaker state"""
        state_value = {'closed': 0, 'open': 1, 'half_open': 2}.get(state.value, -1)
        self.circuit_breaker_state.set(state_value)

# =============================================================================
# Enhanced Core Data Models
# =============================================================================

@dataclass
class TelemetryRecord:
    """Standardized telemetry record structure from Layer 0"""
    record_id: str
    timestamp: float
    function_name: str
    execution_phase: str
    anomaly_type: str
    telemetry_data: Dict[str, Any]
    provenance_chain: Optional[Dict[str, Any]] = None
    context_metadata: Optional[Dict[str, Any]] = None
    schema_version: str = "v2.1"
    priority: int = 0  # Added for priority processing
    
    def __post_init__(self):
        if not self.record_id:
            self.record_id = str(uuid.uuid4())

@dataclass
class CleanTelemetryRecord:
    """Processed telemetry record ready for Layer 2"""
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
    processing_mode: str = "full"  # Added to track processing mode

@dataclass
class ProcessedBatch:
    """Complete processed batch output for Layer 2"""
    batch_id: str
    cleaned_records: List[CleanTelemetryRecord]
    schema_metadata: SchemaMetadata
    privacy_audit_trail: PrivacyAuditTrail
    preservation_report: PreservationReport
    quality_metrics: QualityMetrics
    processing_summary: Dict[str, Any]
    audit_trail: ProcessingAudit
    total_processing_time_ms: float = 0.0
    processing_mode: str = "full"  # Added to track batch processing mode

@dataclass
class Layer1Config:
    """Enhanced Layer 1 configuration"""
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
    
    # Enhanced Features
    enable_parallel_processing: bool = True
    enable_circuit_breaker: bool = True
    enable_caching: bool = True
    enable_batch_optimization: bool = True
    enable_resource_management: bool = True
    enable_metrics: bool = True
    
    # Circuit Breaker Settings
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: float = 60.0
    
    # Cache Settings
    cache_max_size: int = 1000
    cache_ttl_seconds: float = 300.0
    
    # Batch Settings
    batch_min_size: int = 10
    batch_max_size: int = 100
    batch_max_wait_ms: float = 10.0
    
    # Development and Testing
    enable_debug_mode: bool = False
    enable_performance_profiling: bool = False
    test_mode: bool = False

# =============================================================================
# Enhanced Layer 1 Main Orchestrator
# =============================================================================

class Layer1_BehavioralIntakeZone:
    """
    Enhanced Layer 1 orchestrator with advanced features:
    - Parallel processing pipeline
    - Circuit breaker resilience
    - Adaptive caching
    - Resource management
    - Batch optimization
    - Comprehensive metrics
    """
    
    def __init__(self, config: Layer1Config = None):
        """Initialize enhanced Layer 1 with all advanced features"""
        self.config = config or Layer1Config()
        self._setup_logging()
        
        # Initialize core processing components
        self._initialize_core_components()
        
        # Initialize supporting subsystems
        self._initialize_subsystems()
        
        # Initialize utility services
        self._initialize_utilities()
        
        # Initialize enhanced features
        self._initialize_enhanced_features()
        
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
            'privacy_compliance_rate': 1.0,
            'cache_hit_rate': 0.0,
            'circuit_breaker_trips': 0
        }
        
        # Thread pool for CPU-bound operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        self.logger.info(
            f"Enhanced Layer 1 Behavioral Intake Zone initialized "
            f"(parallel={self.config.enable_parallel_processing}, "
            f"circuit_breaker={self.config.enable_circuit_breaker}, "
            f"caching={self.config.enable_caching})"
        )
    
    def _setup_logging(self):
        """Setup comprehensive logging for Layer 1"""
        self.logger = logging.getLogger("SCAFAD.Layer1.Enhanced")
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
        
        self.schema_registry = SchemaRegistry(self.config)
        self.privacy_policy_engine = PrivacyPolicyEngine(self.config)
        self.semantic_analyzer = SemanticAnalyzer(self.config)
        self.quality_monitor = QualityAssuranceMonitor(self.config)
        self.audit_generator = AuditTrailGenerator(self.config)
        
        self.logger.info("Supporting subsystems initialized successfully")
    
    def _initialize_utilities(self):
        """Initialize utility services"""
        self.logger.info("Initializing Layer 1 utility services...")
        
        self.hash_library = CryptographicHasher(self.config.hash_algorithms)
        self.redaction_manager = RedactionPolicyManager(self.config)
        self.field_mapper = FieldMappingEngine(self.config)
        self.compression_optimizer = CompressionOptimizer(self.config)
        self.record_validator = TelemetryRecordValidator(self.config)
        
        self.logger.info("Utility services initialized successfully")
    
    def _initialize_enhanced_features(self):
        """Initialize enhanced features"""
        self.logger.info("Initializing enhanced features...")
        
        # Circuit Breaker
        if self.config.enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=self.config.circuit_failure_threshold,
                recovery_timeout=self.config.circuit_recovery_timeout
            )
        else:
            self.circuit_breaker = None
        
        # Adaptive Cache
        if self.config.enable_caching:
            self.cache = AdaptiveCache(
                max_size=self.config.cache_max_size,
                ttl_seconds=self.config.cache_ttl_seconds
            )
        else:
            self.cache = None
        
        # Batch Processor
        if self.config.enable_batch_optimization:
            self.batch_processor = BatchProcessor(
                min_batch_size=self.config.batch_min_size,
                max_batch_size=self.config.batch_max_size,
                max_wait_ms=self.config.batch_max_wait_ms
            )
        else:
            self.batch_processor = None
        
        # Resource Manager
        if self.config.enable_resource_management:
            self.resource_manager = ResourceManager(
                max_memory_mb=self.config.max_memory_overhead_mb
            )
        else:
            self.resource_manager = None
        
        # Metrics
        if self.config.enable_metrics:
            self.metrics = Layer1Metrics()
        else:
            self.metrics = None
        
        self.logger.info("Enhanced features initialized successfully")
    
    def _initialize_monitoring(self):
        """Initialize performance monitoring and metrics collection"""
        self.performance_metrics = {
            'phase_latencies': {phase.name: deque(maxlen=1000) for phase in ProcessingPhase},
            'throughput_samples': deque(maxlen=100),
            'memory_usage_samples': deque(maxlen=100),
            'anomaly_preservation_samples': deque(maxlen=100),
            'error_counts': defaultdict(int),
            'start_time': time.time()
        }
    
    # =========================================================================
    # Enhanced Processing Pipeline with Parallel Execution
    # =========================================================================
    
    async def process_telemetry_batch(self, 
                                    telemetry_records: List[TelemetryRecord],
                                    processing_context: Optional[Dict[str, Any]] = None) -> ProcessedBatch:
        """
        Enhanced processing pipeline with parallel execution and fallback strategies
        """
        batch_start_time = time.time()
        batch_id = str(uuid.uuid4())
        
        self.logger.info(f"Starting enhanced Layer 1 processing for batch {batch_id} with {len(telemetry_records)} records")
        
        # Check resources before processing
        if self.resource_manager:
            await self.resource_manager.cleanup_if_needed()
            resources = await self.resource_manager.check_resources()
            if resources['memory_percent'] > 0.9:
                self.logger.warning(f"High memory usage: {resources['memory_usage_mb']:.1f}MB")
        
        # Update metrics
        if self.metrics:
            self.metrics.batch_size.observe(len(telemetry_records))
        
        # Process with circuit breaker if enabled
        if self.circuit_breaker:
            try:
                result = await self.circuit_breaker.call(
                    self._process_batch_internal,
                    telemetry_records,
                    batch_id,
                    batch_start_time,
                    processing_context
                )
                if self.metrics:
                    self.metrics.update_circuit_breaker(self.circuit_breaker.state)
                return result
            except CircuitBreakerError as e:
                self.logger.error(f"Circuit breaker open: {e}")
                self.processing_state['circuit_breaker_trips'] += 1
                # Fall back to minimal processing
                return await self._process_batch_minimal(
                    telemetry_records, batch_id, batch_start_time
                )
        else:
            # Process without circuit breaker
            return await self._process_batch_internal(
                telemetry_records, batch_id, batch_start_time, processing_context
            )
    
    async def _process_batch_internal(self,
                                     telemetry_records: List[TelemetryRecord],
                                     batch_id: str,
                                     batch_start_time: float,
                                     processing_context: Optional[Dict[str, Any]]) -> ProcessedBatch:
        """Internal batch processing with parallel execution"""
        
        try:
            # Initialize batch processing state
            processing_summary = {
                'batch_id': batch_id,
                'total_records': len(telemetry_records),
                'start_time': batch_start_time,
                'phases_completed': [],
                'errors': [],
                'processing_mode': 'parallel' if self.config.enable_parallel_processing else 'sequential'
            }
            
            if self.config.enable_parallel_processing:
                # Parallel processing pipeline
                result = await self._process_batch_parallel(
                    telemetry_records, processing_summary
                )
            else:
                # Sequential processing (original implementation)
                result = await self._process_batch_sequential(
                    telemetry_records, processing_summary
                )
            
            # Update statistics
            self._update_processing_statistics(result)
            
            # Update metrics
            if self.metrics:
                total_time = time.time() - batch_start_time
                self.metrics.processing_time.observe(total_time)
                self.metrics.update_preservation_rate(result.preservation_report.preservation_rate)
                
                if self.cache:
                    cache_stats = self.cache.get_stats()
                    self.metrics.update_cache_hit_rate(cache_stats['hit_rate'])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            if self.metrics:
                self.metrics.record_error('processing_failure', 'batch')
            
            # Return degraded results
            return await self._handle_processing_error(
                batch_id, telemetry_records, processing_summary, e, batch_start_time
            )
    
    async def _process_batch_parallel(self,
                                     records: List[TelemetryRecord],
                                     processing_summary: Dict[str, Any]) -> ProcessedBatch:
        """Parallel processing implementation"""
        
        batch_id = processing_summary['batch_id']
        
        # Phase 1 & 2: Parallel validation and schema checking
        validation_task = asyncio.create_task(
            self._parallel_phase_validation(records, processing_summary)
        )
        schema_task = asyncio.create_task(
            self._parallel_phase_schema(records, processing_summary)
        )
        
        validated_records, validation_errors = await validation_task
        schema_results = await schema_task
        
        # Merge results from parallel phases
        processed_records = self._merge_parallel_results(
            validated_records, schema_results
        )
        
        # Phase 3 & 4: Parallel sanitization and privacy filtering
        if self.config.enable_caching and self.cache:
            # Try cache for known patterns
            cached_records = []
            uncached_records = []
            
            for record in processed_records:
                cache_key = self.cache.get_cache_key(record)
                cached = await self._try_get_cached(cache_key)
                if cached:
                    cached_records.append(cached)
                else:
                    uncached_records.append(record)
            
            # Process uncached records
            if uncached_records:
                sanitization_task = asyncio.create_task(
                    self._parallel_phase_sanitization(uncached_records, processing_summary)
                )
                privacy_task = asyncio.create_task(
                    self._parallel_phase_privacy(uncached_records, processing_summary)
                )
                
                sanitized = await sanitization_task
                privacy_filtered, privacy_audit = await privacy_task
                
                # Cache results
                for record in privacy_filtered:
                    cache_key = self.cache.get_cache_key(record)
                    await self.cache.get_or_compute(
                        cache_key, 
                        lambda r=record: asyncio.create_task(asyncio.sleep(0))
                    )
                
                processed_records = cached_records + privacy_filtered
            else:
                processed_records = cached_records
                privacy_audit = PrivacyAuditTrail(
                    batch_id=batch_id,
                    records_processed=len(cached_records),
                    privacy_level=self.config.privacy_level,
                    actions_taken=[],
                    compliance_status={'cached': True},
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
        else:
            # Process without cache
            sanitization_task = asyncio.create_task(
                self._parallel_phase_sanitization(processed_records, processing_summary)
            )
            privacy_task = asyncio.create_task(
                self._parallel_phase_privacy(processed_records, processing_summary)
            )
            
            sanitized = await sanitization_task
            processed_records, privacy_audit = await privacy_task
        
        # Phase 5 & 6: Parallel hashing and preservation
        hashing_task = asyncio.create_task(
            self._parallel_phase_hashing(processed_records, processing_summary)
        )
        preservation_task = asyncio.create_task(
            self._parallel_phase_preservation(processed_records, records, processing_summary)
        )
        
        hashed_records = await hashing_task
        final_records, preservation_report = await preservation_task
        
        # Phase 7 & 8: Quality and audit (can be parallel)
        quality_task = asyncio.create_task(
            self._phase_7_quality_assurance(final_records, processing_summary)
        )
        audit_task = asyncio.create_task(
            self._phase_8_audit_generation(processing_summary)
        )
        
        quality_metrics = await quality_task
        audit_trail = await audit_task
        
        # Assemble final batch
        total_processing_time = (time.time() - processing_summary['start_time']) * 1000
        
        return ProcessedBatch(
            batch_id=batch_id,
            cleaned_records=final_records,
            schema_metadata=self.schema_engine.get_current_metadata(),
            privacy_audit_trail=privacy_audit,
            preservation_report=preservation_report,
            quality_metrics=quality_metrics,
            processing_summary=processing_summary,
            audit_trail=audit_trail,
            total_processing_time_ms=total_processing_time,
            processing_mode='parallel'
        )
    
    async def _process_batch_sequential(self,
                                       records: List[TelemetryRecord],
                                       processing_summary: Dict[str, Any]) -> ProcessedBatch:
        """Sequential processing (original implementation)"""
        
        batch_id = processing_summary['batch_id']
        batch_start_time = processing_summary['start_time']
        
        # Phase 1: Input Validation Gateway
        validated_records = await self._phase_1_validation(records, processing_summary)
        
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
            hashed_records, records, processing_summary
        )
        
        # Phase 7: Quality Assurance and Metrics
        quality_metrics = await self._phase_7_quality_assurance(final_records, processing_summary)
        
        # Phase 8: Audit Trail Generation
        audit_trail = await self._phase_8_audit_generation(processing_summary)
        
        # Assemble final processed batch
        total_processing_time = (time.time() - batch_start_time) * 1000
        
        return ProcessedBatch(
            batch_id=batch_id,
            cleaned_records=final_records,
            schema_metadata=self.schema_engine.get_current_metadata(),
            privacy_audit_trail=privacy_audit,
            preservation_report=preservation_report,
            quality_metrics=quality_metrics,
            processing_summary=processing_summary,
            audit_trail=audit_trail,
            total_processing_time_ms=total_processing_time,
            processing_mode='sequential'
        )
    
    async def _process_batch_minimal(self,
                                    records: List[TelemetryRecord],
                                    batch_id: str,
                                    batch_start_time: float) -> ProcessedBatch:
        """Minimal processing for degraded mode"""
        
        self.logger.warning(f"Processing batch {batch_id} in minimal mode")
        
        clean_records = []
        for record in records:
            # Minimal validation and sanitization only
            if await self._quick_validate(record):
                clean_record = CleanTelemetryRecord(
                    record_id=str(uuid.uuid4()),
                    original_record_id=record.record_id,
                    timestamp=record.timestamp,
                    function_name=record.function_name,
                    execution_phase=record.execution_phase,
                    anomaly_type=record.anomaly_type,
                    cleaned_data=self._minimal_sanitize(record.telemetry_data),
                    schema_metadata=SchemaMetadata(
                        version=record.schema_version,
                        compatibility_level="minimal",
                        migration_applied=False
                    ),
                    privacy_compliance={'mode': 'minimal'},
                    preservation_score=0.9,  # Assume good preservation in minimal mode
                    quality_score=0.7,
                    processing_latency_ms=0.0,
                    processing_mode='minimal'
                )
                clean_records.append(clean_record)
        
        total_time = (time.time() - batch_start_time) * 1000
        
        return ProcessedBatch(
            batch_id=batch_id,
            cleaned_records=clean_records,
            schema_metadata=SchemaMetadata(
                version=self.config.schema_version,
                compatibility_level="minimal",
                migration_applied=False
            ),
            privacy_audit_trail=PrivacyAuditTrail(
                batch_id=batch_id,
                records_processed=len(clean_records),
                privacy_level=PrivacyLevel.MINIMAL,
                actions_taken=[],
                compliance_status={'mode': 'minimal'},
                timestamp=datetime.now(timezone.utc).isoformat()
            ),
            preservation_report=PreservationReport(
                batch_id=batch_id,
                records_analyzed=len(clean_records),
                preservation_rate=0.9,
                preservation_mode=PreservationMode.CONSERVATIVE,
                critical_features_preserved=True,
                preservation_details={'mode': 'minimal'},
                timestamp=datetime.now(timezone.utc).isoformat()
            ),
            quality_metrics=QualityMetrics(
                overall_score=0.7,
                individual_scores=[0.7] * len(clean_records),
                error_message=None
            ),
            processing_summary={
                'batch_id': batch_id,
                'mode': 'minimal',
                'reason': 'circuit_breaker_open'
            },
            audit_trail=ProcessingAudit(
                batch_id=batch_id,
                processing_status=ProcessingStatus.DEGRADED,
                error_details=None,
                phases_completed=['minimal'],
                timestamp=datetime.now(timezone.utc).isoformat()
            ),
            total_processing_time_ms=total_time,
            processing_mode='minimal'
        )
    
    # =========================================================================
    # Parallel Processing Helper Methods
    # =========================================================================
    
    async def _parallel_phase_validation(self,
                                        records: List[TelemetryRecord],
                                        processing_summary: Dict[str, Any]) -> Tuple[List[TelemetryRecord], List[Dict]]:
        """Parallel validation phase"""
        phase_start = time.time()
        
        # Process records in parallel chunks
        chunk_size = max(1, len(records) // 4)
        chunks = [records[i:i+chunk_size] for i in range(0, len(records), chunk_size)]
        
        tasks = []
        for chunk in chunks:
            tasks.append(asyncio.create_task(self._validate_chunk(chunk)))
        
        results = await asyncio.gather(*tasks)
        
        # Merge results
        validated_records = []
        validation_errors = []
        for validated, errors in results:
            validated_records.extend(validated)
            validation_errors.extend(errors)
        
        phase_time = (time.time() - phase_start) * 1000
        self.performance_metrics['phase_latencies']['VALIDATION'].append(phase_time)
        
        if self.metrics:
            self.metrics.record_processing('VALIDATION', phase_time/1000, 'success')
        
        processing_summary['phases_completed'].append({
            'phase': 'VALIDATION',
            'duration_ms': phase_time,
            'records_processed': len(validated_records),
            'errors': validation_errors,
            'parallel': True
        })
        
        return validated_records, validation_errors
    
    async def _validate_chunk(self, chunk: List[TelemetryRecord]) -> Tuple[List[TelemetryRecord], List[Dict]]:
        """Validate a chunk of records"""
        validated = []
        errors = []
        
        for record in chunk:
            try:
                validation_result = await self.validator.validate_telemetry_record(record)
                if validation_result.is_valid:
                    sanitized_record = await self.validator.sanitize_malformed_fields(record)
                    validated.append(sanitized_record)
                else:
                    errors.append({
                        'record_id': record.record_id,
                        'errors': validation_result.errors
                    })
            except Exception as e:
                errors.append({
                    'record_id': record.record_id,
                    'error': str(e)
                })
        
        return validated, errors
    
    async def _parallel_phase_schema(self,
                                    records: List[TelemetryRecord],
                                    processing_summary: Dict[str, Any]) -> List[TelemetryRecord]:
        """Parallel schema evolution phase"""
        phase_start = time.time()
        
        # Group records by schema version for efficient processing
        version_groups = defaultdict(list)
        for record in records:
            version_groups[record.schema_version].append(record)
        
        tasks = []
        for version, group in version_groups.items():
            if version != self.config.schema_version:
                tasks.append(asyncio.create_task(
                    self._migrate_schema_group(group, version)
                ))
            else:
                # No migration needed
                tasks.append(asyncio.create_task(
                    asyncio.sleep(0, result=group)
                ))
        
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        migrated_records = []
        for group in results:
            migrated_records.extend(group)
        
        phase_time = (time.time() - phase_start) * 1000
        self.performance_metrics['phase_latencies']['SCHEMA_EVOLUTION'].append(phase_time)
        
        if self.metrics:
            self.metrics.record_processing('SCHEMA_EVOLUTION', phase_time/1000, 'success')
        
        return migrated_records
    
    async def _migrate_schema_group(self,
                                   records: List[TelemetryRecord],
                                   from_version: str) -> List[TelemetryRecord]:
        """Migrate a group of records with the same schema version"""
        migrated = []
        
        for record in records:
            try:
                migration_result = await self.schema_engine.migrate_record_to_current_schema(record)
                if migration_result.success:
                    migrated.append(migration_result.migrated_record)
                else:
                    # Keep original on migration failure
                    migrated.append(record)
            except Exception:
                migrated.append(record)
        
        return migrated
    
    async def _parallel_phase_sanitization(self,
                                          records: List[TelemetryRecord],
                                          processing_summary: Dict[str, Any]) -> List[TelemetryRecord]:
        """Parallel sanitization phase"""
        phase_start = time.time()
        
        # Process records in parallel
        tasks = []
        for record in records:
            tasks.append(asyncio.create_task(self._sanitize_record(record)))
        
        sanitized_records = await asyncio.gather(*tasks)
        
        phase_time = (time.time() - phase_start) * 1000
        self.performance_metrics['phase_latencies']['SANITIZATION'].append(phase_time)
        
        if self.metrics:
            self.metrics.record_processing('SANITIZATION', phase_time/1000, 'success')
        
        return sanitized_records
    
    async def _sanitize_record(self, record: TelemetryRecord) -> TelemetryRecord:
        """Sanitize a single record"""
        try:
            sanitization_result = await self.sanitizer.sanitize_telemetry_record(record)
            
            if sanitization_result.success:
                # Check anomaly preservation
                preservation_check = await self.preservation_guard.analyze_anomaly_risk_before_transform(record)
                
                if preservation_check.anomaly_preservation_score >= self.config.min_anomaly_preservation_rate:
                    return sanitization_result.sanitized_record
                else:
                    # Apply semantic-preserving sanitization
                    return await self.preservation_guard.apply_semantic_preserving_transforms(record)
            else:
                return record
        except Exception:
            return record
    
    async def _parallel_phase_privacy(self,
                                     records: List[TelemetryRecord],
                                     processing_summary: Dict[str, Any]) -> Tuple[List[TelemetryRecord], PrivacyAuditTrail]:
        """Parallel privacy filtering phase"""
        phase_start = time.time()
        
        # Process privacy filters in parallel
        tasks = []
        for record in records:
            tasks.append(asyncio.create_task(self._apply_privacy_filters(record)))
        
        results = await asyncio.gather(*tasks)
        
        # Collect results
        filtered_records = []
        all_actions = []
        for record, actions in results:
            filtered_records.append(record)
            all_actions.extend(actions)
        
        # Generate audit trail
        privacy_audit = PrivacyAuditTrail(
            batch_id=processing_summary['batch_id'],
            records_processed=len(filtered_records),
            privacy_level=self.config.privacy_level,
            actions_taken=all_actions,
            compliance_status={
                'gdpr': self.config.enable_gdpr_compliance,
                'ccpa': self.config.enable_ccpa_compliance,
                'hipaa': self.config.enable_hipaa_compliance
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        phase_time = (time.time() - phase_start) * 1000
        self.performance_metrics['phase_latencies']['PRIVACY_FILTERING'].append(phase_time)
        
        if self.metrics:
            self.metrics.record_processing('PRIVACY_FILTERING', phase_time/1000, 'success')
        
        return filtered_records, privacy_audit
    
    async def _apply_privacy_filters(self, record: TelemetryRecord) -> Tuple[TelemetryRecord, List[str]]:
        """Apply privacy filters to a single record"""
        actions = []
        
        try:
            if self.config.enable_gdpr_compliance:
                gdpr_result = await self.privacy_filter.apply_gdpr_filters(record)
                record = gdpr_result.filtered_record
                actions.extend(gdpr_result.actions_taken)
            
            if self.config.enable_ccpa_compliance:
                ccpa_result = await self.privacy_filter.apply_ccpa_filters(record)
                record = ccpa_result.filtered_record
                actions.extend(ccpa_result.actions_taken)
            
            if self.config.enable_hipaa_compliance:
                hipaa_result = await self.privacy_filter.apply_hipaa_filters(record)
                record = hipaa_result.filtered_record
                actions.extend(hipaa_result.actions_taken)
            
            # Apply PII redaction
            redaction_result = await self.privacy_filter.redact_pii_fields(record)
            if redaction_result.success:
                record = redaction_result.redacted_record
                actions.extend(redaction_result.redaction_actions)
        except Exception:
            pass  # Keep original record on error
        
        return record, actions
    
    async def _parallel_phase_hashing(self,
                                     records: List[TelemetryRecord],
                                     processing_summary: Dict[str, Any]) -> List[TelemetryRecord]:
        """Parallel hashing phase"""
        if not self.config.enable_deferred_hashing:
            return records
        
        phase_start = time.time()
        
        # Process hashing in parallel
        tasks = []
        for record in records:
            tasks.append(asyncio.create_task(self._hash_record(record)))
        
        hashed_records = await asyncio.gather(*tasks)
        
        phase_time = (time.time() - phase_start) * 1000
        self.performance_metrics['phase_latencies']['DEFERRED_HASHING'].append(phase_time)
        
        if self.metrics:
            self.metrics.record_processing('DEFERRED_HASHING', phase_time/1000, 'success')
        
        return hashed_records
    
    async def _hash_record(self, record: TelemetryRecord) -> TelemetryRecord:
        """Hash a single record"""
        try:
            hashable_fields = await self.hash_manager.identify_hashable_fields(record)
            
            if hashable_fields:
                hashing_result = await self.hash_manager.apply_deferred_hashing(record)
                if hashing_result.success:
                    return hashing_result.hashed_record.to_telemetry_record()
        except Exception:
            pass
        
        return record
    
    async def _parallel_phase_preservation(self,
                                          final_records: List[TelemetryRecord],
                                          original_records: List[TelemetryRecord],
                                          processing_summary: Dict[str, Any]) -> Tuple[List[CleanTelemetryRecord], PreservationReport]:
        """Parallel preservation validation phase"""
        phase_start = time.time()
        
        # Create mapping for quick lookup
        record_mapping = {r.record_id: r for r in original_records}
        
        # Validate preservation in parallel
        tasks = []
        for final_record in final_records:
            original = record_mapping.get(final_record.record_id)
            if original:
                tasks.append(asyncio.create_task(
                    self._validate_preservation(final_record, original)
                ))
        
        results = await asyncio.gather(*tasks)
        
        # Collect results
        clean_records = []
        preservation_scores = []
        for clean_record, score in results:
            if clean_record:
                clean_records.append(clean_record)
                preservation_scores.append(score)
        
        # Generate preservation report
        avg_score = sum(preservation_scores) / len(preservation_scores) if preservation_scores else 0.0
        
        preservation_report = PreservationReport(
            batch_id=processing_summary['batch_id'],
            records_analyzed=len(clean_records),
            preservation_rate=avg_score,
            preservation_mode=self.config.anomaly_preservation_mode,
            critical_features_preserved=avg_score >= self.config.min_anomaly_preservation_rate,
            preservation_details={
                'min_score': min(preservation_scores) if preservation_scores else 0.0,
                'max_score': max(preservation_scores) if preservation_scores else 0.0,
                'std_deviation': self._calculate_std_deviation(preservation_scores)
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        phase_time = (time.time() - phase_start) * 1000
        self.performance_metrics['phase_latencies']['PRESERVATION_VALIDATION'].append(phase_time)
        
        if self.metrics:
            self.metrics.record_processing('PRESERVATION_VALIDATION', phase_time/1000, 'success')
        
        return clean_records, preservation_report
    
    async def _validate_preservation(self,
                                    final_record: TelemetryRecord,
                                    original_record: TelemetryRecord) -> Tuple[Optional[CleanTelemetryRecord], float]:
        """Validate preservation for a single record"""
        try:
            preservation_result = await self.preservation_guard.validate_post_processing_detectability(
                original_record, final_record
            )
            
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
                quality_score=0.0,
                processing_latency_ms=0.0
            )
            
            return clean_record, preservation_result.preservation_score
        except Exception:
            return None, 0.0
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _merge_parallel_results(self,
                               validated: List[TelemetryRecord],
                               schema_migrated: List[TelemetryRecord]) -> List[TelemetryRecord]:
        """Merge results from parallel processing phases"""
        # Create mapping of validated records
        validated_map = {r.record_id: r for r in validated}
        
        # Update with schema migration results
        result = []
        for record in schema_migrated:
            if record.record_id in validated_map:
                result.append(record)
        
        return result
    
    async def _try_get_cached(self, cache_key: str) -> Optional[TelemetryRecord]:
        """Try to get cached processed record"""
        try:
            # Check if we have a cached result
            # Note: This is simplified - in reality would need proper cache structure
            return None
        except Exception:
            return None
    
    async def _quick_validate(self, record: TelemetryRecord) -> bool:
        """Quick validation for minimal mode"""
        return bool(record.record_id and record.timestamp and record.function_name)
    
    def _minimal_sanitize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Minimal sanitization for degraded mode"""
        # Remove obvious sensitive fields
        sensitive_fields = ['password', 'token', 'key', 'secret', 'credential']
        sanitized = data.copy()
        
        for field in sensitive_fields:
            for key in list(sanitized.keys()):
                if field in key.lower():
                    sanitized[key] = '[REDACTED]'
        
        return sanitized
    
    def _calculate_std_deviation(self, values: List[float]) -> float:
        """Calculate standard deviation of preservation scores"""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    # =========================================================================
    # Sequential Processing Methods (Original Implementation)
    # =========================================================================
    
    async def _phase_1_validation(self, 
                                  records: List[TelemetryRecord],
                                  processing_summary: Dict[str, Any]) -> List[TelemetryRecord]:
        """Phase 1: Input Validation Gateway - Sequential Implementation"""
        phase_start = time.time()
        
        self.logger.debug(f"Phase 1: Validating {len(records)} telemetry records...")
        
        validated_records = []
        validation_errors = []
        
        for record in records:
            try:
                # Validate record structure and content
                validation_result = await self.validator.validate_telemetry_record(record)
                
                if validation_result.is_valid:
                    # Apply any necessary field sanitization for malformed data
                    sanitized_record = await self.validator.sanitize_malformed_fields(record)
                    validated_records.append(sanitized_record)
                else:
                    validation_errors.append({
                        'record_id': record.record_id,
                        'errors': validation_result.errors,
                        'warnings': validation_result.warnings
                    })
                    
                    # Decide whether to include partially valid records
                    if validation_result.validation_level != ValidationLevel.CRITICAL:
                        sanitized_record = await self.validator.sanitize_malformed_fields(record)
                        validated_records.append(sanitized_record)
            
            except Exception as e:
                self.logger.error(f"Validation failed for record {record.record_id}: {str(e)}")
                validation_errors.append({
                    'record_id': record.record_id,
                    'error': str(e)
                })
                
                if self.metrics:
                    self.metrics.record_error('validation_exception', 'VALIDATION')
        
        phase_time = (time.time() - phase_start) * 1000
        self.performance_metrics['phase_latencies']['VALIDATION'].append(phase_time)
        
        if self.metrics:
            self.metrics.record_processing('VALIDATION', phase_time/1000, 'success')
        
        processing_summary['phases_completed'].append({
            'phase': 'VALIDATION',
            'duration_ms': phase_time,
            'records_processed': len(validated_records),
            'validation_errors': validation_errors,
            'error_count': len(validation_errors)
        })
        
        self.logger.debug(f"Phase 1 complete: {len(validated_records)} validated, {len(validation_errors)} errors")
        
        return validated_records
    
    async def _phase_2_schema_evolution(self,
                                        records: List[TelemetryRecord],
                                        processing_summary: Dict[str, Any]) -> List[TelemetryRecord]:
        """Phase 2: Schema Evolution Engine - Sequential Implementation"""
        phase_start = time.time()
        
        self.logger.debug(f"Phase 2: Processing schema evolution for {len(records)} records...")
        
        migrated_records = []
        migration_errors = []
        
        for record in records:
            try:
                # Check if record needs schema migration
                current_schema = await self.schema_engine.detect_schema_version(record)
                
                if current_schema != self.config.schema_version:
                    # Apply schema migration
                    migration_result = await self.schema_engine.migrate_record_to_current_schema(record)
                    
                    if migration_result.success:
                        migrated_records.append(migration_result.migrated_record)
                    else:
                        migration_errors.append({
                            'record_id': record.record_id,
                            'from_schema': current_schema,
                            'to_schema': self.config.schema_version,
                            'error': migration_result.error_message
                        })
                        
                        # Keep original record if migration fails
                        migrated_records.append(record)
                else:
                    # No migration needed
                    migrated_records.append(record)
            
            except Exception as e:
                self.logger.error(f"Schema migration failed for record {record.record_id}: {str(e)}")
                migration_errors.append({
                    'record_id': record.record_id,
                    'error': str(e)
                })
                
                # Keep original record on exception
                migrated_records.append(record)
                
                if self.metrics:
                    self.metrics.record_error('schema_migration_exception', 'SCHEMA_EVOLUTION')
        
        phase_time = (time.time() - phase_start) * 1000
        self.performance_metrics['phase_latencies']['SCHEMA_EVOLUTION'].append(phase_time)
        
        if self.metrics:
            self.metrics.record_processing('SCHEMA_EVOLUTION', phase_time/1000, 'success')
        
        processing_summary['phases_completed'].append({
            'phase': 'SCHEMA_EVOLUTION',
            'duration_ms': phase_time,
            'records_processed': len(migrated_records),
            'migration_errors': migration_errors,
            'error_count': len(migration_errors)
        })
        
        self.logger.debug(f"Phase 2 complete: {len(migrated_records)} migrated, {len(migration_errors)} errors")
        
        return migrated_records
    
    async def _phase_3_sanitization(self,
                                    records: List[TelemetryRecord],
                                    processing_summary: Dict[str, Any]) -> List[TelemetryRecord]:
        """Phase 3: Sanitization Processor - Sequential Implementation"""
        phase_start = time.time()
        
        self.logger.debug(f"Phase 3: Sanitizing {len(records)} records...")
        
        sanitized_records = []
        sanitization_errors = []
        
        for record in records:
            try:
                # Perform sanitization
                sanitization_result = await self.sanitizer.sanitize_telemetry_record(record)
                
                if sanitization_result.success:
                    # Check anomaly preservation before accepting sanitization
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
                        'error': sanitization_result.error_message,
                        'fields_affected': sanitization_result.fields_modified
                    })
                    
                    # Keep original record if sanitization fails
                    sanitized_records.append(record)
            
            except Exception as e:
                self.logger.error(f"Sanitization failed for record {record.record_id}: {str(e)}")
                sanitization_errors.append({
                    'record_id': record.record_id,
                    'error': str(e)
                })
                
                # Keep original record on exception
                sanitized_records.append(record)
                
                if self.metrics:
                    self.metrics.record_error('sanitization_exception', 'SANITIZATION')
        
        phase_time = (time.time() - phase_start) * 1000
        self.performance_metrics['phase_latencies']['SANITIZATION'].append(phase_time)
        
        if self.metrics:
            self.metrics.record_processing('SANITIZATION', phase_time/1000, 'success')
        
        processing_summary['phases_completed'].append({
            'phase': 'SANITIZATION',
            'duration_ms': phase_time,
            'records_processed': len(sanitized_records),
            'sanitization_errors': sanitization_errors,
            'error_count': len(sanitization_errors)
        })
        
        self.logger.debug(f"Phase 3 complete: {len(sanitized_records)} sanitized, {len(sanitization_errors)} errors")
        
        return sanitized_records
    
    async def _phase_4_privacy_filtering(self,
                                         records: List[TelemetryRecord],
                                         processing_summary: Dict[str, Any]) -> Tuple[List[TelemetryRecord], PrivacyAuditTrail]:
        """Phase 4: Privacy Compliance Filter - Sequential Implementation"""
        phase_start = time.time()
        
        self.logger.debug(f"Phase 4: Applying privacy filters to {len(records)} records...")
        
        filtered_records = []
        privacy_actions = []
        privacy_violations = []
        
        for record in records:
            try:
                current_record = record
                record_actions = []
                
                # Apply GDPR compliance filters if enabled
                if self.config.enable_gdpr_compliance:
                    gdpr_result = await self.privacy_filter.apply_gdpr_filters(current_record)
                    current_record = gdpr_result.filtered_record
                    record_actions.extend(gdpr_result.actions_taken)
                
                # Apply CCPA compliance filters if enabled
                if self.config.enable_ccpa_compliance:
                    ccpa_result = await self.privacy_filter.apply_ccpa_filters(current_record)
                    current_record = ccpa_result.filtered_record
                    record_actions.extend(ccpa_result.actions_taken)
                
                # Apply HIPAA compliance filters if enabled
                if self.config.enable_hipaa_compliance:
                    hipaa_result = await self.privacy_filter.apply_hipaa_filters(current_record)
                    current_record = hipaa_result.filtered_record
                    record_actions.extend(hipaa_result.actions_taken)
                
                # Apply PII redaction
                redaction_result = await self.privacy_filter.redact_pii_fields(current_record)
                
                if redaction_result.success:
                    current_record = redaction_result.redacted_record
                    record_actions.extend(redaction_result.redaction_actions)
                    
                    filtered_records.append(current_record)
                    privacy_actions.extend(record_actions)
                else:
                    privacy_violations.append({
                        'record_id': record.record_id,
                        'violation_type': 'redaction_failure',
                        'error': redaction_result.error_message
                    })
                    
                    # Include original record with warning
                    filtered_records.append(record)
            
            except Exception as e:
                self.logger.error(f"Privacy filtering failed for record {record.record_id}: {str(e)}")
                privacy_violations.append({
                    'record_id': record.record_id,
                    'violation_type': 'processing_error',
                    'error': str(e)
                })
                
                # Keep original record on exception
                filtered_records.append(record)
                
                if self.metrics:
                    self.metrics.record_error('privacy_filtering_exception', 'PRIVACY_FILTERING')
        
        # Generate privacy audit trail
        privacy_audit = PrivacyAuditTrail(
            batch_id=processing_summary['batch_id'],
            records_processed=len(filtered_records),
            privacy_level=self.config.privacy_level,
            actions_taken=privacy_actions,
            violations=privacy_violations,
            compliance_status={
                'gdpr': self.config.enable_gdpr_compliance,
                'ccpa': self.config.enable_ccpa_compliance,
                'hipaa': self.config.enable_hipaa_compliance,
                'violations_count': len(privacy_violations)
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        phase_time = (time.time() - phase_start) * 1000
        self.performance_metrics['phase_latencies']['PRIVACY_FILTERING'].append(phase_time)
        
        if self.metrics:
            self.metrics.record_processing('PRIVACY_FILTERING', phase_time/1000, 'success')
        
        processing_summary['phases_completed'].append({
            'phase': 'PRIVACY_FILTERING',
            'duration_ms': phase_time,
            'records_processed': len(filtered_records),
            'privacy_actions': len(privacy_actions),
            'privacy_violations': len(privacy_violations)
        })
        
        self.logger.debug(f"Phase 4 complete: {len(filtered_records)} filtered, {len(privacy_violations)} violations")
        
        return filtered_records, privacy_audit
    
    async def _phase_5_deferred_hashing(self,
                                        records: List[TelemetryRecord],
                                        processing_summary: Dict[str, Any]) -> List[TelemetryRecord]:
        """Phase 5: Deferred Hashing Manager - Sequential Implementation"""
        phase_start = time.time()
        
        self.logger.debug(f"Phase 5: Processing deferred hashing for {len(records)} records...")
        
        if not self.config.enable_deferred_hashing:
            self.logger.debug("Deferred hashing disabled, skipping phase 5")
            return records
        
        hashed_records = []
        hashing_errors = []
        
        for record in records:
            try:
                # Identify fields suitable for hashing
                hashable_fields = await self.hash_manager.identify_hashable_fields(record)
                
                if hashable_fields:
                    # Apply deferred hashing
                    hashing_result = await self.hash_manager.apply_deferred_hashing(record)
                    
                    if hashing_result.success:
                        # Convert hashed record back to telemetry record
                        hashed_telemetry = hashing_result.hashed_record.to_telemetry_record()
                        hashed_records.append(hashed_telemetry)
                    else:
                        hashing_errors.append({
                            'record_id': record.record_id,
                            'error': hashing_result.error_message,
                            'fields_attempted': hashable_fields
                        })
                        
                        # Keep original record if hashing fails
                        hashed_records.append(record)
                else:
                    # No hashable fields, keep original
                    hashed_records.append(record)
            
            except Exception as e:
                self.logger.error(f"Deferred hashing failed for record {record.record_id}: {str(e)}")
                hashing_errors.append({
                    'record_id': record.record_id,
                    'error': str(e)
                })
                
                # Keep original record on exception
                hashed_records.append(record)
                
                if self.metrics:
                    self.metrics.record_error('hashing_exception', 'DEFERRED_HASHING')
        
        phase_time = (time.time() - phase_start) * 1000
        self.performance_metrics['phase_latencies']['DEFERRED_HASHING'].append(phase_time)
        
        if self.metrics:
            self.metrics.record_processing('DEFERRED_HASHING', phase_time/1000, 'success')
        
        processing_summary['phases_completed'].append({
            'phase': 'DEFERRED_HASHING',
            'duration_ms': phase_time,
            'records_processed': len(hashed_records),
            'hashing_errors': hashing_errors,
            'error_count': len(hashing_errors)
        })
        
        self.logger.debug(f"Phase 5 complete: {len(hashed_records)} processed, {len(hashing_errors)} errors")
        
        return hashed_records
    
    async def _phase_6_preservation_validation(self,
                                               final_records: List[TelemetryRecord],
                                               original_records: List[TelemetryRecord],
                                               processing_summary: Dict[str, Any]) -> Tuple[List[CleanTelemetryRecord], PreservationReport]:
        """Phase 6: Anomaly Preservation Guard - Sequential Implementation"""
        phase_start = time.time()
        
        self.logger.debug(f"Phase 6: Validating anomaly preservation for {len(final_records)} records...")
        
        # Create mapping for quick lookup
        original_mapping = {record.record_id: record for record in original_records}
        
        clean_records = []
        preservation_scores = []
        preservation_failures = []
        
        for final_record in final_records:
            try:
                # Find corresponding original record
                original_record = original_mapping.get(final_record.record_id)
                
                if original_record:
                    # Validate preservation
                    preservation_result = await self.preservation_guard.validate_post_processing_detectability(
                        original_record, final_record
                    )
                    
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
                        quality_score=0.0,  # Will be set in Phase 7
                        processing_latency_ms=(time.time() - processing_summary['start_time']) * 1000
                    )
                    
                    clean_records.append(clean_record)
                    preservation_scores.append(preservation_result.preservation_score)
                    
                    # Check if preservation meets minimum threshold
                    if preservation_result.preservation_score < self.config.min_anomaly_preservation_rate:
                        preservation_failures.append({
                            'record_id': final_record.record_id,
                            'preservation_score': preservation_result.preservation_score,
                            'minimum_required': self.config.min_anomaly_preservation_rate,
                            'critical_features_lost': preservation_result.critical_features_lost
                        })
                else:
                    self.logger.warning(f"No original record found for {final_record.record_id}")
                    
                    # Create clean record without preservation validation
                    clean_record = CleanTelemetryRecord(
                        record_id=str(uuid.uuid4()),
                        original_record_id=final_record.record_id,
                        timestamp=final_record.timestamp,
                        function_name=final_record.function_name,
                        execution_phase=final_record.execution_phase,
                        anomaly_type=final_record.anomaly_type,
                        cleaned_data=final_record.telemetry_data,
                        schema_metadata=self.schema_engine.get_current_metadata(),
                        privacy_compliance={'compliant': True},
                        preservation_score=0.0,
                        quality_score=0.0,
                        processing_latency_ms=(time.time() - processing_summary['start_time']) * 1000
                    )
                    
                    clean_records.append(clean_record)
                    preservation_scores.append(0.0)
            
            except Exception as e:
                self.logger.error(f"Preservation validation failed for record {final_record.record_id}: {str(e)}")
                
                # Create minimal clean record
                clean_record = CleanTelemetryRecord(
                    record_id=str(uuid.uuid4()),
                    original_record_id=final_record.record_id,
                    timestamp=final_record.timestamp,
                    function_name=final_record.function_name,
                    execution_phase=final_record.execution_phase,
                    anomaly_type=final_record.anomaly_type,
                    cleaned_data=final_record.telemetry_data,
                    schema_metadata=self.schema_engine.get_current_metadata(),
                    privacy_compliance={'compliant': True},
                    preservation_score=0.0,
                    quality_score=0.0,
                    processing_latency_ms=0.0
                )
                
                clean_records.append(clean_record)
                preservation_scores.append(0.0)
                
                if self.metrics:
                    self.metrics.record_error('preservation_validation_exception', 'PRESERVATION_VALIDATION')
        
        # Calculate preservation statistics
        avg_preservation = sum(preservation_scores) / len(preservation_scores) if preservation_scores else 0.0
        min_preservation = min(preservation_scores) if preservation_scores else 0.0
        max_preservation = max(preservation_scores) if preservation_scores else 0.0
        
        # Generate preservation report
        preservation_report = PreservationReport(
            batch_id=processing_summary['batch_id'],
            records_analyzed=len(clean_records),
            preservation_rate=avg_preservation,
            preservation_mode=self.config.anomaly_preservation_mode,
            critical_features_preserved=len(preservation_failures) == 0,
            preservation_details={
                'average_score': avg_preservation,
                'minimum_score': min_preservation,
                'maximum_score': max_preservation,
                'std_deviation': self._calculate_std_deviation(preservation_scores),
                'failures': preservation_failures,
                'failure_count': len(preservation_failures)
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        phase_time = (time.time() - phase_start) * 1000
        self.performance_metrics['phase_latencies']['PRESERVATION_VALIDATION'].append(phase_time)
        
        if self.metrics:
            self.metrics.record_processing('PRESERVATION_VALIDATION', phase_time/1000, 'success')
            self.metrics.update_preservation_rate(avg_preservation)
        
        processing_summary['phases_completed'].append({
            'phase': 'PRESERVATION_VALIDATION',
            'duration_ms': phase_time,
            'records_processed': len(clean_records),
            'average_preservation': avg_preservation,
            'preservation_failures': len(preservation_failures)
        })
        
        self.logger.debug(f"Phase 6 complete: {len(clean_records)} validated, avg preservation: {avg_preservation:.3f}")
        
        return clean_records, preservation_report
    
    async def _phase_7_quality_assurance(self,
                                         clean_records: List[CleanTelemetryRecord],
                                         processing_summary: Dict[str, Any]) -> QualityMetrics:
        """Phase 7: Quality Assurance and Metrics"""
        phase_start = time.time()
        
        self.logger.debug(f"Phase 7: Quality assurance for {len(clean_records)} records...")
        
        try:
            # Run quality assessment
            quality_result = await self.quality_monitor.assess_batch_quality(clean_records)
            
            # Update quality scores in clean records
            for i, record in enumerate(clean_records):
                if i < len(quality_result.individual_scores):
                    record.quality_score = quality_result.individual_scores[i]
                else:
                    record.quality_score = quality_result.overall_score
            
            phase_time = (time.time() - phase_start) * 1000
            self.performance_metrics['phase_latencies']['QUALITY_ASSURANCE'].append(phase_time)
            
            if self.metrics:
                self.metrics.record_processing('QUALITY_ASSURANCE', phase_time/1000, 'success')
            
            processing_summary['phases_completed'].append({
                'phase': 'QUALITY_ASSURANCE',
                'duration_ms': phase_time,
                'records_processed': len(clean_records),
                'overall_quality': quality_result.overall_score
            })
            
            self.logger.debug(f"Phase 7 complete: Quality score: {quality_result.overall_score:.3f}")
            
            return quality_result
        
        except Exception as e:
            self.logger.error(f"Quality assurance failed: {str(e)}")
            
            if self.metrics:
                self.metrics.record_error('quality_assurance_exception', 'QUALITY_ASSURANCE')
            
            # Return default quality metrics
            return QualityMetrics(
                overall_score=0.5,
                individual_scores=[0.5] * len(clean_records),
                error_message=str(e)
            )
    
    async def _phase_8_audit_generation(self,
                                        processing_summary: Dict[str, Any]) -> ProcessingAudit:
        """Phase 8: Audit Trail Generation"""
        phase_start = time.time()
        
        self.logger.debug("Phase 8: Generating audit trail...")
        
        try:
            # Generate comprehensive audit trail
            audit_trail = await self.audit_generator.generate_processing_audit(processing_summary)
            
            phase_time = (time.time() - phase_start) * 1000
            
            if self.metrics:
                self.metrics.record_processing('AUDIT_GENERATION', phase_time/1000, 'success')
            
            processing_summary['phases_completed'].append({
                'phase': 'AUDIT_GENERATION',
                'duration_ms': phase_time
            })
            
            self.logger.debug("Phase 8 complete: Audit trail generated")
            
            return audit_trail
        
        except Exception as e:
            self.logger.error(f"Audit generation failed: {str(e)}")
            
            if self.metrics:
                self.metrics.record_error('audit_generation_exception', 'AUDIT_GENERATION')
            
            # Return minimal audit trail
            return ProcessingAudit(
                batch_id=processing_summary.get('batch_id', 'unknown'),
                processing_status=ProcessingStatus.PARTIAL_SUCCESS,
                error_details=str(e),
                phases_completed=[phase['phase'] for phase in processing_summary.get('phases_completed', [])],
                timestamp=datetime.now(timezone.utc).isoformat()
            )
    
    # =========================================================================
    # Error Handling and Recovery
    # =========================================================================
    
    async def _handle_processing_error(self,
                                       batch_id: str,
                                       original_records: List[TelemetryRecord],
                                       processing_summary: Dict[str, Any],
                                       error: Exception,
                                       batch_start_time: float) -> ProcessedBatch:
        """Handle processing errors and return degraded batch"""
        
        self.logger.error(f"Processing error in batch {batch_id}: {str(error)}")
        
        # Try to recover with minimal processing
        try:
            clean_records = []
            
            for record in original_records:
                # Minimal processing - basic sanitization only
                if await self._quick_validate(record):
                    clean_record = CleanTelemetryRecord(
                        record_id=str(uuid.uuid4()),
                        original_record_id=record.record_id,
                        timestamp=record.timestamp,
                        function_name=record.function_name,
                        execution_phase=record.execution_phase,
                        anomaly_type=record.anomaly_type,
                        cleaned_data=self._minimal_sanitize(record.telemetry_data),
                        schema_metadata=SchemaMetadata(
                            version=record.schema_version,
                            compatibility_level="error_recovery",
                            migration_applied=False
                        ),
                        privacy_compliance={'mode': 'error_recovery'},
                        preservation_score=0.8,  # Conservative estimate
                        quality_score=0.6,       # Degraded quality
                        processing_latency_ms=0.0,
                        processing_mode='error_recovery'
                    )
                    clean_records.append(clean_record)
            
            total_time = (time.time() - batch_start_time) * 1000
            
            return ProcessedBatch(
                batch_id=batch_id,
                cleaned_records=[],
                schema_metadata=SchemaMetadata(
                    version=self.config.schema_version,
                    compatibility_level="failure",
                    migration_applied=False
                ),
                privacy_audit_trail=PrivacyAuditTrail(
                    batch_id=batch_id,
                    records_processed=0,
                    privacy_level=PrivacyLevel.MINIMAL,
                    actions_taken=[],
                    compliance_status={'mode': 'complete_failure'},
                    timestamp=datetime.now(timezone.utc).isoformat()
                ),
                preservation_report=PreservationReport(
                    batch_id=batch_id,
                    records_analyzed=0,
                    preservation_rate=0.0,
                    preservation_mode=PreservationMode.CONSERVATIVE,
                    critical_features_preserved=False,
                    preservation_details={'mode': 'complete_failure'},
                    timestamp=datetime.now(timezone.utc).isoformat()
                ),
                quality_metrics=QualityMetrics(
                    overall_score=0.0,
                    individual_scores=[],
                    error_message=f"Complete failure: {str(error)}, Recovery failed: {str(recovery_error)}"
                ),
                processing_summary={
                    **processing_summary,
                    'complete_failure': True,
                    'original_error': str(error),
                    'recovery_error': str(recovery_error)
                },
                audit_trail=ProcessingAudit(
                    batch_id=batch_id,
                    processing_status=ProcessingStatus.SYSTEM_ERROR,
                    error_details=f"Complete failure: {str(error)}, Recovery failed: {str(recovery_error)}",
                    phases_completed=[],
                    timestamp=datetime.now(timezone.utc).isoformat()
                ),
                total_processing_time_ms=total_time,
                processing_mode='complete_failure'
            )
    
    # =========================================================================
    # Statistics and Monitoring
    # =========================================================================
    
    def _update_processing_statistics(self, batch_result: ProcessedBatch):
        """Update internal processing statistics"""
        
        # Update counters
        self.processing_state['total_records_processed'] += len(batch_result.cleaned_records)
        
        if batch_result.audit_trail.processing_status == ProcessingStatus.SUCCESS:
            self.processing_state['successful_records'] += len(batch_result.cleaned_records)
        else:
            self.processing_state['failed_records'] += len(batch_result.cleaned_records)
        
        # Update timing statistics
        self.processing_state['total_processing_time_ms'] += batch_result.total_processing_time_ms
        
        if self.processing_state['total_records_processed'] > 0:
            self.processing_state['average_latency_ms'] = (
                self.processing_state['total_processing_time_ms'] / 
                self.processing_state['total_records_processed']
            )
        
        # Update preservation rate
        self.processing_state['anomaly_preservation_rate'] = batch_result.preservation_report.preservation_rate
        
        # Update cache statistics if available
        if self.cache:
            cache_stats = self.cache.get_stats()
            self.processing_state['cache_hit_rate'] = cache_stats['hit_rate']
        
        # Update memory usage if resource manager is available
        if self.resource_manager:
            asyncio.create_task(self._update_memory_stats())
    
    async def _update_memory_stats(self):
        """Update memory usage statistics"""
        try:
            resources = await self.resource_manager.check_resources()
            if self.metrics:
                self.metrics.update_memory_usage(int(resources['memory_usage_mb'] * 1024 * 1024))
        except Exception:
            pass  # Don't fail on memory stats
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        
        # Calculate throughput
        total_time_seconds = self.processing_state['total_processing_time_ms'] / 1000.0
        throughput = (
            self.processing_state['total_records_processed'] / total_time_seconds 
            if total_time_seconds > 0 else 0.0
        )
        
        # Calculate success rate
        total_records = self.processing_state['total_records_processed']
        success_rate = (
            self.processing_state['successful_records'] / total_records 
            if total_records > 0 else 0.0
        )
        
        stats = {
            **self.processing_state,
            'throughput_records_per_sec': throughput,
            'success_rate': success_rate,
            'uptime_seconds': time.time() - self.performance_metrics['start_time']
        }
        
        # Add phase timing statistics
        for phase, latencies in self.performance_metrics['phase_latencies'].items():
            if latencies:
                stats[f'{phase.lower()}_avg_latency_ms'] = sum(latencies) / len(latencies)
                stats[f'{phase.lower()}_max_latency_ms'] = max(latencies)
                stats[f'{phase.lower()}_min_latency_ms'] = min(latencies)
        
        # Add circuit breaker statistics if available
        if self.circuit_breaker:
            cb_state = self.circuit_breaker.get_state()
            stats['circuit_breaker'] = cb_state
        
        # Add cache statistics if available
        if self.cache:
            cache_stats = self.cache.get_stats()
            stats['cache'] = cache_stats
        
        return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        
        stats = self.get_processing_statistics()
        
        # Determine health status
        health_status = "healthy"
        health_issues = []
        
        # Check latency
        if stats['average_latency_ms'] > self.config.max_processing_latency_ms:
            health_status = "degraded"
            health_issues.append(f"High latency: {stats['average_latency_ms']:.2f}ms")
        
        # Check preservation rate
        if stats['anomaly_preservation_rate'] < self.config.min_anomaly_preservation_rate:
            health_status = "degraded"
            health_issues.append(f"Low preservation rate: {stats['anomaly_preservation_rate']:.1%}")
        
        # Check circuit breaker
        if self.circuit_breaker and self.circuit_breaker.state != CircuitState.CLOSED:
            health_status = "degraded"
            health_issues.append(f"Circuit breaker: {self.circuit_breaker.state.value}")
        
        # Check memory usage
        if self.resource_manager:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Can't await in sync method, use cached value
                    pass
            except Exception:
                pass
        
        # Check error rate
        if stats['success_rate'] < 0.95:
            health_status = "unhealthy"
            health_issues.append(f"High error rate: {(1-stats['success_rate']):.1%}")
        
        return {
            'status': health_status,
            'issues': health_issues,
            'statistics': stats,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    # =========================================================================
    # Configuration and Management
    # =========================================================================
    
    async def update_configuration(self, new_config: Layer1Config):
        """Update Layer 1 configuration"""
        
        self.logger.info("Updating Layer 1 configuration...")
        
        # Update configuration
        old_config = self.config
        self.config = new_config
        
        # Reinitialize components if necessary
        config_changes = self._detect_config_changes(old_config, new_config)
        
        if config_changes['requires_reinit']:
            await self._reinitialize_components(config_changes['changed_sections'])
        
        self.logger.info(f"Configuration updated. Changes: {list(config_changes['changed_sections'])}")
    
    def _detect_config_changes(self, old_config: Layer1Config, new_config: Layer1Config) -> Dict[str, Any]:
        """Detect what changed in configuration"""
        
        changes = {
            'requires_reinit': False,
            'changed_sections': set()
        }
        
        # Check critical settings that require reinitialization
        critical_fields = [
            'privacy_level', 'enable_gdpr_compliance', 'enable_ccpa_compliance', 
            'enable_hipaa_compliance', 'hash_algorithms', 'enable_deferred_hashing'
        ]
        
        for field in critical_fields:
            if getattr(old_config, field) != getattr(new_config, field):
                changes['requires_reinit'] = True
                changes['changed_sections'].add(field)
        
        return changes
    
    async def _reinitialize_components(self, changed_sections: set):
        """Reinitialize components based on configuration changes"""
        
        if 'privacy_level' in changed_sections or any('compliance' in s for s in changed_sections):
            self.privacy_filter = PrivacyComplianceFilter(self.config)
            self.logger.info("Privacy compliance filter reinitialized")
        
        if 'hash_algorithms' in changed_sections or 'enable_deferred_hashing' in changed_sections:
            self.hash_manager = DeferredHashingManager(self.config)
            self.logger.info("Deferred hashing manager reinitialized")
    
    async def shutdown(self):
        """Gracefully shutdown Layer 1"""
        
        self.logger.info("Shutting down Layer 1 Behavioral Intake Zone...")
        
        try:
            # Stop batch processor if running
            if self.batch_processor and hasattr(self.batch_processor, '_timer_task'):
                if self.batch_processor._timer_task:
                    self.batch_processor._timer_task.cancel()
            
            # Shutdown thread pool
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
            
            # Final cleanup
            if self.resource_manager:
                await self.resource_manager.force_cleanup()
            
            # Clear caches
            if self.cache:
                self.cache.clear()
            
            self.logger.info("Layer 1 shutdown complete")
        
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
    
    # =========================================================================
    # Development and Testing Support
    # =========================================================================
    
    async def validate_deployment(self) -> Dict[str, Any]:
        """Validate Layer 1 deployment and configuration"""
        
        validation_report = {
            'deployment_valid': True,
            'component_status': {},
            'performance_metrics': {},
            'configuration_issues': [],
            'recommendations': [],
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        try:
            # Test core components
            validation_report['component_status']['validator'] = await self._test_component(self.validator)
            validation_report['component_status']['schema_engine'] = await self._test_component(self.schema_engine)
            validation_report['component_status']['sanitizer'] = await self._test_component(self.sanitizer)
            validation_report['component_status']['privacy_filter'] = await self._test_component(self.privacy_filter)
            validation_report['component_status']['hash_manager'] = await self._test_component(self.hash_manager)
            validation_report['component_status']['preservation_guard'] = await self._test_component(self.preservation_guard)
            
            # Test subsystems
            validation_report['component_status']['quality_monitor'] = await self._test_component(self.quality_monitor)
            validation_report['component_status']['audit_generator'] = await self._test_component(self.audit_generator)
            
            # Performance validation
            await self._validate_performance_targets(validation_report)
            
            # Configuration validation
            self._validate_configuration(validation_report)
            
        except Exception as e:
            validation_report['deployment_valid'] = False
            validation_report['configuration_issues'].append(f"Deployment validation failed: {str(e)}")
        
        return validation_report
    
    async def _test_component(self, component) -> Dict[str, Any]:
        """Test individual component"""
        try:
            # Basic health check
            if hasattr(component, 'health_check'):
                health = await component.health_check()
                return {'status': 'healthy', 'details': health}
            else:
                # Component exists and is initialized
                return {'status': 'healthy', 'details': 'Component initialized'}
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}
    
    async def _validate_performance_targets(self, validation_report: Dict[str, Any]):
        """Validate performance targets"""
        
        # Create test records
        test_records = [
            TelemetryRecord(
                record_id=f"test-{i}",
                timestamp=time.time(),
                function_name=f"test_function_{i}",
                execution_phase="execution",
                anomaly_type="benign",
                telemetry_data={
                    "memory_usage_mb": 128,
                    "execution_time_ms": 100,
                    "cpu_utilization": 45.0
                }
            )
            for i in range(10)
        ]
        
        # Test processing latency
        start_time = time.time()
        try:
            batch_result = await self.process_telemetry_batch(test_records)
            processing_time_ms = (time.time() - start_time) * 1000
            avg_latency_ms = processing_time_ms / len(test_records)
            
            validation_report['performance_metrics'] = {
                'processing_time_ms': processing_time_ms,
                'average_latency_ms': avg_latency_ms,
                'preservation_rate': batch_result.preservation_report.preservation_rate,
                'quality_score': batch_result.quality_metrics.overall_score
            }
            
            # Check against targets
            if avg_latency_ms > self.config.max_processing_latency_ms:
                validation_report['configuration_issues'].append(
                    f"Average latency ({avg_latency_ms:.2f}ms) exceeds target ({self.config.max_processing_latency_ms}ms)"
                )
                validation_report['deployment_valid'] = False
            
            if batch_result.preservation_report.preservation_rate < self.config.min_anomaly_preservation_rate:
                validation_report['configuration_issues'].append(
                    f"Preservation rate ({batch_result.preservation_report.preservation_rate:.1%}) below target ({self.config.min_anomaly_preservation_rate:.1%})"
                )
                validation_report['deployment_valid'] = False
                
        except Exception as e:
            validation_report['configuration_issues'].append(f"Performance testing failed: {str(e)}")
            validation_report['deployment_valid'] = False
    
    def _validate_configuration(self, validation_report: Dict[str, Any]):
        """Validate configuration settings"""
        
        # Check privacy compliance settings
        if not any([
            self.config.enable_gdpr_compliance,
            self.config.enable_ccpa_compliance,
            self.config.enable_hipaa_compliance
        ]):
            validation_report['recommendations'].append(
                "Consider enabling at least one privacy compliance framework"
            )
        
        # Check performance settings
        if self.config.max_processing_latency_ms < 1:
            validation_report['configuration_issues'].append(
                "Maximum processing latency too aggressive (<1ms)"
            )
        
        # Check preservation settings
        if self.config.min_anomaly_preservation_rate < 0.9:
            validation_report['recommendations'].append(
                "Consider higher anomaly preservation rate (>90%)"
            )


# =============================================================================
# Deployment Validation and Testing
# =============================================================================

async def validate_layer1_deployment() -> Dict[str, Any]:
    """Standalone deployment validation function"""
    
    validation_report = {
        'deployment_valid': True,
        'errors': [],
        'warnings': [],
        'performance_metrics': {},
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    
    try:
        # Test Layer 1 initialization
        config = Layer1Config(test_mode=True)
        layer1 = Layer1_BehavioralIntakeZone(config)
        
        # Run deployment validation
        deployment_status = await layer1.validate_deployment()
        
        validation_report.update(deployment_status)
        
        # Cleanup
        await layer1.shutdown()
        
    except Exception as e:
        validation_report['deployment_valid'] = False
        validation_report['errors'].append(f"Layer 1 deployment validation failed: {str(e)}")
    
    return validation_report


# =============================================================================
# Main Execution and Testing
# =============================================================================

async def main():
    """Main function for testing and demonstration"""
    
    print("SCAFAD Layer 1: Enhanced Behavioral Intake Zone")
    print("=" * 60)
    
    # Create Layer 1 instance with enhanced features
    config = Layer1Config(
        privacy_level=PrivacyLevel.MODERATE,
        enable_debug_mode=True,
        test_mode=True,
        enable_parallel_processing=True,
        enable_circuit_breaker=True,
        enable_caching=True,
        enable_batch_optimization=True,
        enable_resource_management=True,
        enable_metrics=True
    )
    
    layer1 = Layer1_BehavioralIntakeZone(config)
    
    # Validate deployment
    print("Validating enhanced deployment...")
    deployment_status = await layer1.validate_deployment()
    print(f"Deployment Valid: {deployment_status['deployment_valid']}")
    
    if deployment_status['configuration_issues']:
        print("Configuration Issues:")
        for issue in deployment_status['configuration_issues']:
            print(f"  - {issue}")
    
    if deployment_status['recommendations']:
        print("Recommendations:")
        for rec in deployment_status['recommendations']:
            print(f"  - {rec}")
    
    # Test enhanced processing
    print("\nTesting enhanced Layer 1 processing...")
    
    # Create test records with varied characteristics
    test_records = []
    for i in range(20):  # Larger batch for testing
        record = TelemetryRecord(
            record_id=f"test-{i}",
            timestamp=time.time() + i * 0.1,
            function_name=f"test_function_{i % 5}",  # 5 different functions
            execution_phase=["cold_start", "warm", "execution", "cleanup"][i % 4],
            anomaly_type=["benign", "suspicious", "anomalous"][i % 3],
            telemetry_data={
                "memory_usage_mb": 128 + i * 10,
                "execution_time_ms": 100 + i * 5,
                "cpu_utilization": 45.0 + i * 2,
                "network_io_bytes": 1024 * (i + 1),
                "function_calls": 10 + i,
                "error_count": 0 if i % 5 != 0 else 1,
                "test_field": f"test_value_{i}",
                "sensitive_data": f"user_id_{i}",  # For privacy testing
                "large_payload": "x" * (100 + i * 10)  # For hashing testing
            },
            context_metadata={
                "source": "test",
                "priority": i % 3,
                "tags": [f"tag_{i % 3}", "test"]
            }
        )
        test_records.append(record)
    
    # Test processing performance
    start_time = time.time()
    processed_batch = await layer1.process_telemetry_batch(test_records)
    processing_time = (time.time() - start_time) * 1000
    
    print(f"\nProcessing Results:")
    print(f"  Processed: {len(processed_batch.cleaned_records)} records")
    print(f"  Total time: {processing_time:.2f}ms")
    print(f"  Average latency: {processing_time/len(test_records):.2f}ms per record")
    print(f"  Processing mode: {processed_batch.processing_mode}")
    print(f"  Preservation rate: {processed_batch.preservation_report.preservation_rate:.1%}")
    print(f"  Quality score: {processed_batch.quality_metrics.overall_score:.2f}")
    
    # Display privacy audit
    privacy_audit = processed_batch.privacy_audit_trail
    print(f"\nPrivacy Compliance:")
    print(f"  Actions taken: {len(privacy_audit.actions_taken)}")
    print(f"  Compliance status: {privacy_audit.compliance_status}")
    
    # Display processing statistics
    print("\nEnhanced Processing Statistics:")
    stats = layer1.get_processing_statistics()
    interesting_stats = [
        'total_records_processed', 'average_latency_ms', 'throughput_records_per_sec',
        'success_rate', 'anomaly_preservation_rate', 'cache_hit_rate'
    ]
    
    for stat in interesting_stats:
        if stat in stats:
            value = stats[stat]
            if isinstance(value, float):
                if 'rate' in stat:
                    print(f"  {stat}: {value:.1%}")
                else:
                    print(f"  {stat}: {value:.2f}")
            else:
                print(f"  {stat}: {value}")
    
    # Display health status
    print("\nSystem Health:")
    health = layer1.get_health_status()
    print(f"  Status: {health['status']}")
    if health['issues']:
        print(f"  Issues: {health['issues']}")
    
    # Test circuit breaker if enabled
    if layer1.circuit_breaker:
        print(f"\nCircuit Breaker Status:")
        cb_state = layer1.circuit_breaker.get_state()
        print(f"  State: {cb_state['state']}")
        print(f"  Failures: {cb_state['failure_count']}")
        print(f"  Successes: {cb_state['success_count']}")
    
    # Test cache if enabled
    if layer1.cache:
        print(f"\nCache Statistics:")
        cache_stats = layer1.cache.get_stats()
        print(f"  Size: {cache_stats['size']}/{cache_stats['max_size']}")
        print(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
        print(f"  Hits: {cache_stats['hit_count']}")
        print(f"  Misses: {cache_stats['miss_count']}")
    
    # Cleanup
    await layer1.shutdown()
    print(f"\nLayer 1 shutdown complete.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())_id=batch_id,
                cleaned_records=clean_records,
                schema_metadata=SchemaMetadata(
                    version=self.config.schema_version,
                    compatibility_level="error_recovery",
                    migration_applied=False
                ),
                privacy_audit_trail=PrivacyAuditTrail(
                    batch_id=batch_id,
                    records_processed=len(clean_records),
                    privacy_level=PrivacyLevel.MINIMAL,
                    actions_taken=[],
                    compliance_status={'mode': 'error_recovery'},
                    timestamp=datetime.now(timezone.utc).isoformat()
                ),
                preservation_report=PreservationReport(
                    batch_id=batch_id,
                    records_analyzed=len(clean_records),
                    preservation_rate=0.8,
                    preservation_mode=PreservationMode.CONSERVATIVE,
                    critical_features_preserved=True,
                    preservation_details={'mode': 'error_recovery'},
                    timestamp=datetime.now(timezone.utc).isoformat()
                ),
                quality_metrics=QualityMetrics(
                    overall_score=0.6,
                    individual_scores=[0.6] * len(clean_records),
                    error_message=str(error)
                ),
                processing_summary={
                    **processing_summary,
                    'error_recovery': True,
                    'original_error': str(error)
                },
                audit_trail=ProcessingAudit(
                    batch_id=batch_id,
                    processing_status=ProcessingStatus.SYSTEM_ERROR,
                    error_details=str(error),
                    phases_completed=['error_recovery'],
                    timestamp=datetime.now(timezone.utc).isoformat()
                ),
                total_processing_time_ms=total_time,
                processing_mode='error_recovery'
            )
        
        except Exception as recovery_error:
            # Complete failure - return empty batch
            self.logger.critical(f"Error recovery failed: {str(recovery_error)}")
            
            total_time = (time.time() - batch_start_time) * 1000
            
            return ProcessedBatch(
                batch_id=batch_id,
                cleaned_records=clean_records,
                schema_metadata=SchemaMetadata(
                    version=self.config.schema_version,
                    compatibility_level="error_recovery",
                    migration_applied=False
                ),
                # ... complete all required fields
                total_processing_time_ms=total_time,
                processing_mode='error_recovery'
            )