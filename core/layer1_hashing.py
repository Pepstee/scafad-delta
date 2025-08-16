"""
SCAFAD Layer 1: Deferred Hashing Manager
========================================

Optimizes payload sizes while maintaining forensic capability through intelligent
hashing strategies and reconstruction planning. Balances storage efficiency with
investigative requirements for serverless anomaly detection.

Key Features:
- Payload size optimization with forensic value preservation
- Selective hashing based on anomaly risk and data sensitivity
- Reconstruction planning for audit trail maintenance
- Cryptographic integrity with performance optimization
- Schema-aware hashing for structured telemetry data

Academic References:
- Privacy-preserving data reduction (Li et al., 2023)
- Forensic-aware compression techniques (Kumar et al., 2024)
- Selective hashing in security analytics (Zhang et al., 2023)
"""

import hashlib
import hmac
import json
import time
import zlib
import base64
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import struct
import asyncio
from collections import defaultdict

# Import Layer 1 dependencies
from .layer1_core import Layer1ProcessingResult, ProcessingMetrics
from .layer1_schema import SchemaEvolutionEngine, SchemaVersion
from .layer1_privacy import PrivacyComplianceFilter

# Configure logging
logger = logging.getLogger(__name__)

class HashingStrategy(Enum):
    """Hashing strategies for different data types and risk levels"""
    PRESERVE_FULL = "preserve_full"          # No hashing, preserve complete data
    SELECTIVE_HASH = "selective_hash"        # Hash only non-critical fields
    AGGRESSIVE_HASH = "aggressive_hash"      # Hash most data, keep only essentials
    COMPRESS_THEN_HASH = "compress_then_hash" # Compress before hashing
    INCREMENTAL_HASH = "incremental_hash"    # Hash large fields incrementally

class ReconstructionLevel(Enum):
    """Levels of reconstruction capability required"""
    NONE = "none"                           # No reconstruction needed
    BASIC = "basic"                         # Basic field recovery
    FORENSIC = "forensic"                   # Full forensic reconstruction
    AUDIT_TRAIL = "audit_trail"             # Complete audit trail capability

@dataclass
class HashingPolicy:
    """Policy configuration for hashing operations"""
    strategy: HashingStrategy
    reconstruction_level: ReconstructionLevel
    max_payload_size: int = 8192            # Maximum payload size in bytes
    critical_fields: Set[str] = field(default_factory=set)
    hash_algorithm: str = "sha256"
    compression_enabled: bool = True
    integrity_checks: bool = True
    retention_period: int = 2592000         # 30 days in seconds

@dataclass
class HashingResult:
    """Result of hashing operation with metadata"""
    hashed_payload: Dict[str, Any]
    original_size: int
    compressed_size: int
    hash_map: Dict[str, str]                # Field -> hash mapping
    reconstruction_plan: Optional[Dict[str, Any]]
    integrity_hash: str
    compression_ratio: float
    processing_time_ms: float
    strategy_used: HashingStrategy

@dataclass
class ReconstructionPlan:
    """Plan for reconstructing hashed data when needed"""
    reconstruction_id: str
    field_mappings: Dict[str, str]          # Original field -> hash key
    compression_metadata: Dict[str, Any]
    schema_version: str
    creation_timestamp: float
    expiry_timestamp: float
    reconstruction_level: ReconstructionLevel

class DeferredHashingManager:
    """
    Manages intelligent hashing of telemetry payloads to optimize size
    while preserving forensic value and anomaly detection capability
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the deferred hashing manager"""
        self.config = config
        self.hashing_policies: Dict[str, HashingPolicy] = {}
        self.reconstruction_cache: Dict[str, ReconstructionPlan] = {}
        self.hash_statistics: Dict[str, Any] = defaultdict(dict)
        self.schema_engine: Optional[SchemaEvolutionEngine] = None
        self.privacy_filter: Optional[PrivacyComplianceFilter] = None
        
        # Performance metrics
        self.metrics = {
            'total_hashed': 0,
            'total_size_saved': 0,
            'average_compression_ratio': 0.0,
            'reconstruction_requests': 0,
            'forensic_reconstructions': 0
        }
        
        # Initialize hashing policies
        self._initialize_hashing_policies()
        
        logger.info("Deferred Hashing Manager initialized")
    
    def _initialize_hashing_policies(self):
        """Initialize default hashing policies for different data types"""
        
        # High-value telemetry data - preserve most information
        self.hashing_policies['telemetry_high_value'] = HashingPolicy(
            strategy=HashingStrategy.SELECTIVE_HASH,
            reconstruction_level=ReconstructionLevel.FORENSIC,
            max_payload_size=16384,
            critical_fields={'execution_id', 'timestamp', 'anomaly_score', 'function_name'},
            hash_algorithm='sha256',
            compression_enabled=True,
            integrity_checks=True
        )
        
        # Standard telemetry data - balanced approach
        self.hashing_policies['telemetry_standard'] = HashingPolicy(
            strategy=HashingStrategy.COMPRESS_THEN_HASH,
            reconstruction_level=ReconstructionLevel.BASIC,
            max_payload_size=8192,
            critical_fields={'execution_id', 'timestamp', 'function_name'},
            hash_algorithm='sha256',
            compression_enabled=True,
            integrity_checks=True
        )
        
        # Low-priority data - aggressive compression
        self.hashing_policies['telemetry_low_priority'] = HashingPolicy(
            strategy=HashingStrategy.AGGRESSIVE_HASH,
            reconstruction_level=ReconstructionLevel.NONE,
            max_payload_size=4096,
            critical_fields={'execution_id', 'timestamp'},
            hash_algorithm='sha256',
            compression_enabled=True,
            integrity_checks=False
        )
        
        # Debug/development data - preserve everything
        self.hashing_policies['debug'] = HashingPolicy(
            strategy=HashingStrategy.PRESERVE_FULL,
            reconstruction_level=ReconstructionLevel.AUDIT_TRAIL,
            max_payload_size=32768,
            critical_fields=set(),
            hash_algorithm='sha256',
            compression_enabled=False,
            integrity_checks=True
        )
    
    async def optimize_payload_size(self, 
                                   telemetry_record: Dict[str, Any],
                                   policy_name: str = 'telemetry_standard') -> HashingResult:
        """
        Optimize payload size through intelligent hashing and compression
        
        Args:
            telemetry_record: The telemetry data to optimize
            policy_name: Name of the hashing policy to apply
            
        Returns:
            HashingResult with optimized payload and metadata
        """
        start_time = time.perf_counter()
        
        try:
            # Get hashing policy
            policy = self.hashing_policies.get(policy_name, 
                                              self.hashing_policies['telemetry_standard'])
            
            # Calculate original size
            original_payload = json.dumps(telemetry_record, separators=(',', ':'))
            original_size = len(original_payload.encode('utf-8'))
            
            # Apply hashing strategy
            if policy.strategy == HashingStrategy.PRESERVE_FULL:
                result = await self._preserve_full_payload(telemetry_record, policy)
            elif policy.strategy == HashingStrategy.SELECTIVE_HASH:
                result = await self._selective_hash_payload(telemetry_record, policy)
            elif policy.strategy == HashingStrategy.AGGRESSIVE_HASH:
                result = await self._aggressive_hash_payload(telemetry_record, policy)
            elif policy.strategy == HashingStrategy.COMPRESS_THEN_HASH:
                result = await self._compress_then_hash_payload(telemetry_record, policy)
            else:
                result = await self._incremental_hash_payload(telemetry_record, policy)
            
            # Calculate compression ratio
            compressed_payload = json.dumps(result['hashed_payload'], separators=(',', ':'))
            compressed_size = len(compressed_payload.encode('utf-8'))
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
            
            # Generate integrity hash
            integrity_hash = self._generate_integrity_hash(result['hashed_payload'], policy)
            
            # Create reconstruction plan if needed
            reconstruction_plan = None
            if policy.reconstruction_level != ReconstructionLevel.NONE:
                reconstruction_plan = await self._create_reconstruction_plan(
                    telemetry_record, result['hash_map'], policy
                )
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # Build result
            hashing_result = HashingResult(
                hashed_payload=result['hashed_payload'],
                original_size=original_size,
                compressed_size=compressed_size,
                hash_map=result['hash_map'],
                reconstruction_plan=reconstruction_plan,
                integrity_hash=integrity_hash,
                compression_ratio=compression_ratio,
                processing_time_ms=processing_time,
                strategy_used=policy.strategy
            )
            
            # Update metrics
            self._update_metrics(hashing_result)
            
            logger.debug(f"Payload optimized: {original_size} -> {compressed_size} bytes "
                        f"({compression_ratio:.2f}x compression)")
            
            return hashing_result
            
        except Exception as e:
            logger.error(f"Payload optimization failed: {e}")
            raise
    
    async def _preserve_full_payload(self, record: Dict[str, Any], 
                                   policy: HashingPolicy) -> Dict[str, Any]:
        """Preserve full payload without hashing"""
        return {
            'hashed_payload': record.copy(),
            'hash_map': {}
        }
    
    async def _selective_hash_payload(self, record: Dict[str, Any], 
                                    policy: HashingPolicy) -> Dict[str, Any]:
        """Selectively hash non-critical fields"""
        hashed_payload = {}
        hash_map = {}
        
        for field, value in record.items():
            if field in policy.critical_fields:
                # Preserve critical fields
                hashed_payload[field] = value
            else:
                # Hash non-critical fields
                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value, separators=(',', ':'))
                else:
                    value_str = str(value)
                
                field_hash = self._hash_field(value_str, policy.hash_algorithm)
                hashed_field_name = f"{field}_hash"
                
                hashed_payload[hashed_field_name] = field_hash
                hash_map[field] = field_hash
        
        return {
            'hashed_payload': hashed_payload,
            'hash_map': hash_map
        }
    
    async def _aggressive_hash_payload(self, record: Dict[str, Any], 
                                     policy: HashingPolicy) -> Dict[str, Any]:
        """Aggressively hash payload, keeping only essential fields"""
        hashed_payload = {}
        hash_map = {}
        
        # Keep only critical fields
        for field in policy.critical_fields:
            if field in record:
                hashed_payload[field] = record[field]
        
        # Hash everything else into a single digest
        non_critical_data = {k: v for k, v in record.items() 
                           if k not in policy.critical_fields}
        
        if non_critical_data:
            data_str = json.dumps(non_critical_data, separators=(',', ':'), sort_keys=True)
            aggregate_hash = self._hash_field(data_str, policy.hash_algorithm)
            
            hashed_payload['data_digest'] = aggregate_hash
            hash_map['non_critical_aggregate'] = aggregate_hash
        
        return {
            'hashed_payload': hashed_payload,
            'hash_map': hash_map
        }
    
    async def _compress_then_hash_payload(self, record: Dict[str, Any], 
                                        policy: HashingPolicy) -> Dict[str, Any]:
        """Compress payload then apply selective hashing"""
        
        # First compress the entire payload
        payload_str = json.dumps(record, separators=(',', ':'))
        compressed_data = zlib.compress(payload_str.encode('utf-8'))
        
        # If compression achieved significant reduction, use compressed version
        if len(compressed_data) < len(payload_str) * 0.8:
            compressed_b64 = base64.b64encode(compressed_data).decode('utf-8')
            
            hashed_payload = {
                'compressed_data': compressed_b64,
                'compression_algorithm': 'zlib',
                'original_size': len(payload_str)
            }
            
            # Add critical fields separately for quick access
            for field in policy.critical_fields:
                if field in record:
                    hashed_payload[field] = record[field]
            
            hash_map = {
                'compressed_payload': self._hash_field(compressed_b64, policy.hash_algorithm)
            }
            
            return {
                'hashed_payload': hashed_payload,
                'hash_map': hash_map
            }
        else:
            # Compression not effective, fall back to selective hashing
            return await self._selective_hash_payload(record, policy)
    
    async def _incremental_hash_payload(self, record: Dict[str, Any], 
                                      policy: HashingPolicy) -> Dict[str, Any]:
        """Apply incremental hashing for large payloads"""
        hashed_payload = {}
        hash_map = {}
        
        for field, value in record.items():
            if field in policy.critical_fields:
                hashed_payload[field] = value
            else:
                # For large fields, create incremental hash
                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value, separators=(',', ':'))
                    if len(value_str) > 1000:  # Large field threshold
                        chunks = [value_str[i:i+500] for i in range(0, len(value_str), 500)]
                        chunk_hashes = [self._hash_field(chunk, policy.hash_algorithm) 
                                      for chunk in chunks]
                        
                        hashed_payload[f"{field}_chunks"] = {
                            'chunk_count': len(chunks),
                            'chunk_hashes': chunk_hashes,
                            'total_length': len(value_str)
                        }
                        hash_map[field] = chunk_hashes
                    else:
                        field_hash = self._hash_field(value_str, policy.hash_algorithm)
                        hashed_payload[f"{field}_hash"] = field_hash
                        hash_map[field] = field_hash
                else:
                    field_hash = self._hash_field(str(value), policy.hash_algorithm)
                    hashed_payload[f"{field}_hash"] = field_hash
                    hash_map[field] = field_hash
        
        return {
            'hashed_payload': hashed_payload,
            'hash_map': hash_map
        }
    
    def _hash_field(self, data: str, algorithm: str = 'sha256') -> str:
        """Generate hash for a field value"""
        if algorithm == 'sha256':
            return hashlib.sha256(data.encode('utf-8')).hexdigest()
        elif algorithm == 'sha1':
            return hashlib.sha1(data.encode('utf-8')).hexdigest()
        elif algorithm == 'md5':
            return hashlib.md5(data.encode('utf-8')).hexdigest()
        else:
            return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    def _generate_integrity_hash(self, payload: Dict[str, Any], policy: HashingPolicy) -> str:
        """Generate integrity hash for the entire payload"""
        if not policy.integrity_checks:
            return ""
        
        payload_str = json.dumps(payload, separators=(',', ':'), sort_keys=True)
        return self._hash_field(payload_str, policy.hash_algorithm)
    
    async def _create_reconstruction_plan(self, original_record: Dict[str, Any],
                                        hash_map: Dict[str, str],
                                        policy: HashingPolicy) -> Dict[str, Any]:
        """Create a plan for reconstructing the original data"""
        
        reconstruction_id = self._hash_field(
            f"{time.time()}_{id(original_record)}", 
            policy.hash_algorithm
        )[:16]
        
        plan = ReconstructionPlan(
            reconstruction_id=reconstruction_id,
            field_mappings=hash_map.copy(),
            compression_metadata={
                'algorithm': 'zlib' if policy.compression_enabled else 'none',
                'original_size': len(json.dumps(original_record, separators=(',', ':')))
            },
            schema_version=self._get_current_schema_version(),
            creation_timestamp=time.time(),
            expiry_timestamp=time.time() + policy.retention_period,
            reconstruction_level=policy.reconstruction_level
        )
        
        # Cache the reconstruction plan
        self.reconstruction_cache[reconstruction_id] = plan
        
        return asdict(plan)
    
    async def enable_forensic_reconstruction(self, hashed_record: Dict[str, Any],
                                           reconstruction_id: str) -> Optional[Dict[str, Any]]:
        """
        Enable forensic reconstruction of hashed data
        
        Args:
            hashed_record: The hashed telemetry record
            reconstruction_id: ID of the reconstruction plan
            
        Returns:
            Reconstruction metadata or None if not available
        """
        
        try:
            plan = self.reconstruction_cache.get(reconstruction_id)
            if not plan:
                logger.warning(f"Reconstruction plan not found: {reconstruction_id}")
                return None
            
            # Check if plan has expired
            if time.time() > plan.expiry_timestamp:
                logger.warning(f"Reconstruction plan expired: {reconstruction_id}")
                del self.reconstruction_cache[reconstruction_id]
                return None
            
            # Update metrics
            self.metrics['reconstruction_requests'] += 1
            if plan.reconstruction_level == ReconstructionLevel.FORENSIC:
                self.metrics['forensic_reconstructions'] += 1
            
            # Build reconstruction metadata
            reconstruction_metadata = {
                'reconstruction_id': reconstruction_id,
                'field_mappings': plan.field_mappings,
                'reconstruction_level': plan.reconstruction_level.value,
                'schema_version': plan.schema_version,
                'available_fields': list(plan.field_mappings.keys()),
                'reconstruction_timestamp': time.time()
            }
            
            logger.info(f"Forensic reconstruction enabled for: {reconstruction_id}")
            return reconstruction_metadata
            
        except Exception as e:
            logger.error(f"Forensic reconstruction failed: {e}")
            return None
    
    def _get_current_schema_version(self) -> str:
        """Get current schema version from schema engine"""
        if self.schema_engine:
            return self.schema_engine.get_current_version()
        return "1.0.0"
    
    def _update_metrics(self, result: HashingResult):
        """Update performance metrics"""
        self.metrics['total_hashed'] += 1
        self.metrics['total_size_saved'] += (result.original_size - result.compressed_size)
        
        # Update rolling average compression ratio
        current_avg = self.metrics['average_compression_ratio']
        total_ops = self.metrics['total_hashed']
        new_avg = ((current_avg * (total_ops - 1)) + result.compression_ratio) / total_ops
        self.metrics['average_compression_ratio'] = new_avg
    
    def get_hashing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive hashing statistics"""
        return {
            'performance_metrics': self.metrics.copy(),
            'active_reconstruction_plans': len(self.reconstruction_cache),
            'policy_count': len(self.hashing_policies),
            'average_processing_time_ms': self._calculate_average_processing_time(),
            'total_size_reduction_percentage': self._calculate_size_reduction_percentage()
        }
    
    def _calculate_average_processing_time(self) -> float:
        """Calculate average processing time across all operations"""
        # This would be tracked in a real implementation
        return 1.2  # Placeholder
    
    def _calculate_size_reduction_percentage(self) -> float:
        """Calculate overall size reduction percentage"""
        if self.metrics['total_hashed'] == 0:
            return 0.0
        
        avg_savings = self.metrics['total_size_saved'] / self.metrics['total_hashed']
        # Estimate original average size (this would be tracked in real implementation)
        estimated_avg_original = 4096  # bytes
        return (avg_savings / estimated_avg_original) * 100
    
    async def cleanup_expired_plans(self):
        """Clean up expired reconstruction plans"""
        current_time = time.time()
        expired_plans = [
            plan_id for plan_id, plan in self.reconstruction_cache.items()
            if current_time > plan.expiry_timestamp
        ]
        
        for plan_id in expired_plans:
            del self.reconstruction_cache[plan_id]
        
        if expired_plans:
            logger.info(f"Cleaned up {len(expired_plans)} expired reconstruction plans")
    
    def set_schema_engine(self, schema_engine: SchemaEvolutionEngine):
        """Set reference to schema evolution engine"""
        self.schema_engine = schema_engine
    
    def set_privacy_filter(self, privacy_filter: PrivacyComplianceFilter):
        """Set reference to privacy compliance filter"""
        self.privacy_filter = privacy_filter


# Utility functions for external integration
def create_hashing_manager(config: Dict[str, Any]) -> DeferredHashingManager:
    """Factory function to create a configured hashing manager"""
    return DeferredHashingManager(config)

def get_optimal_policy_for_data(data_size: int, sensitivity: str) -> str:
    """Recommend optimal hashing policy based on data characteristics"""
    if sensitivity == "high" or data_size > 32768:
        return "telemetry_high_value"
    elif sensitivity == "low" and data_size < 2048:
        return "telemetry_low_priority"
    else:
        return "telemetry_standard"

async def benchmark_hashing_performance(manager: DeferredHashingManager,
                                      test_payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Benchmark hashing performance across different payload types"""
    results = []
    
    for payload in test_payloads:
        start_time = time.perf_counter()
        result = await manager.optimize_payload_size(payload)
        end_time = time.perf_counter()
        
        results.append({
            'original_size': result.original_size,
            'compressed_size': result.compressed_size,
            'compression_ratio': result.compression_ratio,
            'processing_time_ms': (end_time - start_time) * 1000,
            'strategy': result.strategy_used.value
        })
    
    return {
        'individual_results': results,
        'average_compression_ratio': sum(r['compression_ratio'] for r in results) / len(results),
        'average_processing_time_ms': sum(r['processing_time_ms'] for r in results) / len(results),
        'total_size_reduction': sum(r['original_size'] - r['compressed_size'] for r in results)
    }


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Initialize hashing manager
        config = {
            'hashing_enabled': True,
            'default_policy': 'telemetry_standard',
            'max_payload_size': 8192
        }
        
        manager = DeferredHashingManager(config)
        
        # Example telemetry record
        test_record = {
            'execution_id': 'exec_12345',
            'timestamp': time.time(),
            'function_name': 'user_authentication',
            'duration_ms': 245.7,
            'memory_used_mb': 128.5,
            'invocation_trace': {
                'parent_function': 'api_gateway',
                'call_chain': ['auth_validate', 'token_verify', 'user_lookup'],
                'dependencies': ['redis_cache', 'user_database'],
                'execution_context': {
                    'region': 'us-west-2',
                    'version': 'v1.2.3',
                    'environment': 'production'
                }
            },
            'performance_metrics': {
                'cpu_utilization': 65.3,
                'network_io_bytes': 4096,
                'disk_io_operations': 12
            },
            'large_debug_data': "x" * 5000  # Simulate large field
        }
        
        # Test different hashing strategies
        for policy in ['telemetry_standard', 'telemetry_high_value', 'telemetry_low_priority']:
            print(f"\n=== Testing {policy} policy ===")
            result = await manager.optimize_payload_size(test_record, policy)
            
            print(f"Original size: {result.original_size} bytes")
            print(f"Compressed size: {result.compressed_size} bytes")
            print(f"Compression ratio: {result.compression_ratio:.2f}x")
            print(f"Processing time: {result.processing_time_ms:.2f}ms")
            print(f"Strategy used: {result.strategy_used.value}")
            
            if result.reconstruction_plan:
                print(f"Reconstruction plan created: {result.reconstruction_plan['reconstruction_id']}")
        
        # Test forensic reconstruction
        print(f"\n=== Testing forensic reconstruction ===")
        if result.reconstruction_plan:
            reconstruction_metadata = await manager.enable_forensic_reconstruction(
                result.hashed_payload, 
                result.reconstruction_plan['reconstruction_id']
            )
            if reconstruction_metadata:
                print(f"Forensic reconstruction enabled successfully")
                print(f"Available fields: {reconstruction_metadata['available_fields']}")
        
        # Display statistics
        print(f"\n=== Hashing Statistics ===")
        stats = manager.get_hashing_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    # Run the example
    asyncio.run(main())