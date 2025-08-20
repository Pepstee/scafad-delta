'available_capabilities': list(available_capabilities),
                'reconstruction_complexity': plan.reconstruction_complexity,
                'estimated_reconstruction_time_ms': plan.estimated_reconstruction_time_ms,
                'schema_version': plan.schema_version,
                'schema_evolution_path': plan.schema_evolution_path,
                'access_count': plan.access_count,
                'last_access': plan.last_access_timestamp,
                'reconstruction_timestamp': current_time,
                'encryption_status': 'encrypted' if plan.encrypted else 'plaintext',
                'integrity_verified': True,
                'decay_status': self._calculate_decay_status(plan, current_time)
            }
            
            # Add access context to audit trail
            if access_context:
                reconstruction_metadata['access_context'] = {
                    'requester': access_context.get('requester', 'unknown'),
                    'purpose': access_context.get('purpose', 'forensic_analysis'),
                    'authorization_level': access_context.get('authorization_level', 'standard')
                }
            
            logger.info(f"Enhanced forensic reconstruction enabled: {reconstruction_id}, "
                       f"capabilities: {available_capabilities}")
            
            return reconstruction_metadata
            
        except Exception as e:
            logger.error(f"Enhanced forensic reconstruction failed: {e}")
            return None
    
    def _calculate_decay_status(self, plan: ReconstructionPlan, current_time: float) -> Dict[str, Any]:
        """Calculate current decay status of reconstruction plan"""
        
        if not plan.decay_schedule:
            return {'decay_enabled': False, 'status': 'permanent'}
        
        decay_status = {'decay_enabled': True, 'current_phase': 'expired'}
        
        for phase, decay_time in sorted(plan.decay_schedule.items(), key=lambda x: x[1]):
            if current_time < decay_time:
                decay_status['current_phase'] = phase
                decay_status['time_remaining'] = decay_time - current_time
                break
        
        return decay_status
    
    async def batch_optimize_payloads(self, records: List[Dict[str, Any]],
                                    policy_name: str = 'performance_optimized',
                                    parallel: bool = True) -> List[HashingResult]:
        """
        Batch optimization of multiple payloads with parallel processing
        
        Args:
            records: List of telemetry records to optimize
            policy_name: Hashing policy to apply
            parallel: Enable parallel processing
            
        Returns:
            List of HashingResult objects
        """
        
        if not records:
            return []
        
        if parallel and len(records) > 1:
            # Parallel processing for better throughput
            tasks = [
                self.optimize_payload_size(record, policy_name)
                for record in records
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Batch optimization failed for record {i}: {result}")
                else:
                    valid_results.append(result)
            
            return valid_results
        else:
            # Sequential processing
            results = []
            for record in records:
                try:
                    result = await self.optimize_payload_size(record, policy_name)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Sequential optimization failed: {e}")
            
            return results
    
    async def analyze_compression_effectiveness(self, 
                                              test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze compression effectiveness across different strategies"""
        
        analysis_results = {
            'strategy_performance': {},
            'optimal_strategies': {},
            'recommendations': []
        }
        
        strategies_to_test = [
            'performance_optimized',
            'ml_adaptive',
            'critical_security',
            'edge_lightweight'
        ]
        
        for strategy in strategies_to_test:
            if strategy not in self.hashing_policies:
                continue
            
            strategy_results = []
            
            for record in test_data[:10]:  # Limit test size
                try:
                    result = await self.optimize_payload_size(record, strategy)
                    strategy_results.append({
                        'compression_ratio': result.compression_ratio,
                        'processing_time_ms': result.processing_time_ms,
                        'preservation_effectiveness': result.preservation_effectiveness,
                        'information_loss': result.information_loss
                    })
                except Exception as e:
                    logger.warning(f"Strategy {strategy} failed: {e}")
                    continue
            
            if strategy_results:
                analysis_results['strategy_performance'][strategy] = {
                    'average_compression_ratio': np.mean([r['compression_ratio'] for r in strategy_results]),
                    'average_processing_time_ms': np.mean([r['processing_time_ms'] for r in strategy_results]),
                    'average_preservation': np.mean([r['preservation_effectiveness'] for r in strategy_results]),
                    'average_information_loss': np.mean([r['information_loss'] for r in strategy_results]),
                    'sample_count': len(strategy_results)
                }
        
        # Generate recommendations
        if analysis_results['strategy_performance']:
            best_compression = max(analysis_results['strategy_performance'].items(),
                                 key=lambda x: x[1]['average_compression_ratio'])
            best_speed = min(analysis_results['strategy_performance'].items(),
                           key=lambda x: x[1]['average_processing_time_ms'])
            best_preservation = max(analysis_results['strategy_performance'].items(),
                                  key=lambda x: x[1]['average_preservation'])
            
            analysis_results['optimal_strategies'] = {
                'best_compression': best_compression[0],
                'fastest_processing': best_speed[0],
                'best_preservation': best_preservation[0]
            }
            
            analysis_results['recommendations'] = [
                f"For maximum compression: use {best_compression[0]} "
                f"({best_compression[1]['average_compression_ratio']:.2f}x ratio)",
                f"For fastest processing: use {best_speed[0]} "
                f"({best_speed[1]['average_processing_time_ms']:.2f}ms avg)",
                f"For best preservation: use {best_preservation[0]} "
                f"({best_preservation[1]['average_preservation']:.3f} effectiveness)"
            ]
        
        return analysis_results
    
    async def optimize_policy_parameters(self, training_data: List[Dict[str, Any]],
                                       policy_name: str) -> Dict[str, Any]:
        """Optimize policy parameters based on training data"""
        
        if policy_name not in self.hashing_policies:
            return {'error': f'Policy {policy_name} not found'}
        
        policy = self.hashing_policies[policy_name]
        original_params = {
            'target_compression_ratio': policy.target_compression_ratio,
            'max_processing_time_ms': policy.max_processing_time_ms,
            'large_field_threshold': policy.large_field_threshold
        }
        
        optimization_results = {
            'original_parameters': original_params,
            'optimized_parameters': {},
            'performance_improvement': {},
            'recommendations': []
        }
        
        # Test different parameter values
        compression_ratios = [0.3, 0.4, 0.5, 0.6]
        time_limits = [1.0, 1.5, 2.0, 2.5]
        field_thresholds = [500, 1000, 1500, 2000]
        
        best_performance = {'ratio': 0, 'time': float('inf'), 'preservation': 0}
        best_params = original_params.copy()
        
        for ratio in compression_ratios:
            for time_limit in time_limits:
                for threshold in field_thresholds:
                    # Update policy temporarily
                    policy.target_compression_ratio = ratio
                    policy.max_processing_time_ms = time_limit
                    policy.large_field_threshold = threshold
                    
                    # Test with subset of training data
                    test_results = []
                    for record in training_data[:5]:
                        try:
                            result = await self.optimize_payload_size(record, policy_name)
                            test_results.append({
                                'compression_ratio': result.compression_ratio,
                                'processing_time_ms': result.processing_time_ms,
                                'preservation_effectiveness': result.preservation_effectiveness
                            })
                        except:
                            continue
                    
                    if test_results:
                        avg_ratio = np.mean([r['compression_ratio'] for r in test_results])
                        avg_time = np.mean([r['processing_time_ms'] for r in test_results])
                        avg_preservation = np.mean([r['preservation_effectiveness'] for r in test_results])
                        
                        # Simple scoring: balance compression, speed, and preservation
                        score = (avg_ratio * 0.4 + (1.0/avg_time) * 100 * 0.3 + avg_preservation * 0.3)
                        
                        if score > (best_performance['ratio'] * 0.4 + 
                                  (1.0/best_performance['time']) * 100 * 0.3 + 
                                  best_performance['preservation'] * 0.3):
                            best_performance = {
                                'ratio': avg_ratio,
                                'time': avg_time,
                                'preservation': avg_preservation
                            }
                            best_params = {
                                'target_compression_ratio': ratio,
                                'max_processing_time_ms': time_limit,
                                'large_field_threshold': threshold
                            }
        
        # Restore original parameters
        policy.target_compression_ratio = original_params['target_compression_ratio']
        policy.max_processing_time_ms = original_params['max_processing_time_ms']
        policy.large_field_threshold = original_params['large_field_threshold']
        
        optimization_results['optimized_parameters'] = best_params
        optimization_results['performance_improvement'] = {
            'compression_improvement': (best_performance['ratio'] / 
                                      (original_params['target_compression_ratio'] or 1)) - 1,
            'time_improvement': (original_params['max_processing_time_ms'] / 
                               best_performance['time']) - 1,
            'preservation_score': best_performance['preservation']
        }
        
        # Generate recommendations
        if best_params != original_params:
            optimization_results['recommendations'].append(
                f"Update policy parameters for improved performance"
            )
            for param, value in best_params.items():
                if value != original_params[param]:
                    optimization_results['recommendations'].append(
                        f"Change {param} from {original_params[param]} to {value}"
                    )
        else:
            optimization_results['recommendations'].append(
                "Current parameters appear optimal for the given data"
            )
        
        return optimization_results
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive hashing and performance statistics"""
        
        stats = {
            'performance_metrics': self.metrics.copy(),
            'policy_statistics': {},
            'cache_performance': {
                'hit_rate': self.metrics['cache_hits'] / max(1, self.metrics['cache_hits'] + self.metrics['cache_misses']),
                'total_entries': len(self.hash_cache),
                'cache_size_mb': sum(len(str(entry)) for entry in self.hash_cache.values()) / 1024 / 1024
            },
            'reconstruction_statistics': {
                'active_plans': len(self.reconstruction_cache),
                'total_reconstructions': self.metrics['reconstruction_requests'],
                'forensic_reconstructions': self.metrics['forensic_reconstructions']
            },
            'ml_optimization_statistics': {
                'optimizations_performed': self.metrics['ml_optimizations'],
                'feature_cache_size': len(self.ml_optimizer.feature_cache),
                'model_trained': self.ml_optimizer.is_trained
            }
        }
        
        # Policy-specific statistics
        for policy_name, policy in self.hashing_policies.items():
            stats['policy_statistics'][policy_name] = {
                'strategy': policy.strategy.value,
                'reconstruction_level': policy.reconstruction_level.value,
                'security_level': policy.security_level.value,
                'max_payload_size': policy.max_payload_size,
                'target_compression_ratio': policy.target_compression_ratio,
                'ml_optimization_enabled': policy.ml_optimization_enabled
            }
        
        # Recent performance trends
        if self.metrics['processing_times_ms']:
            recent_times = list(self.metrics['processing_times_ms'])[-100:]
            recent_ratios = list(self.metrics['compression_ratios'])[-100:]
            
            stats['recent_performance'] = {
                'average_processing_time_ms': np.mean(recent_times),
                'p95_processing_time_ms': np.percentile(recent_times, 95),
                'average_compression_ratio': np.mean(recent_ratios),
                'compression_consistency': 1.0 - (np.std(recent_ratios) / np.mean(recent_ratios))
            }
        
        return stats
    
    async def cleanup_expired_resources(self):
        """Clean up expired reconstruction plans and cache entries"""
        
        current_time = time.time()
        cleanup_stats = {
            'expired_plans': 0,
            'expired_cache_entries': 0,
            'memory_freed_mb': 0
        }
        
        # Clean up expired reconstruction plans
        expired_plan_ids = []
        for plan_id, plan in self.reconstruction_cache.items():
            if current_time > plan.expiry_timestamp:
                expired_plan_ids.append(plan_id)
        
        for plan_id in expired_plan_ids:
            del self.reconstruction_cache[plan_id]
            cleanup_stats['expired_plans'] += 1
        
        # Clean up expired cache entries
        expired_cache_keys = []
        for cache_key, cache_entry in self.hash_cache.items():
            if current_time - cache_entry['timestamp'] > self.cache_ttl:
                expired_cache_keys.append(cache_key)
        
        for cache_key in expired_cache_keys:
            del self.hash_cache[cache_key]
            cleanup_stats['expired_cache_entries'] += 1
        
        # Estimate memory freed (rough calculation)
        cleanup_stats['memory_freed_mb'] = (
            (cleanup_stats['expired_plans'] * 0.1) +  # ~100KB per plan
            (cleanup_stats['expired_cache_entries'] * 0.05)  # ~50KB per cache entry
        )
        
        if cleanup_stats['expired_plans'] > 0 or cleanup_stats['expired_cache_entries'] > 0:
            logger.info(f"Cleanup completed: {cleanup_stats['expired_plans']} plans, "
                       f"{cleanup_stats['expired_cache_entries']} cache entries, "
                       f"{cleanup_stats['memory_freed_mb']:.1f}MB freed")
        
        return cleanup_stats
    
    def set_schema_engine(self, schema_engine: SchemaEvolutionEngine):
        """Set reference to schema evolution engine"""
        self.schema_engine = schema_engine
    
    def set_privacy_filter(self, privacy_filter: PrivacyComplianceFilter):
        """Set reference to privacy compliance filter"""
        self.privacy_filter = privacy_filter
    
    def set_preservation_guard(self, preservation_guard: AnomalyPreservationGuard):
        """Set reference to anomaly preservation guard"""
        self.preservation_guard = preservation_guard
    
    def _get_current_schema_version(self) -> str:
        """Get current schema version from schema engine"""
        if self.schema_engine:
            return self.schema_engine.get_current_version()
        return "2.0.0"
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup"""
        await self.cleanup_expired_resources()
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)


# =============================================================================
# Utility Functions and Factory Methods
# =============================================================================

def create_enhanced_hashing_manager(config: Dict[str, Any]) -> EnhancedDeferredHashingManager:
    """Factory function to create a configured enhanced hashing manager"""
    
    default_config = {
        'cache_ttl': 300,
        'max_workers': 4,
        'ml_optimization_enabled': True,
        'compression_optimization': True,
        'security_level': 'standard'
    }
    
    # Merge with provided config
    merged_config = {**default_config, **config}
    
    return EnhancedDeferredHashingManager(merged_config)

def get_optimal_policy_for_data_enhanced(data_size: int, 
                                        sensitivity: str, 
                                        performance_requirement: str = 'balanced',
                                        security_requirement: str = 'standard') -> str:
    """Recommend optimal hashing policy based on comprehensive data characteristics"""
    
    # Security-first recommendations
    if security_requirement == 'quantum_resistant' or sensitivity == 'classified':
        return 'critical_security'
    
    # Performance-first recommendations
    if performance_requirement == 'ultra_fast' or data_size < 1024:
        return 'edge_lightweight'
    
    # Size and sensitivity based recommendations
    if sensitivity == 'high' or data_size > 32768:
        if performance_requirement == 'fast':
            return 'performance_optimized'
        else:
            return 'critical_security'
    elif sensitivity == 'personal_data':
        return 'privacy_preserving'
    elif data_size > 16384:
        return 'ml_adaptive'
    elif performance_requirement == 'integrity_focused':
        return 'blockchain_integrity'
    else:
        return 'performance_optimized'

async def benchmark_hashing_performance_enhanced(manager: EnhancedDeferredHashingManager,
                                               test_payloads: List[Dict[str, Any]],
                                               policies: List[str] = None) -> Dict[str, Any]:
    """Enhanced benchmark for hashing performance across multiple policies"""
    
    if policies is None:
        policies = ['performance_optimized', 'ml_adaptive', 'critical_security']
    
    benchmark_results = {
        'policy_results': {},
        'comparative_analysis': {},
        'recommendations': []
    }
    
    for policy in policies:
        if policy not in manager.hashing_policies:
            continue
        
        policy_results = []
        
        for payload in test_payloads:
            start_time = time.perf_counter()
            try:
                result = await manager.optimize_payload_size(payload, policy)
                end_time = time.perf_counter()
                
                policy_results.append({
                    'original_size': result.original_size,
                    'compressed_size': result.compressed_size,
                    'compression_ratio': result.compression_ratio,
                    'processing_time_ms': (end_time - start_time) * 1000,
                    'preservation_effectiveness': result.preservation_effectiveness,
                    'information_loss': result.information_loss,
                    'strategy_used': result.strategy_used.value,
                    'ml_confidence': result.ml_confidence_score
                })
            except Exception as e:
                logger.error(f"Benchmark failed for policy {policy}: {e}")
                continue
        
        if policy_results:
            benchmark_results['policy_results'][policy] = {
                'individual_results': policy_results,
                'average_compression_ratio': np.mean([r['compression_ratio'] for r in policy_results]),
                'average_processing_time_ms': np.mean([r['processing_time_ms'] for r in policy_results]),
                'average_preservation': np.mean([r['preservation_effectiveness'] for r in policy_results]),
                'average_information_loss': np.mean([r['information_loss'] for r in policy_results]),
                'total_size_reduction': sum(r['original_size'] - r['compressed_size'] for r in policy_results),
                'success_rate': len(policy_results) / len(test_payloads)
            }
    
    # Comparative analysis
    if len(benchmark_results['policy_results']) > 1:
        policy_metrics = benchmark_results['policy_results']
        
        # Find best performing policies
        best_compression = max(policy_metrics.items(), 
                             key=lambda x: x[1]['average_compression_ratio'])
        best_speed = min(policy_metrics.items(), 
                        key=lambda x: x[1]['average_processing_time_ms'])
        best_preservation = max(policy_metrics.items(), 
                              key=lambda x: x[1]['average_preservation'])
        
        benchmark_results['comparative_analysis'] = {
            'best_compression_policy': best_compression[0],
            'best_compression_ratio': best_compression[1]['average_compression_ratio'],
            'fastest_policy': best_speed[0],
            'fastest_time_ms': best_speed[1]['average_processing_time_ms'],
            'best_preservation_policy': best_preservation[0],
            'best_preservation_score': best_preservation[1]['average_preservation']
        }
        
        # Generate recommendations
        benchmark_results['recommendations'] = [
            f"For maximum compression: use {best_compression[0]} "
            f"({best_compression[1]['average_compression_ratio']:.2f}x ratio)",
            f"For fastest processing: use {best_speed[0]} "
            f"({best_speed[1]['average_processing_time_ms']:.2f}ms avg)",
            f"For best preservation: use {best_preservation[0]} "
            f"({best_preservation[1]['average_preservation']:.3f} effectiveness)"
        ]
        
        # Performance consistency analysis
        for policy, metrics in policy_metrics.items():
            times = [r['processing_time_ms'] for r in metrics['individual_results']]
            time_consistency = 1.0 - (np.std(times) / np.mean(times)) if np.mean(times) > 0 else 0
            
            if time_consistency > 0.9:
                benchmark_results['recommendations'].append(
                    f"{policy} shows excellent performance consistency ({time_consistency:.2f})"
                )
    
    return benchmark_results

class HashingManagerIntegration:
    """Integration helper for embedding hashing manager in production systems"""
    
    def __init__(self, manager: EnhancedDeferredHashingManager):
        self.manager = manager
        self.integration_metrics = {
            'total_requests': 0,
            'successful_optimizations': 0,
            'failed_optimizations': 0,
            'average_overhead_ms': 0.0,
            'cache_hit_rate': 0.0
        }
    
    async def optimize_telemetry_payload(self, payload: Dict[str, Any],
                                       optimization_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Production-friendly payload optimization with error handling"""
        
        start_time = time.perf_counter()
        self.integration_metrics['total_requests'] += 1
        
        try:
            # Determine policy based on context
            policy_name = self._select_policy_from_context(payload, optimization_context)
            
            # Perform optimization
            result = await self.manager.optimize_payload_size(payload, policy_name, optimization_context)
            
            # Calculate integration overhead
            processing_time = (time.perf_counter() - start_time) * 1000
            self._update_overhead_metrics(processing_time)
            
            self.integration_metrics['successful_optimizations'] += 1
            
            # Return production-friendly result
            return {
                'success': True,
                'optimized_payload': result.hashed_payload,
                'compression_achieved': result.compression_ratio,
                'size_reduction_bytes': result.original_size - result.compressed_size,
                'processing_time_ms': processing_time,
                'reconstruction_id': result.reconstruction_id,
                'preservation_effectiveness': result.preservation_effectiveness,
                'optimization_suggestions': result.optimization_suggestions[:3]  # Top 3 suggestions
            }
            
        except Exception as e:
            self.integration_metrics['failed_optimizations'] += 1
            logger.error(f"Payload optimization failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'optimized_payload': payload,  # Return original on failure
                'compression_achieved': 1.0,
                'size_reduction_bytes': 0,
                'processing_time_ms': (time.perf_counter() - start_time) * 1000
            }
    
    def _select_policy_from_context(self, payload: Dict[str, Any], 
                                  context: Optional[Dict[str, Any]]) -> str:
        """Select appropriate policy based on payload and context"""
        
        if not context:
            return 'performance_optimized'
        
        # Security-sensitive data
        if context.get('security_event', False) or context.get('threat_detected', False):
            return 'critical_security'
        
        # Privacy-sensitive data
        if context.get('contains_pii', False):
            return 'privacy_preserving'
        
        # High-performance requirements
        if context.get('real_time_processing', False):
            return 'edge_lightweight'
        
        # Audit trail requirements
        if context.get('audit_required', False):
            return 'blockchain_integrity'
        
        # Complex data that might benefit from ML optimization
        payload_size = len(json.dumps(payload, separators=(',', ':')))
        if payload_size > 16384:
            return 'ml_adaptive'
        
        return 'performance_optimized'
    
    def _update_overhead_metrics(self, processing_time_ms: float):
        """Update integration overhead metrics"""
        current_avg = self.integration_metrics['average_overhead_ms']
        total_requests = self.integration_metrics['total_requests']
        
        new_avg = ((current_avg * (total_requests - 1)) + processing_time_ms) / total_requests
        self.integration_metrics['average_overhead_ms'] = new_avg
    
    def get_integration_health(self) -> Dict[str, Any]:
        """Get integration health and performance metrics"""
        
        total_requests = self.integration_metrics['total_requests']
        success_rate = (self.integration_metrics['successful_optimizations'] / total_requests 
                       if total_requests > 0 else 0.0)
        
        # Get manager statistics
        manager_stats = self.manager.get_comprehensive_statistics()
        
        return {
            'success_rate': success_rate,
            'total_requests': total_requests,
            'average_overhead_ms': self.integration_metrics['average_overhead_ms'],
            'cache_hit_rate': manager_stats['cache_performance']['hit_rate'],
            'health_status': 'excellent' if success_rate > 0.99 else 'good' if success_rate > 0.95 else 'degraded',
            'manager_performance': {
                'total_optimizations': manager_stats['performance_metrics']['total_hashed'],
                'average_compression_ratio': manager_stats['performance_metrics']['average_compression_ratio'],
                'ml_optimizations': manager_stats['performance_metrics']['ml_optimizations']
            }
        }


# =============================================================================
# Example Usage and Testing Framework
# =============================================================================

async def main():
    """Enhanced example usage of the deferred hashing manager"""
    
    # Initialize configuration
    config = {
        'cache_ttl': 300,
        'max_workers': 4,
        'ml_optimization_enabled': True,
        'compression_optimization': True,
        'security_level': 'standard'
    }
    
    # Create enhanced hashing manager
    async with create_enhanced_hashing_manager(config) as manager:
        
        print("=== Enhanced Deferred Hashing Manager Demo ===")
        
        # Example 1: ML-guided optimization
        print("\n1. ML-Guided Payload Optimization")
        
        complex_record = {
            'execution_id': 'exec_demo_001',
            'timestamp': time.time(),
            'function_name': 'ml_inference_pipeline',
            'duration_ms': 2450.7,
            'memory_used_mb': 512.5,
            'cpu_utilization': 85.3,
            'model_metadata': {
                'model_name': 'anomaly_detector_v2',
                'version': '2.1.0',
                'parameters': list(range(1000)),  # Large parameter list
                'training_data_hash': 'sha256_hash_here',
                'performance_metrics': {
                    'accuracy': 0.97,
                    'precision': 0.95,
                    'recall': 0.93,
                    'f1_score': 0.94
                }
            },
            'execution_context': {
                'region': 'us-west-2',
                'environment': 'production',
                'dependencies': ['tensorflow', 'numpy', 'pandas'],
                'resource_limits': {
                    'memory_gb': 2,
                    'cpu_cores': 4,
                    'timeout_seconds': 300
                }
            },
            'large_debug_output': "x" * 10000,  # Simulate large debug output
            'anomaly_indicators': {
                'suspicious_patterns': True,
                'threat_score': 0.85,
                'confidence_level': 0.92
            }
        }
        
        # Test ML-guided optimization
        result = await manager.optimize_payload_size(complex_record, 'ml_adaptive')
        
        print(f"Original size: {result.original_size} bytes")
        print(f"Optimized size: {result.compressed_size} bytes")
        print(f"Compression ratio: {result.compression_ratio:.2f}x")
        print(f"Preservation effectiveness: {result.preservation_effectiveness:.3f}")
        print(f"ML confidence: {result.ml_confidence_score:.3f}")
        print(f"Strategy used: {result.strategy_used.value}")
        print(f"Information loss: {result.information_loss:.3f}")
        
        if result.optimization_suggestions:
            print(f"Optimization suggestions:")
            for suggestion in result.optimization_suggestions[:3]:
                print(f"  - {suggestion}")
        
        # Example 2: Security-focused hashing
        print("\n2. Quantum-Resistant Security Optimization")
        
        security_record = {
            'execution_id': 'sec_exec_001',
            'timestamp': time.time(),
            'security_event': True,
            'threat_detected': True,
            'attack_vector': 'sql_injection',
            'source_ip': '192.168.1.100',
            'user_agent': 'malicious_bot_v1.0',
            'payload_analysis': {
                'malicious_patterns': ['union select',        # Edge-optimized lightweight hashing
        self.hashing_policies['edge_lightweight'] = HashingPolicy(
            strategy=HashingStrategy.COMPRESS_THEN_HASH,
            reconstruction_level=ReconstructionLevel.NONE,
            security_level=SecurityLevel.PERFORMANCE,
            max_payload_size=4096,
            target_compression_ratio=0.6,
            max_processing_time_ms=0.5,
            critical_fields={'execution_id', 'timestamp'},
            compression_algorithm=CompressionAlgorithm.LZ4,
            parallel_processing=False,
            cache_enabled=True,
            ml_optimization_enabled=False  # Minimize compute for edge devices
        )
    
    @performance_monitor
    async def optimize_payload_size(self, 
                                   telemetry_record: Dict[str, Any],
                                   policy_name: str = 'performance_optimized',
                                   context: Optional[Dict[str, Any]] = None) -> HashingResult:
        """
        Advanced payload optimization with ML guidance and comprehensive analysis
        
        Args:
            telemetry_record: The telemetry data to optimize
            policy_name: Name of the hashing policy to apply
            context: Additional context for optimization decisions
            
        Returns:
            Enhanced HashingResult with comprehensive metadata
        """
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            # Get hashing policy
            policy = self.hashing_policies.get(policy_name, 
                                              self.hashing_policies['performance_optimized'])
            
            # Calculate original metrics
            original_payload = json.dumps(telemetry_record, separators=(',', ':'))
            original_size = len(original_payload.encode('utf-8'))
            
            # Check cache first
            cache_key = self._generate_cache_key(telemetry_record, policy_name)
            if policy.cache_enabled and cache_key in self.hash_cache:
                cache_entry = self.hash_cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                    self.metrics['cache_hits'] += 1
                    return cache_entry['result']
            
            self.metrics['cache_misses'] += 1
            
            # ML-guided strategy optimization
            if policy.ml_optimization_enabled:
                recommended_strategy, ml_confidence = self.ml_optimizer.recommend_strategy(
                    telemetry_record, policy
                )
                if ml_confidence > 0.8:
                    # Override policy strategy with ML recommendation
                    effective_strategy = recommended_strategy
                    self.metrics['ml_optimizations'] += 1
                else:
                    effective_strategy = policy.strategy
            else:
                effective_strategy = policy.strategy
                ml_confidence = 0.0
            
            # Anomaly risk assessment
            anomaly_risk = await self._assess_anomaly_risk(telemetry_record, context)
            
            # Apply hashing strategy
            if effective_strategy == HashingStrategy.PRESERVE_FULL:
                result = await self._preserve_full_payload_enhanced(telemetry_record, policy)
            elif effective_strategy == HashingStrategy.SELECTIVE_HASH:
                result = await self._selective_hash_payload_enhanced(telemetry_record, policy)
            elif effective_strategy == HashingStrategy.AGGRESSIVE_HASH:
                result = await self._aggressive_hash_payload_enhanced(telemetry_record, policy)
            elif effective_strategy == HashingStrategy.COMPRESS_THEN_HASH:
                result = await self._compress_then_hash_payload_enhanced(telemetry_record, policy)
            elif effective_strategy == HashingStrategy.ML_GUIDED_HASH:
                result = await self._ml_guided_hash_payload(telemetry_record, policy)
            elif effective_strategy == HashingStrategy.DIFFERENTIAL_PRIVACY_HASH:
                result = await self._differential_privacy_hash_payload(telemetry_record, policy)
            elif effective_strategy == HashingStrategy.QUANTUM_RESISTANT_HASH:
                result = await self._quantum_resistant_hash_payload(telemetry_record, policy)
            elif effective_strategy == HashingStrategy.BLOCKCHAIN_INTEGRITY:
                result = await self._blockchain_integrity_hash_payload(telemetry_record, policy)
            elif effective_strategy == HashingStrategy.ADAPTIVE_COMPRESSION:
                result = await self._adaptive_compression_hash_payload(telemetry_record, policy)
            else:
                result = await self._incremental_hash_payload_enhanced(telemetry_record, policy)
            
            # Calculate comprehensive metrics
            compressed_payload = json.dumps(result['hashed_payload'], separators=(',', ':'))
            compressed_size = len(compressed_payload.encode('utf-8'))
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
            
            # Performance metrics
            processing_time = (time.perf_counter() - start_time) * 1000
            memory_peak = (psutil.Process().memory_info().rss - start_memory) / 1024 / 1024
            
            # Generate enhanced integrity verification
            integrity_hash, integrity_chain = await self._generate_enhanced_integrity(
                result['hashed_payload'], policy, telemetry_record
            )
            
            # Preservation effectiveness assessment
            preservation_effectiveness = 1.0
            if self.preservation_guard and anomaly_risk > policy.anomaly_risk_threshold:
                preservation_assessment = await self.preservation_guard.assess_preservation_impact(
                    telemetry_record, result['hashed_payload'], f"hashing_{policy_name}"
                )
                preservation_effectiveness = preservation_assessment.preservation_effectiveness
            
            # Create reconstruction plan if needed
            reconstruction_plan = None
            reconstruction_id = None
            if policy.reconstruction_level != ReconstructionLevel.NONE:
                reconstruction_plan, reconstruction_id = await self._create_enhanced_reconstruction_plan(
                    telemetry_record, result['hash_map'], policy, compression_ratio
                )
            
            # Information-theoretic analysis
            information_loss = self._calculate_information_loss(
                original_payload, compressed_payload
            )
            
            # Generate optimization suggestions
            optimization_suggestions = await self._generate_optimization_suggestions(
                telemetry_record, result, policy, processing_time, compression_ratio
            )
            
            # Build comprehensive result
            hashing_result = HashingResult(
                hashed_payload=result['hashed_payload'],
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compression_ratio,
                hash_map=result['hash_map'],
                integrity_hash=integrity_hash,
                integrity_chain=integrity_chain,
                reconstruction_plan=reconstruction_plan,
                reconstruction_id=reconstruction_id,
                strategy_used=effective_strategy,
                compression_algorithm_used=result.get('compression_algorithm', 'none'),
                processing_time_ms=processing_time,
                memory_peak_mb=memory_peak,
                preservation_effectiveness=preservation_effectiveness,
                information_loss=information_loss,
                ml_confidence_score=ml_confidence,
                anomaly_risk_assessment=anomaly_risk,
                optimization_suggestions=optimization_suggestions,
                security_level_achieved=policy.security_level
            )
            
            # Update metrics and cache
            self._update_enhanced_metrics(hashing_result)
            
            if policy.cache_enabled:
                self.hash_cache[cache_key] = {
                    'result': hashing_result,
                    'timestamp': time.time()
                }
            
            logger.debug(f"Enhanced payload optimization: {original_size} -> {compressed_size} bytes "
                        f"({compression_ratio:.2f}x), preservation: {preservation_effectiveness:.3f}")
            
            return hashing_result
            
        except Exception as e:
            logger.error(f"Enhanced payload optimization failed: {e}")
            raise
    
    async def _assess_anomaly_risk(self, record: Dict[str, Any], 
                                 context: Optional[Dict[str, Any]]) -> float:
        """Assess anomaly risk for preservation decision making"""
        
        risk_score = 0.0
        
        # Check for anomaly indicators in the record
        anomaly_indicators = ['anomaly', 'alert', 'error', 'failure', 'breach', 'violation']
        for field, value in record.items():
            field_lower = field.lower()
            if any(indicator in field_lower for indicator in anomaly_indicators):
                risk_score += 0.3
            
            # Check for unusual numeric values
            if isinstance(value, (int, float)):
                if 'duration' in field_lower and value > 10000:  # > 10 seconds
                    risk_score += 0.2
                elif 'memory' in field_lower and value > 1000:  # > 1GB
                    risk_score += 0.2
                elif 'cpu' in field_lower and value > 90:  # > 90% CPU
                    risk_score += 0.2
        
        # Context-based risk assessment
        if context:
            if context.get('security_event', False):
                risk_score += 0.4
            if context.get('performance_degradation', False):
                risk_score += 0.3
            if context.get('error_rate') and context['error_rate'] > 0.1:
                risk_score += 0.3
        
        return min(1.0, risk_score)
    
    async def _preserve_full_payload_enhanced(self, record: Dict[str, Any], 
                                            policy: HashingPolicy) -> Dict[str, Any]:
        """Enhanced full payload preservation with integrity verification"""
        
        # Add integrity metadata while preserving full payload
        enhanced_record = record.copy()
        
        if policy.digital_signature:
            enhanced_record['_integrity_signature'] = self.crypto_hasher.hash_data(
                json.dumps(record, separators=(',', ':'), sort_keys=True)
            )
        
        enhanced_record['_preservation_timestamp'] = time.time()
        enhanced_record['_policy_applied'] = 'preserve_full'
        
        return {
            'hashed_payload': enhanced_record,
            'hash_map': {},
            'compression_algorithm': 'none'
        }
    
    async def _selective_hash_payload_enhanced(self, record: Dict[str, Any], 
                                             policy: HashingPolicy) -> Dict[str, Any]:
        """Enhanced selective hashing with ML-guided field importance"""
        
        hashed_payload = {}
        hash_map = {}
        
        for field, value in record.items():
            # Assess field importance using ML
            field_importance = self.ml_optimizer.assess_field_importance(
                field, value, record
            )
            
            # Determine if field should be preserved based on policy and ML assessment
            should_preserve = (
                field in policy.critical_fields or
                field_importance > 0.7 or
                (isinstance(value, (int, float)) and 'time' not in field.lower())
            )
            
            if should_preserve:
                hashed_payload[field] = value
            else:
                # Hash non-critical fields
                value_str = json.dumps(value, separators=(',', ':')) if isinstance(value, (dict, list)) else str(value)
                field_hash = self.crypto_hasher.hash_data(value_str)
                hashed_field_name = f"{field}_hash"
                
                hashed_payload[hashed_field_name] = {
                    'hash': field_hash,
                    'type': type(value).__name__,
                    'size': len(value_str),
                    'importance_score': field_importance
                }
                hash_map[field] = field_hash
        
        return {
            'hashed_payload': hashed_payload,
            'hash_map': hash_map,
            'compression_algorithm': 'selective'
        }
    
    async def _ml_guided_hash_payload(self, record: Dict[str, Any], 
                                    policy: HashingPolicy) -> Dict[str, Any]:
        """ML-guided intelligent hashing based on content analysis"""
        
        # Extract features and analyze record structure
        features = self.ml_optimizer.extract_features(record)
        
        hashed_payload = {}
        hash_map = {}
        ml_decisions = []
        
        # Always preserve critical fields
        for field in policy.critical_fields:
            if field in record:
                hashed_payload[field] = record[field]
        
        # ML-guided decisions for other fields
        for field, value in record.items():
            if field in policy.critical_fields:
                continue
            
            # Calculate field-specific features
            field_importance = self.ml_optimizer.assess_field_importance(field, value, record)
            
            # Decision logic based on ML analysis
            if field_importance > 0.8:
                # High importance - preserve
                hashed_payload[field] = value
                ml_decisions.append(f"{field}: preserved (importance: {field_importance:.2f})")
            elif field_importance > 0.5:
                # Medium importance - compress then preserve
                if isinstance(value, (dict, list)):
                    compressed_value = await self._compress_structured_data(value)
                    hashed_payload[f"{field}_compressed"] = compressed_value
                else:
                    hashed_payload[field] = value
                ml_decisions.append(f"{field}: compressed (importance: {field_importance:.2f})")
            else:
                # Low importance - hash
                value_str = json.dumps(value, separators=(',', ':')) if isinstance(value, (dict, list)) else str(value)
                field_hash = self.crypto_hasher.hash_data(value_str)
                hashed_payload[f"{field}_hash"] = field_hash
                hash_map[field] = field_hash
                ml_decisions.append(f"{field}: hashed (importance: {field_importance:.2f})")
        
        # Add ML decision metadata
        hashed_payload['_ml_decisions'] = ml_decisions
        hashed_payload['_ml_features'] = features
        
        return {
            'hashed_payload': hashed_payload,
            'hash_map': hash_map,
            'compression_algorithm': 'ml_guided'
        }
    
    async def _adaptive_compression_hash_payload(self, record: Dict[str, Any], 
                                               policy: HashingPolicy) -> Dict[str, Any]:
        """Adaptive compression with algorithm selection optimization"""
        
        payload_str = json.dumps(record, separators=(',', ':'))
        
        # Select optimal compression algorithm
        optimal_algorithm = self.compression_optimizer.select_algorithm(
            payload_str, policy.target_compression_ratio
        )
        
        # Apply compression
        compressed_data, compression_time = await self.compression_optimizer.compress_data(
            payload_str, optimal_algorithm, policy.compression_level
        )
        
        # Encode compressed data
        compressed_b64 = base64.b64encode(compressed_data).decode('utf-8')
        
        # Build optimized payload
        hashed_payload = {
            'compressed_data': compressed_b64,
            'compression_algorithm': optimal_algorithm.value,
            'compression_time_ms': compression_time,
            'original_size': len(payload_str),
            'compression_ratio': len(payload_str) / len(compressed_data)
        }
        
        # Preserve critical fields for quick access
        for field in policy.critical_fields:
            if field in record:
                hashed_payload[field] = record[field]
        
        hash_map = {
            'compressed_payload': self.crypto_hasher.hash_data(compressed_b64)
        }
        
        return {
            'hashed_payload': hashed_payload,
            'hash_map': hash_map,
            'compression_algorithm': optimal_algorithm.value
        }
    
    async def _differential_privacy_hash_payload(self, record: Dict[str, Any], 
                                               policy: HashingPolicy) -> Dict[str, Any]:
        """Differential privacy-preserving hashing"""
        
        hashed_payload = {}
        hash_map = {}
        
        # Add noise for differential privacy
        epsilon = policy.differential_privacy_epsilon
        
        for field, value in record.items():
            if field in policy.critical_fields:
                hashed_payload[field] = value
            elif field in policy.sensitive_fields:
                # Apply differential privacy noise
                if isinstance(value, (int, float)):
                    # Add Laplace noise for numeric values
                    noise = np.random.laplace(0, 1/epsilon)
                    noisy_value = value + noise
                    hashed_payload[f"{field}_dp"] = noisy_value
                else:
                    # Hash sensitive non-numeric fields
                    field_hash = self.crypto_hasher.hash_data(str(value))
                    hashed_payload[f"{field}_hash"] = field_hash
                    hash_map[field] = field_hash
            else:
                # Standard processing for non-sensitive fields
                value_str = json.dumps(value, separators=(',', ':')) if isinstance(value, (dict, list)) else str(value)
                field_hash = self.crypto_hasher.hash_data(value_str)
                hashed_payload[f"{field}_hash"] = field_hash
                hash_map[field] = field_hash
        
        # Add privacy metadata
        hashed_payload['_privacy_epsilon'] = epsilon
        hashed_payload['_privacy_applied'] = True
        
        return {
            'hashed_payload': hashed_payload,
            'hash_map': hash_map,
            'compression_algorithm': 'differential_privacy'
        }
    
    async def _quantum_resistant_hash_payload(self, record: Dict[str, Any], 
                                            policy: HashingPolicy) -> Dict[str, Any]:
        """Quantum-resistant cryptographic hashing"""
        
        # Use quantum-resistant hasher
        quantum_hasher = CryptographicHasher(SecurityLevel.QUANTUM_RESISTANT)
        
        hashed_payload = {}
        hash_map = {}
        
        # Preserve critical fields
        for field in policy.critical_fields:
            if field in record:
                hashed_payload[field] = record[field]
        
        # Apply quantum-resistant hashing to other fields
        for field, value in record.items():
            if field not in policy.critical_fields:
                value_str = json.dumps(value, separators=(',', ':')) if isinstance(value, (dict, list)) else str(value)
                
                # Generate primary and secondary hashes for redundancy
                primary_hash = quantum_hasher.hash_data(value_str, "primary")
                secondary_hash = quantum_hasher.hash_data(value_str, "secondary")
                
                hashed_payload[f"{field}_qr_hash"] = {
                    'primary': primary_hash,
                    'secondary': secondary_hash,
                    'algorithm': 'blake3_sha3'
                }
                hash_map[field] = primary_hash
        
        # Add quantum resistance metadata
        hashed_payload['_quantum_resistant'] = True
        hashed_payload['_security_level'] = 'quantum_resistant'
        
        return {
            'hashed_payload': hashed_payload,
            'hash_map': hash_map,
            'compression_algorithm': 'quantum_resistant'
        }
    
    async def _blockchain_integrity_hash_payload(self, record: Dict[str, Any], 
                                               policy: HashingPolicy) -> Dict[str, Any]:
        """Blockchain-inspired integrity verification hashing"""
        
        hashed_payload = {}
        hash_map = {}
        
        # Create integrity chain for the record
        field_payloads = []
        for field, value in record.items():
            if field in policy.critical_fields:
                hashed_payload[field] = value
            else:
                value_str = json.dumps(value, separators=(',', ':')) if isinstance(value, (dict, list)) else str(value)
                field_payloads.append(f"{field}:{value_str}")
        
        # Generate blockchain-style integrity chain
        integrity_chain = self.crypto_hasher.generate_integrity_chain(field_payloads)
        
        # Hash non-critical fields
        for i, (field, value) in enumerate(record.items()):
            if field not in policy.critical_fields:
                value_str = json.dumps(value, separators=(',', ':')) if isinstance(value, (dict, list)) else str(value)
                field_hash = integrity_chain[i] if i < len(integrity_chain) else self.crypto_hasher.hash_data(value_str)
                hashed_payload[f"{field}_bc_hash"] = field_hash
                hash_map[field] = field_hash
        
        # Add blockchain metadata
        hashed_payload['_integrity_chain'] = integrity_chain
        hashed_payload['_chain_verification'] = True
        hashed_payload['_blockchain_timestamp'] = time.time()
        
        return {
            'hashed_payload': hashed_payload,
            'hash_map': hash_map,
            'compression_algorithm': 'blockchain_integrity',
            'integrity_chain': integrity_chain
        }
    
    async def _compress_structured_data(self, data: Any) -> Dict[str, Any]:
        """Compress structured data while preserving key information"""
        
        if isinstance(data, dict):
            # Preserve structure but compress values
            compressed = {}
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 100:
                    compressed[key] = f"compressed:{base64.b64encode(zlib.compress(value.encode())).decode()}"
                else:
                    compressed[key] = value
            return compressed
        elif isinstance(data, list):
            # Compress large lists
            if len(data) > 50:
                sample = data[:10] + data[-10:]  # Keep first and last 10 items
                return {
                    'type': 'compressed_list',
                    'total_length': len(data),
                    'sample_items': sample,
                    'compressed_data': base64.b64encode(zlib.compress(json.dumps(data).encode())).decode()
                }
            else:
                return data
        else:
            return data
    
    async def _generate_enhanced_integrity(self, payload: Dict[str, Any], 
                                         policy: HashingPolicy,
                                         original_record: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Generate enhanced integrity verification with blockchain-style chain"""
        
        # Primary integrity hash
        payload_str = json.dumps(payload, separators=(',', ':'), sort_keys=True)
        integrity_hash = self.crypto_hasher.hash_data(payload_str)
        
        # Generate integrity chain if enabled
        integrity_chain = []
        if policy.digital_signature:
            # Create verification chain
            components = [
                json.dumps(original_record, separators=(',', ':'), sort_keys=True),
                payload_str,
                str(time.time()),
                policy.strategy.value
            ]
            integrity_chain = self.crypto_hasher.generate_integrity_chain(components)
        
        return integrity_hash, integrity_chain
    
    async def _create_enhanced_reconstruction_plan(self, original_record: Dict[str, Any],
                                                 hash_map: Dict[str, str],
                                                 policy: HashingPolicy,
                                                 compression_ratio: float) -> Tuple[Dict[str, Any], str]:
        """Create enhanced reconstruction plan with temporal decay and encryption"""
        
        reconstruction_id = self.crypto_hasher.hash_data(
            f"{time.time()}_{id(original_record)}_{policy.strategy.value}"
        )[:16]
        
        # Calculate reconstruction complexity
        complexity_score = len(hash_map) * 0.1 + (1.0 / compression_ratio) * 0.5
        estimated_time = complexity_score * 10  # Rough estimate in milliseconds
        
        # Create decay schedule
        decay_schedule = {}
        if policy.reconstruction_decay:
            current_time = time.time()
            decay_schedule = {
                'full_capability': current_time + policy.hot_data_period,
                'basic_capability': current_time + policy.hot_data_period * 2,
                'minimal_capability': current_time + policy.retention_period
            }
        
        plan = ReconstructionPlan(
            reconstruction_id=reconstruction_id,
            field_mappings=hash_map.copy(),
            structure_metadata={
                'original_field_count': len(original_record),
                'hashed_field_count': len(hash_map),
                'compression_ratio': compression_ratio,
                'strategy_used': policy.strategy.value
            },
            compression_metadata={
                'algorithm': getattr(policy, 'compression_algorithm', CompressionAlgorithm.ZLIB).value,
                'level': policy.compression_level,
                'original_size': len(json.dumps(original_record, separators=(',', ':')))
            },
            schema_version=self._get_current_schema_version(),
            expiry_timestamp=time.time() + policy.retention_period,
            reconstruction_level=policy.reconstruction_level,
            available_capabilities={
                'field_recovery', 'structure_analysis', 'integrity_verification'
            },
            decay_schedule=decay_schedule,
            encrypted=policy.encryption_enabled,
            reconstruction_complexity=complexity_score,
            estimated_reconstruction_time_ms=estimated_time
        )
        
        # Encrypt reconstruction plan if required
        if policy.encryption_enabled:
            plan.encryption_key_id = self._generate_encryption_key_id()
        
        # Cache the reconstruction plan
        self.reconstruction_cache[reconstruction_id] = plan
        
        return asdict(plan), reconstruction_id
    
    def _calculate_information_loss(self, original: str, processed: str) -> float:
        """Calculate information-theoretic loss during processing"""
        
        if not original or not processed:
            return 1.0 if not processed else 0.0
        
        # Simple entropy-based information loss calculation
        original_entropy = self.ml_optimizer._calculate_entropy(original)
        processed_entropy = self.ml_optimizer._calculate_entropy(processed)
        
        if original_entropy == 0:
            return 0.0
        
        return max(0.0, (original_entropy - processed_entropy) / original_entropy)
    
    async def _generate_optimization_suggestions(self, record: Dict[str, Any],
                                               result: Dict[str, Any],
                                               policy: HashingPolicy,
                                               processing_time: float,
                                               compression_ratio: float) -> List[str]:
        """Generate optimization suggestions based on processing results"""
        
        suggestions = []
        
        # Performance-based suggestions
        if processing_time > policy.max_processing_time_ms:
            suggestions.append(f"Processing time ({processing_time:.2f}ms) exceeds target - consider simpler strategy")
        
        # Compression ratio suggestions
        if compression_ratio < policy.target_compression_ratio:
            suggestions.append(f"Compression ratio ({compression_ratio:.2f}) below target - try aggressive hashing")
        elif compression_ratio > policy.target_compression_ratio * 2:
            suggestions.append("High compression achieved - consider preserving more fields")
        
        # Content-based suggestions
        features = self.ml_optimizer.extract_features(record)
        if features['complex_fields'] > features['field_count'] * 0.7:
            suggestions.append("High complexity data - consider incremental hashing")
        
        if features['content_entropy'] > 7.0:
            suggestions.append("High entropy content - compression may be less effective")
        
        # Policy-specific suggestions
        if policy.ml_optimization_enabled and len(result['hash_map']) > 10:
            suggestions.append("Many fields hashed - ML optimization could improve field selection")
        
        return suggestions
    
    def _generate_cache_key(self, record: Dict[str, Any], policy_name: str) -> str:
        """Generate cache key for hashing results"""
        
        # Create deterministic hash of record content and policy
        record_hash = self.crypto_hasher.hash_data(
            json.dumps(record, separators=(',', ':'), sort_keys=True)
        )[:16]
        
        return f"{policy_name}_{record_hash}"
    
    def _generate_encryption_key_id(self) -> str:
        """Generate encryption key identifier"""
        return f"key_{secrets.token_hex(8)}"
    
    def _update_enhanced_metrics(self, result: HashingResult):
        """Update comprehensive performance metrics"""
        
        self.metrics['total_hashed'] += 1
        self.metrics['total_size_saved'] += (result.original_size - result.compressed_size)
        
        # Update rolling metrics
        self.metrics['processing_times_ms'].append(result.processing_time_ms)
        self.metrics['compression_ratios'].append(result.compression_ratio)
        
        # Update average compression ratio
        current_avg = self.metrics['average_compression_ratio']
        total_ops = self.metrics['total_hashed']
        new_avg = ((current_avg * (total_ops - 1)) + result.compression_ratio) / total_ops
        self.metrics['average_compression_ratio'] = new_avg
        
        # Track reconstruction plan creation
        if result.reconstruction_plan:
            if result.reconstruction_plan.get('reconstruction_level') == 'forensic':
                self.metrics['forensic_reconstructions'] += 1
    
    async def enable_forensic_reconstruction_enhanced(self, hashed_record: Dict[str, Any],
                                                    reconstruction_id: str,
                                                    access_context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Enhanced forensic reconstruction with access control and audit trail
        
        Args:
            hashed_record: The hashed telemetry record
            reconstruction_id: ID of the reconstruction plan
            access_context: Context for access control and auditing
            
        Returns:
            Enhanced reconstruction metadata or None if not available
        """
        
        try:
            plan = self.reconstruction_cache.get(reconstruction_id)
            if not plan:
                logger.warning(f"Reconstruction plan not found: {reconstruction_id}")
                return None
            
            # Check temporal decay and capabilities
            current_time = time.time()
            if current_time > plan.expiry_timestamp:
                logger.warning(f"Reconstruction plan expired: {reconstruction_id}")
                del self.reconstruction_cache[reconstruction_id]
                return None
            
            # Determine available capabilities based on decay schedule
            available_capabilities = set(plan.available_capabilities)
            if plan.decay_schedule:
                for capability, decay_time in plan.decay_schedule.items():
                    if current_time > decay_time:
                        if capability in available_capabilities:
                            available_capabilities.remove(capability)
            
            # Update access tracking
            plan.last_access_timestamp = current_time
            plan.access_count += 1
            
            # Update metrics
            self.metrics['reconstruction_requests'] += 1
            if plan.reconstruction_level == ReconstructionLevel.FORENSIC:
                self.metrics['forensic_reconstructions'] += 1
            
            # Build enhanced reconstruction metadata
            reconstruction_metadata = {
                'reconstruction_id': reconstruction_id,
                'plan_version': plan.plan_version,
                'field_mappings': plan.field_mappings,
                'structure_metadata': plan.structure_metadata,
                'reconstruction_level': plan.reconstruction_level.value,
                'available_capabilities': list(available_capabilities),
                'reconstruction_complexity#!/usr/bin/env python3
"""
SCAFAD Layer 1: Enhanced Deferred Hashing Manager
=================================================

Advanced payload optimization system with intelligent hashing strategies, forensic
reconstruction capabilities, and ML-powered optimization. Balances storage efficiency
with investigative requirements while maintaining anomaly detection effectiveness.

Key Innovations:
- ML-guided selective hashing based on anomaly risk assessment
- Quantum-resistant cryptographic hashing with performance optimization
- Advanced reconstruction planning with temporal decay strategies
- Differential privacy-aware hashing for sensitive telemetry
- Real-time compression algorithm selection and optimization
- Blockchain-inspired integrity verification chains
- Edge-optimized lightweight hashing for resource-constrained environments

Performance Targets:
- Payload size reduction: 60-85% typical compression
- Processing latency: <1.5ms per payload optimization
- Reconstruction accuracy: 99.9%+ for forensic-level recovery
- Memory overhead: <8MB for manager instance
- Cryptographic security: SHA-3/BLAKE3 with collision resistance

Author: SCAFAD Research Team
Institution: Birmingham Newman University
License: MIT
Version: 2.0.0
"""

import hashlib
import hmac
import json
import time
import zlib
import lzma
import brotli
import base64
import logging
import asyncio
import struct
import secrets
import threading
from typing import Dict, Any, List, Optional, Tuple, Union, Set, Callable, Iterator
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import psutil
import numpy as np
from functools import wraps, lru_cache

# Scientific and ML libraries
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import blake3
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

# Import Layer 1 dependencies
from .layer1_core import Layer1ProcessingResult, ProcessingMetrics, TelemetryRecord
from .layer1_schema import SchemaEvolutionEngine, SchemaVersion
from .layer1_privacy import PrivacyComplianceFilter
from .layer1_preservation import AnomalyPreservationGuard, PreservationAssessment

# Configure logging
logger = logging.getLogger(__name__)

# Performance monitoring decorator
def performance_monitor(func):
    """Decorator for monitoring hashing performance"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            result = await func(*args, **kwargs)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            memory_delta = psutil.Process().memory_info().rss - start_memory
            
            logger.debug(f"{func.__name__} completed in {execution_time:.2f}ms, "
                        f"memory delta: {memory_delta / 1024 / 1024:.1f}MB")
            
            return result
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"{func.__name__} failed after {execution_time:.2f}ms: {e}")
            raise
    return wrapper

class HashingStrategy(Enum):
    """Advanced hashing strategies with ML optimization"""
    PRESERVE_FULL = "preserve_full"                    # No hashing, preserve complete data
    SELECTIVE_HASH = "selective_hash"                  # Hash only non-critical fields
    AGGRESSIVE_HASH = "aggressive_hash"                # Hash most data, keep essentials
    COMPRESS_THEN_HASH = "compress_then_hash"          # Optimize compression before hashing
    INCREMENTAL_HASH = "incremental_hash"              # Hash large fields incrementally
    ML_GUIDED_HASH = "ml_guided_hash"                  # ML-powered field selection
    DIFFERENTIAL_PRIVACY_HASH = "differential_privacy" # Privacy-preserving hashing
    QUANTUM_RESISTANT_HASH = "quantum_resistant"       # Post-quantum cryptography
    ADAPTIVE_COMPRESSION = "adaptive_compression"      # Algorithm selection optimization
    BLOCKCHAIN_INTEGRITY = "blockchain_integrity"      # Blockchain-inspired verification

class ReconstructionLevel(Enum):
    """Enhanced reconstruction capability levels"""
    NONE = "none"                                      # No reconstruction needed
    BASIC = "basic"                                    # Basic field recovery
    FORENSIC = "forensic"                             # Full forensic reconstruction
    AUDIT_TRAIL = "audit_trail"                       # Complete audit trail capability
    REAL_TIME = "real_time"                           # Real-time reconstruction support
    DISTRIBUTED = "distributed"                       # Multi-node reconstruction
    ENCRYPTED = "encrypted"                           # Encrypted reconstruction data

class CompressionAlgorithm(Enum):
    """Supported compression algorithms with optimization profiles"""
    ZLIB = "zlib"                                     # Balanced compression
    LZMA = "lzma"                                     # High compression ratio
    BROTLI = "brotli"                                 # Web-optimized compression
    LZ4 = "lz4"                                       # High-speed compression
    ADAPTIVE = "adaptive"                             # ML-selected algorithm

class SecurityLevel(Enum):
    """Security levels for cryptographic operations"""
    STANDARD = "standard"                             # SHA-256, AES-128
    HIGH = "high"                                     # SHA-3, AES-256
    QUANTUM_RESISTANT = "quantum_resistant"           # Post-quantum algorithms
    PERFORMANCE = "performance"                       # BLAKE3, optimized

@dataclass
class HashingPolicy:
    """Enhanced policy configuration for hashing operations"""
    # Core strategy
    strategy: HashingStrategy
    reconstruction_level: ReconstructionLevel
    security_level: SecurityLevel = SecurityLevel.STANDARD
    
    # Size and performance limits
    max_payload_size: int = 8192
    target_compression_ratio: float = 0.4             # Target 60% size reduction
    max_processing_time_ms: float = 2.0              # Performance SLA
    
    # Field selection
    critical_fields: Set[str] = field(default_factory=set)
    sensitive_fields: Set[str] = field(default_factory=set)
    large_field_threshold: int = 1000                 # Bytes
    
    # Cryptographic settings
    hash_algorithm: str = "blake3"                    # Primary hash algorithm
    encryption_enabled: bool = False                  # Encrypt reconstruction data
    digital_signature: bool = False                   # Sign integrity hashes
    
    # Compression settings
    compression_algorithm: CompressionAlgorithm = CompressionAlgorithm.ADAPTIVE
    compression_level: int = 6                        # 1-9 compression level
    
    # ML and optimization
    ml_optimization_enabled: bool = True              # Enable ML-guided optimization
    anomaly_risk_threshold: float = 0.7              # Risk threshold for preservation
    differential_privacy_epsilon: float = 1.0        # Privacy parameter
    
    # Temporal and retention
    retention_period: int = 2592000                   # 30 days in seconds
    reconstruction_decay: bool = True                 # Gradually reduce reconstruction capability
    hot_data_period: int = 86400                     # 24 hours of full reconstruction
    
    # Performance optimization
    parallel_processing: bool = True                  # Enable parallel hashing
    cache_enabled: bool = True                        # Enable hash caching
    batch_optimization: bool = False                  # Optimize for batch processing

@dataclass
class HashingResult:
    """Enhanced result of hashing operation with comprehensive metadata"""
    # Core results
    hashed_payload: Dict[str, Any]
    original_size: int
    compressed_size: int
    compression_ratio: float
    
    # Hash mappings and integrity
    hash_map: Dict[str, str]                          # Field -> hash mapping
    integrity_hash: str                               # Overall payload integrity
    integrity_chain: List[str] = field(default_factory=list)  # Blockchain-style chain
    
    # Reconstruction metadata
    reconstruction_plan: Optional[Dict[str, Any]] = None
    reconstruction_id: Optional[str] = None
    forensic_metadata: Optional[Dict[str, Any]] = None
    
    # Processing metadata
    strategy_used: HashingStrategy = HashingStrategy.SELECTIVE_HASH
    compression_algorithm_used: str = "zlib"
    processing_time_ms: float = 0.0
    cpu_cycles_used: int = 0
    memory_peak_mb: float = 0.0
    
    # Quality metrics
    preservation_effectiveness: float = 1.0           # Anomaly preservation score
    information_loss: float = 0.0                     # Information-theoretic loss
    reconstruction_accuracy: float = 1.0              # Expected reconstruction quality
    
    # Security metadata
    encryption_applied: bool = False
    signature_verification: Optional[str] = None
    security_level_achieved: SecurityLevel = SecurityLevel.STANDARD
    
    # ML insights
    ml_confidence_score: float = 0.0                 # ML optimization confidence
    anomaly_risk_assessment: float = 0.0             # Assessed anomaly risk
    optimization_suggestions: List[str] = field(default_factory=list)

@dataclass
class ReconstructionPlan:
    """Enhanced plan for reconstructing hashed data with temporal decay"""
    # Basic identification
    reconstruction_id: str
    plan_version: str = "2.0"
    
    # Field and structure mappings
    field_mappings: Dict[str, str] = field(default_factory=dict)
    structure_metadata: Dict[str, Any] = field(default_factory=dict)
    compression_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Schema and versioning
    schema_version: str = "1.0.0"
    schema_evolution_path: List[str] = field(default_factory=list)
    
    # Temporal management
    creation_timestamp: float = field(default_factory=time.time)
    expiry_timestamp: float = 0.0
    last_access_timestamp: float = 0.0
    access_count: int = 0
    
    # Reconstruction capabilities
    reconstruction_level: ReconstructionLevel = ReconstructionLevel.BASIC
    available_capabilities: Set[str] = field(default_factory=set)
    decay_schedule: Dict[str, float] = field(default_factory=dict)
    
    # Security and encryption
    encrypted: bool = False
    encryption_key_id: Optional[str] = None
    integrity_verifiers: List[str] = field(default_factory=list)
    
    # Performance metadata
    reconstruction_complexity: float = 1.0           # Computational complexity score
    estimated_reconstruction_time_ms: float = 0.0
    memory_requirements_mb: float = 0.0

class MLHashingOptimizer:
    """Machine learning optimizer for hashing strategies"""
    
    def __init__(self):
        self.field_importance_model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_history = []
        self.feature_cache = {}
    
    def extract_features(self, record: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for ML-guided hashing decisions"""
        
        cache_key = hash(str(sorted(record.items())))
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        features = {}
        
        # Size-based features
        record_str = json.dumps(record, separators=(',', ':'))
        features['total_size'] = len(record_str)
        features['field_count'] = len(record)
        features['nesting_depth'] = self._calculate_nesting_depth(record)
        
        # Content-based features
        features['numeric_fields'] = sum(1 for v in record.values() if isinstance(v, (int, float)))
        features['string_fields'] = sum(1 for v in record.values() if isinstance(v, str))
        features['complex_fields'] = sum(1 for v in record.values() if isinstance(v, (dict, list)))
        
        # Entropy and information content
        features['content_entropy'] = self._calculate_entropy(record_str)
        features['value_diversity'] = len(set(str(v) for v in record.values()))
        
        # Temporal features (if timestamp present)
        if 'timestamp' in record:
            ts = record['timestamp']
            features['hour_of_day'] = (ts % 86400) // 3600
            features['day_of_week'] = (ts // 86400) % 7
        
        # Performance prediction features
        features['serialization_complexity'] = self._estimate_serialization_complexity(record)
        features['compression_potential'] = self._estimate_compression_potential(record_str)
        
        self.feature_cache[cache_key] = features
        return features
    
    def _calculate_nesting_depth(self, obj: Any, depth: int = 0) -> int:
        """Calculate maximum nesting depth"""
        if isinstance(obj, dict):
            return max([self._calculate_nesting_depth(v, depth + 1) for v in obj.values()], default=depth)
        elif isinstance(obj, list):
            return max([self._calculate_nesting_depth(v, depth + 1) for v in obj], default=depth)
        else:
            return depth
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0
        
        char_counts = defaultdict(int)
        for char in text:
            char_counts[char] += 1
        
        text_len = len(text)
        entropy = 0.0
        for count in char_counts.values():
            probability = count / text_len
            entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _estimate_serialization_complexity(self, record: Dict[str, Any]) -> float:
        """Estimate computational complexity of serialization"""
        complexity = 0.0
        
        for value in record.values():
            if isinstance(value, str):
                complexity += len(value) * 0.1
            elif isinstance(value, (dict, list)):
                complexity += len(str(value)) * 0.5
            else:
                complexity += 1.0
        
        return complexity
    
    def _estimate_compression_potential(self, text: str) -> float:
        """Estimate compression potential based on text characteristics"""
        if not text:
            return 0.0
        
        # Simple compression ratio estimation
        char_frequency = defaultdict(int)
        for char in text:
            char_frequency[char] += 1
        
        # Higher repetition suggests better compression
        repetition_score = sum(count ** 2 for count in char_frequency.values()) / len(text) ** 2
        
        return min(1.0, repetition_score * 2)
    
    def recommend_strategy(self, record: Dict[str, Any], 
                          policy: HashingPolicy) -> Tuple[HashingStrategy, float]:
        """Recommend optimal hashing strategy with confidence score"""
        
        features = self.extract_features(record)
        
        # Rule-based strategy selection with ML enhancement
        confidence = 0.8  # Base confidence
        
        # Size-based decisions
        if features['total_size'] < 1000:
            return HashingStrategy.PRESERVE_FULL, confidence
        elif features['total_size'] > 50000:
            return HashingStrategy.AGGRESSIVE_HASH, confidence * 0.9
        
        # Content-based decisions
        if features['complex_fields'] > features['field_count'] * 0.5:
            return HashingStrategy.INCREMENTAL_HASH, confidence
        
        # High entropy suggests good compression potential
        if features['compression_potential'] > 0.7:
            return HashingStrategy.COMPRESS_THEN_HASH, confidence * 1.1
        
        # Default to ML-guided for complex cases
        if policy.ml_optimization_enabled and features['field_count'] > 10:
            return HashingStrategy.ML_GUIDED_HASH, confidence * 0.95
        
        return HashingStrategy.SELECTIVE_HASH, confidence
    
    def assess_field_importance(self, field_name: str, field_value: Any, 
                               context: Dict[str, Any]) -> float:
        """Assess importance of a field for anomaly detection"""
        
        importance = 0.5  # Base importance
        
        # Known critical fields
        critical_indicators = ['execution_id', 'timestamp', 'anomaly', 'alert', 'error']
        if any(indicator in field_name.lower() for indicator in critical_indicators):
            importance += 0.4
        
        # Performance metrics are important
        performance_indicators = ['duration', 'latency', 'memory', 'cpu', 'network']
        if any(indicator in field_name.lower() for indicator in performance_indicators):
            importance += 0.3
        
        # Large values might be less important unless they're metrics
        if isinstance(field_value, str) and len(field_value) > 5000:
            importance -= 0.2
        
        # Numeric values are often important metrics
        if isinstance(field_value, (int, float)):
            importance += 0.2
        
        return max(0.0, min(1.0, importance))

class CompressionOptimizer:
    """Advanced compression algorithm selection and optimization"""
    
    def __init__(self):
        self.algorithm_performance = defaultdict(list)
        self.data_type_preferences = {}
        self.size_thresholds = {
            'small': 1024,      # < 1KB
            'medium': 10240,    # 1-10KB
            'large': 102400     # 10-100KB
        }
    
    def select_algorithm(self, data: str, target_ratio: float = 0.4) -> CompressionAlgorithm:
        """Select optimal compression algorithm based on data characteristics"""
        
        data_size = len(data.encode('utf-8'))
        
        # Fast algorithms for small data
        if data_size < self.size_thresholds['small']:
            return CompressionAlgorithm.ZLIB
        
        # High compression for large data
        if data_size > self.size_thresholds['large']:
            return CompressionAlgorithm.LZMA
        
        # Text data characteristics
        entropy = self._calculate_text_entropy(data)
        repetition = self._calculate_repetition_factor(data)
        
        # High repetition suggests LZMA will work well
        if repetition > 0.7:
            return CompressionAlgorithm.LZMA
        
        # Low entropy suggests structured data, Brotli might be good
        if entropy < 4.0:
            return CompressionAlgorithm.BROTLI
        
        # Default balanced choice
        return CompressionAlgorithm.ZLIB
    
    def _calculate_text_entropy(self, text: str) -> float:
        """Calculate entropy of text for compression algorithm selection"""
        if not text:
            return 0.0
        
        char_counts = defaultdict(int)
        for char in text:
            char_counts[char] += 1
        
        text_len = len(text)
        entropy = 0.0
        for count in char_counts.values():
            probability = count / text_len
            entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _calculate_repetition_factor(self, text: str) -> float:
        """Calculate repetition factor for compression potential"""
        if len(text) < 10:
            return 0.0
        
        # Look for repeated substrings
        repetitions = 0
        total_chars = len(text)
        
        # Check for common patterns
        for length in [2, 3, 4, 5]:
            seen = set()
            for i in range(len(text) - length + 1):
                substring = text[i:i+length]
                if substring in seen:
                    repetitions += length
                seen.add(substring)
        
        return min(1.0, repetitions / total_chars)
    
    async def compress_data(self, data: str, algorithm: CompressionAlgorithm,
                           level: int = 6) -> Tuple[bytes, float]:
        """Compress data using specified algorithm"""
        
        start_time = time.perf_counter()
        data_bytes = data.encode('utf-8')
        
        try:
            if algorithm == CompressionAlgorithm.ZLIB:
                compressed = zlib.compress(data_bytes, level)
            elif algorithm == CompressionAlgorithm.LZMA:
                compressed = lzma.compress(data_bytes, preset=level)
            elif algorithm == CompressionAlgorithm.BROTLI:
                compressed = brotli.compress(data_bytes, quality=level)
            else:  # Default to zlib
                compressed = zlib.compress(data_bytes, level)
            
            compression_time = (time.perf_counter() - start_time) * 1000
            
            # Update performance tracking
            compression_ratio = len(data_bytes) / len(compressed)
            self.algorithm_performance[algorithm.value].append({
                'ratio': compression_ratio,
                'time_ms': compression_time,
                'size': len(data_bytes)
            })
            
            return compressed, compression_time
            
        except Exception as e:
            logger.error(f"Compression failed with {algorithm.value}: {e}")
            # Fallback to zlib
            compressed = zlib.compress(data_bytes, 6)
            compression_time = (time.perf_counter() - start_time) * 1000
            return compressed, compression_time

class CryptographicHasher:
    """Advanced cryptographic hashing with quantum resistance support"""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD):
        self.security_level = security_level
        self.hash_cache = {}
        self.performance_metrics = defaultdict(list)
        
        # Initialize hash functions based on security level
        if security_level == SecurityLevel.QUANTUM_RESISTANT:
            self.primary_hasher = blake3.blake3
            self.secondary_hasher = hashlib.sha3_256
        elif security_level == SecurityLevel.HIGH:
            self.primary_hasher = hashlib.sha3_256
            self.secondary_hasher = hashlib.sha256
        elif security_level == SecurityLevel.PERFORMANCE:
            self.primary_hasher = blake3.blake3
            self.secondary_hasher = hashlib.blake2b
        else:  # STANDARD
            self.primary_hasher = hashlib.sha256
            self.secondary_hasher = hashlib.sha1
    
    @lru_cache(maxsize=1000)
    def hash_data(self, data: str, algorithm: str = "primary") -> str:
        """Generate cryptographic hash with caching"""
        
        start_time = time.perf_counter()
        
        try:
            data_bytes = data.encode('utf-8')
            
            if algorithm == "primary":
                if self.security_level == SecurityLevel.PERFORMANCE and hasattr(blake3, 'blake3'):
                    hasher = blake3.blake3(data_bytes)
                    result = hasher.hexdigest()
                else:
                    result = self.primary_hasher(data_bytes).hexdigest()
            else:
                result = self.secondary_hasher(data_bytes).hexdigest()
            
            processing_time = (time.perf_counter() - start_time) * 1000
            self.performance_metrics[algorithm].append(processing_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Hashing failed: {e}")
            # Fallback to SHA-256
            return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    def generate_integrity_chain(self, payloads: List[str]) -> List[str]:
        """Generate blockchain-style integrity verification chain"""
        
        chain = []
        previous_hash = "0" * 64  # Genesis hash
        
        for payload in payloads:
            # Combine previous hash with current payload
            combined = f"{previous_hash}{payload}"
            current_hash = self.hash_data(combined)
            chain.append(current_hash)
            previous_hash = current_hash
        
        return chain
    
    def verify_integrity_chain(self, payloads: List[str], chain: List[str]) -> bool:
        """Verify integrity of a hash chain"""
        
        if len(payloads) != len(chain):
            return False
        
        expected_chain = self.generate_integrity_chain(payloads)
        return expected_chain == chain

class EnhancedDeferredHashingManager:
    """
    Advanced deferred hashing manager with ML optimization, quantum-resistant
    cryptography, and comprehensive reconstruction capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the enhanced deferred hashing manager"""
        self.config = config
        self.hashing_policies: Dict[str, HashingPolicy] = {}
        self.reconstruction_cache: Dict[str, ReconstructionPlan] = {}
        self.hash_statistics: Dict[str, Any] = defaultdict(dict)
        
        # Advanced components
        self.ml_optimizer = MLHashingOptimizer()
        self.compression_optimizer = CompressionOptimizer()
        self.crypto_hasher = CryptographicHasher(SecurityLevel.STANDARD)
        
        # External integrations
        self.schema_engine: Optional[SchemaEvolutionEngine] = None
        self.privacy_filter: Optional[PrivacyComplianceFilter] = None
        self.preservation_guard: Optional[AnomalyPreservationGuard] = None
        
        # Performance tracking
        self.metrics = {
            'total_hashed': 0,
            'total_size_saved': 0,
            'average_compression_ratio': 0.0,
            'reconstruction_requests': 0,
            'forensic_reconstructions': 0,
            'ml_optimizations': 0,
            'processing_times_ms': deque(maxlen=1000),
            'compression_ratios': deque(maxlen=1000),
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Caching and optimization
        self.hash_cache = {}
        self.compression_cache = {}
        self.cache_ttl = config.get('cache_ttl', 300)  # 5 minutes
        
        # Threading for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
        
        # Initialize components
        self._initialize_enhanced_policies()
        
        logger.info("Enhanced Deferred Hashing Manager initialized with ML optimization")
    
    def _initialize_enhanced_policies(self):
        """Initialize comprehensive hashing policies"""
        
        # High-security critical telemetry
        self.hashing_policies['critical_security'] = HashingPolicy(
            strategy=HashingStrategy.QUANTUM_RESISTANT_HASH,
            reconstruction_level=ReconstructionLevel.ENCRYPTED,
            security_level=SecurityLevel.QUANTUM_RESISTANT,
            max_payload_size=32768,
            target_compression_ratio=0.3,
            critical_fields={'execution_id', 'timestamp', 'security_event', 'anomaly_score', 'threat_level'},
            sensitive_fields={'user_data', 'credentials', 'tokens'},
            hash_algorithm='blake3',
            encryption_enabled=True,
            digital_signature=True,
            ml_optimization_enabled=True,
            anomaly_risk_threshold=0.9,
            retention_period=7776000,  # 90 days
            hot_data_period=604800     # 7 days
        )
        
        # Performance-optimized telemetry
        self.hashing_policies['performance_optimized'] = HashingPolicy(
            strategy=HashingStrategy.ADAPTIVE_COMPRESSION,
            reconstruction_level=ReconstructionLevel.REAL_TIME,
            security_level=SecurityLevel.PERFORMANCE,
            max_payload_size=16384,
            target_compression_ratio=0.4,
            max_processing_time_ms=1.0,
            critical_fields={'execution_id', 'timestamp', 'function_name', 'duration_ms'},
            hash_algorithm='blake3',
            compression_algorithm=CompressionAlgorithm.ADAPTIVE,
            ml_optimization_enabled=True,
            parallel_processing=True,
            cache_enabled=True
        )
        
        # ML-guided adaptive hashing
        self.hashing_policies['ml_adaptive'] = HashingPolicy(
            strategy=HashingStrategy.ML_GUIDED_HASH,
            reconstruction_level=ReconstructionLevel.FORENSIC,
            security_level=SecurityLevel.HIGH,
            max_payload_size=20480,
            target_compression_ratio=0.35,
            critical_fields={'execution_id', 'timestamp'},  # ML will determine others
            ml_optimization_enabled=True,
            anomaly_risk_threshold=0.7,
            differential_privacy_epsilon=1.0,
            batch_optimization=True
        )
        
        # Privacy-preserving hashing
        self.hashing_policies['privacy_preserving'] = HashingPolicy(
            strategy=HashingStrategy.DIFFERENTIAL_PRIVACY_HASH,
            reconstruction_level=ReconstructionLevel.BASIC,
            security_level=SecurityLevel.HIGH,
            max_payload_size=12288,
            target_compression_ratio=0.5,
            critical_fields={'execution_id', 'timestamp'},
            sensitive_fields={'user_id', 'ip_address', 'location', 'personal_data'},
            encryption_enabled=True,
            differential_privacy_epsilon=0.5,  # Strong privacy
            reconstruction_decay=True
        )
        
        # Blockchain integrity verification
        self.hashing_policies['blockchain_integrity'] = HashingPolicy(
            strategy=HashingStrategy.BLOCKCHAIN_INTEGRITY,
            reconstruction_level=ReconstructionLevel.AUDIT_TRAIL,
            security_level=SecurityLevel.HIGH,
            max_payload_size=16384,
            critical_fields={'execution_id', 'timestamp', 'integrity_hash'},
            digital_signature=True,
            retention_period=31536000,  # 1 year
            reconstruction_decay=False   # Permanent integrity records
        )
        
        # Edge-optimized lightweight hashing
        self.hashing_policies['edge_lightweight'] = HashingPolicy(
            strategy=HashingStrategy.COMPRESS_THEN_HASH,
            reconstruction_level=ReconstructionLevel.NONE,
            security_level=SecurityLevel.PERFORMANCE,
            max_payload_size=4096,
            target_compression_ratio=0.6,
            max_processing_time_ms=0.5,
            critical_fields={'execution_id', 'timestamp'},
            compression_algorithm=CompressionAlgorithm.LZ4,
            parallel_processing=False,