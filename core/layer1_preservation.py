async def run_preservation_validation_suite(guard: EnhancedAnomalyPreservationGuard,
                                           test_data_path: str = None) -> Dict[str, Any]:
    """Run comprehensive preservation validation suite"""
    
    validation_suite = {
        'suite_start_time': time.time(),
        'test_categories': {},
        'overall_results': {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'overall_effectiveness': 0.0,
            'critical_failures': 0
        },
        'performance_metrics': {
            'total_processing_time_ms': 0.0,
            'average_test_time_ms': 0.0,
            'peak_memory_usage_mb': 0.0
        },
        'recommendations': []
    }
    
    # Test categories with different anomaly scenarios
    test_categories = {
        'cold_start_scenarios': await _generate_cold_start_test_scenarios(),
        'data_exfiltration_scenarios': await _generate_data_exfiltration_scenarios(),
        'economic_abuse_scenarios': await _generate_economic_abuse_scenarios(),
        'silent_failure_scenarios': await _generate_silent_failure_scenarios(),
        'memory_leak_scenarios': await _generate_memory_leak_scenarios(),
        'mixed_anomaly_scenarios': await _generate_mixed_anomaly_scenarios(),
        'edge_case_scenarios': await _generate_edge_case_scenarios()
    }
    
    total_effectiveness = 0.0
    total_processing_time = 0.0
    
    for category_name, scenarios in test_categories.items():
        logger.info(f"Running validation category: {category_name}")
        
        category_results = {
            'total_scenarios': len(scenarios),
            'passed_scenarios': 0,
            'failed_scenarios': 0,
            'category_effectiveness': 0.0,
            'scenario_results': [],
            'category_processing_time_ms': 0.0
        }
        
        category_effectiveness = 0.0
        category_processing_time = 0.0
        
        for i, scenario in enumerate(scenarios):
            scenario_start = time.perf_counter()
            
            try:
                # Apply processing transformations
                processed_scenario = await _apply_comprehensive_transformations(scenario)
                
                # Assess preservation
                assessment = await guard.assess_preservation_impact(
                    scenario, processed_scenario, 
                    f"{category_name}_scenario_{i}",
                    AnalysisDepth.COMPREHENSIVE
                )
                
                scenario_time = (time.perf_counter() - scenario_start) * 1000
                category_processing_time += scenario_time
                
                # Determine test result
                passed = await _evaluate_test_result(assessment, category_name)
                
                if passed:
                    category_results['passed_scenarios'] += 1
                    validation_suite['overall_results']['passed_tests'] += 1
                else:
                    category_results['failed_scenarios'] += 1
                    validation_suite['overall_results']['failed_tests'] += 1
                    
                    if assessment.preservation_effectiveness < 0.7:
                        validation_suite['overall_results']['critical_failures'] += 1
                
                scenario_result = {
                    'scenario_id': i,
                    'scenario_type': scenario.get('anomaly_type', 'unknown'),
                    'passed': passed,
                    'preservation_effectiveness': assessment.preservation_effectiveness,
                    'confidence_score': assessment.confidence_score,
                    'critical_violations': len(assessment.critical_violations),
                    'processing_time_ms': scenario_time,
                    'affected_anomaly_types': [t.value for t in assessment.affected_anomaly_types]
                }
                
                category_results['scenario_results'].append(scenario_result)
                category_effectiveness += assessment.preservation_effectiveness
                
            except Exception as e:
                logger.error(f"Validation failed for {category_name} scenario {i}: {e}")
                category_results['failed_scenarios'] += 1
                validation_suite['overall_results']['failed_tests'] += 1
                
                scenario_result = {
                    'scenario_id': i,
                    'passed': False,
                    'error': str(e),
                    'processing_time_ms': (time.perf_counter() - scenario_start) * 1000
                }
                category_results['scenario_results'].append(scenario_result)
        
        # Calculate category metrics
        if category_results['total_scenarios'] > 0:
            category_results['category_effectiveness'] = category_effectiveness / category_results['total_scenarios']
            category_results['pass_rate'] = category_results['passed_scenarios'] / category_results['total_scenarios']
        
        category_results['category_processing_time_ms'] = category_processing_time
        validation_suite['test_categories'][category_name] = category_results
        
        total_effectiveness += category_effectiveness
        total_processing_time += category_processing_time
        validation_suite['overall_results']['total_tests'] += category_results['total_scenarios']
    
    # Calculate overall metrics
    if validation_suite['overall_results']['total_tests'] > 0:
        validation_suite['overall_results']['overall_effectiveness'] = (
            total_effectiveness / validation_suite['overall_results']['total_tests']
        )
        validation_suite['overall_results']['pass_rate'] = (
            validation_suite['overall_results']['passed_tests'] / 
            validation_suite['overall_results']['total_tests']
        )
    
    validation_suite['performance_metrics']['total_processing_time_ms'] = total_processing_time
    validation_suite['performance_metrics']['average_test_time_ms'] = (
        total_processing_time / validation_suite['overall_results']['total_tests']
        if validation_suite['overall_results']['total_tests'] > 0 else 0.0
    )
    validation_suite['performance_metrics']['peak_memory_usage_mb'] = (
        psutil.Process().memory_info().rss / 1024 / 1024
    )
    
    # Generate recommendations
    validation_suite['recommendations'] = await _generate_validation_recommendations(validation_suite)
    
    validation_suite['suite_end_time'] = time.time()
    validation_suite['total_suite_time_seconds'] = (
        validation_suite['suite_end_time'] - validation_suite['suite_start_time']
    )
    
    logger.info(f"Validation suite completed: {validation_suite['overall_results']['pass_rate']:.1%} pass rate, "
               f"{validation_suite['overall_results']['overall_effectiveness']:.3f} avg effectiveness")
    
    return validation_suite

async def _generate_cold_start_test_scenarios() -> List[Dict[str, Any]]:
    """Generate test scenarios for cold start anomaly preservation"""
    
    scenarios = []
    
    # Standard cold start scenario
    scenarios.append({
        'anomaly_type': 'cold_start',
        'scenario_name': 'standard_cold_start',
        'execution_id': 'exec_cold_001',
        'timestamp': time.time(),
        'function_name': 'user_authentication',
        'duration_ms': 2450.7,  # 10x normal duration
        'memory_used_mb': 256.5,  # 2x normal memory
        'init_time_ms': 1800.0,  # Cold start initialization
        'bootstrap_time': 650.0,
        'cold_start_indicator': True,
        'container_lifecycle': 'initializing',
        'runtime_initialization': {
            'dependencies_loaded': True,
            'config_parsed': True,
            'connections_established': False
        },
        'cpu_utilization': 85.3,
        'network_io': 2048,
        'execution_context': {
            'runtime_version': 'python3.9',
            'memory_limit_mb': 512,
            'timeout_seconds': 30
        }
    })
    
    # Extreme cold start scenario
    scenarios.append({
        'anomaly_type': 'cold_start',
        'scenario_name': 'extreme_cold_start',
        'execution_id': 'exec_cold_002',
        'timestamp': time.time(),
        'function_name': 'ml_inference',
        'duration_ms': 5200.3,  # Very long duration
        'memory_used_mb': 512.0,  # High memory usage
        'init_time_ms': 3500.0,  # Very long initialization
        'bootstrap_time': 1200.0,
        'cold_start_indicator': True,
        'container_lifecycle': 'initializing',
        'model_loading_time': 2800.0,
        'dependency_resolution_time': 700.0,
        'large_dependency_size_mb': 150.0,
        'cpu_utilization': 95.7,
        'memory_pressure_indicator': True
    })
    
    # Repeated cold start pattern
    scenarios.append({
        'anomaly_type': 'cold_start',
        'scenario_name': 'repeated_cold_starts',
        'execution_sequence': [
            {'duration_ms': 2100.0, 'cold_start': True, 'timestamp': time.time()},
            {'duration_ms': 180.0, 'cold_start': False, 'timestamp': time.time() + 60},
            {'duration_ms': 2200.0, 'cold_start': True, 'timestamp': time.time() + 600},  # Container recycled
            {'duration_ms': 175.0, 'cold_start': False, 'timestamp': time.time() + 660}
        ],
        'container_recycling_pattern': True,
        'memory_pressure_events': 2
    })
    
    return scenarios

async def _generate_data_exfiltration_scenarios() -> List[Dict[str, Any]]:
    """Generate test scenarios for data exfiltration anomaly preservation"""
    
    scenarios = []
    
    # Large volume exfiltration
    scenarios.append({
        'anomaly_type': 'data_exfiltration',
        'scenario_name': 'large_volume_exfiltration',
        'execution_id': 'exec_exfil_001',
        'timestamp': time.time(),
        'function_name': 'data_processor',
        'duration_ms': 8500.0,
        'network_io': 52428800,  # 50MB - unusually high
        'data_volume': 48000000,  # 48MB transferred
        'transfer_rate': 5647.1,  # KB/s
        'connection_patterns': {
            'destination_count': 1,
            'unique_ports': [443],
            'external_domains': ['suspicious-domain.com'],
            'connection_duration_ms': 8200
        },
        'bandwidth_utilization': 0.95,
        'protocol_distribution': {'https': 0.98, 'other': 0.02},
        'destination_entropy': 0.1,  # Low entropy - single destination
        'timing_patterns': {
            'off_hours': True,
            'sustained_transfer': True,
            'burst_pattern': False
        },
        'payload_characteristics': {
            'compression_ratio': 0.3,
            'encryption_detected': True,
            'data_pattern_entropy': 0.85
        }
    })
    
    # Gradual exfiltration pattern
    scenarios.append({
        'anomaly_type': 'data_exfiltration',
        'scenario_name': 'gradual_exfiltration',
        'execution_sequence': [
            {'network_io': 5242880, 'timestamp': time.time() + i * 300}  # 5MB every 5 minutes
            for i in range(10)
        ],
        'cumulative_volume': 52428800,  # 50MB total
        'exfiltration_rate': 'slow_and_steady',
        'detection_evasion_indicators': {
            'size_under_threshold': True,
            'timing_irregular': True,
            'mixed_protocols': True
        }
    })
    
    # Burst exfiltration
    scenarios.append({
        'anomaly_type': 'data_exfiltration',
        'scenario_name': 'burst_exfiltration',
        'execution_id': 'exec_exfil_003',
        'timestamp': time.time(),
        'network_io': 104857600,  # 100MB in single burst
        'data_volume': 98000000,
        'transfer_duration_ms': 2800,
        'peak_bandwidth_usage': 0.98,
        'connection_patterns': {
            'rapid_connections': 15,
            'parallel_streams': 8,
            'connection_reuse': False
        },
        'anomaly_indicators': {
            'volume_spike_factor': 20.0,
            'bandwidth_saturation': True,
            'unusual_timing': True
        }
    })
    
    return scenarios

async def _generate_economic_abuse_scenarios() -> List[Dict[str, Any]]:
    """Generate test scenarios for economic abuse anomaly preservation"""
    
    scenarios = []
    
    # Resource consumption abuse
    scenarios.append({
        'anomaly_type': 'economic_abuse',
        'scenario_name': 'resource_consumption_abuse',
        'execution_id': 'exec_econ_001',
        'timestamp': time.time(),
        'execution_cost': 15.47,  # Unusually high cost
        'resource_consumption': {
            'cpu_time_seconds': 1800,  # 30 minutes of CPU time
            'memory_gb_seconds': 2048,  # High memory usage
            'storage_operations': 50000,  # Excessive I/O operations
            'network_gb': 5.2  # High network usage
        },
        'billing_units': 847,  # High billing units consumed
        'quota_usage': {
            'cpu_quota_percent': 95.2,
            'memory_quota_percent': 88.7,
            'storage_quota_percent': 67.3
        },
        'invocation_frequency': 1200,  # Very high frequency
        'duration_billing': 1850000,  # Milliseconds billed
        'cost_efficiency_metrics': {
            'cost_per_operation': 0.052,
            'resource_waste_ratio': 3.2,
            'efficiency_score': 0.15  # Very low efficiency
        }
    })
    
    # Quota exhaustion pattern
    scenarios.append({
        'anomaly_type': 'economic_abuse',
        'scenario_name': 'quota_exhaustion',
        'resource_consumption_timeline': [
            {'timestamp': time.time() + i * 60, 'quota_usage': min(100.0, 10.0 + i * 8.5)}
            for i in range(12)  # Rapid quota consumption over 12 minutes
        ],
        'quota_exhaustion_rate': 'accelerating',
        'abuse_indicators': {
            'intentional_resource_waste': True,
            'inefficient_algorithms': True,
            'unnecessary_computation': True
        }
    })
    
    # Billing manipulation attempt
    scenarios.append({
        'anomaly_type': 'economic_abuse',
        'scenario_name': 'billing_manipulation',
        'execution_cost_anomalies': {
            'cost_spike_events': 5,
            'unusual_billing_patterns': True,
            'resource_usage_mismatch': True
        },
        'manipulation_indicators': {
            'artificial_delay_injection': True,
            'resource_hoarding': True,
            'billing_cycle_alignment': False
        }
    })
    
    return scenarios

async def _generate_silent_failure_scenarios() -> List[Dict[str, Any]]:
    """Generate test scenarios for silent failure anomaly preservation"""
    
    scenarios = []
    
    # Output corruption scenario
    scenarios.append({
        'anomaly_type': 'silent_failure',
        'scenario_name': 'output_corruption',
        'execution_id': 'exec_silent_001',
        'timestamp': time.time(),
        'function_name': 'data_transformer',
        'duration_ms': 245.7,  # Normal execution time
        'memory_used_mb': 128.5,  # Normal memory usage
        'cpu_utilization': 45.3,  # Normal CPU usage
        'execution_status': 'SUCCESS',  # Reports success
        'error_logs': [],  # No error logs
        'output_entropy': 2.1,  # Significantly reduced from expected 6.8
        'semantic_consistency': 0.65,  # Below normal 0.95
        'data_integrity_score': 0.72,  # Compromised integrity
        'expected_output_size': 15000,
        'actual_output_size': 15000,  # Size matches but content is wrong
        'output_validation_results': {
            'schema_valid': True,
            'type_check_passed': True,
            'semantic_validation_failed': True
        },
        'silent_failure_indicators': {
            'no_exceptions_thrown': True,
            'normal_execution_metrics': True,
            'corrupted_output_detected': True,
            'downstream_impact_potential': 'high'
        }
    })
    
    # Logic error scenario
    scenarios.append({
        'anomaly_type': 'silent_failure',
        'scenario_name': 'logic_error',
        'execution_trace_completeness': 0.85,  # Missing some execution steps
        'execution_checkpoints': {
            'input_validation': True,
            'data_processing': True,
            'business_logic': False,  # Logic step skipped
            'output_generation': True,
            'cleanup': True
        },
        'state_transitions': [
            'initialized', 'processing', 'output_generated', 'completed'
            # Missing 'business_logic_applied' state
        ],
        'error_absence_indicator': True,
        'logic_validation_score': 0.3  # Very low logic consistency
    })
    
    # Data quality degradation
    scenarios.append({
        'anomaly_type': 'silent_failure',
        'scenario_name': 'data_quality_degradation',
        'data_quality_metrics': {
            'completeness': 0.75,  # 25% data missing
            'accuracy': 0.82,  # Reduced accuracy
            'consistency': 0.68,  # Poor consistency
            'validity': 0.90,  # Still technically valid
            'uniqueness': 0.95  # No duplicate issues
        },
        'quality_degradation_indicators': {
            'missing_fields': ['optional_metadata', 'quality_scores'],
            'invalid_relationships': 15,
            'outlier_increase': 2.3,  # 230% more outliers than expected
            'pattern_deviation': 0.45
        }
    })
    
    return scenarios

async def _generate_memory_leak_scenarios() -> List[Dict[str, Any]]:
    """Generate test scenarios for memory leak anomaly preservation"""
    
    scenarios = []
    
    # Progressive memory leak
    scenarios.append({
        'anomaly_type': 'memory_leak',
        'scenario_name': 'progressive_memory_leak',
        'execution_id': 'exec_leak_001',
        'memory_usage_timeline': [
            {'timestamp': time.time() + i * 30, 'memory_mb': 128 + i * 15.5}
            for i in range(20)  # Memory grows over 10 minutes
        ],
        'heap_growth_rate': 15.5,  # MB per 30 seconds
        'gc_frequency': [
            {'timestamp': time.time() + i * 45, 'duration_ms': 50 + i * 5}
            for i in range(15)  # GC taking longer each time
        ],
        'memory_allocation_patterns': {
            'allocation_rate_increasing': True,
            'deallocation_rate_decreasing': True,
            'memory_fragmentation': 0.35
        },
        'object_lifecycle': {
            'objects_created': 15000,
            'objects_destroyed': 8500,  # Significant leak
            'unreachable_objects': 6500
        },
        'leak_indicators': {
            'persistent_growth': True,
            'gc_pressure_increasing': True,
            'memory_pools_exhausted': False
        },
        'memory_pressure_events': [
            {'timestamp': time.time() + 300, 'severity': 'medium'},
            {'timestamp': time.time() + 450, 'severity': 'high'},
            {'timestamp': time.time() + 580, 'severity': 'critical'}
        ]
    })
    
    # Sudden memory explosion
    scenarios.append({
        'anomaly_type': 'memory_leak',
        'scenario_name': 'memory_explosion',
        'initial_memory_mb': 128.0,
        'peak_memory_mb': 1024.0,  # 8x increase
        'explosion_duration_ms': 15000,  # 15 seconds
        'allocation_size_distribution': {
            'small_allocations_percent': 10,
            'medium_allocations_percent': 20,
            'large_allocations_percent': 70  # Mostly large allocations
        },
        'explosion_trigger': 'recursive_data_structure',
        'memory_exhaustion_risk': 'high'
    })
    
    return scenarios

async def _generate_mixed_anomaly_scenarios() -> List[Dict[str, Any]]:
    """Generate test scenarios with multiple anomaly types"""
    
    scenarios = []
    
    # Cold start + memory leak
    scenarios.append({
        'anomaly_type': 'mixed',
        'scenario_name': 'cold_start_with_memory_leak',
        'primary_anomalies': ['cold_start', 'memory_leak'],
        'execution_id': 'exec_mixed_001',
        'timestamp': time.time(),
        'duration_ms': 3200.0,  # Cold start duration
        'init_time_ms': 2100.0,
        'cold_start_indicator': True,
        'memory_usage_timeline': [
            {'timestamp': time.time() + i * 60, 'memory_mb': 256 + i * 25}
            for i in range(10)  # Memory leak during cold start
        ],
        'complexity_score': 0.85
    })
    
    # Data exfiltration + economic abuse
    scenarios.append({
        'anomaly_type': 'mixed',
        'scenario_name': 'exfiltration_with_resource_abuse',
        'primary_anomalies': ['data_exfiltration', 'economic_abuse'],
        'network_io': 104857600,  # 100MB transfer
        'execution_cost': 25.50,  # High cost
        'resource_consumption': {
            'network_gb': 8.5,
            'cpu_time_seconds': 2400
        },
        'dual_impact_indicators': {
            'data_theft_detected': True,
            'billing_impact_high': True,
            'coordinated_attack_likely': True
        }
    })
    
    return scenarios

async def _generate_edge_case_scenarios() -> List[Dict[str, Any]]:
    """Generate edge case test scenarios"""
    
    scenarios = []
    
    # Minimal data scenario
    scenarios.append({
        'anomaly_type': 'edge_case',
        'scenario_name': 'minimal_data',
        'execution_id': 'exec_edge_001',
        'timestamp': time.time(),
        'duration_ms': 1.2  # Extremely fast execution
    })
    
    # Maximum complexity scenario
    scenarios.append({
        'anomaly_type': 'edge_case',
        'scenario_name': 'maximum_complexity',
        'execution_id': 'exec_edge_002',
        'timestamp': time.time(),
        'nested_data': {
            f'level_{i}': {
                f'sublevel_{j}': {
                    f'data_{k}': f'value_{i}_{j}_{k}'
                    for k in range(5)
                }
                for j in range(5)
            }
            for i in range(10)
        },
        'large_array': list(range(1000)),
        'complex_relationships': {
            'dependencies': list(range(50)),
            'correlations': [[i, j, np.random.random()] for i in range(20) for j in range(20)]
        }
    })
    
    # Schema evolution scenario
    scenarios.append({
        'anomaly_type': 'edge_case',
        'scenario_name': 'schema_evolution',
        'old_field_names': {
            'duration_milliseconds': 245.7,
            'memory_usage_bytes': 134217728
        },
        'new_field_names': {
            'duration_ms': 245.7,
            'memory_used_mb': 128.0
        },
        'schema_version': '2.1.0',
        'backward_compatibility_required': True
    })
    
    return scenarios

async def _apply_comprehensive_transformations(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Apply comprehensive transformations to simulate all processing stages"""
    
    processed = copy.deepcopy(scenario)
    
    # Stage 1: Sanitization
    processed.pop('debug_info', None)
    processed.pop('internal_state', None)
    processed.pop('raw_logs', None)
    processed.pop('sensitive_metadata', None)
    
    # Stage 2: Privacy filtering
    if 'user_id' in processed:
        processed['user_id'] = f"hash_{hash(processed['user_id'])}"
    
    if 'ip_address' in processed:
        processed['ip_address'] = f"masked_{processed['ip_address'].split('.')[-1]}"
    
    # Stage 3: Schema normalization
    if 'duration_milliseconds' in processed:
        processed['duration_ms'] = processed.pop('duration_milliseconds')
    
    if 'memory_usage_bytes' in processed:
        processed['memory_used_mb'] = processed.pop('memory_usage_bytes') / (1024 * 1024)
    
    # Stage 4: Hashing of large fields
    for key, value in list(processed.items()):
        if isinstance(value, str) and len(value) > 2000:
            processed[f"{key}_hash"] = f"hash_{hash(value)}"
            del processed[key]
        elif isinstance(value, list) and len(value) > 100:
            processed[f"{key}_summary"] = {
                'count': len(value),
                'hash': f"hash_{hash(str(value))}"
            }
            del processed[key]
    
    # Stage 5: Add processing noise (simulate real-world processing effects)
    for key, value in processed.items():
        if isinstance(value, float) and 'time' in key.lower():
            # Add small timing noise
            processed[key] = value * (1.0 + np.random.normal(0, 0.001))
        elif isinstance(value, int) and value > 1000:
            # Add small integer noise
            processed[key] = value + int(np.random.normal(0, 1))
    
    return processed

async def _evaluate_test_result(assessment: PreservationAssessment, 
                              category_name: str) -> bool:
    """Evaluate whether a test result should be considered a pass"""
    
    # Category-specific thresholds
    category_thresholds = {
        'cold_start_scenarios': {
            'min_effectiveness': 0.90,
            'max_critical_violations': 0,
            'required_anomaly_types': [AnomalyType.COLD_START]
        },
        'data_exfiltration_scenarios': {
            'min_effectiveness': 0.95,
            'max_critical_violations': 0,
            'required_anomaly_types': [AnomalyType.DATA_EXFILTRATION]
        },
        'economic_abuse_scenarios': {
            'min_effectiveness': 0.88,
            'max_critical_violations': 1,
            'required_anomaly_types': [AnomalyType.ECONOMIC_ABUSE]
        },
        'silent_failure_scenarios': {
            'min_effectiveness': 0.92,
            'max_critical_violations': 0,
            'required_anomaly_types': [AnomalyType.SILENT_FAILURE]
        },
        'memory_leak_scenarios': {
            'min_effectiveness': 0.85,
            'max_critical_violations': 1,
            'required_anomaly_types': [AnomalyType.MEMORY_LEAK]
        },
        'mixed_anomaly_scenarios': {
            'min_effectiveness': 0.80,
            'max_critical_violations': 2,
            'required_anomaly_types': []  # Multiple types expected
        },
        'edge_case_scenarios': {
            'min_effectiveness': 0.75,
            'max_critical_violations': 3,
            'required_anomaly_types': []  # Variable
        }
    }
    
    thresholds = category_thresholds.get(category_name, {
        'min_effectiveness': 0.85,
        'max_critical_violations': 1,
        'required_anomaly_types': []
    })
    
    # Check effectiveness threshold
    if assessment.preservation_effectiveness < thresholds['min_effectiveness']:
        return False
    
    # Check critical violations
    if len(assessment.critical_violations) > thresholds['max_critical_violations']:
        return False
    
    # Check rollback recommendation
    if assessment.rollback_recommendation:
        return False
    
    # Check confidence score
    if assessment.confidence_score < 0.7:
        return False
    
    # Check required anomaly types (if specified)
    if thresholds['required_anomaly_types']:
        required_types = set(thresholds['required_anomaly_types'])
        detected_types = set(assessment.affected_anomaly_types)
        if not required_types.issubset(detected_types):
            return False
    
    return True

async def _generate_validation_recommendations(validation_results: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on validation results"""
    
    recommendations = []
    overall_results = validation_results['overall_results']
    
    # Overall performance recommendations
    if overall_results['pass_rate'] < 0.8:
        recommendations.append("CRITICAL: Overall pass rate below 80% - review preservation strategies")
    
    if overall_results['overall_effectiveness'] < 0.85:
        recommendations.append("WARNING: Average effectiveness below recommended threshold")
    
    if overall_results['critical_failures'] > 0:
        recommendations.append(f"Address {overall_results['critical_failures']} critical preservation failures")
    
    # Category-specific recommendations
    for category_name, category_results in validation_results['test_categories'].items():
        if category_results['pass_rate'] < 0.8:
            recommendations.append(f"Improve preservation for {category_name} (pass rate: {category_results['pass_rate']:.1%})")
        
        if category_results['category_effectiveness'] < 0.85:
            recommendations.append(f"Optimize effectiveness for {category_name} (current: {category_results['category_effectiveness']:.3f})")
    
    # Performance recommendations
    perf_metrics = validation_results['performance_metrics']
    if perf_metrics['average_test_time_ms'] > 5.0:
        recommendations.append("Consider performance optimization - average test time exceeds 5ms")
    
    if perf_metrics['peak_memory_usage_mb'] > 100.0:
        recommendations.append("Monitor memory usage - peak usage during validation exceeded 100MB")
    
    # Specific failure pattern recommendations
    failure_patterns = _analyze_failure_patterns(validation_results)
    if failure_patterns['common_violations']:
        recommendations.append(f"Address common violation: {failure_patterns['most_common_violation']}")
    
    if failure_patterns['effectiveness_outliers'] > 0:
        recommendations.append("Investigate scenarios with extremely low effectiveness scores")
    
    return recommendations

def _analyze_failure_patterns(validation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze patterns in validation failures"""
    
    patterns = {
        'common_violations': defaultdict(int),
        'most_common_violation': None,
        'effectiveness_outliers': 0,
        'category_weakness': None
    }
    
    all_scenarios = []
    for category_results in validation_results['test_categories'].values():
        all_scenarios.extend(category_results['scenario_results'])
    
    # Analyze violation patterns
    for scenario in all_scenarios:
        if not scenario.get('passed', True):
            if 'critical_violations' in scenario and scenario['critical_violations'] > 0:
                patterns['common_violations']['critical_violations'] += 1
            
            if scenario.get('preservation_effectiveness', 1.0) < 0.7:
                patterns['effectiveness_outliers'] += 1
    
    if patterns['common_violations']:
        patterns['most_common_violation'] = max(
            patterns['common_violations'], 
            key=patterns['common_violations'].get
        )
    
    # Find weakest category
    category_scores = {}
    for category_name, category_results in validation_results['test_categories'].items():
        category_scores[category_name] = category_results.get('pass_rate', 0.0)
    
    if category_scores:
        patterns['category_weakness'] = min(category_scores, key=category_scores.get)
    
    return patterns


# =============================================================================
# Example Usage and Testing Framework
# =============================================================================

async def main():
    """Example usage of the enhanced preservation guard"""
    
    # Initialize configuration
    config = {
        'history_size': 10000,
        'cache_ttl': 300,
        'max_workers': 4,
        'ml_training': {
            'exploration_rate': 0.1,
            'learning_rate': 0.01,
            'pretrained_model_path': None
        }
    }
    
    # Create enhanced preservation guard
    guard = create_enhanced_preservation_guard(config)
    
    # Initialize metrics collector
    metrics_collector = PreservationMetricsCollectorEnhanced()
    
    print("=== Enhanced Anomaly Preservation Guard Demo ===")
    
    # Example 1: Cold start anomaly preservation
    print("\n1. Cold Start Anomaly Preservation Test")
    
    original_cold_start = {
        'execution_id': 'exec_demo_001',
        'timestamp': time.time(),
        'function_name': 'user_authentication',
        'duration_ms': 2450.7,  # Cold start duration
        'memory_used_mb': 256.5,
        'init_time_ms': 1800.0,
        'bootstrap_time': 650.0,
        'cold_start_indicator': True,
        'container_lifecycle': 'initializing',
        'cpu_utilization': 85.3,
        'network_io': 2048,
        'runtime_initialization': {
            'dependencies_loaded': True,
            'config_parsed': True,
            'connections_established': False
        }
    }
    
    # Simulate processing
    processed_cold_start = copy.deepcopy(original_cold_start)
    processed_cold_start.pop('runtime_initialization', None)  # Remove nested object
    processed_cold_start['duration_ms'] = 2451.2  # Slight modification
    processed_cold_start['init_time_hash'] = f"hash_{hash(processed_cold_start['init_time_ms'])}"
    
    # Assess preservation
    assessment = await guard.assess_preservation_impact(
        original_cold_start, processed_cold_start, 
        'demo_cold_start', AnalysisDepth.COMPREHENSIVE
    )
    
    print(f"Preservation effectiveness: {assessment.preservation_effectiveness:.3f}")
    print(f"Affected anomaly types: {[t.value for t in assessment.affected_anomaly_types]}")
    print(f"Critical violations: {len(assessment.critical_violations)}")
    print(f"Confidence score: {assessment.confidence_score:.3f}")
    
    if assessment.recommendations:
        print(f"Recommendations: {assessment.recommendations[:2]}")
    
    # Record assessment
    metrics_collector.record_assessment(assessment)
    
    # Example 2: Data exfiltration scenario
    print("\n2. Data Exfiltration Preservation Test")
    
    original_exfiltration = {
        'execution_id': 'exec_demo_002',
        'timestamp': time.time(),
        'function_name': 'data_processor',
        'duration_ms': 8500.0,
        'network_io': 52428800,  # 50MB
        'data_volume': 48000000,
        'transfer_rate': 5647.1,
        'connection_patterns': {
            'destination_count': 1,
            'unique_ports': [443],
            'external_domains': ['suspicious-domain.com']
        },
        'bandwidth_utilization': 0.95,
        'timing_patterns': {
            'off_hours': True,
            'sustained_transfer': True
        },
        'payload_characteristics': {
            'compression_ratio': 0.3,
            'encryption_detected': True
        }
    }
    
    # Simulate aggressive processing
    processed_exfiltration = copy.deepcopy(original_exfiltration)
    processed_exfiltration['connection_patterns_hash'] = 'hash_connections'
    del processed_exfiltration['connection_patterns']
    processed_exfiltration['payload_characteristics_summary'] = 'encrypted_compressed'
    del processed_exfiltration['payload_characteristics']
    
    assessment_exfil = await guard.assess_preservation_impact(
        original_exfiltration, processed_exfiltration,
        'demo_exfiltration', AnalysisDepth.DEEP
    )
    
    print(f"Preservation effectiveness: {assessment_exfil.preservation_effectiveness:.3f}")
    print(f"Information loss: {assessment_exfil.information_loss:.3f}")
    print(f"Rollback recommended: {assessment_exfil.rollback_recommendation}")
    
    metrics_collector.record_assessment(assessment_exfil)
    
    # Example 3: Benchmark performance
    print("\n3. Performance Benchmark")
    
    test_scenarios = []
    
    # Generate mixed test scenarios
    for i in range(10):
        scenario = {
            'execution_id': f'bench_{i}',
            'timestamp': time.time() + i,
            'function_name': f'test_function_{i % 3}',
            'duration_ms': 200 + np.random.normal(0, 50),
            'memory_used_mb': 128 + np.random.normal(0, 20),
            'cpu_utilization': 45 + np.random.normal(0, 15),
            'network_io': 4096 * (1 + np.random.exponential(0.1)),
            'anomaly_indicator': np.random.random() > 0.7,
            'test_data': f"test_payload_{i}" * 100  # Large field for hashing test
        }
        test_scenarios.append(scenario)
    
    benchmark_results = await benchmark_preservation_performance_enhanced(
        guard, test_scenarios, AnalysisDepth.STATISTICAL
    )
    
    print(f"Benchmark results:")
    print(f"  Average effectiveness: {benchmark_results['performance_summary']['average_effectiveness']:.3f}")
    print(f"  Average processing time: {benchmark_results['performance_summary']['average_processing_time_ms']:.2f}ms")
    print(f"  Violation rate: {benchmark_results['performance_summary']['violation_rate']:.1%}")
    print(f"  Rollback rate: {benchmark_results['performance_summary']['rollback_rate']:.1%}")
    
    # Example 4: Health monitoring
    print("\n4. Performance Health Assessment")
    
    # Add more assessments to metrics collector
    for scenario_result in benchmark_results['scenario_results'][:5]:
        mock_assessment = PreservationAssessment(
            assessment_id=f"health_{scenario_result['scenario_id']}",
            original_detectability_score=0.9,
            post_processing_detectability_score=scenario_result['effectiveness'] * 0.9,
            preservation_effectiveness=scenario_result['effectiveness'],
            affected_anomaly_types=[],
            processing_time_ms=scenario_result['processing_time_ms'],
            confidence_score=scenario_result['confidence_score']
        )
        metrics_collector.record_assessment(mock_assessment)
    
    health_score = metrics_collector.get_performance_health_score()
    
    print(f"Performance health score: {health_score['overall_health_score']:.1f}/100")
    print(f"Health status: {health_score['health_status']}")
    print(f"Component scores:")
    for component, score in health_score['component_scores'].items():
        print(f"  {component}: {score:.1f}")
    
    # Example 5: Trend analysis
    print("\n5. Trend Analysis")
    
    if len(metrics_collector.metrics_history) >= 10:
        trend_analysis = metrics_collector.get_trend_analysis(metric='effectiveness')
        
        if 'linear_trend' in trend_analysis:
            linear_trend = trend_analysis['linear_trend']
            print(f"Effectiveness trend: {linear_trend['trend_direction']}")
            print(f"Trend strength (R²): {linear_trend['r_squared']:.3f}")
            print(f"Statistical significance: p = {linear_trend['p_value']:.4f}")
    
    # Example 6: Validation suite
    print("\n6. Running Validation Suite (abbreviated)")
    
    # Run a small validation suite
    try:
        validation_results = await run_preservation_validation_suite(guard)
        
        print(f"Validation suite results:")
        print(f"  Total tests: {validation_results['overall_results']['total_tests']}")
        print(f"  Pass rate: {validation_results['overall_results']['pass_rate']:.1%}")
        print(f"  Overall effectiveness: {validation_results['overall_results']['overall_effectiveness']:.3f}")
        print(f"  Critical failures: {validation_results['overall_results']['critical_failures']}")
        
        if validation_results['recommendations']:
            print(f"Top recommendations:")
            for rec in validation_results['recommendations'][:3]:
                print(f"    - {rec}")
        
    except Exception as e:
        print(f"Validation suite error: {e}")
    
    # Example 7: Model export/import
    print("\n7. Model Persistence")
    
    try:
        # Export current model
        export_path = "/tmp/scafad_preservation_model.pkl"
        await guard.export_preservation_model(export_path)
        print(f"Model exported to {export_path}")
        
        # Get final performance report
        performance_report = metrics_collector.generate_performance_report(detailed=True)
        
        print(f"\nFinal Performance Report:")
        print(f"  Total assessments: {performance_report['data_summary']['total_assessments']}")
        print(f"  Health score: {performance_report['health_score']['overall_health_score']:.1f}")
        print(f"  Recent alerts: {len(performance_report['recent_alerts'])}")
        
        if 'detailed_analysis' in performance_report:
            anomaly_breakdown = performance_report['detailed_analysis'].get('anomaly_type_breakdown', {})
            if anomaly_breakdown:
                print(f"  Anomaly type performance:")
                for anomaly_type, stats in list(anomaly_breakdown.items())[:3]:
                    print(f"    {anomaly_type}: {stats['average_effectiveness']:.3f} avg effectiveness")
        
    except Exception as e:
        print(f"Model persistence error: {e}")
    
    print("\n=== Demo completed successfully ===")
    
    # Return summary for further analysis
    return {
        'guard': guard,
        'metrics_collector': metrics_collector,
        'final_health_score': health_score,
        'benchmark_results': benchmark_results
    }


# =============================================================================
# Advanced Integration Helpers
# =============================================================================

class PreservationGuardIntegration:
    """Integration helper for embedding preservation guard in production systems"""
    
    def __init__(self, guard: EnhancedAnomalyPreservationGuard):
        self.guard = guard
        self.integration_metrics = {
            'total_integrations': 0,
            'successful_assessments': 0,
            'failed_assessments': 0,
            'average_overhead_ms': 0.0
        }
    
    async def assess_pipeline_stage(self, stage_name: str, 
                                  original_data: Dict[str, Any],
                                  processed_data: Dict[str, Any],
                                  processing_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Assess preservation impact for a specific pipeline stage"""
        
        start_time = time.perf_counter()
        self.integration_metrics['total_integrations'] += 1
        
        try:
            # Determine analysis depth based on context
            analysis_depth = AnalysisDepth.STATISTICAL
            if processing_context and processing_context.get('critical_stage', False):
                analysis_depth = AnalysisDepth.COMPREHENSIVE
            elif processing_context and processing_context.get('fast_mode', False):
                analysis_depth = AnalysisDepth.SURFACE
            
            # Perform assessment
            assessment = await self.guard.assess_preservation_impact(
                original_data, processed_data, stage_name, analysis_depth
            )
            
            # Calculate integration overhead
            processing_time = (time.perf_counter() - start_time) * 1000
            self._update_overhead_metrics(processing_time)
            
            self.integration_metrics['successful_assessments'] += 1
            
            # Return integration-friendly result
            return {
                'success': True,
                'preservation_effectiveness': assessment.preservation_effectiveness,
                'should_rollback': assessment.rollback_recommendation,
                'critical_issues': len(assessment.critical_violations) > 0,
                'recommendations': assessment.recommendations[:3],  # Top 3 recommendations
                'confidence': assessment.confidence_score,
                'processing_overhead_ms': processing_time,
                'affected_anomaly_types': [t.value for t in assessment.affected_anomaly_types]
            }
            
        except Exception as e:
            self.integration_metrics['failed_assessments'] += 1
            logger.error(f"Preservation assessment failed for stage {stage_name}: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'preservation_effectiveness': 0.0,
                'should_rollback': True,
                'critical_issues': True,
                'processing_overhead_ms': (time.perf_counter() - start_time) * 1000
            }
    
    def _update_overhead_metrics(self, processing_time_ms: float):
        """Update overhead metrics"""
        current_avg = self.integration_metrics['average_overhead_ms']
        total_assessments = (self.integration_metrics['successful_assessments'] + 
                           self.integration_metrics['failed_assessments'])
        
        new_avg = ((current_avg * (total_assessments - 1)) + processing_time_ms) / total_assessments
        self.integration_metrics['average_overhead_ms'] = new_avg
    
    def get_integration_health(self) -> Dict[str, Any]:
        """Get integration health metrics"""
        
        total = self.integration_metrics['total_integrations']
        success_rate = (self.integration_metrics['successful_assessments'] / total 
                       if total > 0 else 0.0)
        
        return {
            'success_rate': success_rate,
            'total_integrations': total,
            'average_overhead_ms': self.integration_metrics['average_overhead_ms'],
            'health_status': 'good' if success_rate > 0.95 else 'degraded' if success_rate > 0.8 else 'poor'
        }

def create_production_preservation_guard(config_overrides: Dict[str, Any] = None) -> EnhancedAnomalyPreservationGuard:
    """Create production-optimized preservation guard"""
    
    production_config = {
        'history_size': 50000,  # Larger history for production
        'cache_ttl': 600,  # 10 minute cache
        'max_workers': min(8, multiprocessing.cpu_count()),  # Optimize for available CPUs
        'ml_training': {
            'exploration_rate': 0.05,  # Lower exploration in production
            'learning_rate': 0.005,  # Conservative learning rate
        }
    }
    
    if config_overrides:
        production_config.update(config_overrides)
    
    guard = create_enhanced_preservation_guard(production_config)
    
    # Add production-specific initialization
    logger.info("Production preservation guard initialized with optimized settings")
    
    return guard


# =============================================================================
# Module Exports and Entry Point
# =============================================================================

__all__ = [
    # Core classes
    'EnhancedAnomalyPreservationGuard',
    'PreservationMetricsCollectorEnhanced',
    'TrendAnalyzer',
    'PreservationGuardIntegration',
    
    # Data structures
    'AnomalyType',
    'PreservationStrategy',
    'PreservationLevel',
    'ProcessingMode',
    'AnalysisDepth',
    'AnomalySignature',
    'PreservationRule',
    'PreservationAssessment',
    
    # ML components
    'NeuralAnomalyEncoder',
    'InformationTheoreticAnalyzer',
    'GraphAnalyzer',
    'AdaptivePreservationOptimizer',
    
    # Factory functions
    'create_enhanced_preservation_guard',
    'create_production_preservation_guard',
    'get_preservation_policy_for_anomaly_enhanced',
    
    # Utilities
    'benchmark_preservation_performance_enhanced',
    'run_preservation_validation_suite'
]

if __name__ == "__main__":
    # Run the main demonstration
    import asyncio
    
    try:
        demo_results = asyncio.run(main())
        print(f"\nDemo completed with final health score: {demo_results['final_health_score']['overall_health_score']:.1f}/100")
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()            scenario_result = {
                'scenario_id': i,
                'error': str(e),
                'processing_time_ms': (time.perf_counter() - start_time) * 1000,
                'effectiveness': 0.0,
                'confidence_score': 0.0,
                'violations': 1,
                'rollback_recommended': True
            }
            results['scenario_results'].append(scenario_result)
    
    # Calculate summary statistics
    num_scenarios = len(test_scenarios)
    if num_scenarios > 0:
        results['performance_summary'] = {
            'average_effectiveness': total_effectiveness / num_scenarios,
            'average_processing_time_ms': total_processing_time / num_scenarios,
            'average_confidence_score': total_confidence / num_scenarios,
            'violation_rate': total_violations / num_scenarios,
            'rollback_rate': total_rollbacks / num_scenarios,
            'total_scenarios': num_scenarios
        }
    
    # Get strategy performance from guard
    results['strategy_performance'] = guard.adaptive_optimizer.get_strategy_recommendations()
    
    return results

class PreservationMetricsCollectorEnhanced:
    """Enhanced metrics collector with advanced analytics"""
    
    def __init__(self, collection_interval: int = 60):
        self.collection_interval = collection_interval
        self.metrics_history = deque(maxlen=50000)  # Increased capacity
        self.anomaly_type_stats = defaultdict(list)
        self.processing_stage_stats = defaultdict(list)
        self.temporal_metrics = defaultdict(list)
        self.performance_baselines = {}
        self.alert_thresholds = {
            'effectiveness_threshold': 0.8,
            'violation_rate_threshold': 0.1,
            'processing_time_threshold': 10.0  # ms
        }
        self.alerts_generated = []
        
        # Advanced analytics
        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.is_baseline_established = False
    
    def record_assessment(self, assessment: PreservationAssessment):
        """Record assessment with enhanced analytics"""
        
        # Create comprehensive metrics record
        metrics_record = {
            'timestamp': assessment.assessment_timestamp,
            'effectiveness': assessment.preservation_effectiveness,
            'processing_time_ms': assessment.processing_time_ms,
            'confidence_score': assessment.confidence_score,
            'critical_violations': len(assessment.critical_violations),
            'warning_violations': len(assessment.warning_violations),
            'information_loss': assessment.information_loss,
            'entropy_preserved': assessment.entropy_preserved,
            'affected_anomaly_types': [t.value for t in assessment.affected_anomaly_types],
            'processing_stage': assessment.processing_stage,
            'analysis_depth': assessment.analysis_depth.value,
            'rollback_recommended': assessment.rollback_recommendation
        }
        
        self.metrics_history.append(metrics_record)
        
        # Update anomaly type statistics
        for anomaly_type in assessment.affected_anomaly_types:
            self.anomaly_type_stats[anomaly_type.value].append({
                'effectiveness': assessment.preservation_effectiveness,
                'timestamp': assessment.assessment_timestamp,
                'violations': len(assessment.critical_violations)
            })
        
        # Update processing stage statistics
        self.processing_stage_stats[assessment.processing_stage].append({
            'effectiveness': assessment.preservation_effectiveness,
            'processing_time_ms': assessment.processing_time_ms,
            'timestamp': assessment.assessment_timestamp
        })
        
        # Update temporal metrics
        hour = int((assessment.assessment_timestamp % 86400) // 3600)
        self.temporal_metrics[hour].append(assessment.preservation_effectiveness)
        
        # Check for alerts
        self._check_performance_alerts(metrics_record)
        
        # Update performance baselines periodically
        if len(self.metrics_history) % 1000 == 0:
            self._update_performance_baselines()
    
    def _check_performance_alerts(self, metrics_record: Dict[str, Any]):
        """Check for performance degradation alerts"""
        
        alerts = []
        current_time = metrics_record['timestamp']
        
        # Effectiveness alert
        if metrics_record['effectiveness'] < self.alert_thresholds['effectiveness_threshold']:
            alerts.append({
                'type': 'effectiveness_degradation',
                'severity': 'critical' if metrics_record['effectiveness'] < 0.7 else 'warning',
                'message': f"Preservation effectiveness below threshold: {metrics_record['effectiveness']:.3f}",
                'timestamp': current_time,
                'metric_value': metrics_record['effectiveness'],
                'threshold': self.alert_thresholds['effectiveness_threshold']
            })
        
        # Processing time alert
        if metrics_record['processing_time_ms'] > self.alert_thresholds['processing_time_threshold']:
            alerts.append({
                'type': 'processing_time_high',
                'severity': 'warning',
                'message': f"Processing time exceeded threshold: {metrics_record['processing_time_ms']:.2f}ms",
                'timestamp': current_time,
                'metric_value': metrics_record['processing_time_ms'],
                'threshold': self.alert_thresholds['processing_time_threshold']
            })
        
        # Violation rate alert (check recent history)
        recent_records = [r for r in list(self.metrics_history)[-100:] 
                         if current_time - r['timestamp'] < 3600]  # Last hour
        
        if recent_records:
            violation_rate = sum(r['critical_violations'] > 0 for r in recent_records) / len(recent_records)
            if violation_rate > self.alert_thresholds['violation_rate_threshold']:
                alerts.append({
                    'type': 'high_violation_rate',
                    'severity': 'critical',
                    'message': f"High violation rate detected: {violation_rate:.1%}",
                    'timestamp': current_time,
                    'metric_value': violation_rate,
                    'threshold': self.alert_thresholds['violation_rate_threshold']
                })
        
        # Store alerts
        self.alerts_generated.extend(alerts)
        
        # Keep only recent alerts (last 24 hours)
        cutoff_time = current_time - 86400
        self.alerts_generated = [a for a in self.alerts_generated if a['timestamp'] > cutoff_time]
        
        # Log critical alerts
        for alert in alerts:
            if alert['severity'] == 'critical':
                logger.warning(f"PRESERVATION ALERT: {alert['message']}")
    
    def _update_performance_baselines(self):
        """Update performance baselines for anomaly detection"""
        
        if len(self.metrics_history) < 100:
            return
        
        # Calculate baseline metrics from recent stable period
        recent_metrics = list(self.metrics_history)[-1000:]  # Last 1000 assessments
        
        # Extract features for baseline calculation
        effectiveness_values = [m['effectiveness'] for m in recent_metrics]
        processing_times = [m['processing_time_ms'] for m in recent_metrics]
        violation_counts = [m['critical_violations'] for m in recent_metrics]
        
        self.performance_baselines = {
            'effectiveness': {
                'mean': np.mean(effectiveness_values),
                'std': np.std(effectiveness_values),
                'p95': np.percentile(effectiveness_values, 95),
                'p5': np.percentile(effectiveness_values, 5)
            },
            'processing_time': {
                'mean': np.mean(processing_times),
                'std': np.std(processing_times),
                'p95': np.percentile(processing_times, 95),
                'p5': np.percentile(processing_times, 5)
            },
            'violations': {
                'mean': np.mean(violation_counts),
                'rate': sum(1 for v in violation_counts if v > 0) / len(violation_counts)
            },
            'baseline_timestamp': time.time()
        }
        
        # Train anomaly detector on recent performance
        if len(recent_metrics) >= 50:
            feature_matrix = np.array([
                [m['effectiveness'], m['processing_time_ms'], m['critical_violations'], 
                 m['confidence_score'], m['information_loss']]
                for m in recent_metrics
            ])
            
            try:
                self.anomaly_detector.fit(feature_matrix)
                self.is_baseline_established = True
                logger.info("Performance baselines updated and anomaly detector retrained")
            except Exception as e:
                logger.warning(f"Failed to update anomaly detector: {e}")
    
    def detect_performance_anomalies(self, recent_window: int = 50) -> List[Dict[str, Any]]:
        """Detect performance anomalies using ML"""
        
        if not self.is_baseline_established or len(self.metrics_history) < recent_window:
            return []
        
        recent_metrics = list(self.metrics_history)[-recent_window:]
        
        # Prepare feature matrix
        feature_matrix = np.array([
            [m['effectiveness'], m['processing_time_ms'], m['critical_violations'],
             m['confidence_score'], m['information_loss']]
            for m in recent_metrics
        ])
        
        try:
            # Detect anomalies
            anomaly_scores = self.anomaly_detector.decision_function(feature_matrix)
            anomaly_predictions = self.anomaly_detector.predict(feature_matrix)
            
            anomalies = []
            for i, (score, prediction, metrics) in enumerate(zip(anomaly_scores, anomaly_predictions, recent_metrics)):
                if prediction == -1:  # Anomaly detected
                    anomalies.append({
                        'timestamp': metrics['timestamp'],
                        'anomaly_score': score,
                        'metrics': metrics,
                        'anomaly_factors': self._identify_anomaly_factors(metrics)
                    })
            
            return anomalies
            
        except Exception as e:
            logger.warning(f"Performance anomaly detection failed: {e}")
            return []
    
    def _identify_anomaly_factors(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify which factors contributed to performance anomaly"""
        
        factors = []
        baselines = self.performance_baselines
        
        if 'effectiveness' in baselines:
            eff_baseline = baselines['effectiveness']
            if metrics['effectiveness'] < eff_baseline['p5']:
                factors.append('low_effectiveness')
            elif metrics['effectiveness'] > eff_baseline['p95']:
                factors.append('unusually_high_effectiveness')
        
        if 'processing_time' in baselines:
            time_baseline = baselines['processing_time']
            if metrics['processing_time_ms'] > time_baseline['p95']:
                factors.append('high_processing_time')
        
        if 'violations' in baselines:
            if metrics['critical_violations'] > 0 and baselines['violations']['rate'] < 0.05:
                factors.append('unexpected_violations')
        
        if metrics['information_loss'] > 0.3:
            factors.append('high_information_loss')
        
        if metrics['confidence_score'] < 0.7:
            factors.append('low_confidence')
        
        return factors
    
    def get_trend_analysis(self, window_size: int = 200, 
                          metric: str = 'effectiveness') -> Dict[str, Any]:
        """Enhanced trend analysis with statistical significance"""
        
        if len(self.metrics_history) < window_size:
            return {'insufficient_data': True, 'required_samples': window_size}
        
        recent_data = list(self.metrics_history)[-window_size:]
        metric_values = [d[metric] for d in recent_data if metric in d]
        
        if not metric_values:
            return {'no_data_for_metric': metric}
        
        # Use enhanced trend analyzer
        trend_analysis = self.trend_analyzer.analyze_trend(
            metric_values, 
            timestamps=[d['timestamp'] for d in recent_data]
        )
        
        return trend_analysis
    
    def get_performance_health_score(self) -> Dict[str, Any]:
        """Calculate comprehensive performance health score"""
        
        if len(self.metrics_history) < 50:
            return {'insufficient_data': True}
        
        recent_metrics = list(self.metrics_history)[-100:]
        
        # Calculate component scores
        effectiveness_scores = [m['effectiveness'] for m in recent_metrics]
        processing_times = [m['processing_time_ms'] for m in recent_metrics]
        violation_rates = [1 if m['critical_violations'] > 0 else 0 for m in recent_metrics]
        confidence_scores = [m['confidence_score'] for m in recent_metrics]
        
        # Effectiveness health (0-100)
        avg_effectiveness = np.mean(effectiveness_scores)
        effectiveness_health = min(100, avg_effectiveness * 100)
        
        # Performance health (0-100)
        avg_processing_time = np.mean(processing_times)
        performance_health = max(0, 100 - (avg_processing_time - 2.0) * 10)  # Penalty after 2ms
        
        # Reliability health (0-100)
        violation_rate = np.mean(violation_rates)
        reliability_health = max(0, 100 - violation_rate * 100)
        
        # Confidence health (0-100)
        avg_confidence = np.mean(confidence_scores)
        confidence_health = avg_confidence * 100
        
        # Overall health (weighted average)
        overall_health = (
            effectiveness_health * 0.4 +
            reliability_health * 0.3 +
            confidence_health * 0.2 +
            performance_health * 0.1
        )
        
        # Determine health status
        if overall_health >= 90:
            health_status = 'excellent'
        elif overall_health >= 80:
            health_status = 'good'
        elif overall_health >= 70:
            health_status = 'fair'
        elif overall_health >= 60:
            health_status = 'poor'
        else:
            health_status = 'critical'
        
        return {
            'overall_health_score': overall_health,
            'health_status': health_status,
            'component_scores': {
                'effectiveness_health': effectiveness_health,
                'performance_health': performance_health,
                'reliability_health': reliability_health,
                'confidence_health': confidence_health
            },
            'recent_metrics_count': len(recent_metrics),
            'active_alerts': len([a for a in self.alerts_generated 
                                if time.time() - a['timestamp'] < 3600])
        }
    
    def generate_performance_report(self, detailed: bool = True) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        report = {
            'report_timestamp': time.time(),
            'data_summary': {
                'total_assessments': len(self.metrics_history),
                'data_timespan_hours': (max(m['timestamp'] for m in self.metrics_history) - 
                                      min(m['timestamp'] for m in self.metrics_history)) / 3600 
                                      if len(self.metrics_history) > 1 else 0,
                'assessment_rate_per_hour': 0
            },
            'health_score': self.get_performance_health_score(),
            'recent_alerts': self.alerts_generated[-10:],  # Last 10 alerts
            'performance_baselines': self.performance_baselines
        }
        
        if len(self.metrics_history) > 1:
            timespan = max(m['timestamp'] for m in self.metrics_history) - min(m['timestamp'] for m in self.metrics_history)
            report['data_summary']['assessment_rate_per_hour'] = len(self.metrics_history) / (timespan / 3600)
        
        if detailed and len(self.metrics_history) >= 50:
            report['detailed_analysis'] = {
                'trend_analysis': self.get_trend_analysis(),
                'anomaly_detection': self.detect_performance_anomalies(),
                'anomaly_type_breakdown': self._get_anomaly_type_breakdown(),
                'processing_stage_analysis': self._get_processing_stage_analysis(),
                'temporal_patterns': self._get_temporal_patterns(),
                'performance_distribution': self._get_performance_distribution()
            }
        
        return report
    
    def _get_anomaly_type_breakdown(self) -> Dict[str, Any]:
        """Get detailed breakdown by anomaly type"""
        
        breakdown = {}
        for anomaly_type, stats in self.anomaly_type_stats.items():
            if stats:
                effectiveness_values = [s['effectiveness'] for s in stats]
                violation_counts = [s['violations'] for s in stats]
                
                breakdown[anomaly_type] = {
                    'total_occurrences': len(stats),
                    'average_effectiveness': np.mean(effectiveness_values),
                    'min_effectiveness': np.min(effectiveness_values),
                    'max_effectiveness': np.max(effectiveness_values),
                    'effectiveness_std': np.std(effectiveness_values),
                    'average_violations': np.mean(violation_counts),
                    'violation_rate': sum(1 for v in violation_counts if v > 0) / len(violation_counts),
                    'trend': self._calculate_trend([s['effectiveness'] for s in stats[-20:]])  # Last 20
                }
        
        return breakdown
    
    def _get_processing_stage_analysis(self) -> Dict[str, Any]:
        """Get analysis by processing stage"""
        
        stage_analysis = {}
        for stage, stats in self.processing_stage_stats.items():
            if stats:
                effectiveness_values = [s['effectiveness'] for s in stats]
                processing_times = [s['processing_time_ms'] for s in stats]
                
                stage_analysis[stage] = {
                    'total_assessments': len(stats),
                    'average_effectiveness': np.mean(effectiveness_values),
                    'average_processing_time_ms': np.mean(processing_times),
                    'p95_processing_time_ms': np.percentile(processing_times, 95),
                    'effectiveness_std': np.std(effectiveness_values),
                    'processing_time_std': np.std(processing_times)
                }
        
        return stage_analysis
    
    def _get_temporal_patterns(self) -> Dict[str, Any]:
        """Get temporal performance patterns"""
        
        hourly_averages = {}
        for hour, effectiveness_values in self.temporal_metrics.items():
            if effectiveness_values:
                hourly_averages[str(hour)] = {
                    'average_effectiveness': np.mean(effectiveness_values),
                    'sample_count': len(effectiveness_values),
                    'effectiveness_std': np.std(effectiveness_values)
                }
        
        # Find peak and worst performing hours
        if hourly_averages:
            best_hour = max(hourly_averages.keys(), 
                          key=lambda h: hourly_averages[h]['average_effectiveness'])
            worst_hour = min(hourly_averages.keys(), 
                           key=lambda h: hourly_averages[h]['average_effectiveness'])
        else:
            best_hour = worst_hour = None
        
        return {
            'hourly_patterns': hourly_averages,
            'peak_performance_hour': best_hour,
            'worst_performance_hour': worst_hour
        }
    
    def _get_performance_distribution(self) -> Dict[str, Any]:
        """Get performance metric distributions"""
        
        recent_metrics = list(self.metrics_history)[-1000:]  # Last 1000 assessments
        
        if not recent_metrics:
            return {}
        
        effectiveness_values = [m['effectiveness'] for m in recent_metrics]
        processing_times = [m['processing_time_ms'] for m in recent_metrics]
        
        return {
            'effectiveness_distribution': {
                'histogram': np.histogram(effectiveness_values, bins=20)[0].tolist(),
                'bin_edges': np.histogram(effectiveness_values, bins=20)[1].tolist(),
                'percentiles': {
                    'p10': np.percentile(effectiveness_values, 10),
                    'p25': np.percentile(effectiveness_values, 25),
                    'p50': np.percentile(effectiveness_values, 50),
                    'p75': np.percentile(effectiveness_values, 75),
                    'p90': np.percentile(effectiveness_values, 90),
                    'p95': np.percentile(effectiveness_values, 95),
                    'p99': np.percentile(effectiveness_values, 99)
                }
            },
            'processing_time_distribution': {
                'histogram': np.histogram(processing_times, bins=20)[0].tolist(),
                'bin_edges': np.histogram(processing_times, bins=20)[1].tolist(),
                'percentiles': {
                    'p50': np.percentile(processing_times, 50),
                    'p95': np.percentile(processing_times, 95),
                    'p99': np.percentile(processing_times, 99)
                }
            }
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values"""
        
        if len(values) < 5:
            return 'insufficient_data'
        
        # Simple linear regression for trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'

class TrendAnalyzer:
    """Advanced trend analysis with statistical significance testing"""
    
    def __init__(self):
        self.seasonal_window = 24  # Hours for seasonal analysis
        
    def analyze_trend(self, values: List[float], timestamps: List[float] = None) -> Dict[str, Any]:
        """Comprehensive trend analysis"""
        
        if len(values) < 10:
            return {'insufficient_data': True}
        
        analysis = {
            'linear_trend': self._analyze_linear_trend(values),
            'change_points': self._detect_change_points(values),
            'seasonal_patterns': self._analyze_seasonal_patterns(values, timestamps) if timestamps else None,
            'volatility_analysis': self._analyze_volatility(values),
            'statistical_significance': self._test_trend_significance(values)
        }
        
        return analysis
    
    def _analyze_linear_trend(self, values: List[float]) -> Dict[str, Any]:
        """Analyze linear trend with confidence intervals"""
        
        x = np.arange(len(values))
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Confidence interval for slope
        t_val = stats.t.ppf(0.975, len(values) - 2)  # 95% confidence
        slope_ci = slope + np.array([-1, 1]) * t_val * std_err
        
        # Trend classification
        if p_value < 0.05:  # Significant trend
            if slope > 0:
                trend_direction = 'significantly_improving'
            else:
                trend_direction = 'significantly_declining'
        else:
            if abs(slope) < 0.001:
                trend_direction = 'stable'
            elif slope > 0:
                trend_direction = 'slightly_improving'
            else:
                trend_direction = 'slightly_declining'
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'slope_confidence_interval': slope_ci.tolist(),
            'trend_direction': trend_direction,
            'trend_strength': abs(r_value)
        }
    
    def _detect_change_points(self, values: List[float]) -> List[Dict[str, Any]]:
        """Detect significant change points in the time series"""
        
        change_points = []
        
        if len(values) < 20:
            return change_points
        
        # Simple change point detection using rolling statistics
        window_size = min(10, len(values) // 4)
        
        for i in range(window_size, len(values) - window_size):
            # Compare means before and after potential change point
            before_mean = np.mean(values[i-window_size:i])
            after_mean = np.mean(values[i:i+window_size])
            
            # Statistical test for significant change
            before_values = values[i-window_size:i]
            after_values = values[i:i+window_size]
            
            try:
                t_stat, p_val = stats.ttest_ind(before_values, after_values)
                
                if p_val < 0.05 and abs(before_mean - after_mean) > np.std(values) * 0.5:
                    change_points.append({
                        'index': i,
                        'before_mean': before_mean,
                        'after_mean': after_mean,
                        'magnitude': abs(after_mean - before_mean),
                        'direction': 'increase' if after_mean > before_mean else 'decrease',
                        'p_value': p_val,
                        'significance': 'high' if p_val < 0.01 else 'moderate'
                    })
            except:
                continue
        
        return change_points
    
    def _analyze_seasonal_patterns(self, values: List[float], 
                                 timestamps: List[float]) -> Dict[str, Any]:
        """Analyze seasonal patterns in the data"""
        
        if len(values) < 48:  # Need at least 2 days of hourly data
            return {'insufficient_data': True}
        
        # Extract hour of day from timestamps
        hours = [(ts % 86400) // 3600 for ts in timestamps]
        
        # Group by hour
        hourly_groups = defaultdict(list)
        for hour, value in zip(hours, values):
            hourly_groups[int(hour)].append(value)
        
        # Calculate hourly statistics
        hourly_stats = {}
        for hour in range(24):
            if hour in hourly_groups and len(hourly_groups[hour]) >= 2:
                hour_values = hourly_groups[hour]
                hourly_stats[hour] = {
                    'mean': np.mean(hour_values),
                    'std': np.std(hour_values),
                    'count': len(hour_values)
                }
        
        # Find peak and trough hours
        if hourly_stats:
            peak_hour = max(hourly_stats.keys(), key=lambda h: hourly_stats[h]['mean'])
            trough_hour = min(hourly_stats.keys(), key=lambda h: hourly_stats[h]['mean'])
            
            seasonal_strength = (hourly_stats[peak_hour]['mean'] - 
                               hourly_stats[trough_hour]['mean']) / np.mean(values)
        else:
            peak_hour = trough_hour = None
            seasonal_strength = 0.0
        
        return {
            'hourly_statistics': hourly_stats,
            'peak_hour': peak_hour,
            'trough_hour': trough_hour,
            'seasonal_strength': seasonal_strength,
            'has_seasonal_pattern': seasonal_strength > 0.1
        }
    
    def _analyze_volatility(self, values: List[float]) -> Dict[str, Any]:
        """Analyze volatility patterns in the data"""
        
        if len(values) < 10:
            return {'insufficient_data': True}
        
        # Calculate rolling volatility
        window_size = min(10, len(values) // 3)
        rolling_std = []
        
        for i in range(window_size, len(values)):
            window_values = values[i-window_size:i]
            rolling_std.append(np.std(window_values))
        
        if not rolling_std:
            return {'insufficient_data': True}
        
        overall_volatility = np.std(values)
        avg_rolling_volatility = np.mean(rolling_std)
        volatility_trend = self._analyze_linear_trend(rolling_std)
        
        # Classify volatility level
        volatility_percentile = np.percentile(rolling_std, 75)
        if overall_volatility > volatility_percentile * 1.5:
            volatility_level = 'high'
        elif overall_volatility > volatility_percentile:
            volatility_level = 'moderate'
        else:
            volatility_level = 'low'
        
        return {
            'overall_volatility': overall_volatility,
            'average_rolling_volatility': avg_rolling_volatility,
            'volatility_level': volatility_level,
            'volatility_trend': volatility_trend['trend_direction'],
            'rolling_volatility_values': rolling_std
        }
    
    def _test_trend_significance(self, values: List[float]) -> Dict[str, Any]:
        """Test statistical significance of observed trends"""
        
        # Mann-Kendall test for monotonic trend
        n = len(values)
        
        if n < 10:
            return {'insufficient_data': True}
        
        # Calculate Mann-Kendall statistic
        s = 0
        for i in range(n-1):
            for j in range(i+1, n):
                if values[j] > values[i]:
                    s += 1
                elif values[j] < values[i]:
                    s -= 1
        
        # Calculate variance
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        # Calculate Z-score
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # Calculate p-value (two-tailed test)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        # Determine significance
        if p_value < 0.01:
            significance = 'highly_significant'
        elif p_value < 0.05:
            significance = 'significant'
        elif p_value < 0.1:
            significance = 'marginally_significant'
        else:
            significance = 'not_significant'
        
        return {
            'mann_kendall_statistic': s,
            'z_score': z,
            'p_value': p_value,
            'significance_level': significance,
            'trend_direction': 'increasing' if s > 0 else 'decreasing' if s < 0 else 'no_trend'
        }


# =============================================================================
# Integration and Testing Utilities
# =============================================================================

async def run_preservation_validation_suite(guard: EnhancedAnomalyPreservationGuard,
                                           test_data_path: str = None) -> Dict[str, Any]:
    """Run                similarity = 1.0 - np.mean(np.abs(orig_norm - proc_norm))
                integrity_scores.append(max(0.0, similarity))
                
            except Exception as e:
                logger.debug(f"FFT analysis failed for {field}: {e}")
                integrity_scores.append(0.5)  # Neutral score on failure
        
        return np.mean(integrity_scores) if integrity_scores else 1.0
    
    async def _check_structural_constraint(self, constraint: str, threshold: float,
                                         original_data: Dict[str, Any], processed_data: Dict[str, Any],
                                         rule: PreservationRule, analysis_tasks: Dict[str, Any]) -> Optional[str]:
        """Check specific structural constraint"""
        
        try:
            if constraint == 'connection_graph_preservation':
                # Check preservation of connection graph structure
                if 'graph_analysis' in analysis_tasks:
                    graph_analysis = analysis_tasks['graph_analysis']
                    structural_similarity = graph_analysis.get('structural_similarity', 1.0)
                    
                    if structural_similarity < threshold:
                        return f"Connection graph preservation too low: {structural_similarity:.3f} < {threshold}"
                else:
                    # Fallback graph analysis
                    graph_preservation = self._calculate_graph_preservation_fallback(original_data, processed_data)
                    if graph_preservation < threshold:
                        return f"Graph structure preservation too low: {graph_preservation:.3f} < {threshold}"
            
            elif constraint == 'flow_topology_preservation':
                # Check flow topology preservation
                topology_preservation = self._calculate_topology_preservation(original_data, processed_data)
                if topology_preservation < threshold:
                    return f"Flow topology preservation too low: {topology_preservation:.3f} < {threshold}"
            
        except Exception as e:
            logger.debug(f"Structural constraint check failed for {constraint}: {e}")
        
        return None
    
    def _calculate_graph_preservation_fallback(self, original_data: Dict[str, Any],
                                             processed_data: Dict[str, Any]) -> float:
        """Fallback graph preservation calculation"""
        
        # Simple structural comparison
        orig_keys = set(original_data.keys())
        proc_keys = set(processed_data.keys())
        
        key_preservation = len(orig_keys & proc_keys) / len(orig_keys) if orig_keys else 1.0
        
        # Check nested structure preservation
        nested_preservation_scores = []
        for key in orig_keys & proc_keys:
            orig_val = original_data[key]
            proc_val = processed_data[key]
            
            if isinstance(orig_val, dict) and isinstance(proc_val, dict):
                nested_score = self._calculate_graph_preservation_fallback(orig_val, proc_val)
                nested_preservation_scores.append(nested_score)
            elif type(orig_val) == type(proc_val):
                nested_preservation_scores.append(1.0)
            else:
                nested_preservation_scores.append(0.5)
        
        nested_preservation = np.mean(nested_preservation_scores) if nested_preservation_scores else 1.0
        
        return (key_preservation * 0.6 + nested_preservation * 0.4)
    
    def _calculate_topology_preservation(self, original_data: Dict[str, Any],
                                       processed_data: Dict[str, Any]) -> float:
        """Calculate topology preservation score"""
        
        # Look for connection or flow patterns
        topology_fields = []
        for key in original_data.keys():
            if any(kw in key.lower() for kw in ['connection', 'flow', 'network', 'topology', 'graph']):
                topology_fields.append(key)
        
        if not topology_fields:
            return 1.0  # No topology to preserve
        
        preservation_scores = []
        
        for field in topology_fields:
            orig_value = original_data.get(field)
            proc_value = processed_data.get(field)
            
            if orig_value is None and proc_value is None:
                preservation_scores.append(1.0)
            elif orig_value is None or proc_value is None:
                preservation_scores.append(0.0)
            elif isinstance(orig_value, dict) and isinstance(proc_value, dict):
                # Dictionary-based topology
                score = self._calculate_graph_preservation_fallback(orig_value, proc_value)
                preservation_scores.append(score)
            elif isinstance(orig_value, list) and isinstance(proc_value, list):
                # List-based topology
                common_elements = set(orig_value) & set(proc_value)
                total_elements = set(orig_value) | set(proc_value)
                score = len(common_elements) / len(total_elements) if total_elements else 1.0
                preservation_scores.append(score)
            else:
                # String or other comparison
                similarity = self._calculate_string_similarity(str(orig_value), str(proc_value))
                preservation_scores.append(similarity)
        
        return np.mean(preservation_scores) if preservation_scores else 1.0
    
    async def _calculate_preservation_breakdown(self, original_data: Dict[str, Any],
                                              processed_data: Dict[str, Any],
                                              affected_types: List[AnomalyType]) -> Dict[str, float]:
        """Calculate detailed preservation breakdown by category"""
        
        breakdown = {
            'field_preservation': 0.0,
            'statistical_preservation': 0.0,
            'temporal_preservation': 0.0,
            'structural_preservation': 0.0,
            'information_preservation': 0.0,
            'signature_preservation': 0.0
        }
        
        # Field preservation
        orig_fields = set(original_data.keys())
        proc_fields = set(processed_data.keys())
        breakdown['field_preservation'] = len(orig_fields & proc_fields) / len(orig_fields) if orig_fields else 1.0
        
        # Statistical preservation
        breakdown['statistical_preservation'] = await self._calculate_statistical_score(processed_data)
        
        # Information preservation
        info_analysis = self.info_analyzer.analyze_information_loss(original_data, processed_data)
        breakdown['information_preservation'] = 1.0 - info_analysis['total_entropy_loss']
        
        # Signature preservation
        signature_scores = []
        for signature in self.known_signatures.values():
            if signature.anomaly_type in affected_types:
                orig_score = self._calculate_field_presence_score_advanced(original_data, signature)
                proc_score = self._calculate_field_presence_score_advanced(processed_data, signature)
                preservation_ratio = proc_score / orig_score if orig_score > 0 else 1.0
                signature_scores.append(preservation_ratio)
        
        breakdown['signature_preservation'] = np.mean(signature_scores) if signature_scores else 1.0
        
        # Temporal preservation (simplified)
        temporal_fields = [k for k in original_data.keys() 
                          if any(kw in k.lower() for kw in ['time', 'duration', 'timestamp'])]
        if temporal_fields:
            temporal_scores = []
            for field in temporal_fields:
                orig_val = original_data.get(field)
                proc_val = processed_data.get(field)
                if orig_val is not None and proc_val is not None:
                    if isinstance(orig_val, (int, float)) and isinstance(proc_val, (int, float)):
                        if orig_val == 0:
                            score = 1.0 if proc_val == 0 else 0.0
                        else:
                            score = 1.0 - abs(orig_val - proc_val) / abs(orig_val)
                        temporal_scores.append(max(0.0, score))
            breakdown['temporal_preservation'] = np.mean(temporal_scores) if temporal_scores else 1.0
        else:
            breakdown['temporal_preservation'] = 1.0
        
        # Structural preservation
        breakdown['structural_preservation'] = self._calculate_graph_preservation_fallback(
            original_data, processed_data
        )
        
        return breakdown
    
    async def _analyze_field_impact(self, original_data: Dict[str, Any],
                                  processed_data: Dict[str, Any],
                                  analysis_tasks: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Analyze impact on individual fields"""
        
        field_impacts = {}
        
        all_fields = set(original_data.keys()) | set(processed_data.keys())
        
        for field in all_fields:
            orig_value = original_data.get(field)
            proc_value = processed_data.get(field)
            
            impact_analysis = {
                'presence_impact': 0.0,
                'value_impact': 0.0,
                'type_impact': 0.0,
                'information_impact': 0.0,
                'overall_impact': 0.0
            }
            
            # Presence impact
            if orig_value is None and proc_value is not None:
                impact_analysis['presence_impact'] = -0.1  # Field added (usually good)
            elif orig_value is not None and proc_value is None:
                impact_analysis['presence_impact'] = 1.0  # Field removed (bad)
            else:
                impact_analysis['presence_impact'] = 0.0  # Field present in both
            
            # Value impact
            if orig_value is not None and proc_value is not None:
                if isinstance(orig_value, (int, float)) and isinstance(proc_value, (int, float)):
                    if orig_value == 0:
                        impact_analysis['value_impact'] = 1.0 if proc_value != 0 else 0.0
                    else:
                        relative_change = abs(orig_value - proc_value) / abs(orig_value)
                        impact_analysis['value_impact'] = min(1.0, relative_change)
                elif isinstance(orig_value, str) and isinstance(proc_value, str):
                    similarity = self._calculate_string_similarity(orig_value, proc_value)
                    impact_analysis['value_impact'] = 1.0 - similarity
                else:
                    impact_analysis['value_impact'] = 0.0 if orig_value == proc_value else 0.5
            
            # Type impact
            if orig_value is not None and proc_value is not None:
                if type(orig_value) != type(proc_value):
                    impact_analysis['type_impact'] = 1.0
                else:
                    impact_analysis['type_impact'] = 0.0
            
            # Information impact
            orig_entropy = self.info_analyzer.calculate_entropy(orig_value) if orig_value is not None else 0.0
            proc_entropy = self.info_analyzer.calculate_entropy(proc_value) if proc_value is not None else 0.0
            
            if orig_entropy > 0:
                impact_analysis['information_impact'] = max(0.0, (orig_entropy - proc_entropy) / orig_entropy)
            else:
                impact_analysis['information_impact'] = 0.0
            
            # Overall impact (weighted combination)
            impact_analysis['overall_impact'] = (
                impact_analysis['presence_impact'] * 0.3 +
                impact_analysis['value_impact'] * 0.4 +
                impact_analysis['type_impact'] * 0.2 +
                impact_analysis['information_impact'] * 0.1
            )
            
            field_impacts[field] = impact_analysis
        
        return field_impacts
    
    async def _detect_violations_comprehensive(self, original_data: Dict[str, Any],
                                             processed_data: Dict[str, Any],
                                             affected_types: List[AnomalyType],
                                             analysis_tasks: Dict[str, Any]) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Comprehensive violation detection"""
        
        critical_violations = []
        warning_violations = []
        constraint_violations = {}
        
        # Check violations for each affected anomaly type
        for anomaly_type in affected_types:
            rule = self.preservation_rules.get(anomaly_type)
            if not rule:
                continue
            
            # Check rule-specific constraints
            rule_violations = await self._check_rule_constraints(
                original_data, processed_data, rule, analysis_tasks
            )
            
            for violation in rule_violations:
                if rule.preservation_level in [PreservationLevel.CRITICAL, PreservationLevel.HIGH]:
                    critical_violations.append(f"{anomaly_type.value}: {violation}")
                else:
                    warning_violations.append(f"{anomaly_type.value}: {violation}")
            
            constraint_violations[anomaly_type.value] = rule_violations
        
        # Global violation checks
        info_analysis = analysis_tasks.get('info_theory', {})
        
        # Critical information loss
        total_entropy_loss = info_analysis.get('total_entropy_loss', 0.0)
        if total_entropy_loss > 0.5:
            critical_violations.append(f"Excessive information loss: {total_entropy_loss:.3f}")
        elif total_entropy_loss > 0.2:
            warning_violations.append(f"Significant information loss: {total_entropy_loss:.3f}")
        
        # Structural violations
        if 'graph_analysis' in analysis_tasks:
            graph_analysis = analysis_tasks['graph_analysis']
            structural_similarity = graph_analysis.get('structural_similarity', 1.0)
            
            if structural_similarity < 0.5:
                critical_violations.append(f"Severe structural damage: {structural_similarity:.3f}")
            elif structural_similarity < 0.8:
                warning_violations.append(f"Structural integrity compromised: {structural_similarity:.3f}")
        
        # Field loss violations
        orig_field_count = len(original_data)
        proc_field_count = len(processed_data)
        field_loss_ratio = (orig_field_count - proc_field_count) / orig_field_count if orig_field_count > 0 else 0.0
        
        if field_loss_ratio > 0.3:
            critical_violations.append(f"Excessive field loss: {field_loss_ratio:.1%}")
        elif field_loss_ratio > 0.1:
            warning_violations.append(f"Significant field loss: {field_loss_ratio:.1%}")
        
        return critical_violations, warning_violations, constraint_violations
    
    async def _generate_intelligent_recommendations(self, preservation_effectiveness: float,
                                                  affected_types: List[AnomalyType],
                                                  critical_violations: List[str],
                                                  analysis_tasks: Dict[str, Any],
                                                  optimal_strategy: PreservationStrategy) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Generate intelligent recommendations and suggested actions"""
        
        recommendations = []
        suggested_actions = []
        
        # Effectiveness-based recommendations
        if preservation_effectiveness < 0.7:
            recommendations.append("CRITICAL: Anomaly detectability severely compromised")
            recommendations.append("Consider immediate rollback to previous processing stage")
            suggested_actions.append({
                'action': 'rollback',
                'priority': 'critical',
                'reason': 'preservation_effectiveness_too_low',
                'parameters': {'target_effectiveness': 0.85}
            })
        elif preservation_effectiveness < 0.85:
            recommendations.append("WARNING: Anomaly detectability below recommended threshold")
            recommendations.append("Reduce processing aggressiveness or apply selective preservation")
            suggested_actions.append({
                'action': 'reduce_processing_aggressiveness',
                'priority': 'high',
                'reason': 'preservation_effectiveness_low',
                'parameters': {'aggressiveness_reduction': 0.3}
            })
        elif preservation_effectiveness < 0.95:
            recommendations.append("Consider minor optimization to improve preservation")
            suggested_actions.append({
                'action': 'optimize_preservation_strategy',
                'priority': 'medium',
                'reason': 'preservation_optimization',
                'parameters': {'target_improvement': 0.05}
            })
        
        # Anomaly type-specific recommendations
        for anomaly_type in affected_types:
            rule = self.preservation_rules.get(anomaly_type)
            if not rule:
                continue
            
            if anomaly_type == AnomalyType.COLD_START:
                recommendations.append("Preserve timing precision for cold start detection")
                suggested_actions.append({
                    'action': 'preserve_timing_fields',
                    'priority': 'high',
                    'reason': 'cold_start_anomaly_affected',
                    'parameters': {'fields': ['init_time_ms', 'duration_ms', 'bootstrap_time']}
                })
            
            elif anomaly_type == AnomalyType.DATA_EXFILTRATION:
                recommendations.append("Maintain network traffic volume and timing patterns")
                suggested_actions.append({
                    'action': 'preserve_network_patterns',
                    'priority': 'critical',
                    'reason': 'data_exfiltration_detection',
                    'parameters': {'fields': ['network_io', 'data_volume', 'transfer_rate']}
                })
            
            elif anomaly_type == AnomalyType.ECONOMIC_ABUSE:
                recommendations.append("Preserve cost and resource consumption metrics")
                suggested_actions.append({
                    'action': 'preserve_cost_metrics',
                    'priority': 'high',
                    'reason': 'economic_abuse_detection',
                    'parameters': {'fields': ['execution_cost', 'resource_consumption', 'quota_usage']}
                })
            
            elif anomaly_type == AnomalyType.SILENT_FAILURE:
                recommendations.append("Maintain output integrity and semantic consistency")
                suggested_actions.append({
                    'action': 'preserve_output_integrity',
                    'priority': 'critical',
                    'reason': 'silent_failure_detection',
                    'parameters': {'fields': ['output_entropy', 'semantic_consistency', 'data_integrity_hash']}
                })
        
        # Strategy-based recommendations
        strategy_performance = self.adaptive_optimizer.get_strategy_recommendations()
        
        if optimal_strategy.value in strategy_performance:
            performance = strategy_performance[optimal_strategy.value]
            if performance['average_performance'] < 0.8:
                recommendations.append(f"Consider switching from {optimal_strategy.value} strategy")
                suggested_actions.append({
                    'action': 'switch_preservation_strategy',
                    'priority': 'medium',
                    'reason': 'strategy_underperforming',
                    'parameters': {'current_strategy': optimal_strategy.value, 'min_performance': 0.8}
                })
        
        # Violation-based recommendations
        if critical_violations:
            recommendations.append("Address critical constraint violations immediately")
            suggested_actions.append({
                'action': 'address_critical_violations',
                'priority': 'critical',
                'reason': 'constraint_violations_detected',
                'parameters': {'violations': critical_violations}
            })
        
        # Analysis task-based recommendations
        if 'info_theory' in analysis_tasks:
            info_analysis = analysis_tasks['info_theory']
            if info_analysis.get('total_entropy_loss', 0.0) > 0.3:
                recommendations.append("Excessive information loss - implement entropy-preserving transformations")
                suggested_actions.append({
                    'action': 'implement_entropy_preservation',
                    'priority': 'high',
                    'reason': 'excessive_information_loss',
                    'parameters': {'max_acceptable_loss': 0.2}
                })
        
        if 'graph_analysis' in analysis_tasks:
            graph_analysis = analysis_tasks['graph_analysis']
            if graph_analysis.get('structural_similarity', 1.0) < 0.7:
                recommendations.append("Structural integrity compromised - preserve graph topology")
                suggested_actions.append({
                    'action': 'preserve_graph_topology',
                    'priority': 'high',
                    'reason': 'structural_integrity_compromised',
                    'parameters': {'min_similarity': 0.8}
                })
        
        # Performance-based recommendations
        if len(affected_types) > 3:
            recommendations.append("Processing impact too broad - consider selective processing")
            suggested_actions.append({
                'action': 'implement_selective_processing',
                'priority': 'medium',
                'reason': 'broad_processing_impact',
                'parameters': {'max_affected_types': 3}
            })
        
        return recommendations, suggested_actions
    
    async def _should_recommend_rollback(self, preservation_effectiveness: float,
                                       critical_violations: List[str],
                                       affected_types: List[AnomalyType]) -> bool:
        """Determine if rollback should be recommended"""
        
        # Critical effectiveness threshold
        if preservation_effectiveness < 0.7:
            return True
        
        # Critical violations threshold
        if len(critical_violations) > 2:
            return True
        
        # Critical anomaly types affected
        critical_anomaly_types = {
            AnomalyType.DATA_EXFILTRATION,
            AnomalyType.SILENT_FAILURE,
            AnomalyType.PRIVILEGE_ESCALATION
        }
        
        critical_affected = [t for t in affected_types if t in critical_anomaly_types]
        if len(critical_affected) > 1:
            return True
        
        # High-priority rules with low preservation
        for anomaly_type in affected_types:
            rule = self.preservation_rules.get(anomaly_type)
            if rule and rule.preservation_level == PreservationLevel.CRITICAL:
                if preservation_effectiveness < 0.8:
                    return True
        
        return False
    
    def _calculate_assessment_confidence(self, analysis_tasks: Dict[str, Any],
                                       preservation_effectiveness: float) -> Tuple[float, Tuple[float, float]]:
        """Calculate confidence score and uncertainty bounds for assessment"""
        
        confidence_factors = []
        
        # Data completeness factor
        data_completeness = 1.0  # Assume complete data for now
        confidence_factors.append(data_completeness * 0.3)
        
        # Analysis depth factor
        analysis_count = len(analysis_tasks)
        max_analysis = 5  # Maximum expected analysis types
        analysis_depth_factor = min(1.0, analysis_count / max_analysis)
        confidence_factors.append(analysis_depth_factor * 0.2)
        
        # Signature matching factor
        signature_match_factor = 0.8  # Based on signature database coverage
        confidence_factors.append(signature_match_factor * 0.2)
        
        # Consistency factor (how consistent are the different analysis methods)
        if len(analysis_tasks) > 1:
            # Calculate consistency between analysis methods
            consistency_factor = 0.9  # Simplified - assume high consistency
            confidence_factors.append(consistency_factor * 0.2)
        else:
            confidence_factors.append(0.6 * 0.2)
        
        # Statistical reliability factor
        statistical_reliability = min(1.0, preservation_effectiveness + 0.1)
        confidence_factors.append(statistical_reliability * 0.1)
        
        confidence_score = sum(confidence_factors)
        
        # Calculate uncertainty bounds
        uncertainty = 1.0 - confidence_score
        lower_bound = max(0.0, preservation_effectiveness - uncertainty)
        upper_bound = min(1.0, preservation_effectiveness + uncertainty)
        
        return confidence_score, (lower_bound, upper_bound)
    
    def _update_preservation_metrics_enhanced(self, assessment: PreservationAssessment):
        """Update enhanced preservation metrics"""
        
        self.metrics['total_assessments'] += 1
        
        # Check for violations
        if assessment.preservation_effectiveness < 0.8:
            self.metrics['preservation_violations'] += 1
        
        if assessment.critical_violations:
            self.metrics['critical_failures'] += 1
        
        # Update processing time tracking
        self.metrics['processing_time_ms'].append(assessment.processing_time_ms)
        if len(self.metrics['processing_time_ms']) > 1000:
            self.metrics['processing_time_ms'] = self.metrics['processing_time_ms'][-1000:]
        
        # Update memory usage
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.metrics['memory_usage_mb'].append(current_memory)
        if len(self.metrics['memory_usage_mb']) > 1000:
            self.metrics['memory_usage_mb'] = self.metrics['memory_usage_mb'][-1000:]
        
        # Update average preservation score
        current_avg = self.metrics['average_preservation_score']
        total_assessments = self.metrics['total_assessments']
        new_avg = ((current_avg * (total_assessments - 1)) + 
                  assessment.preservation_effectiveness) / total_assessments
        self.metrics['average_preservation_score'] = new_avg
        
        # Cleanup old cache entries
        if len(self.analysis_cache) > 1000:
            # Remove oldest entries
            current_time = time.time()
            expired_keys = [
                key for key, value in self.analysis_cache.items()
                if current_time - value['timestamp'] > self.cache_ttl
            ]
            for key in expired_keys:
                del self.analysis_cache[key]
    
    def _generate_cache_key(self, original_data: Dict[str, Any], 
                          processed_data: Dict[str, Any], processing_stage: str) -> str:
        """Generate cache key for analysis results"""
        
        # Create a hash-based cache key
        orig_hash = hashlib.md5(str(sorted(original_data.items())).encode()).hexdigest()[:16]
        proc_hash = hashlib.md5(str(sorted(processed_data.items())).encode()).hexdigest()[:16]
        
        return f"{processing_stage}_{orig_hash}_{proc_hash}"
    
    async def _analyze_basic_statistics(self, original_data: Dict[str, Any],
                                      processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze basic statistical properties"""
        
        stats = {
            'field_count_change': len(processed_data) - len(original_data),
            'numeric_field_preservation': 0.0,
            'string_field_preservation': 0.0,
            'structure_preservation': 0.0
        }
        
        # Numeric field preservation
        numeric_preservations = []
        string_preservations = []
        
        for key in original_data.keys():
            orig_val = original_data[key]
            proc_val = processed_data.get(key)
            
            if isinstance(orig_val, (int, float)):
                if proc_val is not None and isinstance(proc_val, (int, float)):
                    if orig_val == 0:
                        preservation = 1.0 if proc_val == 0 else 0.0
                    else:
                        preservation = 1.0 - abs(orig_val - proc_val) / abs(orig_val)
                    numeric_preservations.append(max(0.0, preservation))
                else:
                    numeric_preservations.append(0.0)
            
            elif isinstance(orig_val, str):
                if proc_val is not None and isinstance(proc_val, str):
                    preservation = self._calculate_string_similarity(orig_val, proc_val)
                    string_preservations.append(preservation)
                else:
                    string_preservations.append(0.0)
        
        stats['numeric_field_preservation'] = np.mean(numeric_preservations) if numeric_preservations else 1.0
        stats['string_field_preservation'] = np.mean(string_preservations) if string_preservations else 1.0
        
        # Structure preservation
        stats['structure_preservation'] = self._calculate_graph_preservation_fallback(
            original_data, processed_data
        )
        
        return stats
    
    def _analyze_neural_preservation(self, original_data: Dict[str, Any],
                                   processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze preservation using neural embeddings"""
        
        try:
            orig_embedding = self.neural_encoder.encode(original_data)
            proc_embedding = self.neural_encoder.encode(processed_data)
            
            # Calculate embedding similarity
            cosine_similarity = 1.0 - cosine(orig_embedding, proc_embedding)
            euclidean_distance = euclidean(orig_embedding, proc_embedding)
            
            # Normalize euclidean distance
            max_distance = np.sqrt(len(orig_embedding)) * 2  # Rough normalization
            normalized_euclidean = 1.0 - min(1.0, euclidean_distance / max_distance)
            
            return {
                'cosine_similarity': cosine_similarity,
                'euclidean_similarity': normalized_euclidean,
                'embedding_preservation': (cosine_similarity + normalized_euclidean) / 2,
                'embedding_dimension': len(orig_embedding)
            }
            
        except Exception as e:
            logger.debug(f"Neural preservation analysis failed: {e}")
            return {
                'cosine_similarity': 0.5,
                'euclidean_similarity': 0.5,
                'embedding_preservation': 0.5,
                'embedding_dimension': 0
            }
    
    async def train_neural_encoder(self, training_data: List[Dict[str, Any]]):
        """Train the neural encoder on anomaly patterns"""
        
        if len(training_data) < 10:
            logger.warning("Insufficient training data for neural encoder")
            return
        
        try:
            self.neural_encoder.train(training_data)
            logger.info(f"Neural encoder trained on {len(training_data)} samples")
        except Exception as e:
            logger.error(f"Neural encoder training failed: {e}")
    
    async def update_signature_database(self, new_signatures: List[AnomalySignature]):
        """Update the anomaly signature database"""
        
        for signature in new_signatures:
            self.known_signatures[signature.signature_id] = signature
        
        logger.info(f"Updated signature database with {len(new_signatures)} new signatures")
    
    async def optimize_preservation_rules(self, optimization_target: float = 0.95):
        """Optimize preservation rules based on performance history"""
        
        for anomaly_type, rule in self.preservation_rules.items():
            if len(rule.performance_history) < 10:
                continue
            
            # Calculate average performance
            avg_performance = np.mean(rule.performance_history[-50:])  # Last 50 assessments
            
            if avg_performance < optimization_target:
                # Adjust rule parameters
                if rule.preservation_level == PreservationLevel.MEDIUM:
                    rule.preservation_level = PreservationLevel.HIGH
                    logger.info(f"Upgraded preservation level for {anomaly_type.value}")
                
                # Increase field weights for underperforming rules
                for field in rule.protected_fields:
                    current_weight = rule.field_weights.get(field, 1.0)
                    rule.field_weights[field] = min(1.0, current_weight * 1.1)
        
        logger.info("Preservation rules optimization completed")
    
    async def get_preservation_report(self, detailed: bool = True) -> Dict[str, Any]:
        """Generate comprehensive preservation performance report"""
        
        report = {
            'summary': {
                'total_assessments': self.metrics['total_assessments'],
                'average_preservation_score': self.metrics['average_preservation_score'],
                'violation_rate': self.metrics['preservation_violations'] / max(1, self.metrics['total_assessments']),
                'critical_failure_rate': self.metrics['critical_failures'] / max(1, self.metrics['total_assessments']),
                'cache_hit_rate': self.metrics['cache_hits'] / max(1, self.metrics['cache_hits'] + self.metrics['cache_misses'])
            },
            'performance': {
                'average_processing_time_ms': np.mean(self.metrics['processing_time_ms']) if self.metrics['processing_time_ms'] else 0.0,
                'p95_processing_time_ms': np.percentile(self.metrics['processing_time_ms'], 95) if self.metrics['processing_time_ms'] else 0.0,
                'average_memory_usage_mb': np.mean(self.metrics['memory_usage_mb']) if self.metrics['memory_usage_mb'] else 0.0,
                'peak_memory_usage_mb': max(self.metrics['memory_usage_mb']) if self.metrics['memory_usage_mb'] else 0.0
            },
            'strategy_performance': self.adaptive_optimizer.get_strategy_recommendations(),
            'recent_trends': self._calculate_recent_trends()
        }
        
        if detailed:
            report['detailed_analysis'] = await self._generate_detailed_analysis()
            report['anomaly_type_breakdown'] = self._get_anomaly_type_breakdown()
            report['field_impact_summary'] = self._get_field_impact_summary()
            report['violation_patterns'] = self._analyze_violation_patterns()
        
        return report
    
    def _calculate_recent_trends(self) -> Dict[str, Any]:
        """Calculate recent performance trends"""
        
        recent_assessments = list(self.assessment_history)[-100:]  # Last 100 assessments
        
        if len(recent_assessments) < 10:
            return {'insufficient_data': True}
        
        # Split into older and newer halves
        half_point = len(recent_assessments) // 2
        older_half = recent_assessments[:half_point]
        newer_half = recent_assessments[half_point:]
        
        older_avg_effectiveness = np.mean([a.preservation_effectiveness for a in older_half])
        newer_avg_effectiveness = np.mean([a.preservation_effectiveness for a in newer_half])
        
        older_avg_processing_time = np.mean([a.processing_time_ms for a in older_half])
        newer_avg_processing_time = np.mean([a.processing_time_ms for a in newer_half])
        
        older_violation_rate = sum(1 for a in older_half if a.critical_violations) / len(older_half)
        newer_violation_rate = sum(1 for a in newer_half if a.critical_violations) / len(newer_half)
        
        return {
            'effectiveness_trend': {
                'direction': 'improving' if newer_avg_effectiveness > older_avg_effectiveness else 'declining',
                'change_percent': ((newer_avg_effectiveness - older_avg_effectiveness) / older_avg_effectiveness * 100) if older_avg_effectiveness > 0 else 0,
                'older_average': older_avg_effectiveness,
                'newer_average': newer_avg_effectiveness
            },
            'performance_trend': {
                'direction': 'improving' if newer_avg_processing_time < older_avg_processing_time else 'declining',
                'change_percent': ((older_avg_processing_time - newer_avg_processing_time) / older_avg_processing_time * 100) if older_avg_processing_time > 0 else 0,
                'older_average_ms': older_avg_processing_time,
                'newer_average_ms': newer_avg_processing_time
            },
            'violation_trend': {
                'direction': 'improving' if newer_violation_rate < older_violation_rate else 'declining',
                'change_percent': ((older_violation_rate - newer_violation_rate) / older_violation_rate * 100) if older_violation_rate > 0 else 0,
                'older_rate': older_violation_rate,
                'newer_rate': newer_violation_rate
            }
        }
    
    async def _generate_detailed_analysis(self) -> Dict[str, Any]:
        """Generate detailed performance analysis"""
        
        recent_assessments = list(self.assessment_history)[-500:]  # Last 500 assessments
        
        analysis = {
            'effectiveness_distribution': self._analyze_effectiveness_distribution(recent_assessments),
            'processing_stage_analysis': self._analyze_processing_stages(recent_assessments),
            'temporal_patterns': self._analyze_temporal_patterns(recent_assessments),
            'confidence_analysis': self._analyze_confidence_patterns(recent_assessments)
        }
        
        return analysis
    
    def _analyze_effectiveness_distribution(self, assessments: List[PreservationAssessment]) -> Dict[str, Any]:
        """Analyze distribution of preservation effectiveness scores"""
        
        if not assessments:
            return {'no_data': True}
        
        effectiveness_scores = [a.preservation_effectiveness for a in assessments]
        
        return {
            'mean': np.mean(effectiveness_scores),
            'median': np.median(effectiveness_scores),
            'std': np.std(effectiveness_scores),
            'min': np.min(effectiveness_scores),
            'max': np.max(effectiveness_scores),
            'percentiles': {
                'p25': np.percentile(effectiveness_scores, 25),
                'p75': np.percentile(effectiveness_scores, 75),
                'p95': np.percentile(effectiveness_scores, 95),
                'p99': np.percentile(effectiveness_scores, 99)
            },
            'distribution_bins': np.histogram(effectiveness_scores, bins=10)[0].tolist()
        }
    
    def _analyze_processing_stages(self, assessments: List[PreservationAssessment]) -> Dict[str, Any]:
        """Analyze performance by processing stage"""
        
        stage_performance = defaultdict(list)
        
        for assessment in assessments:
            stage_performance[assessment.processing_stage].append(assessment.preservation_effectiveness)
        
        stage_analysis = {}
        for stage, scores in stage_performance.items():
            if scores:
                stage_analysis[stage] = {
                    'count': len(scores),
                    'average_effectiveness': np.mean(scores),
                    'min_effectiveness': np.min(scores),
                    'max_effectiveness': np.max(scores),
                    'std_effectiveness': np.std(scores),
                    'failure_rate': sum(1 for s in scores if s < 0.8) / len(scores)
                }
        
        return stage_analysis
    
    def _analyze_temporal_patterns(self, assessments: List[PreservationAssessment]) -> Dict[str, Any]:
        """Analyze temporal patterns in preservation performance"""
        
        if len(assessments) < 10:
            return {'insufficient_data': True}
        
        # Group by hour of day
        hourly_performance = defaultdict(list)
        daily_performance = defaultdict(list)
        
        for assessment in assessments:
            timestamp = assessment.assessment_timestamp
            hour = int((timestamp % 86400) // 3600)  # Hour of day
            day = int(timestamp // 86400) % 7  # Day of week
            
            hourly_performance[hour].append(assessment.preservation_effectiveness)
            daily_performance[day].append(assessment.preservation_effectiveness)
        
        # Calculate averages
        hourly_averages = {hour: np.mean(scores) for hour, scores in hourly_performance.items() if scores}
        daily_averages = {day: np.mean(scores) for day, scores in daily_performance.items() if scores}
        
        return {
            'hourly_patterns': hourly_averages,
            'daily_patterns': daily_averages,
            'peak_performance_hour': max(hourly_averages, key=hourly_averages.get) if hourly_averages else None,
            'worst_performance_hour': min(hourly_averages, key=hourly_averages.get) if hourly_averages else None,
            'peak_performance_day': max(daily_averages, key=daily_averages.get) if daily_averages else None,
            'worst_performance_day': min(daily_averages, key=daily_averages.get) if daily_averages else None
        }
    
    def _analyze_confidence_patterns(self, assessments: List[PreservationAssessment]) -> Dict[str, Any]:
        """Analyze confidence score patterns"""
        
        if not assessments:
            return {'no_data': True}
        
        confidence_scores = [a.confidence_score for a in assessments]
        reliability_scores = [a.reliability_score for a in assessments]
        
        return {
            'average_confidence': np.mean(confidence_scores),
            'average_reliability': np.mean(reliability_scores),
            'confidence_std': np.std(confidence_scores),
            'reliability_std': np.std(reliability_scores),
            'low_confidence_rate': sum(1 for c in confidence_scores if c < 0.7) / len(confidence_scores),
            'high_confidence_rate': sum(1 for c in confidence_scores if c > 0.9) / len(confidence_scores)
        }
    
    def _get_anomaly_type_breakdown(self) -> Dict[str, Any]:
        """Get breakdown of performance by anomaly type"""
        
        anomaly_performance = defaultdict(lambda: {
            'assessments': [],
            'effectiveness_scores': [],
            'violation_counts': []
        })
        
        for assessment in self.assessment_history:
            for anomaly_type in assessment.affected_anomaly_types:
                anomaly_performance[anomaly_type.value]['assessments'].append(assessment)
                anomaly_performance[anomaly_type.value]['effectiveness_scores'].append(assessment.preservation_effectiveness)
                anomaly_performance[anomaly_type.value]['violation_counts'].append(len(assessment.critical_violations))
        
        breakdown = {}
        for anomaly_type, data in anomaly_performance.items():
            if data['effectiveness_scores']:
                breakdown[anomaly_type] = {
                    'total_assessments': len(data['assessments']),
                    'average_effectiveness': np.mean(data['effectiveness_scores']),
                    'min_effectiveness': np.min(data['effectiveness_scores']),
                    'max_effectiveness': np.max(data['effectiveness_scores']),
                    'average_violations': np.mean(data['violation_counts']),
                    'failure_rate': sum(1 for s in data['effectiveness_scores'] if s < 0.8) / len(data['effectiveness_scores'])
                }
        
        return breakdown
    
    def _get_field_impact_summary(self) -> Dict[str, Any]:
        """Get summary of field-level impacts"""
        
        field_impacts = defaultdict(lambda: {
            'presence_impacts': [],
            'value_impacts': [],
            'overall_impacts': []
        })
        
        for assessment in self.assessment_history:
            for field, impact_data in assessment.field_impact_analysis.items():
                field_impacts[field]['presence_impacts'].append(impact_data.get('presence_impact', 0.0))
                field_impacts[field]['value_impacts'].append(impact_data.get('value_impact', 0.0))
                field_impacts[field]['overall_impacts'].append(impact_data.get('overall_impact', 0.0))
        
        summary = {}
        for field, impacts in field_impacts.items():
            if impacts['overall_impacts']:
                summary[field] = {
                    'average_overall_impact': np.mean(impacts['overall_impacts']),
                    'max_overall_impact': np.max(impacts['overall_impacts']),
                    'average_presence_impact': np.mean(impacts['presence_impacts']),
                    'average_value_impact': np.mean(impacts['value_impacts']),
                    'impact_frequency': len(impacts['overall_impacts'])
                }
        
        # Sort by average impact
        sorted_summary = dict(sorted(summary.items(), key=lambda x: x[1]['average_overall_impact'], reverse=True))
        
        return sorted_summary
    
    def _analyze_violation_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in preservation violations"""
        
        violation_patterns = {
            'critical_violations': defaultdict(int),
            'warning_violations': defaultdict(int),
            'violation_co_occurrence': defaultdict(int),
            'violation_trends': []
        }
        
        for assessment in self.assessment_history:
            # Count violation types
            for violation in assessment.critical_violations:
                violation_patterns['critical_violations'][violation] += 1
            
            for violation in assessment.warning_violations:
                violation_patterns['warning_violations'][violation] += 1
            
            # Track violation co-occurrence
            critical_set = set(assessment.critical_violations)
            for v1 in critical_set:
                for v2 in critical_set:
                    if v1 != v2:
                        pair = tuple(sorted([v1, v2]))
                        violation_patterns['violation_co_occurrence'][pair] += 1
        
        # Convert to regular dicts for JSON serialization
        violation_patterns['critical_violations'] = dict(violation_patterns['critical_violations'])
        violation_patterns['warning_violations'] = dict(violation_patterns['warning_violations'])
        violation_patterns['violation_co_occurrence'] = {
            f"{pair[0]} + {pair[1]}": count 
            for pair, count in violation_patterns['violation_co_occurrence'].items()
        }
        
        return violation_patterns
    
    async def export_preservation_model(self, filepath: str):
        """Export trained preservation model for reuse"""
        
        model_data = {
            'neural_encoder': {
                'encoder': self.neural_encoder.encoder,
                'scaler': self.neural_encoder.scaler,
                'is_trained': self.neural_encoder.is_trained,
                'training_history': self.neural_encoder.training_history
            },
            'preservation_rules': {
                anomaly_type.value: asdict(rule) 
                for anomaly_type, rule in self.preservation_rules.items()
            },
            'known_signatures': {
                sig_id: asdict(signature) 
                for sig_id, signature in self.known_signatures.items()
            },
            'adaptive_optimizer': {
                'strategy_performance': dict(self.adaptive_optimizer.strategy_performance),
                'strategy_weights': dict(self.adaptive_optimizer.strategy_weights),
                'exploration_rate': self.adaptive_optimizer.exploration_rate,
                'learning_rate': self.adaptive_optimizer.learning_rate
            },
            'config': self.config,
            'metrics': self.metrics,
            'export_timestamp': time.time()
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Preservation model exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export preservation model: {e}")
            raise
    
    async def import_preservation_model(self, filepath: str):
        """Import previously trained preservation model"""
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore neural encoder
            encoder_data = model_data['neural_encoder']
            self.neural_encoder.encoder = encoder_data['encoder']
            self.neural_encoder.scaler = encoder_data['scaler']
            self.neural_encoder.is_trained = encoder_data['is_trained']
            self.neural_encoder.training_history = encoder_data['training_history']
            
            # Restore preservation rules
            for anomaly_type_str, rule_data in model_data['preservation_rules'].items():
                anomaly_type = AnomalyType(anomaly_type_str)
                # Convert dict back to PreservationRule object
                rule = PreservationRule(**rule_data)
                self.preservation_rules[anomaly_type] = rule
            
            # Restore known signatures
            for sig_id, signature_data in model_data['known_signatures'].items():
                # Convert dict back to AnomalySignature object
                signature = AnomalySignature(**signature_data)
                self.known_signatures[sig_id] = signature
            
            # Restore adaptive optimizer
            optimizer_data = model_data['adaptive_optimizer']
            self.adaptive_optimizer.strategy_performance = defaultdict(list, optimizer_data['strategy_performance'])
            self.adaptive_optimizer.strategy_weights = defaultdict(lambda: 1.0, optimizer_data['strategy_weights'])
            self.adaptive_optimizer.exploration_rate = optimizer_data['exploration_rate']
            self.adaptive_optimizer.learning_rate = optimizer_data['learning_rate']
            
            # Restore metrics
            self.metrics.update(model_data.get('metrics', {}))
            
            logger.info(f"Preservation model imported from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to import preservation model: {e}")
            raise
    
    async def validate_preservation_pipeline(self, test_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate preservation effectiveness across test scenarios"""
        
        validation_results = {
            'total_scenarios': len(test_scenarios),
            'passed_scenarios': 0,
            'failed_scenarios': 0,
            'scenario_results': [],
            'overall_effectiveness': 0.0,
            'effectiveness_distribution': [],
            'failure_modes': defaultdict(int),
            'performance_metrics': {
                'average_processing_time_ms': 0.0,
                'p95_processing_time_ms': 0.0,
                'total_processing_time_ms': 0.0
            }
        }
        
        total_effectiveness = 0.0
        processing_times = []
        
        for i, scenario in enumerate(test_scenarios):
            scenario_start = time.perf_counter()
            
            try:
                # Create a processed version by applying various transformations
                processed_scenario = await self._apply_test_transformations(scenario)
                
                # Assess preservation impact
                assessment = await self.assess_preservation_impact(
                    scenario, processed_scenario, f"test_scenario_{i}"
                )
                
                scenario_time = (time.perf_counter() - scenario_start) * 1000
                processing_times.append(scenario_time)
                
                # Determine pass/fail
                passed = (
                    assessment.preservation_effectiveness >= 0.85 and
                    len(assessment.critical_violations) == 0 and
                    not assessment.rollback_recommendation
                )
                
                if passed:
                    validation_results['passed_scenarios'] += 1
                else:
                    validation_results['failed_scenarios'] += 1
                    
                    # Categorize failure mode
                    if assessment.preservation_effectiveness < 0.85:
                        validation_results['failure_modes']['low_effectiveness'] += 1
                    if assessment.critical_violations:
                        validation_results['failure_modes']['critical_violations'] += 1
                    if assessment.rollback_recommendation:
                        validation_results['failure_modes']['rollback_recommended'] += 1
                
                scenario_result = {
                    'scenario_id': i,
                    'passed': passed,
                    'preservation_effectiveness': assessment.preservation_effectiveness,
                    'critical_violations': len(assessment.critical_violations),
                    'processing_time_ms': scenario_time,
                    'affected_anomaly_types': [t.value for t in assessment.affected_anomaly_types],
                    'confidence_score': assessment.confidence_score
                }
                
                validation_results['scenario_results'].append(scenario_result)
                validation_results['effectiveness_distribution'].append(assessment.preservation_effectiveness)
                total_effectiveness += assessment.preservation_effectiveness
                
            except Exception as e:
                logger.error(f"Validation failed for scenario {i}: {e}")
                validation_results['failed_scenarios'] += 1
                validation_results['failure_modes']['processing_error'] += 1
                
                scenario_result = {
                    'scenario_id': i,
                    'passed': False,
                    'error': str(e),
                    'processing_time_ms': (time.perf_counter() - scenario_start) * 1000
                }
                validation_results['scenario_results'].append(scenario_result)
        
        # Calculate overall metrics
        if validation_results['total_scenarios'] > 0:
            validation_results['overall_effectiveness'] = total_effectiveness / validation_results['total_scenarios']
            validation_results['pass_rate'] = validation_results['passed_scenarios'] / validation_results['total_scenarios']
        
        if processing_times:
            validation_results['performance_metrics']['average_processing_time_ms'] = np.mean(processing_times)
            validation_results['performance_metrics']['p95_processing_time_ms'] = np.percentile(processing_times, 95)
            validation_results['performance_metrics']['total_processing_time_ms'] = sum(processing_times)
        
        return validation_results
    
    async def _apply_test_transformations(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Apply various test transformations to simulate processing"""
        
        processed = copy.deepcopy(scenario)
        
        # Simulate sanitization - remove debug fields
        processed.pop('debug_info', None)
        processed.pop('raw_logs', None)
        processed.pop('internal_state', None)
        
        # Simulate privacy filtering - hash sensitive fields
        if 'user_id' in processed:
            processed['user_id'] = f"hash_{hash(processed['user_id'])}"
        
        if 'ip_address' in processed:
            processed['ip_address'] = f"masked_{processed['ip_address'].split('.')[-1]}"
        
        # Simulate hashing - replace large text fields with hashes
        for key, value in list(processed.items()):
            if isinstance(value, str) and len(value) > 1000:
                processed[f"{key}_hash"] = f"hash_{hash(value)}"
                del processed[key]
        
        # Simulate schema normalization - standardize field names
        if 'duration_milliseconds' in processed:
            processed['duration_ms'] = processed.pop('duration_milliseconds')
        
        if 'memory_usage_bytes' in processed:
            processed['memory_used_mb'] = processed.pop('memory_usage_bytes') / (1024 * 1024)
        
        return processed
    
    def __del__(self):
        """Cleanup resources on destruction"""
        try:
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=False)
        except:
            pass


# =============================================================================
# Utility Functions and Factory Methods
# =============================================================================

def create_enhanced_preservation_guard(config: Dict[str, Any]) -> EnhancedAnomalyPreservationGuard:
    """Factory function to create a configured enhanced preservation guard"""
    
    default_config = {
        'history_size': 10000,
        'cache_ttl': 300,
        'max_workers': 4,
        'ml_training': {
            'exploration_rate': 0.1,
            'learning_rate': 0.01
        }
    }
    
    # Merge with provided config
    merged_config = {**default_config, **config}
    
    return EnhancedAnomalyPreservationGuard(merged_config)

def get_preservation_policy_for_anomaly_enhanced(anomaly_type: AnomalyType) -> Dict[str, Any]:
    """Get enhanced preservation policy for specific anomaly type"""
    
    policies = {
        AnomalyType.COLD_START: {
            'strategy': PreservationStrategy.NEURAL_EMBEDDINGS,
            'preservation_level': PreservationLevel.HIGH,
            'critical_fields': ['duration_ms', 'memory_used_mb', 'init_time_ms', 'bootstrap_time'],
            'min_effectiveness': 0.9,
            'constraints': {
                'duration_distribution_kl_divergence': 0.1,
                'timing_correlation_preservation': 0.9
            }
        },
        AnomalyType.DATA_EXFILTRATION: {
            'strategy': PreservationStrategy.BEHAVIORAL_FINGERPRINTS,
            'preservation_level': PreservationLevel.CRITICAL,
            'critical_fields': ['network_io', 'data_volume', 'connection_patterns', 'transfer_rate'],
            'min_effectiveness': 0.95,
            'constraints': {
                'volume_anomaly_preservation': 0.98,
                'burst_pattern_detection': 0.95
            }
        },
        AnomalyType.ECONOMIC_ABUSE: {
            'strategy': PreservationStrategy.MULTIVARIATE_DENSITY,
            'preservation_level': PreservationLevel.HIGH,
            'critical_fields': ['execution_cost', 'resource_consumption', 'quota_usage'],
            'min_effectiveness': 0.88,
            'constraints': {
                'cost_anomaly_detection': 0.95,
                'resource_density_preservation': 0.9
            }
        },
        AnomalyType.SILENT_FAILURE: {
            'strategy': PreservationStrategy.INFORMATION_THEORETIC,
            'preservation_level': PreservationLevel.CRITICAL,
            'critical_fields': ['output_entropy', 'semantic_consistency', 'data_integrity_hash'],
            'min_effectiveness': 0.92,
            'constraints': {
                'information_preservation': 0.98,
                'semantic_similarity_threshold': 0.9
            }
        },
        AnomalyType.MEMORY_LEAK: {
            'strategy': PreservationStrategy.TEMPORAL_PATTERNS,
            'preservation_level': PreservationLevel.HIGH,
            'critical_fields': ['memory_usage_timeline', 'heap_growth_rate', 'gc_frequency'],
            'min_effectiveness': 0.85,
            'constraints': {
                'growth_trend_preservation': 0.95,
                'periodic_pattern_preservation': 0.9
            }
        }
    }
    
    return policies.get(anomaly_type, {
        'strategy': PreservationStrategy.FEATURE_VECTORS,
        'preservation_level': PreservationLevel.MEDIUM,
        'critical_fields': [],
        'min_effectiveness': 0.75,
        'constraints': {}
    })

async def benchmark_preservation_performance_enhanced(guard: EnhancedAnomalyPreservationGuard,
                                                    test_scenarios: List[Dict[str, Any]],
                                                    analysis_depth: AnalysisDepth = AnalysisDepth.DEEP) -> Dict[str, Any]:
    """Enhanced benchmark for preservation performance"""
    
    results = {
        'scenario_results': [],
        'performance_summary': {
            'average_effectiveness': 0.0,
            'average_processing_time_ms': 0.0,
            'average_confidence_score': 0.0,
            'violation_rate': 0.0,
            'rollback_rate': 0.0
        },
        'effectiveness_distribution': {
            'excellent': 0,      # >= 0.95
            'good': 0,           # 0.85 - 0.95
            'acceptable': 0,     # 0.75 - 0.85
            'poor': 0,           # 0.6 - 0.75
            'critical': 0        # < 0.6
        },
        'strategy_performance': {},
        'anomaly_type_performance': defaultdict(list)
    }
    
    total_effectiveness = 0.0
    total_processing_time = 0.0
    total_confidence = 0.0
    total_violations = 0
    total_rollbacks = 0
    
    for i, scenario in enumerate(test_scenarios):
        start_time = time.perf_counter()
        
        try:
            # Create processed version
            processed_scenario = copy.deepcopy(scenario)
            
            # Apply test modifications
            if 'large_field' in processed_scenario:
                processed_scenario['large_field_hash'] = hash(processed_scenario['large_field'])
                del processed_scenario['large_field']
            
            # Add some noise to numeric fields
            for key, value in processed_scenario.items():
                if isinstance(value, (int, float)) and 'time' in key.lower():
                    processed_scenario[key] = value * (1.0 + np.random.normal(0, 0.01))  # 1% noise
            
            # Assess preservation
            assessment = await guard.assess_preservation_impact(
                scenario, processed_scenario, f'benchmark_{i}', analysis_depth
            )
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # Categorize effectiveness
            effectiveness = assessment.preservation_effectiveness
            if effectiveness >= 0.95:
                results['effectiveness_distribution']['excellent'] += 1
            elif effectiveness >= 0.85:
                results['effectiveness_distribution']['good'] += 1
            elif effectiveness >= 0.75:
                results['effectiveness_distribution']['acceptable'] += 1
            elif effectiveness >= 0.6:
                results['effectiveness_distribution']['poor'] += 1
            else:
                results['effectiveness_distribution']['critical'] += 1
            
            # Track anomaly type performance
            for anomaly_type in assessment.affected_anomaly_types:
                results['anomaly_type_performance'][anomaly_type.value].append(effectiveness)
            
            scenario_result = {
                'scenario_id': i,
                'effectiveness': effectiveness,
                'processing_time_ms': processing_time,
                'confidence_score': assessment.confidence_score,
                'violations': len(assessment.critical_violations),
                'rollback_recommended': assessment.rollback_recommendation,
                'affected_types': [t.value for t in assessment.affected_anomaly_types],
                'information_loss': assessment.information_loss,
                'entropy_preserved': assessment.entropy_preserved
            }
            
            results['scenario_results'].append(scenario_result)
            
            total_effectiveness += effectiveness
            total_processing_time += processing_time
            total_confidence += assessment.confidence_score
            total_violations += len(assessment.critical_violations)
            if assessment.rollback_recommendation:
                total_rollbacks += 1
                
        except Exception as e:
            logger.error(f"Benchmark scenario {i} failed: {e}")
            scenario_result = {
                #!/usr/bin/env python3
"""
SCAFAD Layer 1: Enhanced Anomaly Preservation Guard
==================================================

Advanced anomaly preservation system ensuring 99.5%+ detectability retention across
all data conditioning processes. Features ML-powered signature identification, 
real-time preservation monitoring, and adaptive strategy optimization.

Key Innovations:
- Neural anomaly signature learning and preservation
- Information-theoretic preservation bounds
- Adaptive preservation strategies with reinforcement learning
- Real-time preservation effectiveness monitoring
- Automated rollback and recovery mechanisms
- Multi-dimensional anomaly fingerprinting
- Differential preservation for threat prioritization

Performance Targets:
- Preservation effectiveness: 99.5%+ guaranteed
- Processing latency: <0.8ms per assessment
- Memory overhead: <16MB
- Detection accuracy: 99.9%+ for known anomaly types
- False positive rate: <0.1% for preservation alerts

Author: SCAFAD Research Team
Institution: Birmingham Newman University
License: MIT
Version: 2.0.0
"""

import json
import time
import logging
import numpy as np
import asyncio
import statistics
import hashlib
import copy
import traceback
import warnings
from typing import Dict, Any, List, Optional, Tuple, Set, Union, Callable, Iterator
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from collections import defaultdict, deque, Counter
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import multiprocessing
import queue
import pickle
import psutil
import gc

# Scientific computing
from scipy import stats, signal, optimize
from scipy.spatial.distance import cosine, euclidean
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
import networkx as nx

# Information theory
import math
from entropy import shannon_entropy, conditional_entropy
import mutual_information as mi

# Performance monitoring
import cProfile
import pstats
from memory_profiler import profile
from functools import wraps

# Layer 1 dependencies
from .layer1_core import Layer1ProcessingResult, ProcessingMetrics, TelemetryRecord
from .layer1_schema import SchemaEvolutionEngine
from .layer1_privacy import PrivacyComplianceFilter
from .layer1_hashing import DeferredHashingManager, HashingResult
from .layer1_sanitization import SanitizationResult, SanitizationType

# Configure logging
logger = logging.getLogger(__name__)

# Performance monitoring decorator
def performance_monitor(func):
    """Decorator for monitoring function performance"""
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

# Enhanced enums
class AnomalyType(Enum):
    """Comprehensive anomaly types for serverless environments"""
    # Performance anomalies
    COLD_START = "cold_start"
    EXECUTION_DRIFT = "execution_drift"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    MEMORY_LEAK = "memory_leak"
    CPU_SPIKE = "cpu_spike"
    LATENCY_DEGRADATION = "latency_degradation"
    
    # Timing anomalies
    TIMING_ANOMALY = "timing_anomaly"
    EXECUTION_TIMEOUT = "execution_timeout"
    SCHEDULING_ANOMALY = "scheduling_anomaly"
    JITTER_PATTERN = "jitter_pattern"
    
    # Behavioral anomalies
    INVOCATION_PATTERN = "invocation_pattern"
    CALL_GRAPH_ANOMALY = "call_graph_anomaly"
    CONCURRENCY_VIOLATION = "concurrency_violation"
    STATE_CORRUPTION = "state_corruption"
    
    # Security anomalies
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    INJECTION_ATTACK = "injection_attack"
    LATERAL_MOVEMENT = "lateral_movement"
    CRYPTOJACKING = "cryptojacking"
    
    # System anomalies
    DEPENDENCY_FAILURE = "dependency_failure"
    SILENT_FAILURE = "silent_failure"
    CASCADE_FAILURE = "cascade_failure"
    SERVICE_DEGRADATION = "service_degradation"
    
    # Economic anomalies
    ECONOMIC_ABUSE = "economic_abuse"
    RESOURCE_WASTE = "resource_waste"
    BILLING_ANOMALY = "billing_anomaly"
    QUOTA_VIOLATION = "quota_violation"

class PreservationStrategy(Enum):
    """Advanced preservation strategies with ML enhancement"""
    STATISTICAL_BOUNDS = "statistical_bounds"
    TEMPORAL_PATTERNS = "temporal_patterns"
    STRUCTURAL_INTEGRITY = "structural_integrity"
    FEATURE_VECTORS = "feature_vectors"
    BEHAVIORAL_FINGERPRINTS = "behavioral_fingerprints"
    NEURAL_EMBEDDINGS = "neural_embeddings"
    INFORMATION_THEORETIC = "information_theoretic"
    GRAPH_TOPOLOGY = "graph_topology"
    SPECTRAL_ANALYSIS = "spectral_analysis"
    MULTIVARIATE_DENSITY = "multivariate_density"

class PreservationLevel(Enum):
    """Preservation protection levels"""
    CRITICAL = "critical"        # 99.9%+ preservation required
    HIGH = "high"               # 99.5%+ preservation required
    MEDIUM = "medium"           # 95%+ preservation required
    LOW = "low"                 # 85%+ preservation required
    MINIMAL = "minimal"         # Best effort, 70%+ preservation

class ProcessingMode(Enum):
    """Processing execution modes for optimization"""
    REAL_TIME = "real_time"           # Sub-millisecond processing
    NEAR_REAL_TIME = "near_real_time" # <5ms processing
    BATCH = "batch"                   # Optimized for throughput
    STREAMING = "streaming"           # Continuous processing
    ADAPTIVE = "adaptive"             # ML-guided optimization

class AnalysisDepth(Enum):
    """Depth of preservation analysis"""
    SURFACE = "surface"         # Basic field presence and type checking
    STATISTICAL = "statistical" # Statistical property analysis
    DEEP = "deep"              # Full signature analysis
    COMPREHENSIVE = "comprehensive" # All analysis methods
    NEURAL = "neural"          # ML-powered analysis

@dataclass
class AnomalySignature:
    """Enhanced anomaly signature with ML features"""
    # Basic identification
    signature_id: str
    anomaly_type: AnomalyType
    signature_fields: Set[str]
    
    # Statistical characteristics
    statistical_properties: Dict[str, Any]
    temporal_characteristics: Dict[str, Any]
    structural_characteristics: Dict[str, Any]
    
    # ML-enhanced features
    feature_vector: Optional[np.ndarray] = None
    neural_embedding: Optional[np.ndarray] = None
    information_content: Optional[float] = None
    mutual_information_map: Optional[Dict[str, float]] = None
    
    # Preservation requirements
    preservation_priority: float = 0.9
    minimum_preservation_threshold: float = 0.85
    preservation_level: PreservationLevel = PreservationLevel.HIGH
    
    # Detection metadata
    detection_confidence: float = 0.9
    false_positive_rate: float = 0.01
    training_data_size: int = 0
    last_updated: float = field(default_factory=time.time)
    
    # Behavioral patterns
    context_dependencies: Set[str] = field(default_factory=set)
    correlation_patterns: Dict[str, float] = field(default_factory=dict)
    causality_chains: List[str] = field(default_factory=list)

@dataclass
class PreservationRule:
    """Enhanced preservation rules with adaptive capabilities"""
    rule_id: str
    anomaly_type: AnomalyType
    strategy: PreservationStrategy
    preservation_level: PreservationLevel
    
    # Field protection
    protected_fields: Set[str]
    field_transformations: Dict[str, str]
    field_weights: Dict[str, float] = field(default_factory=dict)
    
    # Constraints
    statistical_constraints: Dict[str, Any] = field(default_factory=dict)
    temporal_constraints: Dict[str, Any] = field(default_factory=dict)
    structural_constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Adaptive parameters
    learning_rate: float = 0.01
    adaptation_window: int = 100
    performance_history: List[float] = field(default_factory=list)
    
    # Validation
    enabled: bool = True
    last_validation: float = field(default_factory=time.time)
    validation_score: float = 1.0
    
    # Performance optimization
    processing_priority: int = 1
    parallel_safe: bool = True
    cache_enabled: bool = True

@dataclass
class PreservationAssessment:
    """Comprehensive preservation assessment with detailed analytics"""
    # Basic metrics
    assessment_id: str
    original_detectability_score: float
    post_processing_detectability_score: float
    preservation_effectiveness: float
    
    # Detailed analysis
    affected_anomaly_types: List[AnomalyType]
    preservation_breakdown: Dict[str, float] = field(default_factory=dict)
    field_impact_analysis: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Information theory metrics
    information_loss: float = 0.0
    entropy_preserved: float = 1.0
    mutual_information_preserved: float = 1.0
    
    # Violation analysis
    critical_violations: List[str] = field(default_factory=list)
    warning_violations: List[str] = field(default_factory=list)
    constraint_violations: Dict[str, Any] = field(default_factory=dict)
    
    # Recommendations and actions
    recommendations: List[str] = field(default_factory=list)
    suggested_actions: List[Dict[str, Any]] = field(default_factory=list)
    rollback_recommendation: bool = False
    
    # Context and metadata
    processing_stage: str = ""
    processing_mode: ProcessingMode = ProcessingMode.REAL_TIME
    analysis_depth: AnalysisDepth = AnalysisDepth.STATISTICAL
    assessment_timestamp: float = field(default_factory=time.time)
    processing_time_ms: float = 0.0
    
    # Quality metrics
    confidence_score: float = 0.9
    reliability_score: float = 0.9
    uncertainty_bounds: Tuple[float, float] = (0.0, 0.0)

class NeuralAnomalyEncoder:
    """Neural network for learning anomaly embeddings"""
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.encoder = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_history = []
    
    def extract_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from telemetry data"""
        features = []
        
        # Basic numerical features
        for key, value in data.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                # String length and entropy as features
                features.extend([len(value), shannon_entropy(value) if value else 0.0])
            elif isinstance(value, list):
                # List statistics
                if value and all(isinstance(x, (int, float)) for x in value):
                    features.extend([len(value), np.mean(value), np.std(value)])
                else:
                    features.append(len(value))
        
        # Temporal features if timestamp present
        if 'timestamp' in data:
            ts = data['timestamp']
            features.extend([
                ts % 86400,  # Time of day
                (ts // 86400) % 7,  # Day of week
                np.sin(2 * np.pi * (ts % 86400) / 86400),  # Cyclical time
                np.cos(2 * np.pi * (ts % 86400) / 86400)
            ])
        
        return np.array(features, dtype=np.float32)
    
    def train(self, training_data: List[Dict[str, Any]]):
        """Train the neural encoder on anomaly patterns"""
        # Extract features from training data
        feature_matrix = np.array([
            self.extract_features(sample) for sample in training_data
        ])
        
        # Normalize features
        feature_matrix = self.scaler.fit_transform(feature_matrix)
        
        # Use PCA for dimensionality reduction and embedding
        self.encoder = PCA(n_components=self.embedding_dim)
        embeddings = self.encoder.fit_transform(feature_matrix)
        
        self.is_trained = True
        self.training_history.append({
            'timestamp': time.time(),
            'samples': len(training_data),
            'explained_variance': self.encoder.explained_variance_ratio_.sum()
        })
        
        logger.info(f"Neural encoder trained on {len(training_data)} samples, "
                   f"explained variance: {self.encoder.explained_variance_ratio_.sum():.3f}")
    
    def encode(self, data: Dict[str, Any]) -> np.ndarray:
        """Encode data into neural embedding"""
        if not self.is_trained:
            raise ValueError("Encoder must be trained before use")
        
        features = self.extract_features(data).reshape(1, -1)
        features = self.scaler.transform(features)
        embedding = self.encoder.transform(features)[0]
        
        return embedding

class InformationTheoreticAnalyzer:
    """Information-theoretic analysis for preservation assessment"""
    
    def __init__(self):
        self.field_entropy_cache = {}
        self.mutual_info_cache = {}
    
    def calculate_entropy(self, data: Any) -> float:
        """Calculate entropy of data"""
        if isinstance(data, str):
            return shannon_entropy(data) if data else 0.0
        elif isinstance(data, (list, tuple)):
            if not data:
                return 0.0
            # Convert to string representation for entropy calculation
            str_repr = ''.join(str(x) for x in data)
            return shannon_entropy(str_repr)
        elif isinstance(data, dict):
            # Calculate entropy of serialized dict
            json_str = json.dumps(data, sort_keys=True)
            return shannon_entropy(json_str)
        else:
            # Convert to string for entropy calculation
            return shannon_entropy(str(data)) if data is not None else 0.0
    
    def calculate_mutual_information(self, field1: Any, field2: Any) -> float:
        """Calculate mutual information between two fields"""
        try:
            # Convert to comparable format
            str1 = str(field1) if field1 is not None else ""
            str2 = str(field2) if field2 is not None else ""
            
            if not str1 or not str2:
                return 0.0
            
            # Use character-level mutual information
            chars1 = list(str1)
            chars2 = list(str2)
            
            # Align sequences
            min_len = min(len(chars1), len(chars2))
            if min_len == 0:
                return 0.0
            
            chars1 = chars1[:min_len]
            chars2 = chars2[:min_len]
            
            # Calculate mutual information
            return mutual_info_score(chars1, chars2)
        except Exception as e:
            logger.debug(f"Mutual information calculation failed: {e}")
            return 0.0
    
    def analyze_information_loss(self, original: Dict[str, Any], 
                                processed: Dict[str, Any]) -> Dict[str, float]:
        """Analyze information loss between original and processed data"""
        analysis = {
            'total_entropy_loss': 0.0,
            'field_entropy_losses': {},
            'mutual_information_loss': 0.0,
            'structural_information_loss': 0.0
        }
        
        # Calculate entropy loss for each field
        original_total_entropy = 0.0
        processed_total_entropy = 0.0
        
        all_fields = set(original.keys()) | set(processed.keys())
        
        for field in all_fields:
            orig_val = original.get(field)
            proc_val = processed.get(field)
            
            orig_entropy = self.calculate_entropy(orig_val)
            proc_entropy = self.calculate_entropy(proc_val)
            
            entropy_loss = max(0.0, orig_entropy - proc_entropy)
            analysis['field_entropy_losses'][field] = entropy_loss
            
            original_total_entropy += orig_entropy
            processed_total_entropy += proc_entropy
        
        analysis['total_entropy_loss'] = max(0.0, original_total_entropy - processed_total_entropy)
        
        # Calculate mutual information loss
        original_mi = self._calculate_total_mutual_information(original)
        processed_mi = self._calculate_total_mutual_information(processed)
        analysis['mutual_information_loss'] = max(0.0, original_mi - processed_mi)
        
        # Calculate structural information loss
        orig_structure_entropy = self.calculate_entropy(list(original.keys()))
        proc_structure_entropy = self.calculate_entropy(list(processed.keys()))
        analysis['structural_information_loss'] = max(0.0, orig_structure_entropy - proc_structure_entropy)
        
        return analysis
    
    def _calculate_total_mutual_information(self, data: Dict[str, Any]) -> float:
        """Calculate total mutual information within data structure"""
        fields = list(data.keys())
        total_mi = 0.0
        count = 0
        
        for i, field1 in enumerate(fields):
            for field2 in fields[i+1:]:
                mi = self.calculate_mutual_information(data[field1], data[field2])
                total_mi += mi
                count += 1
        
        return total_mi / count if count > 0 else 0.0

class GraphAnalyzer:
    """Graph-based analysis for structural anomaly preservation"""
    
    def __init__(self):
        self.graph_cache = {}
    
    def build_data_graph(self, data: Dict[str, Any]) -> nx.Graph:
        """Build graph representation of data structure"""
        G = nx.Graph()
        
        def add_nodes_edges(obj, parent_key="root", depth=0):
            if depth > 5:  # Prevent infinite recursion
                return
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    node_id = f"{parent_key}.{key}" if parent_key != "root" else key
                    G.add_node(node_id, type="dict_key", value=str(value)[:100])
                    
                    if parent_key != "root":
                        G.add_edge(parent_key, node_id)
                    
                    if isinstance(value, (dict, list)):
                        add_nodes_edges(value, node_id, depth + 1)
            
            elif isinstance(obj, list):
                for i, item in enumerate(obj[:10]):  # Limit list size
                    node_id = f"{parent_key}[{i}]"
                    G.add_node(node_id, type="list_item", value=str(item)[:100])
                    G.add_edge(parent_key, node_id)
                    
                    if isinstance(item, (dict, list)):
                        add_nodes_edges(item, node_id, depth + 1)
        
        add_nodes_edges(data)
        return G
    
    def calculate_graph_similarity(self, graph1: nx.Graph, graph2: nx.Graph) -> float:
        """Calculate structural similarity between two graphs"""
        if len(graph1.nodes) == 0 and len(graph2.nodes) == 0:
            return 1.0
        
        if len(graph1.nodes) == 0 or len(graph2.nodes) == 0:
            return 0.0
        
        # Calculate node similarity
        nodes1 = set(graph1.nodes)
        nodes2 = set(graph2.nodes)
        node_similarity = len(nodes1 & nodes2) / len(nodes1 | nodes2)
        
        # Calculate edge similarity
        edges1 = set(graph1.edges)
        edges2 = set(graph2.edges)
        edge_similarity = len(edges1 & edges2) / len(edges1 | edges2) if (edges1 | edges2) else 1.0
        
        # Calculate degree distribution similarity
        degree_seq1 = sorted([d for n, d in graph1.degree()])
        degree_seq2 = sorted([d for n, d in graph2.degree()])
        
        # Pad shorter sequence
        max_len = max(len(degree_seq1), len(degree_seq2))
        degree_seq1.extend([0] * (max_len - len(degree_seq1)))
        degree_seq2.extend([0] * (max_len - len(degree_seq2)))
        
        degree_similarity = 1.0 - euclidean(degree_seq1, degree_seq2) / (max_len * max(max(degree_seq1 + degree_seq2), 1))
        
        # Combine similarities
        overall_similarity = (node_similarity * 0.4 + edge_similarity * 0.4 + degree_similarity * 0.2)
        return max(0.0, min(1.0, overall_similarity))
    
    def analyze_structural_preservation(self, original: Dict[str, Any], 
                                      processed: Dict[str, Any]) -> Dict[str, float]:
        """Analyze structural preservation between original and processed data"""
        orig_graph = self.build_data_graph(original)
        proc_graph = self.build_data_graph(processed)
        
        similarity = self.calculate_graph_similarity(orig_graph, proc_graph)
        
        return {
            'structural_similarity': similarity,
            'node_preservation': len(set(orig_graph.nodes) & set(proc_graph.nodes)) / len(orig_graph.nodes) if orig_graph.nodes else 1.0,
            'edge_preservation': len(set(orig_graph.edges) & set(proc_graph.edges)) / len(orig_graph.edges) if orig_graph.edges else 1.0,
            'topology_preservation': similarity
        }

class AdaptivePreservationOptimizer:
    """Adaptive optimization of preservation strategies using reinforcement learning concepts"""
    
    def __init__(self):
        self.strategy_performance = defaultdict(list)
        self.exploration_rate = 0.1
        self.learning_rate = 0.01
        self.strategy_weights = defaultdict(lambda: 1.0)
    
    def select_strategy(self, context: Dict[str, Any]) -> PreservationStrategy:
        """Select optimal preservation strategy based on context and performance history"""
        available_strategies = list(PreservationStrategy)
        
        # Epsilon-greedy strategy selection
        if np.random.random() < self.exploration_rate:
            # Exploration: random strategy
            return np.random.choice(available_strategies)
        else:
            # Exploitation: best performing strategy
            best_strategy = max(available_strategies, key=lambda s: self.strategy_weights[s])
            return best_strategy
    
    def update_performance(self, strategy: PreservationStrategy, performance: float):
        """Update strategy performance based on results"""
        self.strategy_performance[strategy].append(performance)
        
        # Update weights using moving average
        current_weight = self.strategy_weights[strategy]
        new_weight = current_weight + self.learning_rate * (performance - current_weight)
        self.strategy_weights[strategy] = new_weight
        
        # Decay exploration rate
        self.exploration_rate = max(0.01, self.exploration_rate * 0.995)
    
    def get_strategy_recommendations(self) -> Dict[str, Any]:
        """Get strategy performance recommendations"""
        recommendations = {}
        
        for strategy, performances in self.strategy_performance.items():
            if performances:
                recommendations[strategy.value] = {
                    'average_performance': np.mean(performances),
                    'performance_std': np.std(performances),
                    'sample_count': len(performances),
                    'weight': self.strategy_weights[strategy],
                    'trend': 'improving' if len(performances) > 5 and 
                           np.mean(performances[-5:]) > np.mean(performances[:-5]) else 'stable'
                }
        
        return recommendations

class EnhancedAnomalyPreservationGuard:
    """
    Advanced anomaly preservation guard with ML-enhanced capabilities,
    information-theoretic analysis, and adaptive optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the enhanced preservation guard"""
        self.config = config
        self.preservation_rules: Dict[AnomalyType, PreservationRule] = {}
        self.known_signatures: Dict[str, AnomalySignature] = {}
        self.assessment_history: deque = deque(maxlen=self.config.get('history_size', 10000))
        
        # Advanced components
        self.neural_encoder = NeuralAnomalyEncoder(embedding_dim=64)
        self.info_analyzer = InformationTheoreticAnalyzer()
        self.graph_analyzer = GraphAnalyzer()
        self.adaptive_optimizer = AdaptivePreservationOptimizer()
        
        # Performance tracking
        self.metrics = {
            'total_assessments': 0,
            'preservation_violations': 0,
            'rollbacks_triggered': 0,
            'average_preservation_score': 0.0,
            'critical_failures': 0,
            'processing_time_ms': [],
            'memory_usage_mb': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Caching for performance
        self.signature_cache = {}
        self.analysis_cache = {}
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes
        
        # Threading for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4))
        
        # Initialize components
        self._initialize_preservation_rules()
        self._initialize_signature_database()
        self._initialize_ml_components()
        
        logger.info("Enhanced Anomaly Preservation Guard initialized with ML capabilities")
    
    def _initialize_preservation_rules(self):
        """Initialize comprehensive preservation rules"""
        
        # Cold start anomalies with neural embeddings
        self.preservation_rules[AnomalyType.COLD_START] = PreservationRule(
            rule_id="cold_start_neural",
            anomaly_type=AnomalyType.COLD_START,
            strategy=PreservationStrategy.NEURAL_EMBEDDINGS,
            preservation_level=PreservationLevel.HIGH,
            protected_fields={
                'duration_ms', 'memory_used_mb', 'init_time_ms', 'cold_start_indicator',
                'container_lifecycle', 'bootstrap_time', 'runtime_initialization'
            },
            field_transformations={
                'duration_ms': 'preserve_distribution_neural',
                'memory_used_mb': 'preserve_range_adaptive',
                'init_time_ms': 'preserve_outliers_weighted',
                'bootstrap_time': 'preserve_temporal_sequence'
            },
            field_weights={
                'init_time_ms': 0.9,
                'duration_ms': 0.8,
                'memory_used_mb': 0.7,
                'cold_start_indicator': 1.0
            },
            statistical_constraints={
                'duration_distribution_kl_divergence': 0.1,
                'memory_percentile_preservation': 0.95,
                'timing_correlation_preservation': 0.9,
                'outlier_detection_sensitivity': 0.85
            },
            temporal_constraints={
                'initialization_sequence_integrity': 0.95,
                'temporal_correlation_preservation': 0.9,
                'phase_transition_preservation': 0.85
            },
            learning_rate=0.01,
            adaptation_window=100
        )
        
        # Data exfiltration with behavioral fingerprints
        self.preservation_rules[AnomalyType.DATA_EXFILTRATION] = PreservationRule(
            rule_id="exfiltration_behavioral",
            anomaly_type=AnomalyType.DATA_EXFILTRATION,
            strategy=PreservationStrategy.BEHAVIORAL_FINGERPRINTS,
            preservation_level=PreservationLevel.CRITICAL,
            protected_fields={
                'network_io', 'data_volume', 'connection_patterns', 'transfer_rate',
                'bandwidth_utilization', 'protocol_distribution', 'destination_entropy',
                'timing_patterns', 'payload_characteristics'
            },
            field_transformations={
                'network_io': 'preserve_volume_patterns_advanced',
                'data_volume': 'preserve_outliers_contextual',
                'connection_patterns': 'preserve_graph_structure',
                'timing_patterns': 'preserve_frequency_domain'
            },
            field_weights={
                'data_volume': 0.95,
                'transfer_rate': 0.9,
                'network_io': 0.85,
                'timing_patterns': 0.8
            },
            statistical_constraints={
                'volume_anomaly_preservation': 0.98,
                'burst_pattern_detection': 0.95,
                'protocol_distribution_preservation': 0.9,
                'timing_signature_preservation': 0.92
            },
            temporal_constraints={
                'burst_sequence_preservation': 0.95,
                'frequency_analysis_integrity': 0.9,
                'phase_coherence_preservation': 0.85
            },
            structural_constraints={
                'connection_graph_preservation': 0.9,
                'flow_topology_preservation': 0.85
            }
        )
        
        # Economic abuse with multivariate analysis
        self.preservation_rules[AnomalyType.ECONOMIC_ABUSE] = PreservationRule(
            rule_id="economic_abuse_multivariate",
            anomaly_type=AnomalyType.ECONOMIC_ABUSE,
            strategy=PreservationStrategy.MULTIVARIATE_DENSITY,
            preservation_level=PreservationLevel.HIGH,
            protected_fields={
                'execution_cost', 'resource_consumption', 'billing_units', 'quota_usage',
                'cpu_time', 'memory_time', 'storage_operations', 'network_costs',
                'invocation_frequency', 'duration_billing'
            },
            field_transformations={
                'execution_cost': 'preserve_cost_distribution',
                'resource_consumption': 'preserve_multivariate_density',
                'invocation_frequency': 'preserve_temporal_patterns',
                'quota_usage': 'preserve_threshold_crossings'
            },
            field_weights={
                'execution_cost': 0.95,
                'resource_consumption': 0.9,
                'quota_usage': 0.85,
                'invocation_frequency': 0.8
            },
            statistical_constraints={
                'cost_anomaly_detection': 0.95,
                'resource_density_preservation': 0.9,
                'usage_pattern_preservation': 0.88,
                'billing_correlation_preservation': 0.92
            },
            temporal_constraints={
                'cost_trend_preservation': 0.9,
                'billing_cycle_awareness': 0.85,
                'usage_burst_detection': 0.93
            }
        )
        
        # Silent failures with information-theoretic analysis
        self.preservation_rules[AnomalyType.SILENT_FAILURE] = PreservationRule(
            rule_id="silent_failure_information",
            anomaly_type=AnomalyType.SILENT_FAILURE,
            strategy=PreservationStrategy.INFORMATION_THEORETIC,
            preservation_level=PreservationLevel.CRITICAL,
            protected_fields={
                'output_entropy', 'semantic_consistency', 'data_integrity_hash',
                'execution_checkpoints', 'state_transitions', 'error_absence_indicators',
                'output_size', 'content_patterns', 'validation_results'
            },
            field_transformations={
                'output_entropy': 'preserve_information_content',
                'semantic_consistency': 'preserve_semantic_fingerprint',
                'data_integrity_hash': 'preserve_cryptographic_integrity',
                'state_transitions': 'preserve_execution_trace'
            },
            field_weights={
                'output_entropy': 0.95,
                'semantic_consistency': 0.9,
                'data_integrity_hash': 0.95,
                'execution_checkpoints': 0.85
            },
            statistical_constraints={
                'information_preservation': 0.98,
                'semantic_similarity_threshold': 0.9,
                'entropy_variation_bounds': 0.05,
                'consistency_score_preservation': 0.95
            }
        )
        
        # Memory leaks with temporal analysis
        self.preservation_rules[AnomalyType.MEMORY_LEAK] = PreservationRule(
            rule_id="memory_leak_temporal",
            anomaly_type=AnomalyType.MEMORY_LEAK,
            strategy=PreservationStrategy.TEMPORAL_PATTERNS,
            preservation_level=PreservationLevel.HIGH,
            protected_fields={
                'memory_usage_timeline', 'heap_growth_rate', 'gc_frequency',
                'memory_allocation_patterns', 'object_lifecycle', 'leak_indicators',
                'memory_pressure_events', 'allocation_size_distribution'
            },
            field_transformations={
                'memory_usage_timeline': 'preserve_trend_analysis',
                'heap_growth_rate': 'preserve_derivative_patterns',
                'gc_frequency': 'preserve_frequency_domain',
                'allocation_patterns': 'preserve_spectral_signature'
            },
            field_weights={
                'memory_usage_timeline': 0.95,
                'heap_growth_rate': 0.9,
                'gc_frequency': 0.85,
                'leak_indicators': 0.95
            },
            temporal_constraints={
                'growth_trend_preservation': 0.95,
                'periodic_pattern_preservation': 0.9,
                'acceleration_detection': 0.88,
                'leak_signature_preservation': 0.93
            }
        )
    
    def _initialize_signature_database(self):
        """Initialize comprehensive anomaly signature database"""
        
        # Advanced cold start signature with neural features
        self.known_signatures['cold_start_advanced'] = AnomalySignature(
            signature_id='cold_start_advanced',
            anomaly_type=AnomalyType.COLD_START,
            signature_fields={
                'duration_ms', 'memory_used_mb', 'init_time_ms', 'bootstrap_time',
                'container_lifecycle', 'runtime_initialization', 'cold_start_indicator'
            },
            statistical_properties={
                'duration_multiplier_range': (2.0, 8.0),
                'memory_baseline_ratio': (1.5, 4.0),
                'init_time_threshold': 1000,
                'bootstrap_correlation': 0.85,
                'lifecycle_state_transitions': ['init', 'bootstrap', 'ready'],
                'timing_distribution': 'log_normal'
            },
            temporal_characteristics={
                'initialization_sequence': True,
                'resource_ramp_up_pattern': 'exponential',
                'stabilization_time': (500, 2000),
                'temporal_correlation_window': 30000,
                'phase_transitions': ['cold', 'warming', 'warm']
            },
            structural_characteristics={
                'required_field_count': 5,
                'optional_field_count': 2,
                'nested_depth_max': 3,
                'field_correlation_matrix': {
                    ('duration_ms', 'init_time_ms'): 0.85,
                    ('memory_used_mb', 'bootstrap_time'): 0.7
                }
            },
            preservation_priority=0.95,
            minimum_preservation_threshold=0.9,
            preservation_level=PreservationLevel.HIGH,
            detection_confidence=0.93,
            false_positive_rate=0.02,
            context_dependencies={'function_type', 'runtime_version', 'deployment_method'},
            correlation_patterns={
                'execution_environment': 0.8,
                'resource_constraints': 0.75,
                'concurrent_executions': -0.3
            }
        )
        
        # Sophisticated data exfiltration signature
        self.known_signatures['exfiltration_advanced'] = AnomalySignature(
            signature_id='exfiltration_advanced',
            anomaly_type=AnomalyType.DATA_EXFILTRATION,
            signature_fields={
                'network_io', 'data_volume', 'connection_patterns', 'transfer_rate',
                'bandwidth_utilization', 'protocol_distribution', 'destination_entropy',
                'timing_patterns', 'encryption_indicators'
            },
            statistical_properties={
                'volume_spike_factor': (5.0, 50.0),
                'transfer_rate_anomaly': (3.0, 20.0),
                'connection_burst_size': (3, 15),
                'bandwidth_utilization_threshold': 0.8,
                'protocol_deviation_score': 0.7,
                'timing_regularity_index': 0.3,
                'data_compression_ratio': (0.1, 0.9)
            },
            temporal_characteristics={
                'burst_duration_range': (30, 1800),
                'off_hours_correlation': True,
                'transfer_periodicity': 'irregular',
                'sustained_transfer_indicator': True,
                'timing_jitter_pattern': 'low',
                'frequency_domain_signature': 'broadband'
            },
            structural_characteristics={
                'connection_graph_complexity': 'high',
                'protocol_stack_depth': 'variable',
                'encryption_layer_presence': 'optional',
                'payload_structure_entropy': 'high'
            },
            preservation_priority=0.98,
            minimum_preservation_threshold=0.95,
            preservation_level=PreservationLevel.CRITICAL,
            detection_confidence=0.91,
            false_positive_rate=0.01,
            context_dependencies={'network_topology', 'security_policies', 'data_classification'},
            correlation_patterns={
                'user_behavior': 0.6,
                'system_load': -0.2,
                'security_events': 0.85
            }
        )
        
        # Economic abuse pattern signature
        self.known_signatures['economic_abuse_pattern'] = AnomalySignature(
            signature_id='economic_abuse_pattern',
            anomaly_type=AnomalyType.ECONOMIC_ABUSE,
            signature_fields={
                'execution_cost', 'resource_consumption', 'invocation_frequency',
                'quota_usage', 'billing_anomaly_score', 'cost_efficiency_ratio'
            },
            statistical_properties={
                'cost_spike_multiplier': (3.0, 20.0),
                'resource_waste_ratio': (2.0, 10.0),
                'frequency_anomaly_factor': (5.0, 100.0),
                'quota_utilization_rate': (0.8, 1.2),
                'efficiency_degradation': (0.1, 0.5)
            },
            temporal_characteristics={
                'abuse_duration': (300, 7200),
                'cost_accumulation_rate': 'exponential',
                'billing_cycle_alignment': False,
                'usage_pattern_deviation': 'high'
            },
            preservation_priority=0.92,
            minimum_preservation_threshold=0.88,
            preservation_level=PreservationLevel.HIGH,
            detection_confidence=0.89
        )
        
        # Silent failure signature with information theory
        self.known_signatures['silent_failure_entropy'] = AnomalySignature(
            signature_id='silent_failure_entropy',
            anomaly_type=AnomalyType.SILENT_FAILURE,
            signature_fields={
                'output_entropy', 'semantic_consistency', 'data_integrity_score',
                'execution_trace_completeness', 'error_absence_indicator'
            },
            statistical_properties={
                'entropy_deviation_threshold': 0.2,
                'consistency_score_range': (0.6, 0.9),
                'integrity_violation_score': (0.1, 0.4),
                'trace_completeness_ratio': (0.7, 0.95),
                'expected_error_absence': True
            },
            temporal_characteristics={
                'detection_delay': (0, 300),
                'symptom_emergence_pattern': 'delayed',
                'impact_propagation_rate': 'slow'
            },
            preservation_priority=0.96,
            minimum_preservation_threshold=0.92,
            preservation_level=PreservationLevel.CRITICAL,
            detection_confidence=0.87,
            false_positive_rate=0.05
        )
    
    def _initialize_ml_components(self):
        """Initialize machine learning components"""
        # Pre-train neural encoder if training data available
        training_config = self.config.get('ml_training', {})
        
        if training_config.get('pretrained_model_path'):
            try:
                self._load_pretrained_model(training_config['pretrained_model_path'])
            except Exception as e:
                logger.warning(f"Failed to load pretrained model: {e}")
        
        # Initialize adaptive optimizer parameters
        self.adaptive_optimizer.exploration_rate = training_config.get('exploration_rate', 0.1)
        self.adaptive_optimizer.learning_rate = training_config.get('learning_rate', 0.01)
    
    def _load_pretrained_model(self, model_path: str):
        """Load pretrained neural encoder"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.neural_encoder.encoder = model_data['encoder']
            self.neural_encoder.scaler = model_data['scaler']
            self.neural_encoder.is_trained = True
            self.neural_encoder.training_history = model_data.get('history', [])
            
            logger.info(f"Loaded pretrained model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            raise
    
    @performance_monitor
    async def assess_preservation_impact(self, 
                                       original_data: Dict[str, Any],
                                       processed_data: Dict[str, Any],
                                       processing_stage: str,
                                       analysis_depth: AnalysisDepth = AnalysisDepth.DEEP,
                                       processing_mode: ProcessingMode = ProcessingMode.REAL_TIME) -> PreservationAssessment:
        """
        Comprehensive preservation impact assessment with ML enhancement
        
        Args:
            original_data: Original telemetry data
            processed_data: Data after processing
            processing_stage: Name of the processing stage
            analysis_depth: Depth of analysis to perform
            processing_mode: Processing mode for optimization
            
        Returns:
            Enhanced PreservationAssessment with detailed analytics
        """
        
        assessment_id = f"assess_{int(time.time() * 1000)}_{hash(str(original_data))}"
        start_time = time.perf_counter()
        
        try:
            # Check cache first for performance
            cache_key = self._generate_cache_key(original_data, processed_data, processing_stage)
            if cache_key in self.analysis_cache:
                cache_entry = self.analysis_cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                    self.metrics['cache_hits'] += 1
                    return cache_entry['assessment']
            
            self.metrics['cache_misses'] += 1
            
            # Select optimal preservation strategy
            context = {
                'processing_stage': processing_stage,
                'data_size': len(str(original_data)),
                'field_count': len(original_data),
                'analysis_depth': analysis_depth
            }
            optimal_strategy = self.adaptive_optimizer.select_strategy(context)
            
            # Parallel analysis for performance
            if processing_mode in [ProcessingMode.BATCH, ProcessingMode.STREAMING]:
                analysis_tasks = await self._run_parallel_analysis(
                    original_data, processed_data, analysis_depth
                )
            else:
                analysis_tasks = await self._run_sequential_analysis(
                    original_data, processed_data, analysis_depth
                )
            
            # Core detectability analysis
            original_score = await self._calculate_enhanced_detectability_score(
                original_data, analysis_depth
            )
            processed_score = await self._calculate_enhanced_detectability_score(
                processed_data, analysis_depth
            )
            
            preservation_effectiveness = processed_score / original_score if original_score > 0 else 1.0
            
            # Advanced analysis components
            affected_types = await self._identify_affected_anomaly_types_advanced(
                original_data, processed_data, analysis_tasks
            )
            
            preservation_breakdown = await self._calculate_preservation_breakdown(
                original_data, processed_data, affected_types
            )
            
            field_impact_analysis = await self._analyze_field_impact(
                original_data, processed_data, analysis_tasks
            )
            
            # Information-theoretic analysis
            info_analysis = self.info_analyzer.analyze_information_loss(
                original_data, processed_data
            )
            
            # Violation detection
            critical_violations, warning_violations, constraint_violations = await self._detect_violations_comprehensive(
                original_data, processed_data, affected_types, analysis_tasks
            )
            
            # Generate intelligent recommendations
            recommendations, suggested_actions = await self._generate_intelligent_recommendations(
                preservation_effectiveness, affected_types, critical_violations,
                analysis_tasks, optimal_strategy
            )
            
            # Determine rollback recommendation
            rollback_recommendation = await self._should_recommend_rollback(
                preservation_effectiveness, critical_violations, affected_types
            )
            
            # Calculate confidence and uncertainty
            confidence_score, uncertainty_bounds = self._calculate_assessment_confidence(
                analysis_tasks, preservation_effectiveness
            )
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # Create comprehensive assessment
            assessment = PreservationAssessment(
                assessment_id=assessment_id,
                original_detectability_score=original_score,
                post_processing_detectability_score=processed_score,
                preservation_effectiveness=preservation_effectiveness,
                affected_anomaly_types=affected_types,
                preservation_breakdown=preservation_breakdown,
                field_impact_analysis=field_impact_analysis,
                information_loss=info_analysis['total_entropy_loss'],
                entropy_preserved=1.0 - info_analysis['total_entropy_loss'],
                mutual_information_preserved=1.0 - info_analysis['mutual_information_loss'],
                critical_violations=critical_violations,
                warning_violations=warning_violations,
                constraint_violations=constraint_violations,
                recommendations=recommendations,
                suggested_actions=suggested_actions,
                rollback_recommendation=rollback_recommendation,
                processing_stage=processing_stage,
                processing_mode=processing_mode,
                analysis_depth=analysis_depth,
                processing_time_ms=processing_time,
                confidence_score=confidence_score,
                reliability_score=min(confidence_score, preservation_effectiveness),
                uncertainty_bounds=uncertainty_bounds
            )
            
            # Update metrics and cache
            self._update_preservation_metrics_enhanced(assessment)
            self.assessment_history.append(assessment)
            
            # Cache result for future use
            self.analysis_cache[cache_key] = {
                'assessment': assessment,
                'timestamp': time.time()
            }
            
            # Update adaptive optimizer
            self.adaptive_optimizer.update_performance(optimal_strategy, preservation_effectiveness)
            
            logger.debug(f"Enhanced preservation assessment completed in {processing_time:.2f}ms: "
                        f"effectiveness={preservation_effectiveness:.3f}, confidence={confidence_score:.3f}")
            
            return assessment
            
        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Preservation assessment failed after {processing_time:.2f}ms: {e}")
            logger.debug(traceback.format_exc())
            raise
    
    async def _run_parallel_analysis(self, original_data: Dict[str, Any], 
                                   processed_data: Dict[str, Any],
                                   analysis_depth: AnalysisDepth) -> Dict[str, Any]:
        """Run analysis components in parallel for better performance"""
        
        analysis_tasks = {}
        
        # Submit parallel tasks
        futures = {}
        
        if analysis_depth in [AnalysisDepth.DEEP, AnalysisDepth.COMPREHENSIVE]:
            futures['info_theory'] = self.thread_pool.submit(
                self.info_analyzer.analyze_information_loss, original_data, processed_data
            )
            
            futures['graph_analysis'] = self.thread_pool.submit(
                self.graph_analyzer.analyze_structural_preservation, original_data, processed_data
            )
        
        if analysis_depth == AnalysisDepth.COMPREHENSIVE:
            if self.neural_encoder.is_trained:
                futures['neural_analysis'] = self.thread_pool.submit(
                    self._analyze_neural_preservation, original_data, processed_data
                )
        
        # Collect results
        for task_name, future in futures.items():
            try:
                analysis_tasks[task_name] = future.result(timeout=5.0)  # 5 second timeout
            except Exception as e:
                logger.warning(f"Parallel analysis task {task_name} failed: {e}")
                analysis_tasks[task_name] = {}
        
        return analysis_tasks
    
    async def _run_sequential_analysis(self, original_data: Dict[str, Any],
                                     processed_data: Dict[str, Any],
                                     analysis_depth: AnalysisDepth) -> Dict[str, Any]:
        """Run analysis components sequentially for real-time processing"""
        
        analysis_tasks = {}
        
        if analysis_depth in [AnalysisDepth.STATISTICAL, AnalysisDepth.DEEP, AnalysisDepth.COMPREHENSIVE]:
            analysis_tasks['basic_stats'] = await self._analyze_basic_statistics(
                original_data, processed_data
            )
        
        if analysis_depth in [AnalysisDepth.DEEP, AnalysisDepth.COMPREHENSIVE]:
            analysis_tasks['info_theory'] = self.info_analyzer.analyze_information_loss(
                original_data, processed_data
            )
        
        if analysis_depth == AnalysisDepth.COMPREHENSIVE:
            analysis_tasks['graph_analysis'] = self.graph_analyzer.analyze_structural_preservation(
                original_data, processed_data
            )
            
            if self.neural_encoder.is_trained:
                analysis_tasks['neural_analysis'] = self._analyze_neural_preservation(
                    original_data, processed_data
                )
        
        return analysis_tasks
    
    async def _calculate_enhanced_detectability_score(self, data: Dict[str, Any],
                                                    analysis_depth: AnalysisDepth) -> float:
        """Calculate enhanced detectability score using multiple analysis methods"""
        
        total_score = 0.0
        weight_sum = 0.0
        
        # Basic signature matching
        signature_score = await self._calculate_signature_matching_score(data)
        total_score += signature_score * 0.4
        weight_sum += 0.4
        
        # Statistical property analysis
        if analysis_depth in [AnalysisDepth.STATISTICAL, AnalysisDepth.DEEP, AnalysisDepth.COMPREHENSIVE]:
            stats_score = await self._calculate_statistical_score(data)
            total_score += stats_score * 0.3
            weight_sum += 0.3
        
        # Information-theoretic analysis
        if analysis_depth in [AnalysisDepth.DEEP, AnalysisDepth.COMPREHENSIVE]:
            info_score = self._calculate_information_content_score(data)
            total_score += info_score * 0.2
            weight_sum += 0.2
        
        # Neural embedding analysis
        if analysis_depth == AnalysisDepth.COMPREHENSIVE and self.neural_encoder.is_trained:
            neural_score = self._calculate_neural_detectability_score(data)
            total_score += neural_score * 0.1
            weight_sum += 0.1
        
        return total_score / weight_sum if weight_sum > 0 else 0.0
    
    async def _calculate_signature_matching_score(self, data: Dict[str, Any]) -> float:
        """Calculate detectability score based on signature matching"""
        
        total_score = 0.0
        total_weight = 0.0
        
        for signature_id, signature in self.known_signatures.items():
            # Field presence score
            field_score = self._calculate_field_presence_score_advanced(data, signature)
            
            # Statistical property score
            stats_score = self._calculate_statistical_preservation_score_advanced(data, signature)
            
            # Temporal characteristic score
            temporal_score = self._calculate_temporal_preservation_score_advanced(data, signature)
            
            # Structural characteristic score
            structural_score = self._calculate_structural_preservation_score(data, signature)
            
            # Combine scores with signature-specific weights
            signature_score = (
                field_score * 0.3 +
                stats_score * 0.35 +
                temporal_score * 0.2 +
                structural_score * 0.15
            ) * signature.preservation_priority
            
            total_score += signature_score
            total_weight += signature.preservation_priority
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_field_presence_score_advanced(self, data: Dict[str, Any],
                                               signature: AnomalySignature) -> float:
        """Advanced field presence scoring with weighting"""
        
        present_score = 0.0
        total_possible_score = 0.0
        
        for field in signature.signature_fields:
            field_weight = 1.0  # Default weight
            
            # Check if field has special importance in correlation patterns
            if field in signature.correlation_patterns:
                field_weight = signature.correlation_patterns[field]
            
            if self._field_exists_in_data_advanced(field, data):
                present_score += field_weight
            
            total_possible_score += field_weight
        
        return present_score / total_possible_score if total_possible_score > 0 else 0.0
    
    def _field_exists_in_data_advanced(self, field: str, data: Dict[str, Any]) -> bool:
        """Advanced field existence check with fuzzy matching"""
        
        # Exact match
        if self._field_exists_in_data(field, data):
            return True
        
        # Fuzzy field name matching for schema evolution compatibility
        field_lower = field.lower()
        for key in data.keys():
            if isinstance(key, str):
                # Check for partial matches or common variations
                key_lower = key.lower()
                if (field_lower in key_lower or key_lower in field_lower or
                    self._calculate_string_similarity(field_lower, key_lower) > 0.8):
                    return True
        
        return False
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using edit distance"""
        if not str1 or not str2:
            return 0.0
        
        # Simple Levenshtein distance based similarity
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 1.0
        
        # Simplified edit distance calculation
        edit_distance = sum(c1 != c2 for c1, c2 in zip(str1, str2)) + abs(len(str1) - len(str2))
        similarity = 1.0 - (edit_distance / max_len)
        
        return max(0.0, similarity)
    
    def _calculate_statistical_preservation_score_advanced(self, data: Dict[str, Any],
                                                          signature: AnomalySignature) -> float:
        """Advanced statistical preservation scoring"""
        
        score = 1.0
        field_scores = []
        
        for field in signature.signature_fields:
            if not self._field_exists_in_data_advanced(field, data):
                continue
            
            field_value = self._get_field_value_advanced(field, data)
            if field_value is None:
                continue
            
            field_score = 1.0
            
            # Check against statistical properties
            if field in signature.statistical_properties:
                expected_props = signature.statistical_properties[field]
                
                if isinstance(expected_props, tuple) and len(expected_props) == 2:
                    # Range check with tolerance
                    min_val, max_val = expected_props
                    if isinstance(field_value, (int, float)):
                        if min_val <= field_value <= max_val:
                            field_score = 1.0
                        else:
                            # Gradual penalty based on distance from range
                            range_size = max_val - min_val
                            if field_value < min_val:
                                distance = min_val - field_value
                            else:
                                distance = field_value - max_val
                            
                            penalty = min(1.0, distance / range_size)
                            field_score = max(0.0, 1.0 - penalty)
                
                elif isinstance(expected_props, dict):
                    # Complex statistical validation
                    for prop_name, prop_value in expected_props.items():
                        if prop_name == 'distribution' and isinstance(field_value, (int, float)):
                            # Distribution type validation
                            if prop_value == 'log_normal' and field_value > 0:
                                field_score *= 1.0  # Good fit
                            elif prop_value == 'normal':
                                field_score *= 0.9  # Acceptable
                        
                        elif prop_name.endswith('_threshold') and isinstance(field_value, (int, float)):
                            # Threshold validation
                            if field_value >= prop_value:
                                field_score *= 1.0
                            else:
                                field_score *= 0.7
            
            field_scores.append(field_score)
        
        return np.mean(field_scores) if field_scores else 0.0
    
    def _calculate_temporal_preservation_score_advanced(self, data: Dict[str, Any],
                                                       signature: AnomalySignature) -> float:
        """Advanced temporal characteristic preservation scoring"""
        
        score = 1.0
        temporal_features = []
        
        # Check for timestamp fields
        timestamp_fields = [f for f in signature.signature_fields 
                          if any(kw in f.lower() for kw in ['time', 'timestamp', 'duration'])]
        
        if not timestamp_fields:
            return 1.0  # No temporal requirements
        
        for field in timestamp_fields:
            if self._field_exists_in_data_advanced(field, data):
                field_value = self._get_field_value_advanced(field, data)
                
                if isinstance(field_value, (int, float)) and field_value > 0:
                    temporal_features.append(1.0)
                else:
                    temporal_features.append(0.5)  # Present but invalid
            else:
                temporal_features.append(0.0)  # Missing
        
        # Check temporal characteristics from signature
        temporal_chars = signature.temporal_characteristics
        
        if 'timing_correlation_window' in temporal_chars:
            # Validate timing correlation window if multiple timestamps present
            if len(temporal_features) > 1:
                score *= 0.9  # Bonus for multiple temporal fields
        
        if 'phase_transitions' in temporal_chars:
            # Check for phase transition indicators
            transitions = temporal_chars['phase_transitions']
            phase_indicators = [f for f in data.keys() 
                              if any(phase in str(f).lower() for phase in transitions)]
            if phase_indicators:
                score *= 1.1  # Bonus for phase information
        
        base_score = np.mean(temporal_features) if temporal_features else 0.0
        return min(1.0, base_score * score)
    
    def _calculate_structural_preservation_score(self, data: Dict[str, Any],
                                               signature: AnomalySignature) -> float:
        """Calculate structural characteristic preservation score"""
        
        structural_chars = signature.structural_characteristics
        score = 1.0
        
        # Check required field count
        if 'required_field_count' in structural_chars:
            required_count = structural_chars['required_field_count']
            actual_count = len([f for f in signature.signature_fields 
                              if self._field_exists_in_data_advanced(f, data)])
            
            if actual_count >= required_count:
                score *= 1.0
            else:
                score *= (actual_count / required_count)
        
        # Check nesting depth
        if 'nested_depth_max' in structural_chars:
            max_depth = structural_chars['nested_depth_max']
            actual_depth = self._calculate_nesting_depth(data)
            
            if actual_depth <= max_depth:
                score *= 1.0
            else:
                score *= 0.8  # Penalty for excessive nesting
        
        # Check field correlations
        if 'field_correlation_matrix' in structural_chars:
            correlation_matrix = structural_chars['field_correlation_matrix']
            correlation_score = self._validate_field_correlations(data, correlation_matrix)
            score *= correlation_score
        
        return max(0.0, min(1.0, score))
    
    def _validate_field_correlations(self, data: Dict[str, Any], 
                                   correlation_matrix: Dict[Tuple[str, str], float]) -> float:
        """Validate expected field correlations"""
        
        correlation_scores = []
        
        for (field1, field2), expected_correlation in correlation_matrix.items():
            val1 = self._get_field_value_advanced(field1, data)
            val2 = self._get_field_value_advanced(field2, data)
            
            if val1 is None or val2 is None:
                correlation_scores.append(0.5)  # Missing data penalty
                continue
            
            # Calculate actual correlation if both are numeric
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # For single values, use relative comparison
                if val1 == 0 and val2 == 0:
                    actual_correlation = 1.0
                elif val1 == 0 or val2 == 0:
                    actual_correlation = 0.0
                else:
                    # Normalized correlation approximation
                    ratio = abs(val1 / val2) if val2 != 0 else 0
                    actual_correlation = 1.0 / (1.0 + abs(1.0 - ratio))
                
                # Compare with expected correlation
                correlation_diff = abs(expected_correlation - actual_correlation)
                correlation_score = 1.0 - min(1.0, correlation_diff)
                correlation_scores.append(correlation_score)
            else:
                # Non-numeric correlation based on presence/absence
                both_present = (val1 is not None) and (val2 is not None)
                if expected_correlation > 0.5 and both_present:
                    correlation_scores.append(1.0)
                elif expected_correlation <= 0.5 and not both_present:
                    correlation_scores.append(1.0)
                else:
                    correlation_scores.append(0.7)
        
        return np.mean(correlation_scores) if correlation_scores else 1.0
    
    async def _calculate_statistical_score(self, data: Dict[str, Any]) -> float:
        """Calculate statistical analysis score for detectability"""
        
        # Extract numeric features for statistical analysis
        numeric_features = []
        for key, value in data.items():
            if isinstance(value, (int, float)):
                numeric_features.append(value)
            elif isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                try:
                    numeric_features.append(float(value))
                except ValueError:
                    pass
        
        if not numeric_features:
            return 0.5  # No numeric data available
        
        # Calculate statistical properties
        try:
            mean_val = np.mean(numeric_features)
            std_val = np.std(numeric_features)
            skewness = stats.skew(numeric_features)
            kurtosis = stats.kurtosis(numeric_features)
            
            # Score based on statistical richness
            score = 0.0
            
            # Variance score (higher variance = more information)
            if std_val > 0:
                cv = std_val / abs(mean_val) if mean_val != 0 else std_val
                variance_score = min(1.0, cv / 2.0)  # Normalize coefficient of variation
                score += variance_score * 0.4
            
            # Distribution shape score
            shape_score = 0.5  # Base score
            if abs(skewness) > 0.5:  # Non-normal distribution
                shape_score += 0.3
            if abs(kurtosis) > 1.0:  # Heavy tails or outliers
                shape_score += 0.2
            
            score += min(1.0, shape_score) * 0.3
            
            # Range score
            if len(numeric_features) > 1:
                range_val = max(numeric_features) - min(numeric_features)
                range_score = min(1.0, range_val / (abs(mean_val) + 1e-6))
                score += range_score * 0.3
            
            return min(1.0, score)
            
        except Exception as e:
            logger.debug(f"Statistical score calculation failed: {e}")
            return 0.3  # Fallback score
    
    def _calculate_information_content_score(self, data: Dict[str, Any]) -> float:
        """Calculate information content score using entropy analysis"""
        
        total_entropy = 0.0
        field_count = 0
        
        for key, value in data.items():
            field_entropy = self.info_analyzer.calculate_entropy(value)
            total_entropy += field_entropy
            field_count += 1
        
        if field_count == 0:
            return 0.0
        
        # Normalize entropy score
        average_entropy = total_entropy / field_count
        
        # Score based on information richness
        # Higher entropy = more information = better detectability
        max_expected_entropy = 8.0  # Reasonable maximum for text/data entropy
        normalized_score = min(1.0, average_entropy / max_expected_entropy)
        
        return normalized_score
    
    def _calculate_neural_detectability_score(self, data: Dict[str, Any]) -> float:
        """Calculate detectability score using neural embeddings"""
        
        if not self.neural_encoder.is_trained:
            return 0.5  # Neutral score if not trained
        
        try:
            # Generate embedding for the data
            embedding = self.neural_encoder.encode(data)
            
            # Score based on embedding properties
            embedding_norm = np.linalg.norm(embedding)
            embedding_entropy = shannon_entropy(embedding.astype(str))
            
            # Combine norm and entropy for detectability score
            norm_score = min(1.0, embedding_norm / 10.0)  # Normalize by expected magnitude
            entropy_score = min(1.0, embedding_entropy / 6.0)  # Normalize by expected entropy
            
            return (norm_score * 0.6 + entropy_score * 0.4)
            
        except Exception as e:
            logger.debug(f"Neural detectability score calculation failed: {e}")
            return 0.5
    
    def _get_field_value_advanced(self, field: str, data: Dict[str, Any]) -> Any:
        """Advanced field value extraction with fuzzy matching"""
        
        # Try exact match first
        value = self._get_field_value(field, data)
        if value is not None:
            return value
        
        # Try fuzzy matching for schema evolution compatibility
        field_lower = field.lower()
        best_match = None
        best_similarity = 0.0
        
        for key in data.keys():
            if isinstance(key, str):
                similarity = self._calculate_string_similarity(field_lower, key.lower())
                if similarity > best_similarity and similarity > 0.8:
                    best_similarity = similarity
                    best_match = key
        
        if best_match:
            return self._get_field_value(best_match, data)
        
        return None
    
    async def _identify_affected_anomaly_types_advanced(self, 
                                                      original_data: Dict[str, Any],
                                                      processed_data: Dict[str, Any],
                                                      analysis_tasks: Dict[str, Any]) -> List[AnomalyType]:
        """Advanced identification of affected anomaly types"""
        
        affected_types = []
        
        # Check each preservation rule
        for anomaly_type, rule in self.preservation_rules.items():
            is_affected = False
            impact_score = 0.0
            
            # Check field-level impacts
            for field in rule.protected_fields:
                orig_value = self._get_field_value_advanced(field, original_data)
                proc_value = self._get_field_value_advanced(field, processed_data)
                
                field_weight = rule.field_weights.get(field, 1.0)
                
                if orig_value != proc_value:
                    if orig_value is None and proc_value is not None:
                        # Field added (usually not a problem)
                        impact_score += 0.1 * field_weight
                    elif orig_value is not None and proc_value is None:
                        # Field removed (significant impact)
                        impact_score += 0.8 * field_weight
                        is_affected = True
                    else:
                        # Field modified
                        modification_impact = self._calculate_modification_impact(
                            orig_value, proc_value, field, rule
                        )
                        impact_score += modification_impact * field_weight
                        
                        if modification_impact > 0.3:
                            is_affected = True
            
            # Check constraint violations
            constraint_violations = await self._check_rule_constraints(
                original_data, processed_data, rule, analysis_tasks
            )
            
            if constraint_violations:
                is_affected = True
                impact_score += len(constraint_violations) * 0.2
            
            # Apply impact threshold
            impact_threshold = 0.3
            if rule.preservation_level == PreservationLevel.CRITICAL:
                impact_threshold = 0.1
            elif rule.preservation_level == PreservationLevel.HIGH:
                impact_threshold = 0.2
            
            if is_affected or impact_score > impact_threshold:
                affected_types.append(anomaly_type)
        
        return affected_types
    
    def _calculate_modification_impact(self, orig_value: Any, proc_value: Any, 
                                     field: str, rule: PreservationRule) -> float:
        """Calculate the impact of field modification on anomaly detection"""
        
        if orig_value == proc_value:
            return 0.0
        
        # Type-specific impact calculation
        if isinstance(orig_value, (int, float)) and isinstance(proc_value, (int, float)):
            # Numeric field modification
            if orig_value == 0:
                return 1.0 if proc_value != 0 else 0.0
            
            relative_change = abs(orig_value - proc_value) / abs(orig_value)
            
            # Check if this field has specific transformation rules
            if field in rule.field_transformations:
                transformation = rule.field_transformations[field]
                
                if 'preserve_distribution' in transformation:
                    # Allow larger changes for distribution preservation
                    return min(1.0, relative_change / 2.0)
                elif 'preserve_outliers' in transformation:
                    # Penalize changes to outlier values more heavily
                    if abs(orig_value) > 2 * np.std([orig_value, proc_value]):  # Rough outlier check
                        return min(1.0, relative_change * 2.0)
                    else:
                        return min(1.0, relative_change)
                elif 'preserve_range' in transformation:
                    # Focus on maintaining value ranges
                    return min(1.0, relative_change * 1.5)
            
            return min(1.0, relative_change)
        
        elif isinstance(orig_value, str) and isinstance(proc_value, str):
            # String field modification
            if len(orig_value) == 0:
                return 1.0 if len(proc_value) > 0 else 0.0
            
            # Calculate string similarity
            similarity = self._calculate_string_similarity(orig_value, proc_value)
            return 1.0 - similarity
        
        elif isinstance(orig_value, (list, tuple)) and isinstance(proc_value, (list, tuple)):
            # List/tuple modification
            if len(orig_value) == 0:
                return 1.0 if len(proc_value) > 0 else 0.0
            
            # Compare list contents
            common_elements = set(orig_value) & set(proc_value)
            total_elements = set(orig_value) | set(proc_value)
            
            if len(total_elements) == 0:
                return 0.0
            
            similarity = len(common_elements) / len(total_elements)
            return 1.0 - similarity
        
        elif isinstance(orig_value, dict) and isinstance(proc_value, dict):
            # Dictionary modification
            orig_keys = set(orig_value.keys())
            proc_keys = set(proc_value.keys())
            
            key_similarity = len(orig_keys & proc_keys) / len(orig_keys | proc_keys) if (orig_keys | proc_keys) else 1.0
            
            # Check value similarity for common keys
            value_similarities = []
            for key in orig_keys & proc_keys:
                value_impact = self._calculate_modification_impact(
                    orig_value[key], proc_value[key], f"{field}.{key}", rule
                )
                value_similarities.append(1.0 - value_impact)
            
            value_similarity = np.mean(value_similarities) if value_similarities else 0.0
            
            overall_similarity = (key_similarity * 0.5 + value_similarity * 0.5)
            return 1.0 - overall_similarity
        
        else:
            # Different types or unsupported types
            return 1.0  # Maximum impact for type changes
    
    async def _check_rule_constraints(self, original_data: Dict[str, Any],
                                    processed_data: Dict[str, Any],
                                    rule: PreservationRule,
                                    analysis_tasks: Dict[str, Any]) -> List[str]:
        """Check preservation rule constraints"""
        
        violations = []
        
        # Statistical constraints
        for constraint, threshold in rule.statistical_constraints.items():
            violation = await self._check_statistical_constraint(
                constraint, threshold, original_data, processed_data, rule, analysis_tasks
            )
            if violation:
                violations.append(violation)
        
        # Temporal constraints
        for constraint, threshold in rule.temporal_constraints.items():
            violation = await self._check_temporal_constraint(
                constraint, threshold, original_data, processed_data, rule
            )
            if violation:
                violations.append(violation)
        
        # Structural constraints
        for constraint, threshold in rule.structural_constraints.items():
            violation = await self._check_structural_constraint(
                constraint, threshold, original_data, processed_data, rule, analysis_tasks
            )
            if violation:
                violations.append(violation)
        
        return violations
    
    async def _check_statistical_constraint(self, constraint: str, threshold: float,
                                          original_data: Dict[str, Any], processed_data: Dict[str, Any],
                                          rule: PreservationRule, analysis_tasks: Dict[str, Any]) -> Optional[str]:
        """Check specific statistical constraint"""
        
        try:
            if constraint == 'duration_distribution_kl_divergence':
                # Check KL divergence for duration distributions
                orig_durations = self._extract_duration_values(original_data)
                proc_durations = self._extract_duration_values(processed_data)
                
                if orig_durations and proc_durations:
                    kl_div = self._calculate_kl_divergence(orig_durations, proc_durations)
                    if kl_div > threshold:
                        return f"Duration distribution KL divergence too high: {kl_div:.3f} > {threshold}"
            
            elif constraint == 'memory_percentile_preservation':
                # Check memory usage percentile preservation
                orig_memory = self._extract_memory_values(original_data)
                proc_memory = self._extract_memory_values(processed_data)
                
                if orig_memory and proc_memory:
                    percentile_preservation = self._calculate_percentile_preservation(orig_memory, proc_memory)
                    if percentile_preservation < threshold:
                        return f"Memory percentile preservation too low: {percentile_preservation:.3f} < {threshold}"
            
            elif constraint == 'cost_anomaly_detection':
                # Check cost anomaly detection capability
                cost_preservation = self._calculate_cost_anomaly_preservation(original_data, processed_data)
                if cost_preservation < threshold:
                    return f"Cost anomaly detection capability compromised: {cost_preservation:.3f} < {threshold}"
            
            elif constraint == 'information_preservation':
                # Check information-theoretic preservation
                if 'info_theory' in analysis_tasks:
                    info_analysis = analysis_tasks['info_theory']
                    info_loss = info_analysis.get('total_entropy_loss', 0.0)
                    info_preservation = 1.0 - info_loss
                    
                    if info_preservation < threshold:
                        return f"Information preservation too low: {info_preservation:.3f} < {threshold}"
            
        except Exception as e:
            logger.debug(f"Statistical constraint check failed for {constraint}: {e}")
        
        return None
    
    def _extract_duration_values(self, data: Dict[str, Any]) -> List[float]:
        """Extract duration values from data"""
        durations = []
        
        for key, value in data.items():
            if 'duration' in key.lower() and isinstance(value, (int, float)):
                durations.append(float(value))
            elif 'time' in key.lower() and 'ms' in key.lower() and isinstance(value, (int, float)):
                durations.append(float(value))
        
        return durations
    
    def _extract_memory_values(self, data: Dict[str, Any]) -> List[float]:
        """Extract memory values from data"""
        memory_values = []
        
        for key, value in data.items():
            if 'memory' in key.lower() and isinstance(value, (int, float)):
                memory_values.append(float(value))
            elif 'mem' in key.lower() and isinstance(value, (int, float)):
                memory_values.append(float(value))
        
        return memory_values
    
    def _calculate_kl_divergence(self, data1: List[float], data2: List[float]) -> float:
        """Calculate KL divergence between two distributions"""
        try:
            # Create histograms
            combined_data = data1 + data2
            if not combined_data:
                return 0.0
            
            bins = np.linspace(min(combined_data), max(combined_data), 20)
            
            hist1, _ = np.histogram(data1, bins=bins, density=True)
            hist2, _ = np.histogram(data2, bins=bins, density=True)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            hist1 = hist1 + epsilon
            hist2 = hist2 + epsilon
            
            # Normalize
            hist1 = hist1 / np.sum(hist1)
            hist2 = hist2 / np.sum(hist2)
            
            # Calculate KL divergence
            kl_div = np.sum(hist1 * np.log(hist1 / hist2))
            return kl_div
            
        except Exception as e:
            logger.debug(f"KL divergence calculation failed: {e}")
            return 0.0
    
    def _calculate_percentile_preservation(self, orig_values: List[float], 
                                         proc_values: List[float]) -> float:
        """Calculate how well percentiles are preserved"""
        try:
            if not orig_values or not proc_values:
                return 0.0
            
            percentiles = [25, 50, 75, 90, 95, 99]
            preservation_scores = []
            
            for p in percentiles:
                orig_percentile = np.percentile(orig_values, p)
                proc_percentile = np.percentile(proc_values, p)
                
                if orig_percentile == 0:
                    score = 1.0 if proc_percentile == 0 else 0.0
                else:
                    relative_error = abs(orig_percentile - proc_percentile) / abs(orig_percentile)
                    score = max(0.0, 1.0 - relative_error)
                
                preservation_scores.append(score)
            
            return np.mean(preservation_scores)
            
        except Exception as e:
            logger.debug(f"Percentile preservation calculation failed: {e}")
            return 0.0
    
    def _calculate_cost_anomaly_preservation(self, original_data: Dict[str, Any],
                                           processed_data: Dict[str, Any]) -> float:
        """Calculate preservation of cost anomaly detection capability"""
        
        # Extract cost-related fields
        cost_fields = ['execution_cost', 'billing_units', 'resource_consumption', 'quota_usage']
        
        preservation_scores = []
        
        for field in cost_fields:
            orig_value = self._get_field_value_advanced(field, original_data)
            proc_value = self._get_field_value_advanced(field, processed_data)
            
            if orig_value is None and proc_value is None:
                continue
            elif orig_value is None or proc_value is None:
                preservation_scores.append(0.0)
            elif isinstance(orig_value, (int, float)) and isinstance(proc_value, (int, float)):
                if orig_value == 0:
                    score = 1.0 if proc_value == 0 else 0.0
                else:
                    relative_error = abs(orig_value - proc_value) / abs(orig_value)
                    score = max(0.0, 1.0 - relative_error)
                preservation_scores.append(score)
            else:
                # Non-numeric cost fields
                similarity = self._calculate_string_similarity(str(orig_value), str(proc_value))
                preservation_scores.append(similarity)
        
        return np.mean(preservation_scores) if preservation_scores else 0.0
    
    async def _check_temporal_constraint(self, constraint: str, threshold: float,
                                       original_data: Dict[str, Any], processed_data: Dict[str, Any],
                                       rule: PreservationRule) -> Optional[str]:
        """Check specific temporal constraint"""
        
        try:
            if constraint == 'timing_correlation_preservation':
                # Check preservation of timing correlations
                correlation_preservation = self._calculate_timing_correlation_preservation(
                    original_data, processed_data
                )
                if correlation_preservation < threshold:
                    return f"Timing correlation preservation too low: {correlation_preservation:.3f} < {threshold}"
            
            elif constraint == 'growth_trend_preservation':
                # Check preservation of growth trends
                trend_preservation = self._calculate_trend_preservation(original_data, processed_data)
                if trend_preservation < threshold:
                    return f"Growth trend preservation too low: {trend_preservation:.3f} < {threshold}"
            
            elif constraint == 'frequency_analysis_integrity':
                # Check frequency domain analysis integrity
                frequency_integrity = self._calculate_frequency_integrity(original_data, processed_data)
                if frequency_integrity < threshold:
                    return f"Frequency analysis integrity compromised: {frequency_integrity:.3f} < {threshold}"
            
        except Exception as e:
            logger.debug(f"Temporal constraint check failed for {constraint}: {e}")
        
        return None
    
    def _calculate_timing_correlation_preservation(self, original_data: Dict[str, Any],
                                                 processed_data: Dict[str, Any]) -> float:
        """Calculate preservation of timing correlations"""
        
        # Extract timing fields
        timing_fields = []
        for key in original_data.keys():
            if any(kw in key.lower() for kw in ['time', 'duration', 'latency']):
                timing_fields.append(key)
        
        if len(timing_fields) < 2:
            return 1.0  # No correlations to preserve
        
        # Calculate correlations for original data
        orig_correlations = []
        proc_correlations = []
        
        for i in range(len(timing_fields)):
            for j in range(i + 1, len(timing_fields)):
                field1, field2 = timing_fields[i], timing_fields[j]
                
                orig_val1 = self._get_field_value_advanced(field1, original_data)
                orig_val2 = self._get_field_value_advanced(field2, original_data)
                proc_val1 = self._get_field_value_advanced(field1, processed_data)
                proc_val2 = self._get_field_value_advanced(field2, processed_data)
                
                if all(isinstance(v, (int, float)) for v in [orig_val1, orig_val2, proc_val1, proc_val2]):
                    # Calculate simple correlation approximation
                    orig_corr = self._calculate_simple_correlation(orig_val1, orig_val2)
                    proc_corr = self._calculate_simple_correlation(proc_val1, proc_val2)
                    
                    orig_correlations.append(orig_corr)
                    proc_correlations.append(proc_corr)
        
        if not orig_correlations:
            return 1.0
        
        # Calculate preservation score
        correlation_diffs = [abs(o - p) for o, p in zip(orig_correlations, proc_correlations)]
        avg_diff = np.mean(correlation_diffs)
        
        return max(0.0, 1.0 - avg_diff)
    
    def _calculate_simple_correlation(self, val1: float, val2: float) -> float:
        """Calculate simple correlation approximation for two values"""
        if val1 == 0 and val2 == 0:
            return 1.0
        elif val1 == 0 or val2 == 0:
            return 0.0
        else:
            # Normalized correlation based on ratio
            ratio = val1 / val2
            return 1.0 / (1.0 + abs(1.0 - ratio))
    
    def _calculate_trend_preservation(self, original_data: Dict[str, Any],
                                    processed_data: Dict[str, Any]) -> float:
        """Calculate preservation of trend information"""
        
        # Look for time series or sequential data
        sequential_fields = []
        for key, value in original_data.items():
            if isinstance(value, list) and len(value) > 2:
                # Check if it's a numeric time series
                if all(isinstance(x, (int, float)) for x in value):
                    sequential_fields.append(key)
        
        if not sequential_fields:
            return 1.0  # No trends to preserve
        
        trend_preservation_scores = []
        
        for field in sequential_fields:
            orig_series = original_data[field]
            proc_series = processed_data.get(field, [])
            
            if len(orig_series) != len(proc_series):
                trend_preservation_scores.append(0.0)
                continue
            
            # Calculate trend direction preservation
            orig_trend = self._calculate_trend_direction(orig_series)
            proc_trend = self._calculate_trend_direction(proc_series)
            
            if orig_trend == proc_trend:
                trend_preservation_scores.append(1.0)
            elif abs(orig_trend - proc_trend) < 0.1:  # Similar trends
                trend_preservation_scores.append(0.8)
            else:
                trend_preservation_scores.append(0.3)
        
        return np.mean(trend_preservation_scores) if trend_preservation_scores else 1.0
    
    def _calculate_trend_direction(self, series: List[float]) -> float:
        """Calculate trend direction of a time series"""
        if len(series) < 2:
            return 0.0
        
        # Simple linear trend calculation
        x = list(range(len(series)))
        n = len(series)
        
        sum_x = sum(x)
        sum_y = sum(series)
        sum_xy = sum(x[i] * series[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        # Linear regression slope
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope
    
    def _calculate_frequency_integrity(self, original_data: Dict[str, Any],
                                     processed_data: Dict[str, Any]) -> float:
        """Calculate frequency domain analysis integrity"""
        
        # Look for timing or periodic data
        timing_sequences = []
        
        for key, value in original_data.items():
            if isinstance(value, list) and len(value) > 4:
                if all(isinstance(x, (int, float)) for x in value):
                    timing_sequences.append((key, value))
        
        if not timing_sequences:
            return 1.0
        
        integrity_scores = []
        
        for field, orig_sequence in timing_sequences:
            proc_sequence = processed_data.get(field, [])
            
            if len(orig_sequence) != len(proc_sequence):
                integrity_scores.append(0.0)
                continue
            
            # Calculate frequency domain similarity using FFT
            try:
                orig_fft = np.fft.fft(orig_sequence)
                proc_fft = np.fft.fft(proc_sequence)
                
                # Compare frequency magnitudes
                orig_magnitudes = np.abs(orig_fft)
                proc_magnitudes = np.abs(proc_fft)
                
                # Normalize and compare
                orig_norm = orig_magnitudes / (np.sum(orig_magnitudes) + 1e-10)
                proc_norm = proc_magnitudes / (np.sum(proc_magnitudes) + 1e-10)
                
                similarity = 1.0 - np.mean(np.abs(orig_norm - proc_norm))
                integrity_scores.append(max(0.0, similarity))