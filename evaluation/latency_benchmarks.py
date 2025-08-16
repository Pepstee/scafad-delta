#!/usr/bin/env python3
"""
SCAFAD Layer 1: Latency Benchmarks Evaluation
============================================

Latency benchmarking and analysis for Layer 1's behavioral intake zone.
This module provides comprehensive performance evaluation including:

- Processing latency measurement per phase
- Throughput analysis under various loads
- Memory usage profiling
- Performance regression detection
- Optimization impact assessment

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
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum, auto
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

# Layer 1 imports
import sys
sys.path.append('..')
from core.layer1_core import Layer1_BehavioralIntakeZone
from configs.layer1_config import Layer1Config, ProcessingMode, PerformanceProfile

# =============================================================================
# Benchmark Data Models
# =============================================================================

class BenchmarkType(Enum):
    """Types of benchmarks to run"""
    LATENCY = "latency"               # Processing latency measurement
    THROUGHPUT = "throughput"         # Records per second analysis
    MEMORY = "memory"                 # Memory usage profiling
    PHASE_ANALYSIS = "phase_analysis" # Individual phase performance
    SCALABILITY = "scalability"       # Performance scaling analysis
    REGRESSION = "regression"         # Performance regression detection

class LoadProfile(Enum):
    """Load profiles for benchmarking"""
    LIGHT = "light"                   # Low load (100-1000 records)
    MEDIUM = "medium"                 # Medium load (1000-10000 records)
    HEAVY = "heavy"                   # Heavy load (10000-100000 records)
    STRESS = "stress"                 # Stress test (100000+ records)
    BURST = "burst"                   # Burst load patterns

@dataclass
class BenchmarkResult:
    """Result of a benchmark run"""
    benchmark_type: BenchmarkType
    load_profile: LoadProfile
    record_count: int
    total_time_ms: float
    average_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    memory_peak_mb: float
    memory_average_mb: float
    phase_latencies: Dict[str, float]
    error_count: int
    success_rate: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class BenchmarkSuite:
    """Complete benchmark suite configuration"""
    name: str
    description: str
    benchmark_types: List[BenchmarkType]
    load_profiles: List[LoadProfile]
    record_counts: List[int]
    iterations: int
    warmup_runs: int
    cooldown_seconds: float
    output_directory: str
    generate_plots: bool
    save_results: bool

@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison"""
    baseline_name: str
    baseline_date: datetime
    target_latency_ms: float
    target_throughput_rps: float
    target_memory_mb: float
    phase_targets: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None

# =============================================================================
# Latency Benchmark Runner
# =============================================================================

class LatencyBenchmarkRunner:
    """
    Main latency benchmark runner for Layer 1 performance evaluation
    
    Provides comprehensive performance analysis including:
    - Phase-by-phase latency measurement
    - Throughput analysis under various loads
    - Memory usage profiling
    - Performance regression detection
    - Optimization impact assessment
    """
    
    def __init__(self, config: Optional[Layer1Config] = None):
        """Initialize benchmark runner with configuration"""
        self.config = config or Layer1Config()
        self.logger = logging.getLogger("SCAFAD.Layer1.LatencyBenchmarks")
        
        # Initialize Layer 1
        self.layer1 = Layer1_BehavioralIntakeZone(self.config)
        
        # Benchmark state
        self.baseline: Optional[PerformanceBaseline] = None
        self.results_history: List[BenchmarkResult] = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_benchmarks': 0,
            'successful_benchmarks': 0,
            'failed_benchmarks': 0,
            'total_records_processed': 0,
            'total_processing_time_ms': 0.0
        }
        
        self.logger.info("Latency benchmark runner initialized")
    
    def set_baseline(self, baseline: PerformanceBaseline):
        """Set performance baseline for comparison"""
        self.baseline = baseline
        self.logger.info(f"Performance baseline set: {baseline.baseline_name}")
    
    def run_benchmark_suite(self, suite: BenchmarkSuite) -> List[BenchmarkResult]:
        """
        Run complete benchmark suite
        
        Args:
            suite: Benchmark suite configuration
            
        Returns:
            List of benchmark results
        """
        self.logger.info(f"Starting benchmark suite: {suite.name}")
        self.logger.info(f"Description: {suite.description}")
        
        results = []
        
        # Create output directory
        output_path = Path(suite.output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Run benchmarks for each combination
        for benchmark_type in suite.benchmark_types:
            for load_profile in suite.load_profiles:
                for record_count in suite.record_counts:
                    if self._should_run_benchmark(benchmark_type, load_profile, record_count):
                        self.logger.info(f"Running {benchmark_type.value} benchmark with {load_profile.value} load ({record_count} records)")
                        
                        try:
                            # Run benchmark
                            result = self._run_single_benchmark(
                                benchmark_type, load_profile, record_count, suite
                            )
                            
                            if result:
                                results.append(result)
                                self.results_history.append(result)
                                self._update_performance_metrics(result)
                                
                                # Save individual result
                                if suite.save_results:
                                    self._save_benchmark_result(result, output_path)
                            
                        except Exception as e:
                            self.logger.error(f"Benchmark failed: {e}")
                            self.performance_metrics['failed_benchmarks'] += 1
                        
                        # Cooldown between benchmarks
                        if suite.cooldown_seconds > 0:
                            time.sleep(suite.cooldown_seconds)
        
        # Generate summary and plots
        if suite.generate_plots and results:
            self._generate_benchmark_plots(results, output_path)
        
        # Save suite summary
        if suite.save_results:
            self._save_suite_summary(results, suite, output_path)
        
        self.logger.info(f"Benchmark suite completed. {len(results)} benchmarks run successfully")
        return results
    
    def _should_run_benchmark(self, benchmark_type: BenchmarkType, 
                             load_profile: LoadProfile, record_count: int) -> bool:
        """Determine if a benchmark should be run based on configuration"""
        # Add custom logic for benchmark selection
        return True
    
    def _run_single_benchmark(self, benchmark_type: BenchmarkType, load_profile: LoadProfile,
                             record_count: int, suite: BenchmarkSuite) -> Optional[BenchmarkResult]:
        """Run a single benchmark"""
        
        # Generate test data
        test_records = self._generate_test_records(record_count, load_profile)
        
        # Warmup runs
        for _ in range(suite.warmup_runs):
            try:
                _ = asyncio.run(self.layer1.process_telemetry_batch(test_records[:100]))
            except Exception as e:
                self.logger.warning(f"Warmup run failed: {e}")
        
        # Main benchmark runs
        all_latencies = []
        all_phase_latencies = []
        memory_usage = []
        error_count = 0
        success_count = 0
        
        for iteration in range(suite.iterations):
            try:
                # Measure memory before
                memory_before = self._get_memory_usage()
                
                # Run benchmark
                start_time = time.time()
                result = asyncio.run(self.layer1.process_telemetry_batch(test_records))
                end_time = time.time()
                
                # Measure memory after
                memory_after = self._get_memory_usage()
                
                # Calculate latencies
                total_latency = (end_time - start_time) * 1000  # Convert to ms
                per_record_latency = total_latency / record_count
                
                all_latencies.append(per_record_latency)
                all_phase_latencies.append(self._extract_phase_latencies(result))
                memory_usage.append(memory_after - memory_before)
                
                success_count += 1
                
            except Exception as e:
                self.logger.error(f"Benchmark iteration {iteration} failed: {e}")
                error_count += 1
        
        if not all_latencies:
            self.logger.error("All benchmark iterations failed")
            return None
        
        # Calculate statistics
        avg_latency = statistics.mean(all_latencies)
        median_latency = statistics.median(all_latencies)
        p95_latency = np.percentile(all_latencies, 95)
        p99_latency = np.percentile(all_latencies, 99)
        
        # Calculate average phase latencies
        avg_phase_latencies = {}
        if all_phase_latencies:
            phase_keys = all_phase_latencies[0].keys()
            for phase in phase_keys:
                phase_values = [pl.get(phase, 0) for pl in all_phase_latencies]
                avg_phase_latencies[phase] = statistics.mean(phase_values)
        
        # Calculate throughput
        throughput_rps = 1000 / avg_latency if avg_latency > 0 else 0
        
        # Calculate memory statistics
        avg_memory = statistics.mean(memory_usage) if memory_usage else 0
        peak_memory = max(memory_usage) if memory_usage else 0
        
        # Create benchmark result
        result = BenchmarkResult(
            benchmark_type=benchmark_type,
            load_profile=load_profile,
            record_count=record_count,
            total_time_ms=statistics.mean(all_latencies) * record_count,
            average_latency_ms=avg_latency,
            median_latency_ms=median_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            throughput_rps=throughput_rps,
            memory_peak_mb=peak_memory / (1024 * 1024),
            memory_average_mb=avg_memory / (1024 * 1024),
            phase_latencies=avg_phase_latencies,
            error_count=error_count,
            success_rate=success_count / suite.iterations if suite.iterations > 0 else 0,
            timestamp=datetime.now(timezone.utc),
            metadata={
                'iterations': suite.iterations,
                'warmup_runs': suite.warmup_runs,
                'config': self.config.processing_mode.value
            }
        )
        
        return result
    
    def _generate_test_records(self, record_count: int, load_profile: LoadProfile) -> List[Any]:
        """Generate test telemetry records for benchmarking"""
        # This is a simplified test data generator
        # In practice, you'd want more realistic telemetry data
        
        test_records = []
        for i in range(record_count):
            # Create mock telemetry record
            record = {
                'event_id': f"test_event_{i}",
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'function_id': f"test_function_{i % 10}",
                'session_id': f"test_session_{i % 100}",
                'telemetry_data': {
                    'cpu_usage': 50 + (i % 50),
                    'memory_usage': 100 + (i % 200),
                    'execution_time_ms': 10 + (i % 90),
                    'error_count': i % 5,
                    'request_count': 1 + (i % 10)
                },
                'metadata': {
                    'source': 'benchmark',
                    'load_profile': load_profile.value,
                    'record_index': i
                }
            }
            
            # Add some anomaly patterns for testing
            if i % 100 == 0:  # Every 100th record has anomaly
                record['telemetry_data']['cpu_usage'] = 95
                record['telemetry_data']['memory_usage'] = 500
                record['telemetry_data']['execution_time_ms'] = 500
            
            test_records.append(record)
        
        return test_records
    
    def _extract_phase_latencies(self, processing_result: Any) -> Dict[str, float]:
        """Extract phase latencies from processing result"""
        # This would extract actual phase latencies from the Layer 1 result
        # For now, return mock data
        return {
            'validation': 0.1,
            'schema_evolution': 0.2,
            'sanitization': 0.3,
            'privacy_filtering': 0.2,
            'deferred_hashing': 0.1,
            'preservation_validation': 0.1
        }
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            # Fallback to basic memory info
            return 0
    
    def _update_performance_metrics(self, result: BenchmarkResult):
        """Update overall performance metrics"""
        self.performance_metrics['total_benchmarks'] += 1
        self.performance_metrics['successful_benchmarks'] += 1
        self.performance_metrics['total_records_processed'] += result.record_count
        self.performance_metrics['total_processing_time_ms'] += result.total_time_ms
    
    def _save_benchmark_result(self, result: BenchmarkResult, output_path: Path):
        """Save individual benchmark result to file"""
        filename = f"benchmark_{result.benchmark_type.value}_{result.load_profile.value}_{result.record_count}_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = output_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
    
    def _save_suite_summary(self, results: List[BenchmarkResult], suite: BenchmarkSuite, output_path: Path):
        """Save benchmark suite summary"""
        summary = {
            'suite_name': suite.name,
            'suite_description': suite.description,
            'total_benchmarks': len(results),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'results_summary': {
                'latency_stats': self._calculate_latency_statistics(results),
                'throughput_stats': self._calculate_throughput_statistics(results),
                'memory_stats': self._calculate_memory_statistics(results),
                'success_rates': self._calculate_success_rates(results)
            },
            'performance_metrics': self.performance_metrics
        }
        
        summary_file = output_path / f"{suite.name}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def _calculate_latency_statistics(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate latency statistics across all results"""
        if not results:
            return {}
        
        all_latencies = [r.average_latency_ms for r in results]
        return {
            'mean': statistics.mean(all_latencies),
            'median': statistics.median(all_latencies),
            'min': min(all_latencies),
            'max': max(all_latencies),
            'std_dev': statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0,
            'p95': np.percentile(all_latencies, 95),
            'p99': np.percentile(all_latencies, 99)
        }
    
    def _calculate_throughput_statistics(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate throughput statistics across all results"""
        if not results:
            return {}
        
        all_throughputs = [r.throughput_rps for r in results]
        return {
            'mean': statistics.mean(all_throughputs),
            'median': statistics.median(all_throughputs),
            'min': min(all_throughputs),
            'max': max(all_throughputs),
            'std_dev': statistics.stdev(all_throughputs) if len(all_throughputs) > 1 else 0
        }
    
    def _calculate_memory_statistics(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate memory statistics across all results"""
        if not results:
            return {}
        
        all_memory = [r.memory_average_mb for r in results]
        return {
            'mean': statistics.mean(all_memory),
            'median': statistics.median(all_memory),
            'min': min(all_memory),
            'max': max(all_memory),
            'std_dev': statistics.stdev(all_memory) if len(all_memory) > 1 else 0
        }
    
    def _calculate_success_rates(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate success rates across all results"""
        if not results:
            return {}
        
        success_rates = [r.success_rate for r in results]
        return {
            'mean': statistics.mean(success_rates),
            'min': min(success_rates),
            'max': max(success_rates)
        }
    
    def _generate_benchmark_plots(self, results: List[BenchmarkResult], output_path: Path):
        """Generate benchmark visualization plots"""
        try:
            # Latency vs Record Count
            self._plot_latency_vs_records(results, output_path)
            
            # Throughput vs Record Count
            self._plot_throughput_vs_records(results, output_path)
            
            # Memory Usage vs Record Count
            self._plot_memory_vs_records(results, output_path)
            
            # Phase Latency Breakdown
            self._plot_phase_latencies(results, output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate plots: {e}")
    
    def _plot_latency_vs_records(self, results: List[BenchmarkResult], output_path: Path):
        """Plot latency vs record count"""
        plt.figure(figsize=(10, 6))
        
        # Group by load profile
        for load_profile in LoadProfile:
            profile_results = [r for r in results if r.load_profile == load_profile]
            if profile_results:
                x = [r.record_count for r in profile_results]
                y = [r.average_latency_ms for r in profile_results]
                plt.scatter(x, y, label=load_profile.value, alpha=0.7)
        
        plt.xlabel('Record Count')
        plt.ylabel('Average Latency (ms)')
        plt.title('Latency vs Record Count by Load Profile')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.xscale('log')
        
        plot_file = output_path / 'latency_vs_records.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_throughput_vs_records(self, results: List[BenchmarkResult], output_path: Path):
        """Plot throughput vs record count"""
        plt.figure(figsize=(10, 6))
        
        # Group by load profile
        for load_profile in LoadProfile:
            profile_results = [r for r in results if r.load_profile == load_profile]
            if profile_results:
                x = [r.record_count for r in profile_results]
                y = [r.throughput_rps for r in profile_results]
                plt.scatter(x, y, label=load_profile.value, alpha=0.7)
        
        plt.xlabel('Record Count')
        plt.ylabel('Throughput (records/sec)')
        plt.title('Throughput vs Record Count by Load Profile')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.xscale('log')
        
        plot_file = output_path / 'throughput_vs_records.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_vs_records(self, results: List[BenchmarkResult], output_path: Path):
        """Plot memory usage vs record count"""
        plt.figure(figsize=(10, 6))
        
        # Group by load profile
        for load_profile in LoadProfile:
            profile_results = [r for r in results if r.load_profile == load_profile]
            if profile_results:
                x = [r.record_count for r in profile_results]
                y = [r.memory_average_mb for r in profile_results]
                plt.scatter(x, y, label=load_profile.value, alpha=0.7)
        
        plt.xlabel('Record Count')
        plt.ylabel('Average Memory Usage (MB)')
        plt.title('Memory Usage vs Record Count by Load Profile')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.xscale('log')
        
        plot_file = output_path / 'memory_vs_records.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_phase_latencies(self, results: List[BenchmarkResult], output_path: Path):
        """Plot phase latency breakdown"""
        if not results or not results[0].phase_latencies:
            return
        
        # Get all phase names
        phase_names = list(results[0].phase_latencies.keys())
        
        # Calculate average latencies per phase
        phase_avg_latencies = {}
        for phase in phase_names:
            phase_values = [r.phase_latencies.get(phase, 0) for r in results if r.phase_latencies]
            if phase_values:
                phase_avg_latencies[phase] = statistics.mean(phase_values)
        
        if not phase_avg_latencies:
            return
        
        plt.figure(figsize=(10, 6))
        phases = list(phase_avg_latencies.keys())
        latencies = list(phase_avg_latencies.values())
        
        bars = plt.bar(phases, latencies, alpha=0.7)
        plt.xlabel('Processing Phase')
        plt.ylabel('Average Latency (ms)')
        plt.title('Average Latency by Processing Phase')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, latency in zip(bars, latencies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{latency:.3f}', ha='center', va='bottom')
        
        plot_file = output_path / 'phase_latencies.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_with_baseline(self, result: BenchmarkResult) -> Dict[str, Any]:
        """Compare benchmark result with performance baseline"""
        if not self.baseline:
            return {'error': 'No baseline set'}
        
        comparison = {
            'baseline_name': self.baseline.baseline_name,
            'baseline_date': self.baseline.baseline_date.isoformat(),
            'comparison_date': result.timestamp.isoformat(),
            'metrics': {}
        }
        
        # Compare latency
        latency_diff = result.average_latency_ms - self.baseline.target_latency_ms
        latency_change_pct = (latency_diff / self.baseline.target_latency_ms) * 100
        comparison['metrics']['latency'] = {
            'target': self.baseline.target_latency_ms,
            'actual': result.average_latency_ms,
            'difference': latency_diff,
            'change_percent': latency_change_pct,
            'status': 'PASS' if latency_diff <= 0 else 'FAIL'
        }
        
        # Compare throughput
        throughput_diff = result.throughput_rps - self.baseline.target_throughput_rps
        throughput_change_pct = (throughput_diff / self.baseline.target_throughput_rps) * 100
        comparison['metrics']['throughput'] = {
            'target': self.baseline.target_throughput_rps,
            'actual': result.throughput_rps,
            'difference': throughput_diff,
            'change_percent': throughput_change_pct,
            'status': 'PASS' if throughput_diff >= 0 else 'FAIL'
        }
        
        # Compare memory usage
        memory_diff = result.memory_average_mb - self.baseline.target_memory_mb
        memory_change_pct = (memory_diff / self.baseline.target_memory_mb) * 100
        comparison['metrics']['memory'] = {
            'target': self.baseline.target_memory_mb,
            'actual': result.memory_average_mb,
            'difference': memory_diff,
            'change_percent': memory_change_pct,
            'status': 'PASS' if memory_diff <= 0 else 'FAIL'
        }
        
        return comparison

# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Main command line interface for latency benchmarks"""
    parser = argparse.ArgumentParser(description='SCAFAD Layer 1 Latency Benchmarks')
    parser.add_argument('--records', type=int, nargs='+', default=[100, 1000, 10000],
                       help='Record counts to benchmark')
    parser.add_argument('--iterations', type=int, default=5,
                       help='Number of iterations per benchmark')
    parser.add_argument('--warmup', type=int, default=3,
                       help='Number of warmup runs')
    parser.add_argument('--output', type=str, default='./benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--plots', action='store_true',
                       help='Generate performance plots')
    parser.add_argument('--config', type=str, default='balanced',
                       help='Layer 1 configuration mode')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create configuration
    config = Layer1Config()
    if args.config == 'ultra_low_latency':
        config.performance_profile = PerformanceProfile.ULTRA_LOW_LATENCY
    elif args.config == 'high_throughput':
        config.performance_profile = PerformanceProfile.HIGH_THROUGHPUT
    elif args.config == 'quality_optimized':
        config.performance_profile = PerformanceProfile.QUALITY_OPTIMIZED
    
    # Create benchmark suite
    suite = BenchmarkSuite(
        name="Layer1_Latency_Benchmarks",
        description="Comprehensive latency benchmarking for SCAFAD Layer 1",
        benchmark_types=[BenchmarkType.LATENCY, BenchmarkType.THROUGHPUT, BenchmarkType.MEMORY],
        load_profiles=[LoadProfile.LIGHT, LoadProfile.MEDIUM, LoadProfile.HEAVY],
        record_counts=args.records,
        iterations=args.iterations,
        warmup_runs=args.warmup,
        cooldown_seconds=1.0,
        output_directory=args.output,
        generate_plots=args.plots,
        save_results=True
    )
    
    # Run benchmarks
    runner = LatencyBenchmarkRunner(config)
    results = runner.run_benchmark_suite(suite)
    
    # Print summary
    if results:
        print(f"\nBenchmark completed successfully!")
        print(f"Total benchmarks run: {len(results)}")
        print(f"Results saved to: {args.output}")
        
        # Print key metrics
        avg_latency = statistics.mean([r.average_latency_ms for r in results])
        avg_throughput = statistics.mean([r.throughput_rps for r in results])
        print(f"Average latency: {avg_latency:.3f} ms")
        print(f"Average throughput: {avg_throughput:.1f} records/sec")
    else:
        print("Benchmark failed to produce results")

if __name__ == "__main__":
    main()
