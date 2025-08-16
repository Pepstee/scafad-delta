"""
Performance benchmarking tests for Scafad Layer1 system.

Tests system performance characteristics including latency,
throughput, memory usage, and scalability metrics.
"""

import unittest
import sys
import os
import time
import psutil
import gc
import json
from unittest.mock import Mock, patch, MagicMock

# Add the project root to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.layer1_core import Layer1Processor
from core.layer1_validation import DataValidator
from core.layer1_privacy import PrivacyFilter


class TestPerformanceBenchmarks(unittest.TestCase):
    """Test system performance characteristics."""
    
    def setUp(self):
        """Set up test fixtures for performance testing."""
        self.processor = Layer1Processor()
        self.validator = DataValidator()
        self.privacy_filter = PrivacyFilter()
        
        # Performance thresholds
        self.latency_threshold = 0.1  # 100ms
        self.throughput_threshold = 1000  # records per second
        self.memory_threshold = 100 * 1024 * 1024  # 100MB
        
        # Test data sizes
        self.small_data = {"id": "test", "content": "small"}
        self.medium_data = {"id": "test", "content": "x" * 1000}  # ~1KB
        self.large_data = {"id": "test", "content": "x" * 100000}  # ~100KB
        
    def get_memory_usage(self):
        """Get current memory usage in bytes."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss
    
    def measure_latency(self, func, *args, **kwargs):
        """Measure execution latency of a function."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return end_time - start_time, result
    
    def test_basic_processing_latency(self):
        """Test basic data processing latency."""
        print("\n=== Basic Processing Latency Test ===")
        
        # Test small data
        latency, result = self.measure_latency(
            self.processor.process, self.small_data
        )
        print(f"Small data latency: {latency:.6f}s")
        self.assertLess(latency, self.latency_threshold)
        
        # Test medium data
        latency, result = self.measure_latency(
            self.processor.process, self.medium_data
        )
        print(f"Medium data latency: {latency:.6f}s")
        self.assertLess(latency, self.latency_threshold * 2)
        
        # Test large data
        latency, result = self.measure_latency(
            self.processor.process, self.large_data
        )
        print(f"Large data latency: {latency:.6f}s")
        self.assertLess(latency, self.latency_threshold * 5)
    
    def test_validation_performance(self):
        """Test data validation performance."""
        print("\n=== Validation Performance Test ===")
        
        # Add validation rules
        from core.layer1_validation import ValidationRule
        
        def simple_validation(data):
            return 'id' in data
        
        def complex_validation(data):
            # Simulate complex validation logic
            if 'id' not in data:
                return False
            if 'content' in data and len(data['content']) > 10000:
                return False
            return True
        
        self.validator.add_rule(ValidationRule("simple", simple_validation))
        self.validator.add_rule(ValidationRule("complex", complex_validation))
        
        # Test validation performance
        test_data = {"id": "test", "content": "x" * 5000}
        
        latency, result = self.measure_latency(
            self.validator.validate, test_data
        )
        print(f"Validation latency: {latency:.6f}s")
        self.assertLess(latency, self.latency_threshold)
    
    def test_privacy_filtering_performance(self):
        """Test privacy filtering performance."""
        print("\n=== Privacy Filtering Performance Test ===")
        
        # Test data with various PII types
        sensitive_data = {
            "id": "user_123",
            "name": "John Doe",
            "email": "john.doe@example.com",
            "phone": "+1-555-123-4567",
            "credit_card": "4111-1111-1111-1111",
            "address": "123 Main St, Anytown, USA 12345"
        }
        
        latency, result = self.measure_latency(
            self.privacy_filter.process, sensitive_data
        )
        print(f"Privacy filtering latency: {latency:.6f}s")
        self.assertLess(latency, self.latency_threshold * 2)
    
    def test_throughput_performance(self):
        """Test system throughput (records per second)."""
        print("\n=== Throughput Performance Test ===")
        
        # Generate test data batch
        batch_size = 100
        test_batch = [
            {"id": f"record_{i}", "content": f"content_{i}"}
            for i in range(batch_size)
        ]
        
        # Measure batch processing time
        start_time = time.perf_counter()
        results = []
        for record in test_batch:
            result = self.processor.process(record)
            results.append(result)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        throughput = batch_size / total_time
        
        print(f"Batch size: {batch_size}")
        print(f"Total time: {total_time:.6f}s")
        print(f"Throughput: {throughput:.2f} records/second")
        
        self.assertGreater(throughput, self.throughput_threshold)
        self.assertEqual(len(results), batch_size)
    
    def test_memory_usage(self):
        """Test memory usage during processing."""
        print("\n=== Memory Usage Test ===")
        
        # Force garbage collection
        gc.collect()
        
        # Measure baseline memory
        baseline_memory = self.get_memory_usage()
        print(f"Baseline memory: {baseline_memory / 1024 / 1024:.2f} MB")
        
        # Process large dataset
        large_dataset = [
            {"id": f"record_{i}", "content": "x" * 10000}
            for i in range(100)
        ]
        
        results = []
        for record in large_dataset:
            result = self.processor.process(record)
            results.append(result)
        
        # Measure memory after processing
        post_processing_memory = self.get_memory_usage()
        memory_increase = post_processing_memory - baseline_memory
        
        print(f"Post-processing memory: {post_processing_memory / 1024 / 1024:.2f} MB")
        print(f"Memory increase: {memory_increase / 1024 / 1024:.2f} MB")
        
        # Memory increase should be reasonable
        self.assertLess(memory_increase, self.memory_threshold)
        
        # Clean up
        del results
        gc.collect()
    
    def test_scalability_performance(self):
        """Test system scalability with increasing data sizes."""
        print("\n=== Scalability Performance Test ===")
        
        data_sizes = [1, 10, 100, 1000]  # KB
        performance_metrics = {}
        
        for size_kb in data_sizes:
            # Generate data of specified size
            content_size = size_kb * 1024
            test_data = {
                "id": f"size_{size_kb}kb",
                "content": "x" * content_size,
                "metadata": {"size_kb": size_kb}
            }
            
            # Measure processing time
            latency, result = self.measure_latency(
                self.processor.process, test_data
            )
            
            performance_metrics[size_kb] = {
                "latency": latency,
                "throughput": 1 / latency,
                "data_size_kb": size_kb
            }
            
            print(f"{size_kb}KB data: {latency:.6f}s ({1/latency:.2f} records/s)")
        
        # Performance should scale reasonably
        for size_kb in data_sizes[1:]:
            prev_latency = performance_metrics[size_kb - 1]["latency"]
            curr_latency = performance_metrics[size_kb]["latency"]
            
            # Latency increase should be reasonable (not exponential)
            latency_ratio = curr_latency / prev_latency
            print(f"Latency ratio {size_kb}KB/{size_kb-1}KB: {latency_ratio:.2f}")
            
            # Should not increase more than 10x for 10x data increase
            self.assertLess(latency_ratio, 10.0)
    
    def test_concurrent_processing_performance(self):
        """Test concurrent processing performance."""
        print("\n=== Concurrent Processing Performance Test ===")
        
        import threading
        import queue
        
        # Test data
        test_data = {"id": "concurrent_test", "content": "test content"}
        
        # Number of concurrent threads
        num_threads = 10
        results_queue = queue.Queue()
        
        def process_record(thread_id):
            """Process a record in a separate thread."""
            try:
                start_time = time.perf_counter()
                result = self.processor.process(test_data)
                end_time = time.perf_counter()
                
                results_queue.put({
                    "thread_id": thread_id,
                    "latency": end_time - start_time,
                    "success": True
                })
            except Exception as e:
                results_queue.put({
                    "thread_id": thread_id,
                    "error": str(e),
                    "success": False
                })
        
        # Start concurrent processing
        threads = []
        start_time = time.perf_counter()
        
        for i in range(num_threads):
            thread = threading.Thread(target=process_record, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.perf_counter()
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        total_time = end_time - start_time
        successful_results = [r for r in results if r["success"]]
        
        print(f"Concurrent threads: {num_threads}")
        print(f"Total time: {total_time:.6f}s")
        print(f"Successful operations: {len(successful_results)}/{num_threads}")
        
        if successful_results:
            avg_latency = sum(r["latency"] for r in successful_results) / len(successful_results)
            print(f"Average latency: {avg_latency:.6f}s")
            
            # Concurrent processing should be faster than sequential
            sequential_time = sum(r["latency"] for r in successful_results)
            speedup = sequential_time / total_time
            print(f"Speedup: {speedup:.2f}x")
            
            # Should have some speedup (not necessarily linear due to overhead)
            self.assertGreater(speedup, 1.0)
        
        # All operations should succeed
        self.assertEqual(len(successful_results), num_threads)
    
    def test_memory_efficiency(self):
        """Test memory efficiency during long-running operations."""
        print("\n=== Memory Efficiency Test ===")
        
        # Force garbage collection
        gc.collect()
        
        # Measure initial memory
        initial_memory = self.get_memory_usage()
        print(f"Initial memory: {initial_memory / 1024 / 1024:.2f} MB")
        
        # Process many records
        num_records = 1000
        memory_samples = []
        
        for i in range(num_records):
            if i % 100 == 0:  # Sample every 100 records
                memory_samples.append(self.get_memory_usage())
            
            test_data = {
                "id": f"record_{i}",
                "content": f"content_{i}",
                "metadata": {"index": i}
            }
            
            result = self.processor.process(test_data)
            
            # Don't keep references to avoid memory buildup
            del result
        
        # Measure final memory
        final_memory = self.get_memory_usage()
        print(f"Final memory: {final_memory / 1024 / 1024:.2f} MB")
        
        # Memory should not grow excessively
        memory_growth = final_memory - initial_memory
        print(f"Memory growth: {memory_growth / 1024 / 1024:.2f} MB")
        
        # Growth should be reasonable (less than 50MB for 1000 records)
        self.assertLess(memory_growth, 50 * 1024 * 1024)
        
        # Memory should stabilize (not continuously grow)
        if len(memory_samples) > 1:
            early_memory = memory_samples[0]
            late_memory = memory_samples[-1]
            late_growth = late_memory - early_memory
            
            print(f"Late memory growth: {late_growth / 1024 / 1024:.2f} MB")
            
            # Late growth should be minimal
            self.assertLess(late_growth, 10 * 1024 * 1024)
    
    def test_error_handling_performance(self):
        """Test performance of error handling mechanisms."""
        print("\n=== Error Handling Performance Test ===")
        
        # Test data that will cause errors
        invalid_data = None
        malformed_data = {"id": 123, "content": []}  # Invalid types
        
        # Measure error handling performance
        start_time = time.perf_counter()
        
        try:
            self.processor.process(invalid_data)
        except Exception:
            pass
        
        try:
            self.processor.process(malformed_data)
        except Exception:
            pass
        
        end_time = time.perf_counter()
        
        error_handling_time = end_time - start_time
        print(f"Error handling time: {error_handling_time:.6f}s")
        
        # Error handling should be fast
        self.assertLess(error_handling_time, self.latency_threshold)
    
    def test_configuration_performance(self):
        """Test performance impact of different configurations."""
        print("\n=== Configuration Performance Test ===")
        
        test_data = {"id": "config_test", "content": "test content"}
        
        # Test different privacy levels
        privacy_levels = ["low", "medium", "high", "strict"]
        privacy_performance = {}
        
        for level in privacy_levels:
            self.privacy_filter.configure({"privacy_level": level})
            
            latency, result = self.measure_latency(
                self.privacy_filter.process, test_data
            )
            
            privacy_performance[level] = latency
            print(f"Privacy level '{level}': {latency:.6f}s")
        
        # Higher privacy levels may be slower but should still be reasonable
        for level in privacy_levels:
            self.assertLess(privacy_performance[level], self.latency_threshold * 3)
        
        # Test different validation configurations
        validation_configs = [
            {"strict_mode": False, "max_rules": 5},
            {"strict_mode": True, "max_rules": 10},
            {"strict_mode": True, "max_rules": 20}
        ]
        
        for config in validation_configs:
            self.validator.configure(config)
            
            latency, result = self.measure_latency(
                self.validator.validate, test_data
            )
            
            print(f"Validation config {config}: {latency:.6f}s")
            self.assertLess(latency, self.latency_threshold * 2)


class TestPerformanceRegression(unittest.TestCase):
    """Test for performance regressions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = Layer1Processor()
        self.baseline_metrics = {
            "small_data_latency": 0.001,  # 1ms
            "medium_data_latency": 0.005,  # 5ms
            "large_data_latency": 0.050,   # 50ms
            "throughput": 1000,            # records/second
            "memory_growth": 10 * 1024 * 1024  # 10MB
        }
    
    def test_performance_regression_detection(self):
        """Test that performance has not regressed from baseline."""
        print("\n=== Performance Regression Test ===")
        
        # Test small data
        test_data = {"id": "test", "content": "small"}
        start_time = time.perf_counter()
        result = self.processor.process(test_data)
        end_time = time.perf_counter()
        
        current_latency = end_time - start_time
        baseline_latency = self.baseline_metrics["small_data_latency"]
        
        print(f"Current small data latency: {current_latency:.6f}s")
        print(f"Baseline small data latency: {baseline_latency:.6f}s")
        
        # Current performance should not be more than 2x worse than baseline
        performance_ratio = current_latency / baseline_latency
        print(f"Performance ratio: {performance_ratio:.2f}x")
        
        self.assertLess(performance_ratio, 2.0, 
                       "Performance has regressed significantly")


if __name__ == '__main__':
    # Run performance tests
    unittest.main(verbosity=2)
