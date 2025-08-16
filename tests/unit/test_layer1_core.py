"""
Unit tests for layer1_core.py - Main orchestrator module.

Tests the core processing pipeline, error handling, and integration
between different processing stages.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the core directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'core'))

from layer1_core import Layer1Processor, ProcessingPipeline, ProcessingStage


class TestLayer1Processor(unittest.TestCase):
    """Test the main Layer1Processor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = Layer1Processor()
        self.sample_data = {
            "id": "test_123",
            "content": "Sample content for testing",
            "metadata": {"source": "test", "timestamp": "2024-01-01T00:00:00Z"}
        }
    
    def test_processor_initialization(self):
        """Test processor initialization with default settings."""
        self.assertIsNotNone(self.processor)
        self.assertIsInstance(self.processor.pipeline, ProcessingPipeline)
    
    def test_process_data_basic(self):
        """Test basic data processing functionality."""
        result = self.processor.process(self.sample_data)
        self.assertIsNotNone(result)
        self.assertIn('processed_at', result)
    
    def test_process_data_with_custom_config(self):
        """Test data processing with custom configuration."""
        config = {"preserve_anomalies": True, "privacy_level": "high"}
        result = self.processor.process(self.sample_data, config=config)
        self.assertIsNotNone(result)
    
    def test_error_handling_invalid_data(self):
        """Test error handling for invalid input data."""
        invalid_data = None
        with self.assertRaises(ValueError):
            self.processor.process(invalid_data)
    
    def test_error_handling_malformed_data(self):
        """Test error handling for malformed data structures."""
        malformed_data = {"id": 123, "content": []}  # Invalid types
        with self.assertRaises(TypeError):
            self.processor.process(malformed_data)


class TestProcessingPipeline(unittest.TestCase):
    """Test the ProcessingPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = ProcessingPipeline()
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline)
        self.assertEqual(len(self.pipeline.stages), 0)
    
    def test_add_stage(self):
        """Test adding processing stages to the pipeline."""
        stage = ProcessingStage("test_stage", lambda x: x)
        self.pipeline.add_stage(stage)
        self.assertEqual(len(self.pipeline.stages), 1)
    
    def test_pipeline_execution_order(self):
        """Test that pipeline stages execute in the correct order."""
        execution_order = []
        
        def stage1(data):
            execution_order.append(1)
            return data
        
        def stage2(data):
            execution_order.append(2)
            return data
        
        self.pipeline.add_stage(ProcessingStage("stage1", stage1))
        self.pipeline.add_stage(ProcessingStage("stage2", stage2))
        
        test_data = {"test": "data"}
        self.pipeline.execute(test_data)
        
        self.assertEqual(execution_order, [1, 2])


class TestProcessingStage(unittest.TestCase):
    """Test the ProcessingStage class."""
    
    def test_stage_creation(self):
        """Test creating a processing stage."""
        def test_function(data):
            return data
        
        stage = ProcessingStage("test_stage", test_function)
        self.assertEqual(stage.name, "test_stage")
        self.assertEqual(stage.function, test_function)
    
    def test_stage_execution(self):
        """Test stage execution."""
        def add_timestamp(data):
            data['timestamp'] = '2024-01-01T00:00:00Z'
            return data
        
        stage = ProcessingStage("add_timestamp", add_timestamp)
        test_data = {"id": "test"}
        result = stage.execute(test_data)
        
        self.assertIn('timestamp', result)
        self.assertEqual(result['timestamp'], '2024-01-01T00:00:00Z')


class TestIntegration(unittest.TestCase):
    """Test integration between core components."""
    
    def test_full_pipeline_integration(self):
        """Test complete pipeline integration."""
        processor = Layer1Processor()
        
        # Add custom stages
        def validation_stage(data):
            if 'id' not in data:
                raise ValueError("Missing required field: id")
            return data
        
        def processing_stage(data):
            data['processed'] = True
            return data
        
        processor.pipeline.add_stage(ProcessingStage("validation", validation_stage))
        processor.pipeline.add_stage(ProcessingStage("processing", processing_stage))
        
        test_data = {"id": "test_123", "content": "test"}
        result = processor.process(test_data)
        
        self.assertIn('processed', result)
        self.assertTrue(result['processed'])


if __name__ == '__main__':
    unittest.main()
