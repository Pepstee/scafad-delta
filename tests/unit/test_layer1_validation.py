"""
Unit tests for layer1_validation.py - Input validation gateway.

Tests data validation, schema compliance, and error handling
for various input types and formats.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the core directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'core'))

from layer1_validation import DataValidator, ValidationRule, ValidationError


class TestDataValidator(unittest.TestCase):
    """Test the DataValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = DataValidator()
        self.sample_data = {
            "id": "test_123",
            "content": "Sample content",
            "metadata": {"source": "test", "timestamp": "2024-01-01T00:00:00Z"}
        }
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        self.assertIsNotNone(self.validator)
        self.assertEqual(len(self.validator.rules), 0)
    
    def test_add_validation_rule(self):
        """Test adding validation rules."""
        rule = ValidationRule("id_required", lambda data: 'id' in data)
        self.validator.add_rule(rule)
        self.assertEqual(len(self.validator.rules), 1)
    
    def test_validate_data_success(self):
        """Test successful data validation."""
        rule = ValidationRule("id_required", lambda data: 'id' in data)
        self.validator.add_rule(rule)
        
        result = self.validator.validate(self.sample_data)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
    
    def test_validate_data_failure(self):
        """Test data validation failure."""
        rule = ValidationRule("id_required", lambda data: 'id' in data)
        self.validator.add_rule(rule)
        
        invalid_data = {"content": "Missing ID"}
        result = self.validator.validate(invalid_data)
        
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("id_required", str(result.errors[0]))
    
    def test_multiple_validation_rules(self):
        """Test validation with multiple rules."""
        # Rule 1: ID required
        rule1 = ValidationRule("id_required", lambda data: 'id' in data)
        # Rule 2: Content not empty
        rule2 = ValidationRule("content_not_empty", lambda data: data.get('content', '').strip() != '')
        
        self.validator.add_rule(rule1)
        self.validator.add_rule(rule2)
        
        result = self.validator.validate(self.sample_data)
        self.assertTrue(result.is_valid)
    
    def test_validation_error_details(self):
        """Test validation error details and messages."""
        def custom_validation(data):
            if not data.get('content'):
                raise ValidationError("Content is required", field="content")
            return True
        
        rule = ValidationRule("custom_validation", custom_validation)
        self.validator.add_rule(rule)
        
        invalid_data = {"id": "test"}
        result = self.validator.validate(invalid_data)
        
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.errors), 1)
        error = result.errors[0]
        self.assertEqual(error.field, "content")
        self.assertIn("Content is required", str(error))


class TestValidationRule(unittest.TestCase):
    """Test the ValidationRule class."""
    
    def test_rule_creation(self):
        """Test creating a validation rule."""
        def test_function(data):
            return True
        
        rule = ValidationRule("test_rule", test_function)
        self.assertEqual(rule.name, "test_rule")
        self.assertEqual(rule.function, test_function)
    
    def test_rule_execution_success(self):
        """Test successful rule execution."""
        def always_true(data):
            return True
        
        rule = ValidationRule("always_true", always_true)
        test_data = {"test": "data"}
        result = rule.execute(test_data)
        
        self.assertTrue(result)
    
    def test_rule_execution_failure(self):
        """Test rule execution failure."""
        def always_false(data):
            return False
        
        rule = ValidationRule("always_false", always_false)
        test_data = {"test": "data"}
        result = rule.execute(test_data)
        
        self.assertFalse(result)
    
    def test_rule_execution_with_exception(self):
        """Test rule execution that raises an exception."""
        def exception_rule(data):
            raise ValueError("Test exception")
        
        rule = ValidationRule("exception_rule", exception_rule)
        test_data = {"test": "data"}
        
        with self.assertRaises(ValueError):
            rule.execute(test_data)


class TestValidationError(unittest.TestCase):
    """Test the ValidationError class."""
    
    def test_error_creation(self):
        """Test creating a validation error."""
        error = ValidationError("Test error message", field="test_field")
        self.assertEqual(error.message, "Test error message")
        self.assertEqual(error.field, "test_field")
    
    def test_error_string_representation(self):
        """Test error string representation."""
        error = ValidationError("Test error message", field="test_field")
        error_str = str(error)
        self.assertIn("Test error message", error_str)
        self.assertIn("test_field", error_str)
    
    def test_error_without_field(self):
        """Test error creation without specifying a field."""
        error = ValidationError("Test error message")
        self.assertEqual(error.message, "Test error message")
        self.assertIsNone(error.field)


class TestValidationIntegration(unittest.TestCase):
    """Test integration between validation components."""
    
    def test_complex_validation_scenario(self):
        """Test complex validation scenario with multiple rules."""
        validator = DataValidator()
        
        # Complex validation rules
        def has_required_fields(data):
            required = ['id', 'content', 'metadata']
            missing = [field for field in required if field not in data]
            if missing:
                raise ValidationError(f"Missing required fields: {missing}")
            return True
        
        def metadata_structure(data):
            metadata = data.get('metadata', {})
            if not isinstance(metadata, dict):
                raise ValidationError("Metadata must be a dictionary")
            if 'timestamp' not in metadata:
                raise ValidationError("Metadata must contain timestamp")
            return True
        
        def content_length(data):
            content = data.get('content', '')
            if len(content) < 5:
                raise ValidationError("Content must be at least 5 characters")
            return True
        
        validator.add_rule(ValidationRule("required_fields", has_required_fields))
        validator.add_rule(ValidationRule("metadata_structure", metadata_structure))
        validator.add_rule(ValidationRule("content_length", content_length))
        
        # Test valid data
        valid_data = {
            "id": "test_123",
            "content": "This is valid content",
            "metadata": {"source": "test", "timestamp": "2024-01-01T00:00:00Z"}
        }
        
        result = validator.validate(valid_data)
        self.assertTrue(result.is_valid)
        
        # Test invalid data
        invalid_data = {
            "id": "test_123",
            "content": "Short",
            "metadata": "not_a_dict"
        }
        
        result = validator.validate(invalid_data)
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)


if __name__ == '__main__':
    unittest.main()
