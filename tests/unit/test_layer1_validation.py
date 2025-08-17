"""
Comprehensive unit tests for layer1_validation.py

Tests the input validation gateway, validation rules, and validation results
with extensive coverage of edge cases and error conditions.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

# Add the core directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'core'))

from layer1_validation import (
    InputValidationGateway, ValidationResult, ValidationLevel,
    ValidationRule, ValidationError, FieldValidator, SchemaValidator
)


class TestValidationResult:
    """Test the ValidationResult class."""
    
    def test_validation_result_creation(self):
        """Test creating validation results."""
        result = ValidationResult(
            is_valid=True,
            level=ValidationLevel.STRICT,
            errors=[],
            warnings=[],
            metadata={"source": "test"}
        )
        
        assert result.is_valid is True
        assert result.level == ValidationLevel.STRICT
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        assert result.metadata["source"] == "test"
    
    def test_validation_result_with_errors(self):
        """Test validation result with validation errors."""
        errors = [ValidationError("field", "required", "Field is required")]
        result = ValidationResult(
            is_valid=False,
            level=ValidationLevel.STRICT,
            errors=errors,
            warnings=[]
        )
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field == "field"
        assert result.errors[0].code == "required"
    
    def test_validation_result_with_warnings(self):
        """Test validation result with warnings."""
        warnings = [ValidationError("field", "deprecated", "Field is deprecated")]
        result = ValidationResult(
            is_valid=True,
            level=ValidationLevel.STRICT,
            errors=[],
            warnings=warnings
        )
        
        assert result.is_valid is True
        assert len(result.warnings) == 1
        assert result.warnings[0].field == "field"
        assert result.warnings[0].code == "deprecated"
    
    def test_validation_result_serialization(self):
        """Test validation result serialization."""
        result = ValidationResult(
            is_valid=True,
            level=ValidationLevel.STRICT,
            errors=[],
            warnings=[],
            metadata={"timestamp": "2024-01-01T00:00:00Z"}
        )
        
        serialized = result.to_dict()
        assert serialized["is_valid"] is True
        assert serialized["level"] == "strict"
        assert "timestamp" in serialized["metadata"]


class TestValidationError:
    """Test the ValidationError class."""
    
    def test_validation_error_creation(self):
        """Test creating validation errors."""
        error = ValidationError("user_id", "invalid_format", "User ID must be alphanumeric")
        
        assert error.field == "user_id"
        assert error.code == "invalid_format"
        assert error.message == "User ID must be alphanumeric"
        assert error.timestamp is not None
    
    def test_validation_error_serialization(self):
        """Test validation error serialization."""
        error = ValidationError("email", "invalid_email", "Invalid email format")
        
        serialized = error.to_dict()
        assert serialized["field"] == "email"
        assert serialized["code"] == "invalid_email"
        assert serialized["message"] == "Invalid email format"
        assert "timestamp" in serialized


class TestValidationRule:
    """Test the ValidationRule class."""
    
    def test_validation_rule_creation(self):
        """Test creating validation rules."""
        def required_validator(value):
            return value is not None and value != ""
        
        rule = ValidationRule(
            name="required_field",
            validator=required_validator,
            message="Field is required",
            level=ValidationLevel.STRICT
        )
        
        assert rule.name == "required_field"
        assert rule.message == "Field is required"
        assert rule.level == ValidationLevel.STRICT
    
    def test_validation_rule_execution(self):
        """Test validation rule execution."""
        def length_validator(value):
            return len(str(value)) <= 10
        
        rule = ValidationRule(
            name="max_length",
            validator=length_validator,
            message="Value too long",
            level=ValidationLevel.STRICT
        )
        
        # Valid value
        result = rule.validate("short")
        assert result.is_valid is True
        
        # Invalid value
        result = rule.validate("very_long_value_that_exceeds_limit")
        assert result.is_valid is False
        assert result.message == "Value too long"


class TestFieldValidator:
    """Test the FieldValidator class."""
    
    def test_field_validator_creation(self):
        """Test creating field validators."""
        validator = FieldValidator("user_id")
        assert validator.field_name == "user_id"
        assert len(validator.rules) == 0
    
    def test_add_validation_rule(self):
        """Test adding validation rules to field validators."""
        validator = FieldValidator("email")
        
        def email_validator(value):
            return "@" in str(value)
        
        rule = ValidationRule("email_format", email_validator, "Invalid email format")
        validator.add_rule(rule)
        
        assert len(validator.rules) == 1
        assert validator.rules[0].name == "email_format"
    
    def test_field_validation(self):
        """Test field validation with multiple rules."""
        validator = FieldValidator("password")
        
        # Add length rule
        def length_validator(value):
            return len(str(value)) >= 8
        
        # Add complexity rule
        def complexity_validator(value):
            value_str = str(value)
            return any(c.isupper() for c in value_str) and any(c.islower() for c in value_str)
        
        validator.add_rule(ValidationRule("min_length", length_validator, "Too short"))
        validator.add_rule(ValidationRule("complexity", complexity_validator, "Too simple"))
        
        # Valid password
        result = validator.validate("SecurePass123")
        assert result.is_valid is True
        
        # Invalid password - too short
        result = validator.validate("weak")
        assert result.is_valid is False
        assert len(result.errors) == 1
        
        # Invalid password - too simple
        result = validator.validate("password")
        assert result.is_valid is False
        assert len(result.errors) == 1


class TestSchemaValidator:
    """Test the SchemaValidator class."""
    
    def test_schema_validator_creation(self):
        """Test creating schema validators."""
        schema = {
            "required_fields": ["id", "timestamp"],
            "field_types": {
                "id": "string",
                "timestamp": "datetime",
                "value": "number"
            }
        }
        
        validator = SchemaValidator(schema)
        assert validator.schema == schema
        assert len(validator.field_validators) == 0
    
    def test_schema_validation(self):
        """Test schema validation."""
        schema = {
            "required_fields": ["id", "timestamp"],
            "field_types": {
                "id": "string",
                "timestamp": "string",
                "value": "number"
            }
        }
        
        validator = SchemaValidator(schema)
        
        # Valid data
        data = {
            "id": "test_123",
            "timestamp": "2024-01-01T00:00:00Z",
            "value": 42.5
        }
        
        result = validator.validate(data)
        assert result.is_valid is True
        
        # Missing required field
        data = {
            "id": "test_123",
            "value": 42.5
        }
        
        result = validator.validate(data)
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "timestamp" in str(result.errors[0].message)


class TestInputValidationGateway:
    """Test the InputValidationGateway class."""
    
    def test_gateway_initialization(self):
        """Test gateway initialization."""
        gateway = InputValidationGateway()
        assert gateway is not None
        assert hasattr(gateway, 'validators')
    
    def test_register_field_validator(self):
        """Test registering field validators."""
        gateway = InputValidationGateway()
        
        validator = FieldValidator("user_id")
        def required_validator(value):
            return value is not None and value != ""
        
        validator.add_rule(ValidationRule("required", required_validator, "Required field"))
        gateway.register_field_validator(validator)
        
        assert "user_id" in gateway.validators
        assert gateway.validators["user_id"] == validator
    
    def test_register_schema_validator(self):
        """Test registering schema validators."""
        gateway = InputValidationGateway()
        
        schema = {"required_fields": ["id"]}
        validator = SchemaValidator(schema)
        gateway.register_schema_validator(validator)
        
        assert gateway.schema_validator == validator
    
    def test_validate_telemetry_record(self, sample_telemetry_data):
        """Test validating telemetry records."""
        gateway = InputValidationGateway()
        
        # Set up basic validation
        id_validator = FieldValidator("id")
        id_validator.add_rule(ValidationRule("required", lambda v: v is not None, "ID required"))
        gateway.register_field_validator(id_validator)
        
        timestamp_validator = FieldValidator("timestamp")
        timestamp_validator.add_rule(ValidationRule("required", lambda v: v is not None, "Timestamp required"))
        gateway.register_field_validator(timestamp_validator)
        
        # Valid data
        result = gateway.validate_telemetry_record(sample_telemetry_data)
        assert result.is_valid is True
        
        # Invalid data - missing ID
        invalid_data = sample_telemetry_data.copy()
        del invalid_data["id"]
        
        result = gateway.validate_telemetry_record(invalid_data)
        assert result.is_valid is False
        assert len(result.errors) == 1
    
    def test_validate_with_custom_level(self, sample_telemetry_data):
        """Test validation with custom validation levels."""
        gateway = InputValidationGateway()
        
        # Set up validators
        id_validator = FieldValidator("id")
        id_validator.add_rule(ValidationRule("required", lambda v: v is not None, "ID required"))
        gateway.register_field_validator(id_validator)
        
        # Test with different levels
        result = gateway.validate_telemetry_record(sample_telemetry_data, level=ValidationLevel.STRICT)
        assert result.level == ValidationLevel.STRICT
        
        result = gateway.validate_telemetry_record(sample_telemetry_data, level=ValidationLevel.LENIENT)
        assert result.level == ValidationLevel.LENIENT
    
    def test_validation_error_aggregation(self):
        """Test that validation errors are properly aggregated."""
        gateway = InputValidationGateway()
        
        # Set up multiple validators
        id_validator = FieldValidator("id")
        id_validator.add_rule(ValidationRule("required", lambda v: v is not None, "ID required"))
        id_validator.add_rule(ValidationRule("format", lambda v: isinstance(v, str), "ID must be string"))
        
        email_validator = FieldValidator("email")
        email_validator.add_rule(ValidationRule("format", lambda v: "@" in str(v), "Invalid email"))
        
        gateway.register_field_validator(id_validator)
        gateway.register_field_validator(email_validator)
        
        # Data with multiple validation issues
        invalid_data = {
            "email": "invalid-email"
        }
        
        result = gateway.validate_telemetry_record(invalid_data)
        assert result.is_valid is False
        assert len(result.errors) >= 2  # At least ID missing and email format
    
    def test_validation_warnings(self):
        """Test validation warnings for non-critical issues."""
        gateway = InputValidationGateway()
        
        # Set up validator with warning-level rule
        field_validator = FieldValidator("optional_field")
        field_validator.add_rule(ValidationRule(
            "deprecated", 
            lambda v: True,  # Always passes
            "Field is deprecated",
            level=ValidationLevel.WARNING
        ))
        
        gateway.register_field_validator(field_validator)
        
        data = {"optional_field": "old_value"}
        result = gateway.validate_telemetry_record(data)
        
        assert result.is_valid is True  # Warnings don't fail validation
        assert len(result.warnings) == 1
        assert "deprecated" in result.warnings[0].code
    
    def test_validation_performance(self, large_dataset):
        """Test validation performance with large datasets."""
        gateway = InputValidationGateway()
        
        # Set up simple validator
        id_validator = FieldValidator("id")
        id_validator.add_rule(ValidationRule("required", lambda v: v is not None, "ID required"))
        gateway.register_field_validator(id_validator)
        
        # Validate large dataset
        start_time = datetime.now()
        
        for record in large_dataset[:1000]:  # Test with subset
            result = gateway.validate_telemetry_record(record)
            assert result.is_valid is True
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert duration < 10.0  # 10 seconds for 1000 records
    
    def test_validation_edge_cases(self, edge_case_data):
        """Test validation with edge case data."""
        gateway = InputValidationGateway()
        
        # Set up basic validators
        id_validator = FieldValidator("id")
        id_validator.add_rule(ValidationRule("required", lambda v: v is not None, "ID required"))
        id_validator.add_rule(ValidationRule("non_empty", lambda v: str(v).strip() != "", "ID cannot be empty"))
        
        timestamp_validator = FieldValidator("timestamp")
        timestamp_validator.add_rule(ValidationRule("required", lambda v: v is not None, "Timestamp required"))
        
        gateway.register_field_validator(id_validator)
        gateway.register_field_validator(timestamp_validator)
        
        # Test edge cases
        for data in edge_case_data:
            result = gateway.validate_telemetry_record(data)
            # Edge cases should either pass validation or fail with appropriate errors
            assert isinstance(result, ValidationResult)
    
    def test_validation_with_malformed_data(self, malformed_data_samples):
        """Test validation with malformed data samples."""
        gateway = InputValidationGateway()
        
        # Set up basic validators
        id_validator = FieldValidator("id")
        id_validator.add_rule(ValidationRule("required", lambda v: v is not None, "ID required"))
        gateway.register_field_validator(id_validator)
        
        # Test malformed data
        for data in malformed_data_samples:
            try:
                result = gateway.validate_telemetry_record(data)
                # Should handle gracefully without crashing
                assert isinstance(result, ValidationResult)
            except Exception as e:
                # If validation fails, it should be a controlled failure
                assert "validation" in str(e).lower() or "invalid" in str(e).lower()


class TestValidationIntegration:
    """Test integration between validation components."""
    
    def test_full_validation_pipeline(self, sample_telemetry_data):
        """Test complete validation pipeline."""
        gateway = InputValidationGateway()
        
        # Set up comprehensive validation
        id_validator = FieldValidator("id")
        id_validator.add_rule(ValidationRule("required", lambda v: v is not None, "ID required"))
        id_validator.add_rule(ValidationRule("format", lambda v: isinstance(v, str), "ID must be string"))
        
        timestamp_validator = FieldValidator("timestamp")
        timestamp_validator.add_rule(ValidationRule("required", lambda v: v is not None, "Timestamp required"))
        timestamp_validator.add_rule(ValidationRule("format", lambda v: "T" in str(v), "Invalid timestamp format"))
        
        data_validator = FieldValidator("data")
        data_validator.add_rule(ValidationRule("required", lambda v: v is not None, "Data required"))
        data_validator.add_rule(ValidationRule("type", lambda v: isinstance(v, dict), "Data must be dict"))
        
        gateway.register_field_validator(id_validator)
        gateway.register_field_validator(timestamp_validator)
        gateway.register_field_validator(data_validator)
        
        # Valid data
        result = gateway.validate_telemetry_record(sample_telemetry_data)
        assert result.is_valid is True
        
        # Invalid data
        invalid_data = sample_telemetry_data.copy()
        invalid_data["id"] = None
        invalid_data["timestamp"] = "invalid-timestamp"
        invalid_data["data"] = "not-a-dict"
        
        result = gateway.validate_telemetry_record(invalid_data)
        assert result.is_valid is False
        assert len(result.errors) >= 3  # Multiple validation failures
    
    def test_validation_result_consistency(self):
        """Test that validation results are consistent across multiple runs."""
        gateway = InputValidationGateway()
        
        # Set up validator
        validator = FieldValidator("test_field")
        validator.add_rule(ValidationRule("even_number", lambda v: v % 2 == 0, "Must be even"))
        gateway.register_field_validator(validator)
        
        data = {"test_field": 4}
        
        # Run validation multiple times
        results = []
        for _ in range(10):
            result = gateway.validate_telemetry_record(data)
            results.append(result)
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result.is_valid == first_result.is_valid
            assert len(result.errors) == len(first_result.errors)
            assert len(result.warnings) == len(first_result.warnings)


if __name__ == '__main__':
    pytest.main([__file__])
