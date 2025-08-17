"""
Comprehensive unit tests for layer1_sanitization.py

Tests data sanitization, cleaning, normalization, and quality assurance
with extensive coverage of data transformation and validation.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
import json
import re

# Add the core directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'core'))

from layer1_sanitization import (
    SanitizationProcessor, SanitizationResult, SanitizationRule,
    DataCleaner, NormalizationEngine, QualityAssurance
)


class TestSanitizationRule:
    """Test the SanitizationRule class."""
    
    def test_rule_creation(self):
        """Test creating sanitization rules."""
        rule = SanitizationRule(
            name="remove_html_tags",
            description="Remove HTML tags from text fields",
            field_pattern=".*_text$",
            rule_type="text_cleaning",
            function=lambda x: re.sub(r'<[^>]+>', '', str(x)),
            priority=1
        )
        
        assert rule.name == "remove_html_tags"
        assert rule.description == "Remove HTML tags from text fields"
        assert rule.field_pattern == ".*_text$"
        assert rule.rule_type == "text_cleaning"
        assert rule.priority == 1
        assert rule.function is not None
    
    def test_rule_matching(self):
        """Test rule field pattern matching."""
        rule = SanitizationRule(
            name="test_rule",
            description="Test rule",
            field_pattern="user_.*",
            rule_type="test",
            function=lambda x: x,
            priority=1
        )
        
        # Test matching fields
        assert rule.matches_field("user_id") is True
        assert rule.matches_field("user_name") is True
        assert rule.matches_field("user_email") is True
        
        # Test non-matching fields
        assert rule.matches_field("id") is False
        assert rule.matches_field("name") is False
        assert rule.matches_field("email") is False
    
    def test_rule_execution(self):
        """Test rule execution."""
        def uppercase_rule(value):
            return str(value).upper()
        
        rule = SanitizationRule(
            name="uppercase",
            description="Convert to uppercase",
            field_pattern=".*",
            rule_type="transformation",
            function=uppercase_rule,
            priority=1
        )
        
        # Test execution
        result = rule.execute("hello world")
        assert result == "HELLO WORLD"
        
        # Test with different input types
        result = rule.execute(123)
        assert result == "123"
    
    def test_rule_serialization(self):
        """Test rule serialization."""
        rule = SanitizationRule(
            name="serialization_test",
            description="Test serialization",
            field_pattern="test_.*",
            rule_type="test",
            function=lambda x: x,
            priority=5
        )
        
        serialized = rule.to_dict()
        assert serialized["name"] == "serialization_test"
        assert serialized["description"] == "Test serialization"
        assert serialized["field_pattern"] == "test_.*"
        assert serialized["rule_type"] == "test"
        assert serialized["priority"] == 5


class TestDataCleaner:
    """Test the DataCleaner class."""
    
    def test_cleaner_initialization(self):
        """Test data cleaner initialization."""
        cleaner = DataCleaner()
        assert cleaner is not None
        assert hasattr(cleaner, 'rules')
        assert len(cleaner.rules) == 0
    
    def test_add_sanitization_rule(self):
        """Test adding sanitization rules."""
        cleaner = DataCleaner()
        
        rule = SanitizationRule(
            name="test_rule",
            description="Test rule",
            field_pattern=".*",
            rule_type="test",
            function=lambda x: x,
            priority=1
        )
        
        cleaner.add_rule(rule)
        assert len(cleaner.rules) == 1
        assert cleaner.rules[0] == rule
    
    def test_rule_priority_ordering(self):
        """Test that rules are ordered by priority."""
        cleaner = DataCleaner()
        
        # Add rules with different priorities
        high_priority = SanitizationRule(
            name="high",
            description="High priority",
            field_pattern=".*",
            rule_type="test",
            function=lambda x: x,
            priority=1
        )
        
        low_priority = SanitizationRule(
            name="low",
            description="Low priority",
            field_pattern=".*",
            rule_type="test",
            function=lambda x: x,
            priority=10
        )
        
        medium_priority = SanitizationRule(
            name="medium",
            description="Medium priority",
            field_pattern=".*",
            rule_type="test",
            function=lambda x: x,
            priority=5
        )
        
        # Add in random order
        cleaner.add_rule(medium_priority)
        cleaner.add_rule(high_priority)
        cleaner.add_rule(low_priority)
        
        # Check ordering
        priorities = [rule.priority for rule in cleaner.rules]
        assert priorities == [1, 5, 10]  # Should be sorted by priority
    
    def test_field_specific_cleaning(self):
        """Test cleaning specific fields."""
        cleaner = DataCleaner()
        
        # Add field-specific rules
        def clean_email(value):
            return str(value).lower().strip()
        
        def clean_name(value):
            return str(value).title().strip()
        
        email_rule = SanitizationRule(
            name="clean_email",
            description="Clean email addresses",
            field_pattern=".*email.*",
            rule_type="cleaning",
            function=clean_email,
            priority=1
        )
        
        name_rule = SanitizationRule(
            name="clean_name",
            description="Clean names",
            field_pattern=".*name.*",
            rule_type="cleaning",
            function=clean_name,
            priority=1
        )
        
        cleaner.add_rule(email_rule)
        cleaner.add_rule(name_rule)
        
        # Test data
        test_data = {
            "user_email": "  USER@EXAMPLE.COM  ",
            "user_name": "  john doe  ",
            "other_field": "unchanged"
        }
        
        cleaned_data = cleaner.clean_data(test_data)
        
        assert cleaned_data["user_email"] == "user@example.com"
        assert cleaned_data["user_name"] == "John Doe"
        assert cleaned_data["other_field"] == "unchanged"
    
    def test_data_type_cleaning(self):
        """Test cleaning different data types."""
        cleaner = DataCleaner()
        
        # Add type-specific rules
        def clean_string(value):
            if isinstance(value, str):
                return value.strip()
            return value
        
        def clean_number(value):
            if isinstance(value, (int, float)):
                return abs(value)
            return value
        
        string_rule = SanitizationRule(
            name="clean_strings",
            description="Clean string values",
            field_pattern=".*",
            rule_type="type_cleaning",
            function=clean_string,
            priority=1
        )
        
        number_rule = SanitizationRule(
            name="clean_numbers",
            description="Clean numeric values",
            field_pattern=".*",
            rule_type="type_cleaning",
            function=clean_number,
            priority=2
        )
        
        cleaner.add_rule(string_rule)
        cleaner.add_rule(number_rule)
        
        # Test data with mixed types
        test_data = {
            "text": "  hello  ",
            "number": -42,
            "mixed": "  test  "
        }
        
        cleaned_data = cleaner.clean_data(test_data)
        
        assert cleaned_data["text"] == "hello"
        assert cleaned_data["number"] == 42
        assert cleaned_data["mixed"] == "test"
    
    def test_nested_data_cleaning(self):
        """Test cleaning nested data structures."""
        cleaner = DataCleaner()
        
        # Add simple cleaning rule
        def clean_strings(value):
            if isinstance(value, str):
                return value.strip()
            return value
        
        rule = SanitizationRule(
            name="clean_strings",
            description="Clean all string values",
            field_pattern=".*",
            rule_type="cleaning",
            function=clean_strings,
            priority=1
        )
        
        cleaner.add_rule(rule)
        
        # Test nested data
        test_data = {
            "user": {
                "name": "  john  ",
                "profile": {
                    "bio": "  developer  ",
                    "tags": ["  python  ", "  data  "]
                }
            },
            "settings": {
                "theme": "  dark  "
            }
        }
        
        cleaned_data = cleaner.clean_data(test_data)
        
        # Check nested cleaning
        assert cleaned_data["user"]["name"] == "john"
        assert cleaned_data["user"]["profile"]["bio"] == "developer"
        assert cleaned_data["user"]["profile"]["tags"] == ["python", "data"]
        assert cleaned_data["settings"]["theme"] == "dark"
    
    def test_cleaning_error_handling(self):
        """Test error handling during cleaning."""
        cleaner = DataCleaner()
        
        # Add rule that might fail
        def risky_cleaning(value):
            if value == "fail":
                raise ValueError("Cleaning failed")
            return value.upper()
        
        rule = SanitizationRule(
            name="risky",
            description="Risky cleaning rule",
            field_pattern=".*",
            rule_type="test",
            function=risky_cleaning,
            priority=1
        )
        
        cleaner.add_rule(rule)
        
        # Test with data that should work
        safe_data = {"field": "hello"}
        cleaned_data = cleaner.clean_data(safe_data)
        assert cleaned_data["field"] == "HELLO"
        
        # Test with data that should fail
        risky_data = {"field": "fail"}
        
        # Should handle errors gracefully
        try:
            cleaned_data = cleaner.clean_data(risky_data)
            # If it doesn't fail, the field should be unchanged
            assert cleaned_data["field"] == "fail"
        except Exception as e:
            # If it fails, should be a controlled failure
            assert "cleaning" in str(e).lower() or "failed" in str(e).lower()


class TestNormalizationEngine:
    """Test the NormalizationEngine class."""
    
    def test_engine_initialization(self):
        """Test normalization engine initialization."""
        engine = NormalizationEngine()
        assert engine is not None
        assert hasattr(engine, 'normalizers')
        assert hasattr(engine, 'standard_formats')
    
    def test_add_normalizer(self):
        """Test adding normalization functions."""
        engine = NormalizationEngine()
        
        def date_normalizer(value):
            # Simple date normalization
            if isinstance(value, str) and "T" in value:
                return value.split("T")[0]
            return value
        
        engine.add_normalizer("date", date_normalizer)
        assert "date" in engine.normalizers
        assert engine.normalizers["date"] == date_normalizer
    
    def test_date_normalization(self):
        """Test date and timestamp normalization."""
        engine = NormalizationEngine()
        
        def normalize_timestamp(value):
            if isinstance(value, str):
                # Convert various formats to ISO format
                if "T" in value and "Z" in value:
                    return value  # Already ISO format
                elif "T" in value:
                    return value + "Z"  # Add timezone
                else:
                    # Assume date only, add time
                    return value + "T00:00:00Z"
            return value
        
        engine.add_normalizer("timestamp", normalize_timestamp)
        
        # Test various date formats
        test_data = {
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01T12:00:00",
            "deleted_at": "2024-01-01T12:00:00Z"
        }
        
        normalized_data = engine.normalize_data(test_data, ["timestamp"])
        
        assert normalized_data["created_at"] == "2024-01-01T00:00:00Z"
        assert normalized_data["updated_at"] == "2024-01-01T12:00:00Z"
        assert normalized_data["deleted_at"] == "2024-01-01T12:00:00Z"
    
    def test_string_normalization(self):
        """Test string normalization."""
        engine = NormalizationEngine()
        
        def normalize_string(value):
            if isinstance(value, str):
                # Convert to lowercase and remove extra whitespace
                return " ".join(value.lower().split())
            return value
        
        engine.add_normalizer("string", normalize_string)
        
        # Test string normalization
        test_data = {
            "title": "  Hello   World  ",
            "description": "  This is a   TEST  ",
            "category": "  TECHNOLOGY  "
        }
        
        normalized_data = engine.normalize_data(test_data, ["string"])
        
        assert normalized_data["title"] == "hello world"
        assert normalized_data["description"] == "this is a test"
        assert normalized_data["category"] == "technology"
    
    def test_number_normalization(self):
        """Test numeric normalization."""
        engine = NormalizationEngine()
        
        def normalize_number(value):
            if isinstance(value, (int, float)):
                # Round to 2 decimal places
                return round(float(value), 2)
            return value
        
        engine.add_normalizer("number", normalize_number)
        
        # Test number normalization
        test_data = {
            "price": 19.9999,
            "quantity": 5,
            "rating": 4.5678,
            "text": "not a number"
        }
        
        normalized_data = engine.normalize_data(test_data, ["number"])
        
        assert normalized_data["price"] == 20.0
        assert normalized_data["quantity"] == 5.0
        assert normalized_data["rating"] == 4.57
        assert normalized_data["text"] == "not a number"
    
    def test_enum_normalization(self):
        """Test enum and categorical normalization."""
        engine = NormalizationEngine()
        
        # Define standard values
        status_mapping = {
            "active": "ACTIVE",
            "inactive": "INACTIVE",
            "pending": "PENDING",
            "ACTIVE": "ACTIVE",
            "Inactive": "INACTIVE",
            "PENDING": "PENDING"
        }
        
        def normalize_status(value):
            return status_mapping.get(str(value).lower(), "UNKNOWN")
        
        engine.add_normalizer("status", normalize_status)
        
        # Test status normalization
        test_data = {
            "user_status": "active",
            "order_status": "Inactive",
            "payment_status": "PENDING",
            "unknown_status": "invalid"
        }
        
        normalized_data = engine.normalize_data(test_data, ["status"])
        
        assert normalized_data["user_status"] == "ACTIVE"
        assert normalized_data["order_status"] == "INACTIVE"
        assert normalized_data["payment_status"] == "PENDING"
        assert normalized_data["unknown_status"] == "UNKNOWN"
    
    def test_complex_normalization(self):
        """Test complex normalization scenarios."""
        engine = NormalizationEngine()
        
        # Add multiple normalizers
        def normalize_phone(value):
            if isinstance(value, str):
                # Remove all non-digits
                digits = re.sub(r'\D', '', value)
                if len(digits) == 10:
                    return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
                elif len(digits) == 11 and digits[0] == '1':
                    return f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
                return digits
            return value
        
        def normalize_address(value):
            if isinstance(value, str):
                # Standardize address format
                return value.upper().replace("STREET", "ST").replace("AVENUE", "AVE")
            return value
        
        engine.add_normalizer("phone", normalize_phone)
        engine.add_normalizer("address", normalize_address)
        
        # Test complex data
        test_data = {
            "contact": {
                "phone": "555-123-4567",
                "address": "123 Main Street, City, State"
            },
            "emergency_contact": {
                "phone": "15551234567",
                "address": "456 Oak Avenue, Town, State"
            }
        }
        
        normalized_data = engine.normalize_data(test_data, ["phone", "address"])
        
        # Check phone normalization
        assert normalized_data["contact"]["phone"] == "(555) 123-4567"
        assert normalized_data["emergency_contact"]["phone"] == "(555) 123-4567"
        
        # Check address normalization
        assert "ST" in normalized_data["contact"]["address"]
        assert "AVE" in normalized_data["emergency_contact"]["address"]


class TestQualityAssurance:
    """Test the QualityAssurance class."""
    
    def test_qa_initialization(self):
        """Test quality assurance initialization."""
        qa = QualityAssurance()
        assert qa is not None
        assert hasattr(qa, 'quality_checks')
        assert hasattr(qa, 'thresholds')
    
    def test_add_quality_check(self):
        """Test adding quality checks."""
        qa = QualityAssurance()
        
        def completeness_check(data):
            # Check if required fields are present
            required_fields = ["id", "timestamp"]
            missing_fields = [field for field in required_fields if field not in data]
            return len(missing_fields) == 0, f"Missing fields: {missing_fields}"
        
        qa.add_quality_check("completeness", completeness_check)
        assert "completeness" in qa.quality_checks
        assert qa.quality_checks["completeness"] == completeness_check
    
    def test_completeness_check(self):
        """Test data completeness checking."""
        qa = QualityAssurance()
        
        def completeness_check(data):
            required_fields = ["id", "timestamp", "data"]
            missing_fields = [field for field in required_fields if field not in data]
            return len(missing_fields) == 0, f"Missing fields: {missing_fields}"
        
        qa.add_quality_check("completeness", completeness_check)
        
        # Test complete data
        complete_data = {
            "id": "test123",
            "timestamp": "2024-01-01T00:00:00Z",
            "data": {"value": 42}
        }
        
        result = qa.run_quality_check("completeness", complete_data)
        assert result[0] is True  # Check passed
        assert "Missing fields" not in result[1]
        
        # Test incomplete data
        incomplete_data = {
            "id": "test123",
            "timestamp": "2024-01-01T00:00:00Z"
            # Missing "data" field
        }
        
        result = qa.run_quality_check("completeness", incomplete_data)
        assert result[0] is False  # Check failed
        assert "Missing fields" in result[1]
        assert "data" in result[1]
    
    def test_consistency_check(self):
        """Test data consistency checking."""
        qa = QualityAssurance()
        
        def consistency_check(data):
            # Check if data types are consistent
            issues = []
            
            if not isinstance(data.get("id"), str):
                issues.append("ID should be string")
            
            if not isinstance(data.get("timestamp"), str):
                issues.append("Timestamp should be string")
            
            if not isinstance(data.get("data"), dict):
                issues.append("Data should be object")
            
            return len(issues) == 0, f"Consistency issues: {issues}"
        
        qa.add_quality_check("consistency", consistency_check)
        
        # Test consistent data
        consistent_data = {
            "id": "test123",
            "timestamp": "2024-01-01T00:00:00Z",
            "data": {"value": 42}
        }
        
        result = qa.run_quality_check("consistency", consistent_data)
        assert result[0] is True
        
        # Test inconsistent data
        inconsistent_data = {
            "id": 123,  # Should be string
            "timestamp": "2024-01-01T00:00:00Z",
            "data": "not an object"  # Should be dict
        }
        
        result = qa.run_quality_check("consistency", inconsistent_data)
        assert result[0] is False
        assert "ID should be string" in result[1]
        assert "Data should be object" in result[1]
    
    def test_validity_check(self):
        """Test data validity checking."""
        qa = QualityAssurance()
        
        def validity_check(data):
            # Check if values are within valid ranges
            issues = []
            
            # Check ID format
            id_value = data.get("id", "")
            if not re.match(r'^[a-zA-Z0-9_]+$', str(id_value)):
                issues.append("Invalid ID format")
            
            # Check timestamp format
            timestamp = data.get("timestamp", "")
            if not re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$', str(timestamp)):
                issues.append("Invalid timestamp format")
            
            # Check numeric values
            if "data" in data and isinstance(data["data"], dict):
                value = data["data"].get("value")
                if isinstance(value, (int, float)) and (value < 0 or value > 1000):
                    issues.append("Value out of range (0-1000)")
            
            return len(issues) == 0, f"Validity issues: {issues}"
        
        qa.add_quality_check("validity", validity_check)
        
        # Test valid data
        valid_data = {
            "id": "test_123",
            "timestamp": "2024-01-01T00:00:00Z",
            "data": {"value": 500}
        }
        
        result = qa.run_quality_check("validity", valid_data)
        assert result[0] is True
        
        # Test invalid data
        invalid_data = {
            "id": "test@123",  # Invalid characters
            "timestamp": "invalid-timestamp",  # Invalid format
            "data": {"value": 1500}  # Out of range
        }
        
        result = qa.run_quality_check("validity", invalid_data)
        assert result[0] is False
        assert "Invalid ID format" in result[1]
        assert "Invalid timestamp format" in result[1]
        assert "Value out of range" in result[1]
    
    def test_comprehensive_quality_assessment(self):
        """Test comprehensive quality assessment."""
        qa = QualityAssurance()
        
        # Add multiple quality checks
        def completeness_check(data):
            required_fields = ["id", "timestamp", "data"]
            missing_fields = [field for field in required_fields if field not in data]
            return len(missing_fields) == 0, f"Missing fields: {missing_fields}"
        
        def consistency_check(data):
            issues = []
            if not isinstance(data.get("id"), str):
                issues.append("ID should be string")
            if not isinstance(data.get("data"), dict):
                issues.append("Data should be object")
            return len(issues) == 0, f"Consistency issues: {issues}"
        
        def validity_check(data):
            issues = []
            id_value = data.get("id", "")
            if not re.match(r'^[a-zA-Z0-9_]+$', str(id_value)):
                issues.append("Invalid ID format")
            return len(issues) == 0, f"Validity issues: {issues}"
        
        qa.add_quality_check("completeness", completeness_check)
        qa.add_quality_check("consistency", consistency_check)
        qa.add_quality_check("validity", validity_check)
        
        # Test data with mixed quality
        test_data = {
            "id": "test@123",  # Invalid format
            "timestamp": "2024-01-01T00:00:00Z",
            "data": "not an object"  # Wrong type
        }
        
        # Run all quality checks
        results = qa.run_all_quality_checks(test_data)
        
        # Check results
        assert "completeness" in results
        assert "consistency" in results
        assert "validity" in results
        
        # Completeness should pass (all fields present)
        assert results["completeness"][0] is True
        
        # Consistency should fail (data is not object)
        assert results["consistency"][0] is False
        
        # Validity should fail (invalid ID format)
        assert results["validity"][0] is False


class TestSanitizationProcessor:
    """Test the SanitizationProcessor class."""
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = SanitizationProcessor()
        assert processor is not None
        assert hasattr(processor, 'cleaner')
        assert hasattr(processor, 'normalizer')
        assert hasattr(processor, 'quality_assurance')
    
    def test_basic_sanitization(self, sample_telemetry_data):
        """Test basic data sanitization."""
        processor = SanitizationProcessor()
        
        # Add cleaning rule
        def clean_strings(value):
            if isinstance(value, str):
                return value.strip()
            return value
        
        rule = SanitizationRule(
            name="clean_strings",
            description="Clean string values",
            field_pattern=".*",
            rule_type="cleaning",
            function=clean_strings,
            priority=1
        )
        
        processor.cleaner.add_rule(rule)
        
        # Process data
        result = processor.sanitize_data(sample_telemetry_data)
        
        assert result.is_successful is True
        assert result.cleaned_data is not None
        assert result.quality_score >= 0.0
        assert result.quality_score <= 1.0
    
    def test_comprehensive_sanitization(self, sample_telemetry_data):
        """Test comprehensive data sanitization."""
        processor = SanitizationProcessor()
        
        # Add cleaning rules
        def clean_strings(value):
            if isinstance(value, str):
                return value.strip()
            return value
        
        def clean_timestamps(value):
            if isinstance(value, str) and "T" in value:
                return value.split("T")[0] + "T00:00:00Z"
            return value
        
        # Add cleaning rule
        string_rule = SanitizationRule(
            name="clean_strings",
            description="Clean string values",
            field_pattern=".*",
            rule_type="cleaning",
            function=clean_strings,
            priority=1
        )
        
        processor.cleaner.add_rule(string_rule)
        
        # Add normalizer
        processor.normalizer.add_normalizer("timestamp", clean_timestamps)
        
        # Add quality check
        def completeness_check(data):
            required_fields = ["id", "timestamp", "data"]
            missing_fields = [field for field in required_fields if field not in data]
            return len(missing_fields) == 0, f"Missing fields: {missing_fields}"
        
        processor.quality_assurance.add_quality_check("completeness", completeness_check)
        
        # Process data
        result = processor.sanitize_data(sample_telemetry_data)
        
        # Verify results
        assert result.is_successful is True
        assert result.cleaned_data is not None
        assert result.normalized_data is not None
        assert result.quality_metrics is not None
        
        # Check that data was cleaned and normalized
        assert "completeness" in result.quality_metrics
    
    def test_sanitization_with_errors(self):
        """Test sanitization error handling."""
        processor = SanitizationProcessor()
        
        # Add problematic rule
        def error_rule(value):
            if value == "error":
                raise ValueError("Processing error")
            return value
        
        rule = SanitizationRule(
            name="error_rule",
            description="Rule that may cause errors",
            field_pattern=".*",
            rule_type="test",
            function=error_rule,
            priority=1
        )
        
        processor.cleaner.add_rule(rule)
        
        # Test with data that should work
        safe_data = {"field": "safe_value"}
        result = processor.sanitize_data(safe_data)
        assert result.is_successful is True
        
        # Test with data that should cause errors
        error_data = {"field": "error"}
        
        # Should handle errors gracefully
        try:
            result = processor.sanitize_data(error_data)
            # If it doesn't fail, should indicate issues
            assert result.is_successful is False or result.quality_score < 1.0
        except Exception as e:
            # If it fails, should be a controlled failure
            assert "sanitization" in str(e).lower() or "processing" in str(e).lower()
    
    def test_sanitization_performance(self, large_dataset):
        """Test sanitization performance with large datasets."""
        processor = SanitizationProcessor()
        
        # Add simple cleaning rule
        def simple_clean(value):
            if isinstance(value, str):
                return value.strip()
            return value
        
        rule = SanitizationRule(
            name="simple_clean",
            description="Simple string cleaning",
            field_pattern=".*",
            rule_type="cleaning",
            function=simple_clean,
            priority=1
        )
        
        processor.cleaner.add_rule(rule)
        
        # Test with subset of large dataset
        test_data = large_dataset[:1000]
        
        start_time = datetime.now()
        
        for record in test_data:
            result = processor.sanitize_data(record)
            assert result.is_successful is True
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time
        assert duration < 60.0  # 60 seconds for 1000 records
    
    def test_sanitization_result_serialization(self, sample_telemetry_data):
        """Test sanitization result serialization."""
        processor = SanitizationProcessor()
        
        # Process data
        result = processor.sanitize_data(sample_telemetry_data)
        
        # Serialize result
        serialized = result.to_dict()
        
        # Check serialization
        assert "is_successful" in serialized
        assert "cleaned_data" in serialized
        assert "normalized_data" in serialized
        assert "quality_metrics" in serialized
        assert "processing_time" in serialized
        
        # Verify data integrity
        assert serialized["is_successful"] == result.is_successful
        assert serialized["cleaned_data"] == result.cleaned_data


class TestSanitizationIntegration:
    """Test integration between sanitization components."""
    
    def test_full_sanitization_pipeline(self, sample_telemetry_data):
        """Test complete sanitization pipeline."""
        processor = SanitizationProcessor()
        
        # Set up comprehensive cleaning
        def clean_strings(value):
            if isinstance(value, str):
                return value.strip()
            return value
        
        def clean_numbers(value):
            if isinstance(value, (int, float)):
                return abs(value)
            return value
        
        # Add cleaning rules
        string_rule = SanitizationRule(
            name="clean_strings",
            description="Clean string values",
            field_pattern=".*",
            rule_type="cleaning",
            function=clean_strings,
            priority=1
        )
        
        number_rule = SanitizationRule(
            name="clean_numbers",
            description="Clean numeric values",
            field_pattern=".*",
            rule_type="cleaning",
            function=clean_numbers,
            priority=2
        )
        
        processor.cleaner.add_rule(string_rule)
        processor.cleaner.add_rule(number_rule)
        
        # Set up normalization
        def normalize_timestamps(value):
            if isinstance(value, str) and "T" in value:
                return value.split("T")[0] + "T00:00:00Z"
            return value
        
        processor.normalizer.add_normalizer("timestamp", normalize_timestamps)
        
        # Set up quality assurance
        def completeness_check(data):
            required_fields = ["id", "timestamp", "data"]
            missing_fields = [field for field in required_fields if field not in data]
            return len(missing_fields) == 0, f"Missing fields: {missing_fields}"
        
        def consistency_check(data):
            issues = []
            if not isinstance(data.get("id"), str):
                issues.append("ID should be string")
            if not isinstance(data.get("data"), dict):
                issues.append("Data should be object")
            return len(issues) == 0, f"Consistency issues: {issues}"
        
        processor.quality_assurance.add_quality_check("completeness", completeness_check)
        processor.quality_assurance.add_quality_check("consistency", consistency_check)
        
        # Process data
        result = processor.sanitize_data(sample_telemetry_data)
        
        # Verify comprehensive results
        assert result.is_successful is True
        assert result.cleaned_data is not None
        assert result.normalized_data is not None
        assert result.quality_metrics is not None
        
        # Check quality metrics
        assert "completeness" in result.quality_metrics
        assert "consistency" in result.quality_metrics
        
        # Verify data was processed
        assert result.cleaned_data != sample_telemetry_data  # Should be different after cleaning
        assert result.normalized_data != sample_telemetry_data  # Should be different after normalization
        
        # Check quality scores
        for metric_name, (passed, message) in result.quality_metrics.items():
            assert isinstance(passed, bool)
            assert isinstance(message, str)


if __name__ == '__main__':
    pytest.main([__file__])
