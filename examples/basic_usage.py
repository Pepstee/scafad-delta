"""
Basic usage examples for Scafad Layer1 system.

This example demonstrates the fundamental usage patterns for
data processing, validation, and privacy protection.
"""

import sys
import os
import json
from datetime import datetime

# Add the project root to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.layer1_core import Layer1Processor
from core.layer1_validation import DataValidator
from core.layer1_privacy import PrivacyFilter


def basic_data_processing_example():
    """Basic example of data processing through the Layer1 system."""
    
    print("=== Basic Data Processing Example ===\n")
    
    # Initialize the main processor
    processor = Layer1Processor()
    
    # Sample data to process
    sample_data = {
        "id": "user_123",
        "name": "John Doe",
        "email": "john.doe@example.com",
        "phone": "+1-555-123-4567",
        "preferences": {
            "theme": "dark",
            "notifications": True
        },
        "metadata": {
            "source": "user_registration",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    }
    
    print("Input Data:")
    print(json.dumps(sample_data, indent=2))
    print("\n" + "="*50 + "\n")
    
    # Process data with default settings
    result = processor.process(sample_data)
    
    print("Processed Data:")
    print(json.dumps(result, indent=2))
    print("\n" + "="*50 + "\n")
    
    return result


def privacy_protection_example():
    """Example of privacy protection and PII handling."""
    
    print("=== Privacy Protection Example ===\n")
    
    # Initialize privacy filter
    privacy_filter = PrivacyFilter(
        privacy_level="strict",
        detect_pii=True,
        anonymize_data=True
    )
    
    # Sensitive data containing PII
    sensitive_data = {
        "user_id": "user_456",
        "full_name": "Jane Smith",
        "email_address": "jane.smith@company.com",
        "social_security": "123-45-6789",
        "credit_card": "4111-1111-1111-1111",
        "address": "123 Main St, Anytown, USA 12345",
        "phone": "(555) 987-6543"
    }
    
    print("Original Sensitive Data:")
    print(json.dumps(sensitive_data, indent=2))
    print("\n" + "="*50 + "\n")
    
    # Apply privacy protection
    protected_data = privacy_filter.process(sensitive_data)
    
    print("Privacy-Protected Data:")
    print(json.dumps(protected_data, indent=2))
    print("\n" + "="*50 + "\n")
    
    return protected_data


def validation_example():
    """Example of data validation and quality checks."""
    
    print("=== Data Validation Example ===\n")
    
    # Initialize validator
    validator = DataValidator()
    
    # Add validation rules
    from core.layer1_validation import ValidationRule, ValidationError
    
    def validate_required_fields(data):
        """Validate that required fields are present."""
        required_fields = ['id', 'name', 'email']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValidationError(f"Missing required fields: {missing_fields}")
        return True
    
    def validate_email_format(data):
        """Validate email format."""
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if 'email' in data and not re.match(email_pattern, data['email']):
            raise ValidationError("Invalid email format")
        return True
    
    def validate_phone_format(data):
        """Validate phone number format."""
        import re
        phone_pattern = r'^\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})$'
        if 'phone' in data and not re.match(phone_pattern, data['phone']):
            raise ValidationError("Invalid phone number format")
        return True
    
    # Add validation rules
    validator.add_rule(ValidationRule("required_fields", validate_required_fields))
    validator.add_rule(ValidationRule("email_format", validate_email_format))
    validator.add_rule(ValidationRule("phone_format", validate_phone_format))
    
    # Test data for validation
    test_cases = [
        {
            "id": "user_001",
            "name": "Valid User",
            "email": "valid@example.com",
            "phone": "+1-555-123-4567"
        },
        {
            "name": "Missing ID User",
            "email": "invalid-email",
            "phone": "invalid-phone"
        },
        {
            "id": "user_003",
            "name": "Valid User",
            "email": "valid@example.com"
        }
    ]
    
    print("Validation Test Cases:\n")
    
    for i, test_data in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"Input: {json.dumps(test_data, indent=2)}")
        
        try:
            validation_result = validator.validate(test_data)
            if validation_result.is_valid:
                print("✅ Validation PASSED")
            else:
                print("❌ Validation FAILED:")
                for error in validation_result.errors:
                    print(f"  - {error.message}")
        except Exception as e:
            print(f"❌ Validation ERROR: {e}")
        
        print("\n" + "-"*30 + "\n")
    
    return validator


def custom_processing_example():
    """Example of custom processing pipeline configuration."""
    
    print("=== Custom Processing Pipeline Example ===\n")
    
    # Initialize processor with custom configuration
    processor = Layer1Processor()
    
    # Custom processing configuration
    custom_config = {
        "preserve_anomalies": True,
        "privacy_level": "high",
        "hash_sensitive_fields": True,
        "sanitize_content": True,
        "custom_stages": [
            "data_enrichment",
            "quality_assessment",
            "compliance_check"
        ]
    }
    
    # Sample data for custom processing
    data_for_custom_processing = {
        "id": "order_789",
        "customer_id": "cust_456",
        "order_details": "Product purchase with special characters <script>alert('test')</script>",
        "amount": 299.99,
        "payment_method": "credit_card",
        "anomalies": [
            {"type": "unusual_amount", "score": 0.85, "description": "Higher than average order value"}
        ]
    }
    
    print("Input Data for Custom Processing:")
    print(json.dumps(data_for_custom_processing, indent=2))
    print("\nCustom Configuration:")
    print(json.dumps(custom_config, indent=2))
    print("\n" + "="*50 + "\n")
    
    # Process with custom configuration
    custom_result = processor.process(data_for_custom_processing, config=custom_config)
    
    print("Custom Processing Result:")
    print(json.dumps(custom_result, indent=2))
    print("\n" + "="*50 + "\n")
    
    return custom_result


def batch_processing_example():
    """Example of batch processing multiple records."""
    
    print("=== Batch Processing Example ===\n")
    
    # Initialize processor
    processor = Layer1Processor()
    
    # Batch of data records
    batch_data = [
        {
            "id": "record_001",
            "content": "First record content",
            "metadata": {"source": "batch_1", "priority": "high"}
        },
        {
            "id": "record_002",
            "content": "Second record content",
            "metadata": {"source": "batch_1", "priority": "medium"}
        },
        {
            "id": "record_003",
            "content": "Third record content",
            "metadata": {"source": "batch_1", "priority": "low"}
        }
    ]
    
    print("Batch Input Data:")
    for i, record in enumerate(batch_data, 1):
        print(f"Record {i}: {json.dumps(record, indent=2)}")
        print()
    
    print("="*50 + "\n")
    
    # Process batch
    batch_results = []
    for record in batch_data:
        result = processor.process(record)
        batch_results.append(result)
    
    print("Batch Processing Results:")
    for i, result in enumerate(batch_results, 1):
        print(f"Record {i} Result: {json.dumps(result, indent=2)}")
        print()
    
    print("="*50 + "\n")
    
    return batch_results


def main():
    """Run all examples."""
    
    print("Scafad Layer1 System - Usage Examples")
    print("=" * 50)
    print()
    
    try:
        # Run basic examples
        basic_result = basic_data_processing_example()
        privacy_result = privacy_protection_example()
        validator = validation_example()
        custom_result = custom_processing_example()
        batch_results = batch_processing_example()
        
        print("=== Summary ===")
        print("✅ Basic data processing: Completed")
        print("✅ Privacy protection: Completed")
        print("✅ Data validation: Completed")
        print("✅ Custom processing: Completed")
        print("✅ Batch processing: Completed")
        print("\nAll examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
