"""
Comprehensive unit tests for layer1_privacy.py

Tests privacy compliance filtering, redaction policies, and GDPR/CCPA/HIPAA compliance
with extensive coverage of privacy-sensitive data handling.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
import json

# Add the core directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'core'))

from layer1_privacy import (
    PrivacyComplianceFilter, PrivacyAuditTrail, RedactionResult,
    PrivacyLevel, ComplianceFramework, RedactionPolicy, PIIField
)


class TestPrivacyLevel:
    """Test the PrivacyLevel enum."""
    
    def test_privacy_levels(self):
        """Test all privacy levels are defined."""
        levels = list(PrivacyLevel)
        assert len(levels) == 4
        assert PrivacyLevel.MINIMAL in levels
        assert PrivacyLevel.MODERATE in levels
        assert PrivacyLevel.HIGH in levels
        assert PrivacyLevel.MAXIMUM in levels
    
    def test_privacy_level_values(self):
        """Test privacy level string values."""
        assert PrivacyLevel.MINIMAL.value == "minimal"
        assert PrivacyLevel.MODERATE.value == "moderate"
        assert PrivacyLevel.HIGH.value == "high"
        assert PrivacyLevel.MAXIMUM.value == "maximum"


class TestComplianceFramework:
    """Test the ComplianceFramework enum."""
    
    def test_compliance_frameworks(self):
        """Test all compliance frameworks are defined."""
        frameworks = list(ComplianceFramework)
        assert ComplianceFramework.GDPR in frameworks
        assert ComplianceFramework.CCPA in frameworks
        assert ComplianceFramework.HIPAA in frameworks
        assert ComplianceFramework.SOX in frameworks


class TestPIIField:
    """Test the PIIField class."""
    
    def test_pii_field_creation(self):
        """Test creating PII field definitions."""
        field = PIIField(
            name="email",
            sensitivity="high",
            frameworks=[ComplianceFramework.GDPR, ComplianceFramework.CCPA],
            redaction_method="hash",
            retention_days=30
        )
        
        assert field.name == "email"
        assert field.sensitivity == "high"
        assert ComplianceFramework.GDPR in field.frameworks
        assert field.redaction_method == "hash"
        assert field.retention_days == 30
    
    def test_pii_field_serialization(self):
        """Test PII field serialization."""
        field = PIIField(
            name="phone",
            sensitivity="medium",
            frameworks=[ComplianceFramework.HIPAA],
            redaction_method="mask",
            retention_days=90
        )
        
        serialized = field.to_dict()
        assert serialized["name"] == "phone"
        assert serialized["sensitivity"] == "medium"
        assert "HIPAA" in serialized["frameworks"]
        assert serialized["redaction_method"] == "mask"


class TestRedactionPolicy:
    """Test the RedactionPolicy class."""
    
    def test_redaction_policy_creation(self):
        """Test creating redaction policies."""
        policy = RedactionPolicy(
            name="gdpr_compliance",
            frameworks=[ComplianceFramework.GDPR],
            fields={
                "email": PIIField("email", "high", [ComplianceFramework.GDPR]),
                "phone": PIIField("phone", "medium", [ComplianceFramework.GDPR])
            },
            redaction_methods={
                "email": "hash",
                "phone": "mask"
            }
        )
        
        assert policy.name == "gdpr_compliance"
        assert ComplianceFramework.GDPR in policy.frameworks
        assert "email" in policy.fields
        assert "hash" in policy.redaction_methods.values()
    
    def test_redaction_policy_validation(self):
        """Test redaction policy validation."""
        # Valid policy
        valid_policy = RedactionPolicy(
            name="test",
            frameworks=[ComplianceFramework.GDPR],
            fields={},
            redaction_methods={}
        )
        assert valid_policy.is_valid() is True
        
        # Invalid policy - missing frameworks
        invalid_policy = RedactionPolicy(
            name="test",
            frameworks=[],
            fields={},
            redaction_methods={}
        )
        assert invalid_policy.is_valid() is False
    
    def test_redaction_policy_field_matching(self):
        """Test redaction policy field matching."""
        policy = RedactionPolicy(
            name="test",
            frameworks=[ComplianceFramework.GDPR],
            fields={
                "email": PIIField("email", "high", [ComplianceFramework.GDPR]),
                "user_id": PIIField("user_id", "low", [ComplianceFramework.GDPR])
            },
            redaction_methods={
                "email": "hash",
                "user_id": "mask"
            }
        )
        
        # Test field matching
        assert policy.should_redact_field("email") is True
        assert policy.should_redact_field("user_id") is True
        assert policy.should_redact_field("non_pii_field") is False
        
        # Test redaction method selection
        assert policy.get_redaction_method("email") == "hash"
        assert policy.get_redaction_method("user_id") == "mask"
        assert policy.get_redaction_method("unknown") is None


class TestRedactionResult:
    """Test the RedactionResult class."""
    
    def test_redaction_result_creation(self):
        """Test creating redaction results."""
        result = RedactionResult(
            original_value="user@example.com",
            redacted_value="hash_abc123",
            redaction_method="hash",
            field_name="email",
            policy_applied="gdpr_compliance",
            timestamp=datetime.now()
        )
        
        assert result.original_value == "user@example.com"
        assert result.redacted_value == "hash_abc123"
        assert result.redaction_method == "hash"
        assert result.field_name == "email"
        assert result.policy_applied == "gdpr_compliance"
        assert result.timestamp is not None
    
    def test_redaction_result_serialization(self):
        """Test redaction result serialization."""
        result = RedactionResult(
            original_value="test@example.com",
            redacted_value="hash_def456",
            redaction_method="hash",
            field_name="email",
            policy_applied="test_policy",
            timestamp=datetime(2024, 1, 1, 0, 0, 0)
        )
        
        serialized = result.to_dict()
        assert serialized["original_value"] == "test@example.com"
        assert serialized["redacted_value"] == "hash_def456"
        assert serialized["redaction_method"] == "hash"
        assert serialized["field_name"] == "email"


class TestPrivacyAuditTrail:
    """Test the PrivacyAuditTrail class."""
    
    def test_audit_trail_creation(self):
        """Test creating privacy audit trails."""
        audit = PrivacyAuditTrail(
            record_id="test_123",
            timestamp=datetime.now(),
            action="redaction",
            details={
                "fields_redacted": ["email", "phone"],
                "policy_applied": "gdpr_compliance",
                "compliance_status": "compliant"
            }
        )
        
        assert audit.record_id == "test_123"
        assert audit.action == "redaction"
        assert "fields_redacted" in audit.details
        assert audit.details["compliance_status"] == "compliant"
    
    def test_audit_trail_serialization(self):
        """Test audit trail serialization."""
        audit = PrivacyAuditTrail(
            record_id="test_456",
            timestamp=datetime(2024, 1, 1, 0, 0, 0),
            action="validation",
            details={"status": "passed"}
        )
        
        serialized = audit.to_dict()
        assert serialized["record_id"] == "test_456"
        assert serialized["action"] == "validation"
        assert serialized["details"]["status"] == "passed"


class TestPrivacyComplianceFilter:
    """Test the PrivacyComplianceFilter class."""
    
    def test_filter_initialization(self):
        """Test filter initialization."""
        filter = PrivacyComplianceFilter()
        assert filter is not None
        assert hasattr(filter, 'policies')
        assert hasattr(filter, 'audit_trail')
    
    def test_add_redaction_policy(self):
        """Test adding redaction policies."""
        filter = PrivacyComplianceFilter()
        
        policy = RedactionPolicy(
            name="test_policy",
            frameworks=[ComplianceFramework.GDPR],
            fields={
                "email": PIIField("email", "high", [ComplianceFramework.GDPR])
            },
            redaction_methods={"email": "hash"}
        )
        
        filter.add_redaction_policy(policy)
        assert "test_policy" in filter.policies
        assert filter.policies["test_policy"] == policy
    
    def test_apply_privacy_filtering(self, sample_privacy_sensitive_data):
        """Test applying privacy filtering to data."""
        filter = PrivacyComplianceFilter()
        
        # Set up GDPR policy
        gdpr_policy = RedactionPolicy(
            name="gdpr_compliance",
            frameworks=[ComplianceFramework.GDPR],
            fields={
                "email": PIIField("email", "high", [ComplianceFramework.GDPR]),
                "phone": PIIField("phone", "medium", [ComplianceFramework.GDPR]),
                "ip_address": PIIField("ip_address", "medium", [ComplianceFramework.GDPR])
            },
            redaction_methods={
                "email": "hash",
                "phone": "mask",
                "ip_address": "anonymize"
            }
        )
        
        filter.add_redaction_policy(gdpr_policy)
        
        # Apply filtering
        result = filter.apply_privacy_filtering(
            sample_privacy_sensitive_data,
            frameworks=[ComplianceFramework.GDPR],
            level=PrivacyLevel.HIGH
        )
        
        assert result.is_compliant is True
        assert len(result.redacted_fields) >= 3  # email, phone, ip_address
        assert result.audit_trail is not None
    
    def test_privacy_level_filtering(self, sample_privacy_sensitive_data):
        """Test filtering with different privacy levels."""
        filter = PrivacyComplianceFilter()
        
        # Set up comprehensive policy
        policy = RedactionPolicy(
            name="comprehensive",
            frameworks=[ComplianceFramework.GDPR, ComplianceFramework.CCPA],
            fields={
                "email": PIIField("email", "high", [ComplianceFramework.GDPR]),
                "phone": PIIField("phone", "medium", [ComplianceFramework.GDPR]),
                "location": PIIField("location", "low", [ComplianceFramework.CCPA])
            },
            redaction_methods={
                "email": "hash",
                "phone": "mask",
                "location": "generalize"
            }
        )
        
        filter.add_redaction_policy(policy)
        
        # Test different privacy levels
        # MINIMAL - only high sensitivity fields
        result_minimal = filter.apply_privacy_filtering(
            sample_privacy_sensitive_data,
            frameworks=[ComplianceFramework.GDPR],
            level=PrivacyLevel.MINIMAL
        )
        
        # HIGH - all fields
        result_high = filter.apply_privacy_filtering(
            sample_privacy_sensitive_data,
            frameworks=[ComplianceFramework.GDPR],
            level=PrivacyLevel.HIGH
        )
        
        # HIGH should redact more fields than MINIMAL
        assert len(result_high.redacted_fields) >= len(result_minimal.redacted_fields)
    
    def test_framework_specific_filtering(self, sample_privacy_sensitive_data):
        """Test filtering for specific compliance frameworks."""
        filter = PrivacyComplianceFilter()
        
        # GDPR policy
        gdpr_policy = RedactionPolicy(
            name="gdpr",
            frameworks=[ComplianceFramework.GDPR],
            fields={
                "email": PIIField("email", "high", [ComplianceFramework.GDPR])
            },
            redaction_methods={"email": "hash"}
        )
        
        # CCPA policy
        ccpa_policy = RedactionPolicy(
            name="ccpa",
            frameworks=[ComplianceFramework.CCPA],
            fields={
                "location": PIIField("location", "medium", [ComplianceFramework.CCPA])
            },
            redaction_methods={"location": "generalize"}
        )
        
        filter.add_redaction_policy(gdpr_policy)
        filter.add_redaction_policy(ccpa_policy)
        
        # Test GDPR only
        gdpr_result = filter.apply_privacy_filtering(
            sample_privacy_sensitive_data,
            frameworks=[ComplianceFramework.GDPR],
            level=PrivacyLevel.HIGH
        )
        
        # Test CCPA only
        ccpa_result = filter.apply_privacy_filtering(
            sample_privacy_sensitive_data,
            frameworks=[ComplianceFramework.CCPA],
            level=PrivacyLevel.HIGH
        )
        
        # Test both frameworks
        both_result = filter.apply_privacy_filtering(
            sample_privacy_sensitive_data,
            frameworks=[ComplianceFramework.GDPR, ComplianceFramework.CCPA],
            level=PrivacyLevel.HIGH
        )
        
        # Both should redact more fields than individual frameworks
        assert len(both_result.redacted_fields) >= len(gdpr_result.redacted_fields)
        assert len(both_result.redacted_fields) >= len(ccpa_result.redacted_fields)
    
    def test_redaction_methods(self, sample_privacy_sensitive_data):
        """Test different redaction methods."""
        filter = PrivacyComplianceFilter()
        
        # Policy with multiple redaction methods
        policy = RedactionPolicy(
            name="multi_method",
            frameworks=[ComplianceFramework.GDPR],
            fields={
                "email": PIIField("email", "high", [ComplianceFramework.GDPR]),
                "phone": PIIField("phone", "medium", [ComplianceFramework.GDPR]),
                "location": PIIField("location", "low", [ComplianceFramework.GDPR])
            },
            redaction_methods={
                "email": "hash",
                "phone": "mask",
                "location": "generalize"
            }
        )
        
        filter.add_redaction_policy(policy)
        
        result = filter.apply_privacy_filtering(
            sample_privacy_sensitive_data,
            frameworks=[ComplianceFramework.GDPR],
            level=PrivacyLevel.HIGH
        )
        
        # Check that different methods were applied
        methods_used = set()
        for field_result in result.redacted_fields:
            methods_used.add(field_result.redaction_method)
        
        assert len(methods_used) >= 2  # At least 2 different methods
    
    def test_audit_trail_generation(self, sample_privacy_sensitive_data):
        """Test audit trail generation during filtering."""
        filter = PrivacyComplianceFilter()
        
        policy = RedactionPolicy(
            name="audit_test",
            frameworks=[ComplianceFramework.GDPR],
            fields={
                "email": PIIField("email", "high", [ComplianceFramework.GDPR])
            },
            redaction_methods={"email": "hash"}
        )
        
        filter.add_redaction_policy(policy)
        
        result = filter.apply_privacy_filtering(
            sample_privacy_sensitive_data,
            frameworks=[ComplianceFramework.GDPR],
            level=PrivacyLevel.HIGH
        )
        
        # Check audit trail
        assert result.audit_trail is not None
        assert result.audit_trail.record_id is not None
        assert result.audit_trail.action == "privacy_filtering"
        assert "fields_redacted" in result.audit_trail.details
    
    def test_error_handling_invalid_data(self):
        """Test error handling with invalid data."""
        filter = PrivacyComplianceFilter()
        
        # Test with None data
        with pytest.raises(ValueError):
            filter.apply_privacy_filtering(None, [ComplianceFramework.GDPR])
        
        # Test with empty data
        with pytest.raises(ValueError):
            filter.apply_privacy_filtering({}, [ComplianceFramework.GDPR])
        
        # Test with invalid frameworks
        with pytest.raises(ValueError):
            filter.apply_privacy_filtering({"test": "data"}, [])
    
    def test_performance_with_large_data(self, large_dataset):
        """Test privacy filtering performance with large datasets."""
        filter = PrivacyComplianceFilter()
        
        # Set up simple policy
        policy = RedactionPolicy(
            name="performance_test",
            frameworks=[ComplianceFramework.GDPR],
            fields={
                "id": PIIField("id", "low", [ComplianceFramework.GDPR])
            },
            redaction_methods={"id": "mask"}
        )
        
        filter.add_redaction_policy(policy)
        
        # Test with subset of large dataset
        test_data = large_dataset[:1000]
        
        start_time = datetime.now()
        
        for record in test_data:
            result = filter.apply_privacy_filtering(
                record,
                frameworks=[ComplianceFramework.GDPR],
                level=PrivacyLevel.MINIMAL
            )
            assert result.is_compliant is True
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time
        assert duration < 30.0  # 30 seconds for 1000 records
    
    def test_edge_cases(self, edge_case_data):
        """Test privacy filtering with edge case data."""
        filter = PrivacyComplianceFilter()
        
        # Set up basic policy
        policy = RedactionPolicy(
            name="edge_test",
            frameworks=[ComplianceFramework.GDPR],
            fields={
                "id": PIIField("id", "low", [ComplianceFramework.GDPR])
            },
            redaction_methods={"id": "mask"}
        )
        
        filter.add_redaction_policy(policy)
        
        # Test edge cases
        for data in edge_case_data:
            try:
                result = filter.apply_privacy_filtering(
                    data,
                    frameworks=[ComplianceFramework.GDPR],
                    level=PrivacyLevel.MINIMAL
                )
                # Should handle gracefully
                assert isinstance(result.is_compliant, bool)
            except Exception as e:
                # If it fails, should be a controlled failure
                assert "privacy" in str(e).lower() or "invalid" in str(e).lower()


class TestPrivacyIntegration:
    """Test integration between privacy components."""
    
    def test_full_privacy_pipeline(self, sample_privacy_sensitive_data):
        """Test complete privacy filtering pipeline."""
        filter = PrivacyComplianceFilter()
        
        # Set up comprehensive policies
        gdpr_policy = RedactionPolicy(
            name="gdpr_comprehensive",
            frameworks=[ComplianceFramework.GDPR],
            fields={
                "email": PIIField("email", "high", [ComplianceFramework.GDPR]),
                "phone": PIIField("phone", "medium", [ComplianceFramework.GDPR]),
                "ip_address": PIIField("ip_address", "medium", [ComplianceFramework.GDPR]),
                "location": PIIField("location", "low", [ComplianceFramework.GDPR])
            },
            redaction_methods={
                "email": "hash",
                "phone": "mask",
                "ip_address": "anonymize",
                "location": "generalize"
            }
        )
        
        ccpa_policy = RedactionPolicy(
            name="ccpa_comprehensive",
            frameworks=[ComplianceFramework.CCPA],
            fields={
                "location": PIIField("location", "medium", [ComplianceFramework.CCPA]),
                "behavioral_data": PIIField("behavioral_data", "low", [ComplianceFramework.CCPA])
            },
            redaction_methods={
                "location": "generalize",
                "behavioral_data": "aggregate"
            }
        )
        
        filter.add_redaction_policy(gdpr_policy)
        filter.add_redaction_policy(ccpa_policy)
        
        # Apply comprehensive filtering
        result = filter.apply_privacy_filtering(
            sample_privacy_sensitive_data,
            frameworks=[ComplianceFramework.GDPR, ComplianceFramework.CCPA],
            level=PrivacyLevel.MAXIMUM
        )
        
        # Verify results
        assert result.is_compliant is True
        assert len(result.redacted_fields) >= 4  # Multiple fields should be redacted
        assert result.audit_trail is not None
        
        # Check that sensitive fields were redacted
        redacted_field_names = {field.field_name for field in result.redacted_fields}
        assert "email" in redacted_field_names
        assert "phone" in redacted_field_names
        assert "ip_address" in redacted_field_names
    
    def test_policy_conflict_resolution(self, sample_privacy_sensitive_data):
        """Test resolution of conflicting policies."""
        filter = PrivacyComplianceFilter()
        
        # Create conflicting policies for the same field
        policy1 = RedactionPolicy(
            name="policy1",
            frameworks=[ComplianceFramework.GDPR],
            fields={
                "email": PIIField("email", "high", [ComplianceFramework.GDPR])
            },
            redaction_methods={"email": "hash"}
        )
        
        policy2 = RedactionPolicy(
            name="policy2",
            frameworks=[ComplianceFramework.CCPA],
            fields={
                "email": PIIField("email", "medium", [ComplianceFramework.CCPA])
            },
            redaction_methods={"email": "mask"}
        )
        
        filter.add_redaction_policy(policy1)
        filter.add_redaction_policy(policy2)
        
        # Apply both policies
        result = filter.apply_privacy_filtering(
            sample_privacy_sensitive_data,
            frameworks=[ComplianceFramework.GDPR, ComplianceFramework.CCPA],
            level=PrivacyLevel.HIGH
        )
        
        # Should handle conflicts gracefully
        assert result.is_compliant is True
        assert len(result.redacted_fields) >= 1  # At least email should be redacted


if __name__ == '__main__':
    pytest.main([__file__])
