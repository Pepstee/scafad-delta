"""
Privacy compliance tests for Scafad Layer1 system.

Tests privacy regulation compliance, PII handling,
data anonymization, and privacy policy enforcement.
"""

import unittest
import sys
import os
import json
import re
from unittest.mock import Mock, patch, MagicMock

# Add the project root to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.layer1_privacy import PrivacyFilter
from subsystems.privacy_policy_engine import PrivacyPolicyEngine


class TestPrivacyCompliance(unittest.TestCase):
    """Test privacy compliance features."""
    
    def setUp(self):
        """Set up test fixtures for privacy testing."""
        self.privacy_filter = PrivacyFilter()
        self.policy_engine = PrivacyPolicyEngine()
        
        # Test data with various PII types
        self.sensitive_data = {
            "user_id": "user_123",
            "full_name": "John Doe",
            "email": "john.doe@example.com",
            "phone": "+1-555-123-4567",
            "social_security": "123-45-6789",
            "credit_card": "4111-1111-1111-1111",
            "address": "123 Main St, Anytown, USA 12345",
            "date_of_birth": "1985-03-15",
            "ip_address": "192.168.1.100",
            "device_id": "device_abc123"
        }
    
    def test_gdpr_compliance(self):
        """Test GDPR compliance features."""
        print("\n=== GDPR Compliance Test ===")
        
        # Test data minimization
        minimal_data = self.privacy_filter.apply_data_minimization(
            self.sensitive_data,
            purpose="service_delivery"
        )
        
        print("Original data fields:", len(self.sensitive_data))
        print("Minimized data fields:", len(minimal_data))
        
        # Should only include necessary fields for service delivery
        self.assertLess(len(minimal_data), len(self.sensitive_data))
        self.assertIn("user_id", minimal_data)
        self.assertIn("full_name", minimal_data)
        self.assertNotIn("social_security", minimal_data)
        self.assertNotIn("credit_card", minimal_data)
        
        # Test right to be forgotten
        deletion_result = self.privacy_filter.delete_user_data("user_123")
        self.assertTrue(deletion_result.success)
        
        # Test data portability
        export_data = self.privacy_filter.export_user_data("user_123")
        self.assertIsNotNone(export_data)
        self.assertIn("user_data", export_data)
    
    def test_ccpa_compliance(self):
        """Test CCPA compliance features."""
        print("\n=== CCPA Compliance Test ===")
        
        # Test consumer rights
        consumer_rights = self.policy_engine.get_consumer_rights("user_123")
        self.assertIn("access", consumer_rights)
        self.assertIn("deletion", consumer_rights)
        self.assertIn("opt_out", consumer_rights)
        
        # Test data categories disclosure
        data_categories = self.policy_engine.get_data_categories("user_123")
        self.assertIsInstance(data_categories, list)
        self.assertGreater(len(data_categories), 0)
        
        # Test business purpose disclosure
        business_purposes = self.policy_engine.get_business_purposes("user_123")
        self.assertIsInstance(business_purposes, list)
        self.assertGreater(len(business_purposes), 0)
    
    def test_pii_detection(self):
        """Test PII detection capabilities."""
        print("\n=== PII Detection Test ===")
        
        # Test email detection
        detected_pii = self.privacy_filter.detect_pii(self.sensitive_data)
        
        print("Detected PII types:", list(detected_pii.keys()))
        
        # Should detect various PII types
        self.assertIn("emails", detected_pii)
        self.assertIn("phone_numbers", detected_pii)
        self.assertIn("credit_cards", detected_pii)
        self.assertIn("ssn", detected_pii)
        self.assertIn("addresses", detected_pii)
        
        # Verify specific detections
        self.assertIn("john.doe@example.com", detected_pii["emails"])
        self.assertIn("+1-555-123-4567", detected_pii["phone_numbers"])
        self.assertIn("4111-1111-1111-1111", detected_pii["credit_cards"])
        self.assertIn("123-45-6789", detected_pii["ssn"])
    
    def test_data_anonymization(self):
        """Test data anonymization techniques."""
        print("\n=== Data Anonymization Test ===")
        
        # Test k-anonymity
        anonymized_data = self.privacy_filter.anonymize_data(
            self.sensitive_data,
            technique="k_anonymity",
            k_value=5
        )
        
        print("Original data:", json.dumps(self.sensitive_data, indent=2))
        print("Anonymized data:", json.dumps(anonymized_data, indent=2))
        
        # Should not contain original PII
        self.assertNotIn("john.doe@example.com", str(anonymized_data))
        self.assertNotIn("123-45-6789", str(anonymized_data))
        self.assertNotIn("4111-1111-1111-1111", str(anonymized_data))
        
        # Should maintain data structure
        self.assertIn("user_id", anonymized_data)
        self.assertIn("full_name", anonymized_data)
        
        # Test pseudonymization
        pseudonymized_data = self.privacy_filter.pseudonymize_data(
            self.sensitive_data,
            technique="deterministic_hash"
        )
        
        # Should replace PII with pseudonyms
        self.assertNotEqual(
            pseudonymized_data["email"],
            self.sensitive_data["email"]
        )
        self.assertNotEqual(
            pseudonymized_data["phone"],
            self.sensitive_data["phone"]
        )
        
        # Pseudonyms should be consistent for same input
        pseudonymized_data2 = self.privacy_filter.pseudonymize_data(
            self.sensitive_data,
            technique="deterministic_hash"
        )
        self.assertEqual(
            pseudonymized_data["email"],
            pseudonymized_data2["email"]
        )
    
    def test_consent_management(self):
        """Test consent management functionality."""
        print("\n=== Consent Management Test ===")
        
        # Test consent recording
        consent_id = self.policy_engine.record_consent(
            user_id="user_123",
            consent_type="data_processing",
            purposes=["analytics", "service_improvement"],
            data_categories=["usage_data", "performance_metrics"],
            retention_period="2_years"
        )
        
        self.assertIsNotNone(consent_id)
        
        # Test consent validation
        is_valid = self.policy_engine.validate_consent(
            consent_id=consent_id,
            purpose="analytics",
            data_category="usage_data"
        )
        
        self.assertTrue(is_valid)
        
        # Test consent withdrawal
        withdrawal_result = self.policy_engine.withdraw_consent(
            consent_id=consent_id,
            reason="user_request"
        )
        
        self.assertTrue(withdrawal_result.success)
        
        # Consent should no longer be valid
        is_valid_after_withdrawal = self.policy_engine.validate_consent(
            consent_id=consent_id,
            purpose="analytics",
            data_category="usage_data"
        )
        
        self.assertFalse(is_valid_after_withdrawal)
    
    def test_data_retention(self):
        """Test data retention policies."""
        print("\n=== Data Retention Test ===")
        
        # Test retention policy application
        retention_result = self.policy_engine.apply_retention_policy(
            user_id="user_123",
            data_category="usage_data",
            retention_period="1_year"
        )
        
        self.assertTrue(retention_result.success)
        
        # Test data expiration
        expired_data = self.policy_engine.get_expired_data()
        self.assertIsInstance(expired_data, list)
        
        # Test data deletion
        deletion_result = self.policy_engine.delete_expired_data()
        self.assertTrue(deletion_result.success)
        self.assertGreaterEqual(deletion_result.deleted_count, 0)
    
    def test_privacy_policy_enforcement(self):
        """Test privacy policy enforcement."""
        print("\n=== Privacy Policy Enforcement Test ===")
        
        # Test policy application
        policy_result = self.policy_engine.apply_privacy_policy(
            data=self.sensitive_data,
            user_id="user_123",
            purpose="analytics"
        )
        
        self.assertTrue(policy_result.success)
        
        # Test policy compliance check
        compliance_result = self.policy_engine.check_compliance(
            data=policy_result.processed_data,
            policy_name="gdpr_strict"
        )
        
        self.assertTrue(compliance_result.compliant)
        self.assertGreaterEqual(compliance_result.compliance_score, 0.9)
    
    def test_audit_trail(self):
        """Test privacy audit trail functionality."""
        print("\n=== Privacy Audit Trail Test ===")
        
        # Test audit log creation
        audit_entry = self.policy_engine.log_privacy_event(
            event_type="data_access",
            user_id="user_123",
            data_category="personal_info",
            purpose="service_delivery",
            timestamp="2024-01-01T00:00:00Z"
        )
        
        self.assertIsNotNone(audit_entry.id)
        
        # Test audit log retrieval
        audit_log = self.policy_engine.get_audit_log(
            user_id="user_123",
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-02T00:00:00Z"
        )
        
        self.assertIsInstance(audit_log, list)
        self.assertGreater(len(audit_log), 0)
        
        # Test audit log filtering
        filtered_log = self.policy_engine.filter_audit_log(
            event_type="data_access",
            data_category="personal_info"
        )
        
        self.assertIsInstance(filtered_log, list)
    
    def test_data_breach_detection(self):
        """Test data breach detection capabilities."""
        print("\n=== Data Breach Detection Test ===")
        
        # Test anomaly detection
        anomaly_result = self.policy_engine.detect_privacy_anomalies(
            user_id="user_123",
            time_window="24_hours"
        )
        
        self.assertIsInstance(anomaly_result.anomalies, list)
        
        # Test breach notification
        if anomaly_result.anomalies:
            notification_result = self.policy_engine.send_breach_notification(
                user_id="user_123",
                breach_type="unauthorized_access",
                severity="high"
            )
            
            self.assertTrue(notification_result.success)
    
    def test_privacy_by_design(self):
        """Test privacy by design principles."""
        print("\n=== Privacy by Design Test ===")
        
        # Test default privacy settings
        default_settings = self.privacy_filter.get_default_settings()
        
        self.assertEqual(default_settings["privacy_level"], "high")
        self.assertTrue(default_settings["detect_pii"])
        self.assertTrue(default_settings["anonymize_data"])
        
        # Test privacy impact assessment
        pia_result = self.policy_engine.assess_privacy_impact(
            data_processing_activity="user_analytics",
            data_categories=["usage_data", "performance_metrics"],
            user_count=1000,
            retention_period="2_years"
        )
        
        self.assertIsNotNone(pia_result.risk_level)
        self.assertIsNotNone(pia_result.mitigation_measures)
        
        # Test privacy controls
        controls = self.policy_engine.get_privacy_controls(
            risk_level=pia_result.risk_level
        )
        
        self.assertIsInstance(controls, list)
        self.assertGreater(len(controls), 0)


class TestPrivacyRegulations(unittest.TestCase):
    """Test specific privacy regulation compliance."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.policy_engine = PrivacyPolicyEngine()
    
    def test_gdpr_article_30(self):
        """Test GDPR Article 30 compliance (Records of Processing Activities)."""
        print("\n=== GDPR Article 30 Test ===")
        
        # Test processing activities record
        activities = self.policy_engine.get_processing_activities()
        
        self.assertIsInstance(activities, list)
        
        for activity in activities:
            self.assertIn("purpose", activity)
            self.assertIn("legal_basis", activity)
            self.assertIn("data_categories", activity)
            self.assertIn("recipients", activity)
            self.assertIn("retention_period", activity)
    
    def test_gdpr_article_35(self):
        """Test GDPR Article 35 compliance (Data Protection Impact Assessment)."""
        print("\n=== GDPR Article 35 Test ===")
        
        # Test DPIA requirement check
        dpia_required = self.policy_engine.is_dpia_required(
            data_processing_activity="large_scale_monitoring",
            data_categories=["location_data", "behavioral_data"],
            user_count=10000
        )
        
        self.assertTrue(dpia_required)
        
        # Test DPIA execution
        dpia_result = self.policy_engine.execute_dpia(
            data_processing_activity="large_scale_monitoring",
            data_categories=["location_data", "behavioral_data"],
            user_count=10000
        )
        
        self.assertIsNotNone(dpia_result.risk_assessment)
        self.assertIsNotNone(dpia_result.mitigation_measures)
    
    def test_ccpa_section_1798_100(self):
        """Test CCPA Section 1798.100 compliance (General Duties of Businesses)."""
        print("\n=== CCPA Section 1798.100 Test ===")
        
        # Test consumer rights disclosure
        rights_disclosure = self.policy_engine.get_consumer_rights_disclosure()
        
        self.assertIn("right_to_know", rights_disclosure)
        self.assertIn("right_to_delete", rights_disclosure)
        self.assertIn("right_to_opt_out", rights_disclosure)
        self.assertIn("right_to_nondiscrimination", rights_disclosure)
    
    def test_ccpa_section_1798_110(self):
        """Test CCPA Section 1798.110 compliance (Disclosure of Personal Information)."""
        print("\n=== CCPA Section 1798.110 Test ===")
        
        # Test data categories disclosure
        categories_disclosure = self.policy_engine.get_data_categories_disclosure()
        
        self.assertIsInstance(categories_disclosure, dict)
        self.assertIn("personal_identifiers", categories_disclosure)
        self.assertIn("commercial_information", categories_disclosure)
        self.assertIn("internet_activity", categories_disclosure)
    
    def test_lgpd_compliance(self):
        """Test LGPD (Brazil) compliance."""
        print("\n=== LGPD Compliance Test ===")
        
        # Test legal basis validation
        legal_basis = self.policy_engine.validate_lgpd_legal_basis(
            purpose="service_delivery",
            consent_given=True,
            legitimate_interest=False
        )
        
        self.assertTrue(legal_basis.valid)
        
        # Test data subject rights
        lgpd_rights = self.policy_engine.get_lgpd_rights()
        
        self.assertIn("confirmation", lgpd_rights)
        self.assertIn("access", lgpd_rights)
        self.assertIn("correction", lgpd_rights)
        self.assertIn("anonymization", lgpd_rights)
        self.assertIn("portability", lgpd_rights)
        self.assertIn("deletion", lgpd_rights)
    
    def test_pipeda_compliance(self):
        """Test PIPEDA (Canada) compliance."""
        print("\n=== PIPEDA Compliance Test ===")
        
        # Test fair information principles
        principles = self.policy_engine.get_pipeda_principles()
        
        self.assertIn("accountability", principles)
        self.assertIn("identifying_purposes", principles)
        self.assertIn("consent", principles)
        self.assertIn("limiting_collection", principles)
        self.assertIn("limiting_use_disclosure", principles)
        self.assertIn("accuracy", principles)
        self.assertIn("safeguards", principles)
        self.assertIn("openness", principles)
        self.assertIn("individual_access", principles)
        self.assertIn("challenging_compliance", principles)


if __name__ == '__main__':
    # Run privacy compliance tests
    unittest.main(verbosity=2)
