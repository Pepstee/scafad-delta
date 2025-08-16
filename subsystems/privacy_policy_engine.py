#!/usr/bin/env python3
"""
SCAFAD Layer 1: Privacy Policy Engine Subsystem
==============================================

The Privacy Policy Engine manages dynamic privacy policy application, PII detection,
and compliance validation for Layer 1's behavioral intake zone. It provides:

- Dynamic privacy policy management
- PII detection and classification
- GDPR/CCPA/HIPAA compliance validation
- Redaction policy application
- Consent tracking and management
- Data minimization and retention policies

Author: SCAFAD Research Team
Institution: Birmingham Newman University
License: MIT
Version: 1.0.0
"""

import json
import re
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from enum import Enum, auto
from datetime import datetime, timezone, timedelta
import copy
from pathlib import Path

# =============================================================================
# Privacy Policy Data Models
# =============================================================================

class PrivacyRegulation(Enum):
    """Supported privacy regulations"""
    GDPR = "gdpr"           # General Data Protection Regulation
    CCPA = "ccpa"           # California Consumer Privacy Act
    HIPAA = "hipaa"         # Health Insurance Portability and Accountability Act
    SOX = "sox"             # Sarbanes-Oxley Act
    PIPEDA = "pipeda"       # Personal Information Protection and Electronic Documents Act
    LGPD = "lgpd"           # Lei Geral de Proteção de Dados (Brazil)

class PIICategory(Enum):
    """PII classification categories"""
    IDENTIFIERS = "identifiers"           # Names, IDs, SSNs
    CONTACT = "contact"                   # Email, phone, address
    FINANCIAL = "financial"               # Credit cards, bank accounts
    HEALTH = "health"                     # Medical records, health data
    LOCATION = "location"                 # GPS, IP addresses
    BEHAVIORAL = "behavioral"             # Browsing history, preferences
    TECHNICAL = "technical"               # Device IDs, cookies
    SENSITIVE = "sensitive"               # Political, religious, sexual orientation

class RedactionLevel(Enum):
    """Data redaction levels"""
    NONE = "none"                        # No redaction
    PARTIAL = "partial"                  # Partial redaction (e.g., ***@***.com)
    FULL = "full"                        # Complete redaction
    HASH = "hash"                        # Hash-based redaction
    ENCRYPT = "encrypt"                  # Encryption-based redaction

class ConsentStatus(Enum):
    """Consent status values"""
    GRANTED = "granted"                  # Explicit consent given
    DENIED = "denied"                    # Explicit consent denied
    WITHDRAWN = "withdrawn"              # Consent withdrawn
    UNKNOWN = "unknown"                  # Consent status unknown
    IMPLICIT = "implicit"                # Implicit consent (legitimate interest)

@dataclass
class PIIField:
    """PII field definition"""
    field_name: str
    category: PIICategory
    confidence_score: float
    detection_method: str
    redaction_level: RedactionLevel
    retention_policy: str
    compliance_requirements: List[PrivacyRegulation]
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Post-initialization validation"""
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        
        if not isinstance(self.category, PIICategory):
            self.category = PIICategory(self.category)
        
        if not isinstance(self.redaction_level, RedactionLevel):
            self.redaction_level = RedactionLevel(self.redaction_level)

@dataclass
class PrivacyPolicy:
    """Privacy policy definition"""
    policy_id: str
    name: str
    description: str
    regulations: List[PrivacyRegulation]
    effective_date: datetime
    expiration_date: Optional[datetime] = None
    pii_categories: List[PIICategory] = field(default_factory=list)
    redaction_rules: Dict[str, RedactionLevel] = field(default_factory=dict)
    retention_rules: Dict[str, str] = field(default_factory=dict)
    consent_requirements: Dict[str, bool] = field(default_factory=dict)
    data_minimization: bool = True
    purpose_limitation: bool = True
    is_active: bool = True
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Post-initialization validation"""
        if not self.policy_id:
            raise ValueError("Policy ID cannot be empty")
        
        if isinstance(self.effective_date, str):
            self.effective_date = datetime.fromisoformat(self.effective_date)
        
        if self.expiration_date and isinstance(self.expiration_date, str):
            self.expiration_date = datetime.fromisoformat(self.expiration_date)

@dataclass
class ConsentRecord:
    """Consent record for data subjects"""
    subject_id: str
    policy_id: str
    consent_status: ConsentStatus
    granted_at: datetime
    expires_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None
    consent_method: str = "explicit"
    consent_version: str = "1.0"
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Post-initialization validation"""
        if isinstance(self.granted_at, str):
            self.granted_at = datetime.fromisoformat(self.granted_at)
        
        if self.expires_at and isinstance(self.expires_at, str):
            self.expires_at = datetime.fromisoformat(self.expires_at)
        
        if self.withdrawn_at and isinstance(self.withdrawn_at, str):
            self.withdrawn_at = datetime.fromisoformat(self.withdrawn_at)

@dataclass
class PrivacyComplianceReport:
    """Privacy compliance validation report"""
    record_id: str
    policy_id: str
    compliance_status: bool
    violations: List[str]
    pii_fields_detected: List[PIIField]
    redaction_applied: Dict[str, RedactionLevel]
    consent_verified: bool
    retention_compliant: bool
    recommendations: List[str]
    audit_trail: List[str]

# =============================================================================
# Privacy Policy Engine Core Class
# =============================================================================

class PrivacyPolicyEngine:
    """
    Privacy Policy Engine for managing privacy compliance
    
    The Privacy Policy Engine provides comprehensive privacy management for Layer 1,
    ensuring compliance with multiple regulations while maintaining data utility.
    """
    
    def __init__(self, config: Any = None):
        """Initialize the Privacy Policy Engine"""
        self.config = config
        self.logger = logging.getLogger("SCAFAD.Layer1.PrivacyPolicyEngine")
        
        # Policy storage
        self.policies: Dict[str, PrivacyPolicy] = {}
        self.active_policies: Dict[PrivacyRegulation, PrivacyPolicy] = {}
        
        # Consent management
        self.consent_records: Dict[str, List[ConsentRecord]] = {}
        
        # PII detection patterns
        self.pii_patterns: Dict[PIICategory, List[re.Pattern]] = {}
        
        # Performance tracking
        self.engine_stats = {
            'total_policies': 0,
            'total_consent_records': 0,
            'pii_detections': 0,
            'redaction_applications': 0,
            'compliance_checks': 0
        }
        
        self.logger.info("Privacy Policy Engine initialized")
        self._initialize_default_policies()
        self._initialize_pii_patterns()
    
    def register_policy(self, policy: PrivacyPolicy) -> bool:
        """
        Register a new privacy policy
        
        Args:
            policy: PrivacyPolicy to register
            
        Returns:
            bool: True if registration successful
        """
        try:
            # Validate policy
            if not self._validate_policy(policy):
                self.logger.error(f"Policy validation failed for {policy.policy_id}")
                return False
            
            # Check for conflicts
            if policy.policy_id in self.policies:
                self.logger.warning(f"Policy {policy.policy_id} already exists, updating")
            
            # Register policy
            self.policies[policy.policy_id] = policy
            self.engine_stats['total_policies'] += 1
            
            # Update active policies
            if policy.is_active:
                for regulation in policy.regulations:
                    self.active_policies[regulation] = policy
            
            self.logger.info(f"Privacy policy {policy.policy_id} registered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register policy {policy.policy_id}: {str(e)}")
            return False
    
    def get_policy(self, policy_id: str) -> Optional[PrivacyPolicy]:
        """
        Retrieve a privacy policy
        
        Args:
            policy_id: Policy ID to retrieve
            
        Returns:
            PrivacyPolicy or None if not found
        """
        return self.policies.get(policy_id)
    
    def get_active_policy(self, regulation: PrivacyRegulation) -> Optional[PrivacyPolicy]:
        """
        Get active policy for a specific regulation
        
        Args:
            regulation: Privacy regulation
            
        Returns:
            PrivacyPolicy or None if not found
        """
        return self.active_policies.get(regulation)
    
    def list_policies(self) -> List[str]:
        """List all registered policy IDs"""
        return list(self.policies.keys())
    
    def update_policy(self, policy_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing privacy policy
        
        Args:
            policy_id: Policy ID to update
            updates: Dictionary of updates to apply
            
        Returns:
            bool: True if update successful
        """
        try:
            policy = self.policies.get(policy_id)
            if not policy:
                self.logger.error(f"Policy {policy_id} not found")
                return False
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(policy, key):
                    setattr(policy, key, value)
                else:
                    self.logger.warning(f"Unknown policy field: {key}")
            
            # Re-register to update active policies
            return self.register_policy(policy)
            
        except Exception as e:
            self.logger.error(f"Failed to update policy {policy_id}: {str(e)}")
            return False
    
    def deactivate_policy(self, policy_id: str) -> bool:
        """
        Deactivate a privacy policy
        
        Args:
            policy_id: Policy ID to deactivate
            
        Returns:
            bool: True if deactivation successful
        """
        try:
            policy = self.policies.get(policy_id)
            if not policy:
                self.logger.error(f"Policy {policy_id} not found")
                return False
            
            policy.is_active = False
            
            # Remove from active policies
            for regulation in policy.regulations:
                if self.active_policies.get(regulation) == policy:
                    del self.active_policies[regulation]
            
            self.logger.info(f"Policy {policy_id} deactivated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deactivate policy {policy_id}: {str(e)}")
            return False
    
    def record_consent(self, consent: ConsentRecord) -> bool:
        """
        Record consent for a data subject
        
        Args:
            consent: ConsentRecord to register
            
        Returns:
            bool: True if registration successful
        """
        try:
            # Validate consent
            if not self._validate_consent(consent):
                self.logger.error(f"Consent validation failed for subject {consent.subject_id}")
                return False
            
            # Initialize subject consent list if needed
            if consent.subject_id not in self.consent_records:
                self.consent_records[consent.subject_id] = []
            
            # Add consent record
            self.consent_records[consent.subject_id].append(consent)
            self.engine_stats['total_consent_records'] += 1
            
            self.logger.info(f"Consent recorded for subject {consent.subject_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record consent: {str(e)}")
            return False
    
    def get_consent_status(self, subject_id: str, policy_id: str) -> Optional[ConsentStatus]:
        """
        Get current consent status for a subject and policy
        
        Args:
            subject_id: Data subject ID
            policy_id: Policy ID
            
        Returns:
            ConsentStatus or None if not found
        """
        try:
            if subject_id not in self.consent_records:
                return None
            
            # Find most recent consent record for this policy
            policy_consents = [
                c for c in self.consent_records[subject_id] 
                if c.policy_id == policy_id
            ]
            
            if not policy_consents:
                return None
            
            # Sort by granted_at and return most recent
            latest_consent = max(policy_consents, key=lambda c: c.granted_at)
            
            # Check if consent is still valid
            if latest_consent.consent_status == ConsentStatus.WITHDRAWN:
                return ConsentStatus.WITHDRAWN
            
            if latest_consent.expires_at and datetime.now(timezone.utc) > latest_consent.expires_at:
                return ConsentStatus.UNKNOWN
            
            return latest_consent.consent_status
            
        except Exception as e:
            self.logger.error(f"Failed to get consent status: {str(e)}")
            return None
    
    def withdraw_consent(self, subject_id: str, policy_id: str) -> bool:
        """
        Withdraw consent for a data subject
        
        Args:
            subject_id: Data subject ID
            policy_id: Policy ID
            
        Returns:
            bool: True if withdrawal successful
        """
        try:
            if subject_id not in self.consent_records:
                self.logger.error(f"No consent records found for subject {subject_id}")
                return False
            
            # Find consent record to withdraw
            for consent in self.consent_records[subject_id]:
                if consent.policy_id == policy_id and consent.consent_status == ConsentStatus.GRANTED:
                    consent.consent_status = ConsentStatus.WITHDRAWN
                    consent.withdrawn_at = datetime.now(timezone.utc)
                    self.logger.info(f"Consent withdrawn for subject {subject_id}, policy {policy_id}")
                    return True
            
            self.logger.warning(f"No active consent found for subject {subject_id}, policy {policy_id}")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to withdraw consent: {str(e)}")
            return False
    
    def detect_pii(self, data: Any, context: Optional[Dict[str, Any]] = None) -> List[PIIField]:
        """
        Detect PII in data
        
        Args:
            data: Data to analyze for PII
            context: Optional context for detection
            
        Returns:
            List of detected PII fields
        """
        try:
            self.engine_stats['pii_detections'] += 1
            detected_pii = []
            
            if isinstance(data, dict):
                detected_pii.extend(self._detect_pii_in_dict(data, context))
            elif isinstance(data, str):
                detected_pii.extend(self._detect_pii_in_string(data, context))
            elif isinstance(data, list):
                for item in data:
                    detected_pii.extend(self.detect_pii(item, context))
            
            return detected_pii
            
        except Exception as e:
            self.logger.error(f"PII detection failed: {str(e)}")
            return []
    
    def apply_privacy_policies(self, record: Dict[str, Any], 
                             subject_id: Optional[str] = None,
                             regulations: Optional[List[PrivacyRegulation]] = None) -> Tuple[Dict[str, Any], PrivacyComplianceReport]:
        """
        Apply privacy policies to a record
        
        Args:
            record: Record to process
            subject_id: Optional data subject ID for consent checking
            regulations: Optional list of regulations to apply
            
        Returns:
            Tuple of (processed_record, compliance_report)
        """
        try:
            self.engine_stats['compliance_checks'] += 1
            
            # Determine applicable regulations
            if not regulations:
                regulations = list(self.active_policies.keys())
            
            # Initialize compliance report
            compliance_report = PrivacyComplianceReport(
                record_id=record.get('record_id', 'unknown'),
                policy_id='composite',
                compliance_status=True,
                violations=[],
                pii_fields_detected=[],
                redaction_applied={},
                consent_verified=False,
                retention_compliant=True,
                recommendations=[],
                audit_trail=[]
            )
            
            processed_record = copy.deepcopy(record)
            
            # Apply policies for each regulation
            for regulation in regulations:
                policy = self.active_policies.get(regulation)
                if not policy:
                    continue
                
                # Check consent if required
                if subject_id and policy.consent_requirements.get('explicit_consent', False):
                    consent_status = self.get_consent_status(subject_id, policy.policy_id)
                    if consent_status != ConsentStatus.GRANTED:
                        compliance_report.violations.append(f"Missing consent for {regulation.value}")
                        compliance_report.compliance_status = False
                        continue
                    else:
                        compliance_report.consent_verified = True
                
                # Detect PII
                pii_fields = self.detect_pii(processed_record, {'regulation': regulation})
                compliance_report.pii_fields_detected.extend(pii_fields)
                
                # Apply redaction rules
                for pii_field in pii_fields:
                    redaction_level = policy.redaction_rules.get(
                        pii_field.field_name, 
                        pii_field.redaction_level
                    )
                    
                    if redaction_level != RedactionLevel.NONE:
                        processed_record = self._apply_redaction(
                            processed_record, pii_field, redaction_level
                        )
                        compliance_report.redaction_applied[pii_field.field_name] = redaction_level
                        self.engine_stats['redaction_applications'] += 1
                
                # Apply data minimization
                if policy.data_minimization:
                    processed_record = self._apply_data_minimization(processed_record, policy)
                
                # Check retention compliance
                if not self._check_retention_compliance(processed_record, policy):
                    compliance_report.retention_compliant = False
                    compliance_report.violations.append(f"Retention policy violation for {regulation.value}")
            
            # Generate recommendations
            compliance_report.recommendations = self._generate_compliance_recommendations(compliance_report)
            
            return processed_record, compliance_report
            
        except Exception as e:
            self.logger.error(f"Privacy policy application failed: {str(e)}")
            return record, PrivacyComplianceReport(
                record_id=record.get('record_id', 'unknown'),
                policy_id='error',
                compliance_status=False,
                violations=[f"Processing error: {str(e)}"],
                pii_fields_detected=[],
                redaction_applied={},
                consent_verified=False,
                retention_compliant=False,
                recommendations=["Review system logs for errors"],
                audit_trail=[f"Error: {str(e)}"]
            )
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return copy.deepcopy(self.engine_stats)
    
    # =========================================================================
    # Private Helper Methods
    # =========================================================================
    
    def _initialize_default_policies(self):
        """Initialize default privacy policies"""
        # GDPR Policy
        gdpr_policy = PrivacyPolicy(
            policy_id="gdpr_default",
            name="GDPR Default Policy",
            description="Default GDPR compliance policy for Layer 1",
            regulations=[PrivacyRegulation.GDPR],
            effective_date=datetime.now(timezone.utc),
            pii_categories=[
                PIICategory.IDENTIFIERS,
                PIICategory.CONTACT,
                PIICategory.LOCATION,
                PIICategory.BEHAVIORAL
            ],
            redaction_rules={
                'email': RedactionLevel.PARTIAL,
                'phone': RedactionLevel.PARTIAL,
                'ip_address': RedactionLevel.HASH,
                'user_id': RedactionLevel.HASH
            },
            retention_rules={
                'telemetry_data': '90_days',
                'pii_fields': '30_days',
                'audit_logs': '1_year'
            },
            consent_requirements={
                'explicit_consent': True,
                'purpose_limitation': True,
                'data_minimization': True
            }
        )
        
        # CCPA Policy
        ccpa_policy = PrivacyPolicy(
            policy_id="ccpa_default",
            name="CCPA Default Policy",
            description="Default CCPA compliance policy for Layer 1",
            regulations=[PrivacyRegulation.CCPA],
            effective_date=datetime.now(timezone.utc),
            pii_categories=[
                PIICategory.IDENTIFIERS,
                PIICategory.CONTACT,
                PIICategory.FINANCIAL,
                PIICategory.BEHAVIORAL
            ],
            redaction_rules={
                'email': RedactionLevel.PARTIAL,
                'phone': RedactionLevel.PARTIAL,
                'ssn': RedactionLevel.FULL,
                'credit_card': RedactionLevel.FULL
            },
            retention_rules={
                'telemetry_data': '60_days',
                'pii_fields': '30_days',
                'audit_logs': '6_months'
            },
            consent_requirements={
                'explicit_consent': False,
                'opt_out_rights': True,
                'data_minimization': True
            }
        )
        
        # Register default policies
        self.register_policy(gdpr_policy)
        self.register_policy(ccpa_policy)
    
    def _initialize_pii_patterns(self):
        """Initialize PII detection patterns"""
        # Email patterns
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.pii_patterns[PIICategory.CONTACT] = [email_pattern]
        
        # Phone number patterns
        phone_pattern = re.compile(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
        self.pii_patterns[PIICategory.CONTACT].append(phone_pattern)
        
        # IP address patterns
        ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        self.pii_patterns[PIICategory.LOCATION] = [ip_pattern]
        
        # SSN patterns
        ssn_pattern = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
        self.pii_patterns[PIICategory.IDENTIFIERS] = [ssn_pattern]
        
        # Credit card patterns
        cc_pattern = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')
        self.pii_patterns[PIICategory.FINANCIAL] = [cc_pattern]
    
    def _validate_policy(self, policy: PrivacyPolicy) -> bool:
        """Validate privacy policy before registration"""
        try:
            if not policy.policy_id:
                return False
            
            if not policy.regulations:
                return False
            
            if not policy.effective_date:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _validate_consent(self, consent: ConsentRecord) -> bool:
        """Validate consent record before registration"""
        try:
            if not consent.subject_id:
                return False
            
            if not consent.policy_id:
                return False
            
            if not consent.granted_at:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _detect_pii_in_dict(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> List[PIIField]:
        """Detect PII in dictionary data"""
        detected_pii = []
        
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                detected_pii.extend(self.detect_pii(value, context))
            elif isinstance(value, str):
                detected_pii.extend(self._detect_pii_in_string(value, context, field_name=key))
        
        return detected_pii
    
    def _detect_pii_in_string(self, data: str, context: Optional[Dict[str, Any]] = None, 
                             field_name: Optional[str] = None) -> List[PIIField]:
        """Detect PII in string data"""
        detected_pii = []
        
        for category, patterns in self.pii_patterns.items():
            for pattern in patterns:
                if pattern.search(data):
                    confidence_score = self._calculate_confidence_score(data, pattern, category)
                    
                    pii_field = PIIField(
                        field_name=field_name or 'unknown',
                        category=category,
                        confidence_score=confidence_score,
                        detection_method='pattern_matching',
                        redaction_level=RedactionLevel.PARTIAL,
                        retention_policy='30_days',
                        compliance_requirements=[PrivacyRegulation.GDPR, PrivacyRegulation.CCPA]
                    )
                    
                    detected_pii.append(pii_field)
        
        return detected_pii
    
    def _calculate_confidence_score(self, data: str, pattern: re.Pattern, category: PIICategory) -> float:
        """Calculate confidence score for PII detection"""
        # Base confidence based on pattern match
        base_confidence = 0.7
        
        # Adjust based on data characteristics
        if category == PIICategory.CONTACT:
            if '@' in data and '.' in data:
                base_confidence += 0.2
            elif len(data.replace('-', '').replace(' ', '')) == 10:
                base_confidence += 0.1
        
        # Adjust based on context
        if len(data) > 50:  # Likely not a simple identifier
            base_confidence -= 0.1
        
        return min(1.0, max(0.0, base_confidence))
    
    def _apply_redaction(self, record: Dict[str, Any], pii_field: PIIField, 
                         redaction_level: RedactionLevel) -> Dict[str, Any]:
        """Apply redaction to a record"""
        try:
            if redaction_level == RedactionLevel.NONE:
                return record
            
            # Find and redact the field
            record_copy = copy.deepcopy(record)
            self._redact_field_recursive(record_copy, pii_field.field_name, redaction_level)
            
            return record_copy
            
        except Exception as e:
            self.logger.error(f"Redaction failed: {str(e)}")
            return record
    
    def _redact_field_recursive(self, data: Any, field_name: str, redaction_level: RedactionLevel):
        """Recursively redact fields in nested data structures"""
        if isinstance(data, dict):
            for key, value in data.items():
                if key == field_name:
                    data[key] = self._redact_value(value, redaction_level)
                elif isinstance(value, (dict, list)):
                    self._redact_field_recursive(value, field_name, redaction_level)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    self._redact_field_recursive(item, field_name, redaction_level)
    
    def _redact_value(self, value: Any, redaction_level: RedactionLevel) -> Any:
        """Redact a value based on redaction level"""
        if redaction_level == RedactionLevel.FULL:
            return "[REDACTED]"
        elif redaction_level == RedactionLevel.PARTIAL:
            if isinstance(value, str) and '@' in value:
                # Partial email redaction
                parts = value.split('@')
                if len(parts) == 2:
                    username = parts[0]
                    domain = parts[1]
                    if len(username) > 2:
                        redacted_username = username[:2] + '*' * (len(username) - 2)
                    else:
                        redacted_username = '*' * len(username)
                    return f"{redacted_username}@{domain}"
                else:
                    return '*' * len(value)
            else:
                return '*' * min(len(str(value)), 10)
        elif redaction_level == RedactionLevel.HASH:
            return hashlib.sha256(str(value).encode()).hexdigest()[:16]
        else:
            return value
    
    def _apply_data_minimization(self, record: Dict[str, Any], policy: PrivacyPolicy) -> Dict[str, Any]:
        """Apply data minimization to a record"""
        # This would implement data minimization logic
        # For now, return the record as-is
        return record
    
    def _check_retention_compliance(self, record: Dict[str, Any], policy: PrivacyPolicy) -> bool:
        """Check if record complies with retention policies"""
        # This would implement retention compliance checking
        # For now, return True
        return True
    
    def _generate_compliance_recommendations(self, report: PrivacyComplianceReport) -> List[str]:
        """Generate compliance recommendations based on report"""
        recommendations = []
        
        if not report.compliance_status:
            recommendations.append("Review and address all compliance violations")
        
        if report.pii_fields_detected:
            recommendations.append("Consider implementing additional PII detection patterns")
        
        if not report.consent_verified:
            recommendations.append("Implement explicit consent collection mechanisms")
        
        if not report.retention_compliant:
            recommendations.append("Review and update data retention policies")
        
        return recommendations

# =============================================================================
# Privacy Policy Engine Factory Functions
# =============================================================================

def create_default_privacy_policy_engine() -> PrivacyPolicyEngine:
    """Create default privacy policy engine with basic policies"""
    return PrivacyPolicyEngine()

if __name__ == "__main__":
    # Example usage
    engine = create_default_privacy_policy_engine()
    
    print("Registered policies:", engine.list_policies())
    
    # Test PII detection
    test_data = {
        'user_email': 'john.doe@example.com',
        'phone_number': '+1-555-123-4567',
        'ip_address': '192.168.1.100'
    }
    
    pii_fields = engine.detect_pii(test_data)
    print(f"Detected PII fields: {len(pii_fields)}")
    
    for pii_field in pii_fields:
        print(f"- {pii_field.field_name}: {pii_field.category.value} (confidence: {pii_field.confidence_score:.2f})")
    
    # Test privacy policy application
    processed_record, compliance_report = engine.apply_privacy_policies(test_data)
    print(f"Compliance status: {compliance_report.compliance_status}")
    print(f"Redaction applied: {compliance_report.redaction_applied}")
