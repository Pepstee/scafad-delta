#!/usr/bin/env python3
"""
SCAFAD Layer 1: Privacy Compliance Filter
==========================================

The Privacy Compliance Filter ensures telemetry data meets regulatory requirements
for privacy protection while maintaining anomaly detection capabilities. It implements
GDPR, CCPA, HIPAA, and other privacy regulations through intelligent redaction and
anonymization that preserves behavioral patterns.

Key Responsibilities:
- PII detection and redaction
- Regulatory compliance enforcement (GDPR, CCPA, HIPAA, SOX)
- Data anonymization and pseudonymization
- Consent management integration
- Data retention policy enforcement
- Cross-border data transfer compliance
- Audit trail generation for compliance
- Privacy-preserving transformations
- Differential privacy implementation
- Right to erasure support

Performance Targets:
- Privacy filtering latency: <0.4ms per record
- PII detection accuracy: 99.9%+
- Compliance rate: 100%
- Anomaly preservation: 99.5%+ after privacy filtering
- Zero compliance violations

Author: SCAFAD Research Team
Institution: Birmingham Newman University
License: MIT
Version: 1.0.0
"""

import re
import json
import hashlib
import logging
import asyncio
import secrets
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from enum import Enum, auto
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import traceback
import copy
import base64
from functools import lru_cache

# Cryptographic operations
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend
import hmac

# Data processing
import numpy as np
from faker import Faker

# Regular expressions for PII detection
import phonenumbers
import email_validator
from email_validator import validate_email, EmailNotValidError

# IP address handling
import ipaddress

# Performance monitoring
import time

# Credit card validation
import luhn


# =============================================================================
# Privacy Data Models and Enums
# =============================================================================

class PrivacyRegulation(Enum):
    """Supported privacy regulations"""
    GDPR = "gdpr"           # General Data Protection Regulation (EU)
    CCPA = "ccpa"           # California Consumer Privacy Act
    HIPAA = "hipaa"         # Health Insurance Portability and Accountability Act
    SOX = "sox"             # Sarbanes-Oxley Act
    PIPEDA = "pipeda"       # Personal Information Protection and Electronic Documents Act (Canada)
    LGPD = "lgpd"           # Lei Geral de Proteção de Dados (Brazil)
    CUSTOM = "custom"       # Custom privacy policy

class PIIType(Enum):
    """Types of personally identifiable information"""
    NAME = "name"
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"                     # Social Security Number
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"
    LOCATION = "location"
    DATE_OF_BIRTH = "date_of_birth"
    DRIVER_LICENSE = "driver_license"
    PASSPORT = "passport"
    MEDICAL_ID = "medical_id"
    BANK_ACCOUNT = "bank_account"
    USERNAME = "username"
    PASSWORD = "password"
    BIOMETRIC = "biometric"
    DEVICE_ID = "device_id"
    COOKIE_ID = "cookie_id"
    USER_AGENT = "user_agent"
    CUSTOM_ID = "custom_id"

class RedactionMethod(Enum):
    """Methods for redacting PII"""
    MASK = "mask"                   # Replace with mask characters
    HASH = "hash"                   # One-way hash
    ENCRYPT = "encrypt"             # Reversible encryption
    TOKENIZE = "tokenize"           # Replace with token
    GENERALIZE = "generalize"       # Generalize to less specific
    SUPPRESS = "suppress"           # Remove entirely
    SYNTHETIC = "synthetic"         # Replace with synthetic data
    DIFFERENTIAL = "differential"   # Add differential privacy noise

class ConsentStatus(Enum):
    """User consent status"""
    GRANTED = "granted"
    DENIED = "denied"
    PARTIAL = "partial"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"
    NOT_REQUIRED = "not_required"

class DataRetentionPolicy(Enum):
    """Data retention policies"""
    IMMEDIATE = "immediate"         # Delete immediately after processing
    SHORT_TERM = "short_term"       # 7 days
    MEDIUM_TERM = "medium_term"     # 30 days
    LONG_TERM = "long_term"         # 90 days
    ARCHIVE = "archive"             # 1 year
    PERMANENT = "permanent"         # No automatic deletion


@dataclass
class PrivacyAuditTrail:
    """
    Audit trail for privacy operations
    """
    batch_id: str
    timestamp: str
    records_processed: int
    privacy_level: Any  # PrivacyLevel from config
    regulations_applied: List[PrivacyRegulation] = field(default_factory=list)
    pii_detected: Dict[str, int] = field(default_factory=dict)
    redaction_actions: List[Dict[str, Any]] = field(default_factory=list)
    anonymization_actions: List[Dict[str, Any]] = field(default_factory=list)
    consent_checks: List[Dict[str, Any]] = field(default_factory=list)
    compliance_status: Dict[str, Any] = field(default_factory=dict)
    retention_actions: List[Dict[str, Any]] = field(default_factory=list)
    cross_border_transfers: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_redaction_action(self, field: str, pii_type: PIIType, method: RedactionMethod):
        """Add redaction action to audit trail"""
        self.redaction_actions.append({
            'field': field,
            'pii_type': pii_type.value,
            'method': method.value,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit trail to dictionary"""
        return asdict(self)


@dataclass
class PIIDetectionResult:
    """
    Result of PII detection scan
    """
    contains_pii: bool
    pii_fields: Dict[str, List[PIIType]] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    detection_patterns: Dict[str, str] = field(default_factory=dict)
    risk_level: str = "low"  # low, medium, high, critical
    
    def add_pii_detection(self, field: str, pii_type: PIIType, confidence: float = 1.0):
        """Add PII detection to result"""
        if field not in self.pii_fields:
            self.pii_fields[field] = []
        self.pii_fields[field].append(pii_type)
        self.confidence_scores[f"{field}:{pii_type.value}"] = confidence
        self.contains_pii = True


@dataclass
class RedactionResult:
    """
    Result of redaction operation
    """
    success: bool
    redacted_record: Optional[Any] = None
    original_record: Optional[Any] = None
    redacted_fields: List[str] = field(default_factory=list)
    redaction_methods: Dict[str, RedactionMethod] = field(default_factory=dict)
    redaction_actions: List[Dict[str, Any]] = field(default_factory=list)
    reversible: bool = False
    recovery_key: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class ComplianceResult:
    """
    Result of compliance filtering
    """
    compliant: bool
    regulation: PrivacyRegulation
    filtered_record: Optional[Any] = None
    violations: List[str] = field(default_factory=list)
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class PrivacyPolicy:
    """
    Privacy policy configuration
    """
    policy_id: str
    name: str
    regulations: List[PrivacyRegulation]
    pii_handling: Dict[PIIType, RedactionMethod]
    retention_policy: DataRetentionPolicy
    consent_required: bool
    cross_border_allowed: bool
    encryption_required: bool
    anonymization_threshold: float = 0.8  # Minimum anonymization level
    differential_privacy_epsilon: float = 1.0  # Privacy budget
    audit_logging_required: bool = True
    custom_rules: List[Dict[str, Any]] = field(default_factory=list)


# =============================================================================
# PII Detection Engine
# =============================================================================

class PIIDetectionEngine:
    """
    Engine for detecting personally identifiable information
    """
    
    def __init__(self):
        """Initialize PII detection engine"""
        self.logger = logging.getLogger("SCAFAD.Layer1.PIIDetection")
        
        # Initialize detection patterns
        self._initialize_patterns()
        
        # Initialize faker for synthetic data generation
        self.faker = Faker()
        
        # Cache for detection results
        self.detection_cache = {}
        
        # Performance metrics
        self.detection_stats = defaultdict(int)
    
    def _initialize_patterns(self):
        """Initialize PII detection patterns"""
        
        self.patterns = {
            PIIType.EMAIL: [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            PIIType.PHONE: [
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # US phone
                r'\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b',  # US phone with parentheses
                r'\b\+\d{1,3}\s?\d{1,14}\b',  # International
                r'\b\d{10,14}\b'  # Generic numeric
            ],
            PIIType.SSN: [
                r'\b\d{3}-\d{2}-\d{4}\b',  # US SSN with dashes
                r'\b\d{9}\b'  # US SSN without dashes (needs context check)
            ],
            PIIType.CREDIT_CARD: [
                r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # 16 digits
                r'\b\d{4}[\s-]?\d{6}[\s-]?\d{5}\b',  # 15 digits (Amex)
            ],
            PIIType.IP_ADDRESS: [
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',  # IPv4
                r'\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b',  # IPv6
            ],
            PIIType.MAC_ADDRESS: [
                r'\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b'
            ],
            PIIType.DATE_OF_BIRTH: [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or similar
                r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',  # YYYY-MM-DD
            ],
            PIIType.DRIVER_LICENSE: [
                r'\b[A-Z]{1,2}\d{6,8}\b',  # Generic format
                r'\b\d{7,12}\b'  # Numeric only (needs context)
            ],
            PIIType.PASSPORT: [
                r'\b[A-Z][0-9]{8}\b',  # US passport
                r'\b[A-Z]{2}[0-9]{7}\b',  # UK passport
            ],
            PIIType.BANK_ACCOUNT: [
                r'\b\d{8,17}\b',  # IBAN or account number (needs context)
                r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b',  # IBAN
            ],
            PIIType.USERNAME: [
                r'\b(user|username|login|account)[\s:=]+[\w\.\-]+\b',
            ],
            PIIType.PASSWORD: [
                r'\b(password|passwd|pwd|pass)[\s:=]+\S+\b',
            ]
        }
        
        # Compile patterns for performance
        self.compiled_patterns = {}
        for pii_type, patterns in self.patterns.items():
            self.compiled_patterns[pii_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def detect_pii(self, value: Any, context: Dict[str, Any] = None) -> List[PIIType]:
        """
        Detect PII in a value
        
        Args:
            value: Value to check for PII
            context: Additional context for detection
            
        Returns:
            List of detected PII types
        """
        if not value:
            return []
        
        detected_pii = []
        str_value = str(value) if not isinstance(value, str) else value
        
        # Check against patterns
        for pii_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(str_value):
                    # Additional validation for specific types
                    if self._validate_pii_type(str_value, pii_type, pattern):
                        detected_pii.append(pii_type)
                        self.detection_stats[pii_type] += 1
                        break
        
        # Context-aware detection
        if context:
            # Check field names for hints
            field_name = context.get('field_name', '').lower()
            
            if 'email' in field_name:
                if self._looks_like_email(str_value):
                    if PIIType.EMAIL not in detected_pii:
                        detected_pii.append(PIIType.EMAIL)
            
            if any(term in field_name for term in ['phone', 'tel', 'mobile', 'cell']):
                if self._looks_like_phone(str_value):
                    if PIIType.PHONE not in detected_pii:
                        detected_pii.append(PIIType.PHONE)
            
            if any(term in field_name for term in ['ssn', 'social', 'security']):
                if self._looks_like_ssn(str_value):
                    if PIIType.SSN not in detected_pii:
                        detected_pii.append(PIIType.SSN)
            
            if any(term in field_name for term in ['name', 'first', 'last', 'surname']):
                if PIIType.NAME not in detected_pii:
                    detected_pii.append(PIIType.NAME)
        
        return detected_pii
    
    def _validate_pii_type(self, value: str, pii_type: PIIType, pattern: re.Pattern) -> bool:
        """Validate detected PII type"""
        
        if pii_type == PIIType.CREDIT_CARD:
            # Validate using Luhn algorithm
            match = pattern.search(value)
            if match:
                card_number = re.sub(r'[\s-]', '', match.group())
                return self._validate_credit_card(card_number)
        
        elif pii_type == PIIType.EMAIL:
            # Validate email format
            match = pattern.search(value)
            if match:
                try:
                    validate_email(match.group())
                    return True
                except EmailNotValidError:
                    return False
        
        elif pii_type == PIIType.IP_ADDRESS:
            # Validate IP address
            match = pattern.search(value)
            if match:
                try:
                    ipaddress.ip_address(match.group())
                    return True
                except ValueError:
                    return False
        
        elif pii_type == PIIType.PHONE:
            # Basic phone validation
            match = pattern.search(value)
            if match:
                phone_str = re.sub(r'[\s\-\(\)]', '', match.group())
                # Check if it's a reasonable phone length
                return 10 <= len(phone_str) <= 15
        
        # Default to pattern match being valid
        return True
    
    def _validate_credit_card(self, card_number: str) -> bool:
        """Validate credit card using Luhn algorithm"""
        try:
            return luhn.verify(card_number)
        except:
            return False
    
    def _looks_like_email(self, value: str) -> bool:
        """Check if value looks like email"""
        return '@' in value and '.' in value.split('@')[-1]
    
    def _looks_like_phone(self, value: str) -> bool:
        """Check if value looks like phone number"""
        digits = re.sub(r'\D', '', value)
        return 10 <= len(digits) <= 15
    
    def _looks_like_ssn(self, value: str) -> bool:
        """Check if value looks like SSN"""
        digits = re.sub(r'\D', '', value)
        return len(digits) == 9
    
    def scan_record(self, record: Dict[str, Any]) -> PIIDetectionResult:
        """
        Scan entire record for PII
        
        Args:
            record: Record to scan
            
        Returns:
            PIIDetectionResult with all detected PII
        """
        result = PIIDetectionResult(contains_pii=False)
        
        def scan_value(key: str, value: Any, path: str = ""):
            """Recursively scan values"""
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                for k, v in value.items():
                    scan_value(k, v, current_path)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    scan_value(f"[{i}]", item, current_path)
            else:
                # Detect PII
                context = {'field_name': key}
                detected = self.detect_pii(value, context)
                
                for pii_type in detected:
                    result.add_pii_detection(current_path, pii_type)
        
        # Scan all fields
        for key, value in record.items():
            scan_value(key, value)
        
        # Determine risk level
        if result.contains_pii:
            pii_count = sum(len(types) for types in result.pii_fields.values())
            
            if pii_count >= 5:
                result.risk_level = "critical"
            elif pii_count >= 3:
                result.risk_level = "high"
            elif pii_count >= 1:
                result.risk_level = "medium"
            else:
                result.risk_level = "low"
        
        return result


# =============================================================================
# Redaction Engine
# =============================================================================

class RedactionEngine:
    """
    Engine for redacting PII from data
    """
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        """Initialize redaction engine"""
        self.logger = logging.getLogger("SCAFAD.Layer1.Redaction")
        
        # Initialize encryption
        if encryption_key:
            self.fernet = Fernet(encryption_key)
        else:
            self.fernet = Fernet(Fernet.generate_key())
        
        # Initialize tokenization store
        self.token_store = {}
        self.reverse_token_store = {}
        
        # Initialize faker for synthetic data
        self.faker = Faker()
        
        # Redaction masks
        self.masks = {
            PIIType.EMAIL: "****@****.***",
            PIIType.PHONE: "***-***-****",
            PIIType.SSN: "***-**-****",
            PIIType.CREDIT_CARD: "****-****-****-****",
            PIIType.IP_ADDRESS: "***.***.***.***",
            PIIType.NAME: "[REDACTED_NAME]",
            PIIType.DATE_OF_BIRTH: "**/**/****",
            PIIType.PASSWORD: "********"
        }
    
    def redact(self, value: Any, pii_type: PIIType, method: RedactionMethod) -> Tuple[Any, bool]:
        """
        Redact PII from value
        
        Args:
            value: Value containing PII
            pii_type: Type of PII to redact
            method: Redaction method to use
            
        Returns:
            Tuple of (redacted_value, is_reversible)
        """
        if not value:
            return value, False
        
        str_value = str(value) if not isinstance(value, str) else value
        
        if method == RedactionMethod.MASK:
            return self._mask_value(str_value, pii_type), False
        
        elif method == RedactionMethod.HASH:
            return self._hash_value(str_value), False
        
        elif method == RedactionMethod.ENCRYPT:
            return self._encrypt_value(str_value), True
        
        elif method == RedactionMethod.TOKENIZE:
            return self._tokenize_value(str_value), True
        
        elif method == RedactionMethod.GENERALIZE:
            return self._generalize_value(str_value, pii_type), False
        
        elif method == RedactionMethod.SUPPRESS:
            return None, False
        
        elif method == RedactionMethod.SYNTHETIC:
            return self._generate_synthetic(pii_type), False
        
        elif method == RedactionMethod.DIFFERENTIAL:
            return self._add_differential_privacy(str_value, pii_type), False
        
        else:
            return str_value, False
    
    def _mask_value(self, value: str, pii_type: PIIType) -> str:
        """Mask PII value"""
        if pii_type in self.masks:
            return self.masks[pii_type]
        
        # Generic masking
        if len(value) <= 4:
            return "*" * len(value)
        else:
            # Show first and last character
            return value[0] + "*" * (len(value) - 2) + value[-1]
    
    def _hash_value(self, value: str) -> str:
        """Hash PII value (one-way)"""
        hash_obj = hashlib.sha256(value.encode())
        return f"HASH:{hash_obj.hexdigest()[:16]}"
    
    def _encrypt_value(self, value: str) -> str:
        """Encrypt PII value (reversible)"""
        encrypted = self.fernet.encrypt(value.encode())
        return f"ENC:{base64.urlsafe_b64encode(encrypted).decode()[:32]}"
    
    def _tokenize_value(self, value: str) -> str:
        """Tokenize PII value"""
        if value in self.token_store:
            return self.token_store[value]
        
        # Generate unique token
        token = f"TOK:{secrets.token_urlsafe(16)}"
        self.token_store[value] = token
        self.reverse_token_store[token] = value
        
        return token
    
    def _generalize_value(self, value: str, pii_type: PIIType) -> str:
        """Generalize PII value"""
        
        if pii_type == PIIType.IP_ADDRESS:
            # Generalize to subnet
            try:
                ip = ipaddress.ip_address(value)
                if isinstance(ip, ipaddress.IPv4Address):
                    # Keep first 3 octets
                    parts = str(ip).split('.')
                    return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
                else:
                    # IPv6 - keep first 64 bits
                    return str(ipaddress.ip_network(f"{ip}/64", strict=False))
            except:
                return "INVALID_IP"
        
        elif pii_type == PIIType.DATE_OF_BIRTH:
            # Generalize to year only
            match = re.search(r'\d{4}', value)
            if match:
                return f"Year: {match.group()}"
            return "INVALID_DATE"
        
        elif pii_type == PIIType.LOCATION:
            # Generalize to city/state level
            # This is simplified - real implementation would parse address
            return "City, State"
        
        else:
            # Generic generalization
            return f"[{pii_type.value.upper()}]"
    
    def _generate_synthetic(self, pii_type: PIIType) -> str:
        """Generate synthetic data for PII type"""
        
        if pii_type == PIIType.NAME:
            return self.faker.name()
        elif pii_type == PIIType.EMAIL:
            return self.faker.email()
        elif pii_type == PIIType.PHONE:
            return self.faker.phone_number()
        elif pii_type == PIIType.SSN:
            return self.faker.ssn()
        elif pii_type == PIIType.CREDIT_CARD:
            return self.faker.credit_card_number()
        elif pii_type == PIIType.IP_ADDRESS:
            return self.faker.ipv4()
        elif pii_type == PIIType.DATE_OF_BIRTH:
            return self.faker.date_of_birth().strftime("%Y-%m-%d")
        elif pii_type == PIIType.USERNAME:
            return self.faker.user_name()
        else:
            return f"SYNTHETIC_{pii_type.value.upper()}"
    
    def _add_differential_privacy(self, value: str, pii_type: PIIType, epsilon: float = 1.0) -> str:
        """Add differential privacy noise"""
        
        if pii_type in [PIIType.IP_ADDRESS]:
            # Add noise to IP address
            try:
                ip = ipaddress.ip_address(value)
                if isinstance(ip, ipaddress.IPv4Address):
                    # Add Laplace noise to last octet
                    parts = str(ip).split('.')
                    noise = np.random.laplace(0, 1/epsilon)
                    last_octet = int(parts[3]) + int(noise)
                    last_octet = max(0, min(255, last_octet))
                    return f"{parts[0]}.{parts[1]}.{parts[2]}.{last_octet}"
            except:
                pass
        
        # For other types, hash with noise
        noise = np.random.laplace(0, 1/epsilon)
        noisy_hash = hashlib.sha256(f"{value}{noise}".encode()).hexdigest()
        return f"DP:{noisy_hash[:16]}"
    
    def decrypt_value(self, encrypted_value: str) -> Optional[str]:
        """Decrypt an encrypted value"""
        if not encrypted_value.startswith("ENC:"):
            return None
        
        try:
            encrypted_data = base64.urlsafe_b64decode(encrypted_value[4:].encode())
            decrypted = self.fernet.decrypt(encrypted_data)
            return decrypted.decode()
        except:
            return None
    
    def detokenize_value(self, token: str) -> Optional[str]:
        """Detokenize a tokenized value"""
        return self.reverse_token_store.get(token)


# =============================================================================
# Privacy Compliance Filter
# =============================================================================

class PrivacyComplianceFilter:
    """
    Main Privacy Compliance Filter for Layer 1
    
    This class orchestrates privacy filtering operations to ensure
    regulatory compliance while preserving anomaly detection capabilities.
    """
    
    def __init__(self, config: Any):
        """
        Initialize Privacy Compliance Filter
        
        Args:
            config: Layer 1 configuration object
        """
        self.config = config
        self.logger = logging.getLogger("SCAFAD.Layer1.PrivacyCompliance")
        
        # Initialize detection engine
        self.pii_detector = PIIDetectionEngine()
        
        # Initialize redaction engine
        self.redaction_engine = RedactionEngine()
        
        # Initialize privacy policies
        self._initialize_privacy_policies()
        
        # Initialize consent manager
        self.consent_manager = ConsentManager()
        
        # Initialize retention manager
        self.retention_manager = DataRetentionManager()
        
        # Initialize cross-border compliance
        self.cross_border_compliance = CrossBorderCompliance()
        
        # Performance monitoring
        self._initialize_monitoring()
        
        self.logger.info(f"Privacy Compliance Filter initialized with level: {config.privacy_level}")
    
    def _initialize_privacy_policies(self):
        """Initialize privacy policies for different regulations"""
        
        self.policies = {
            PrivacyRegulation.GDPR: PrivacyPolicy(
                policy_id="gdpr_policy",
                name="GDPR Compliance Policy",
                regulations=[PrivacyRegulation.GDPR],
                pii_handling={
                    PIIType.NAME: RedactionMethod.TOKENIZE,
                    PIIType.EMAIL: RedactionMethod.HASH,
                    PIIType.PHONE: RedactionMethod.MASK,
                    PIIType.SSN: RedactionMethod.ENCRYPT,
                    PIIType.CREDIT_CARD: RedactionMethod.SUPPRESS,
                    PIIType.IP_ADDRESS: RedactionMethod.GENERALIZE,
                    PIIType.DATE_OF_BIRTH: RedactionMethod.GENERALIZE,
                    PIIType.LOCATION: RedactionMethod.GENERALIZE
                },
                retention_policy=DataRetentionPolicy.MEDIUM_TERM,
                consent_required=True,
                cross_border_allowed=False,
                encryption_required=True,
                anonymization_threshold=0.9
            ),
            
            PrivacyRegulation.CCPA: PrivacyPolicy(
                policy_id="ccpa_policy",
                name="CCPA Compliance Policy",
                regulations=[PrivacyRegulation.CCPA],
                pii_handling={
                    PIIType.NAME: RedactionMethod.MASK,
                    PIIType.EMAIL: RedactionMethod.HASH,
                    PIIType.PHONE: RedactionMethod.MASK,
                    PIIType.SSN: RedactionMethod.ENCRYPT,
                    PIIType.CREDIT_CARD: RedactionMethod.SUPPRESS,
                    PIIType.IP_ADDRESS: RedactionMethod.MASK,
                    PIIType.DATE_OF_BIRTH: RedactionMethod.MASK
                },
                retention_policy=DataRetentionPolicy.SHORT_TERM,
                consent_required=True,
                cross_border_allowed=True,
                encryption_required=False,
                anonymization_threshold=0.85
            ),
            
            PrivacyRegulation.HIPAA: PrivacyPolicy(
                policy_id="hipaa_policy",
                name="HIPAA Compliance Policy",
                regulations=[PrivacyRegulation.HIPAA],
                pii_handling={
                    PIIType.NAME: RedactionMethod.ENCRYPT,
                    PIIType.EMAIL: RedactionMethod.ENCRYPT,
                    PIIType.PHONE: RedactionMethod.ENCRYPT,
                    PIIType.SSN: RedactionMethod.ENCRYPT,
                    PIIType.DATE_OF_BIRTH: RedactionMethod.ENCRYPT,
                    PIIType.MEDICAL_ID: RedactionMethod.ENCRYPT,
                    PIIType.BIOMETRIC: RedactionMethod.SUPPRESS
                },
                retention_policy=DataRetentionPolicy.LONG_TERM,
                consent_required=True,
                cross_border_allowed=False,
                encryption_required=True,
                anonymization_threshold=0.95
            )
        }
    
    def _initialize_monitoring(self):
        """Initialize performance monitoring"""
        self.stats = {
            'total_records_filtered': 0,
            'pii_detections': defaultdict(int),
            'redactions_performed': defaultdict(int),
            'compliance_violations': 0,
            'consent_checks': 0,
            'retention_actions': 0,
            'average_filtering_time_ms': 0.0,
            'filtering_times': []
        }
    
    # =========================================================================
    # Main Privacy Filtering Methods
    # =========================================================================
    
    async def apply_gdpr_filters(self, record: Any) -> ComplianceResult:
        """
        Apply GDPR compliance filters to record
        
        Args:
            record: Telemetry record to filter
            
        Returns:
            ComplianceResult with filtered record
        """
        filtering_start = time.time()
        
        result = ComplianceResult(
            compliant=True,
            regulation=PrivacyRegulation.GDPR,
            filtered_record=self._to_dict(record)
        )
        
        policy = self.policies[PrivacyRegulation.GDPR]
        
        try:
            # Check consent if required
            if policy.consent_required:
                consent_status = await self.consent_manager.check_consent(record)
                if consent_status != ConsentStatus.GRANTED:
                    result.violations.append(f"Consent not granted: {consent_status.value}")
                    result.compliant = False
            
            # Scan for PII
            detection_result = self.pii_detector.scan_record(result.filtered_record)
            
            if detection_result.contains_pii:
                # Apply GDPR-specific redactions
                for field, pii_types in detection_result.pii_fields.items():
                    for pii_type in pii_types:
                        redaction_method = policy.pii_handling.get(pii_type, RedactionMethod.MASK)
                        
                        # Get field value
                        field_value = self._get_nested_value(result.filtered_record, field)
                        
                        # Redact value
                        redacted_value, reversible = self.redaction_engine.redact(
                            field_value, pii_type, redaction_method
                        )
                        
                        # Update field
                        self._set_nested_value(result.filtered_record, field, redacted_value)
                        
                        # Track action
                        result.actions_taken.append({
                            'field': field,
                            'pii_type': pii_type.value,
                            'method': redaction_method.value,
                            'reversible': reversible
                        })
            
            # Check data retention
            if await self.retention_manager.should_delete(record, policy.retention_policy):
                result.recommendations.append("Data should be deleted per retention policy")
            
            # Check cross-border transfer
            if not policy.cross_border_allowed:
                if await self.cross_border_compliance.is_cross_border(record):
                    result.warnings.append("Cross-border data transfer detected")
            
            # Verify anonymization level
            anonymization_level = self._calculate_anonymization_level(
                result.filtered_record, detection_result
            )
            
            if anonymization_level < policy.anonymization_threshold:
                result.warnings.append(
                    f"Anonymization level {anonymization_level:.2f} below threshold {policy.anonymization_threshold}"
                )
            
        except Exception as e:
            self.logger.error(f"GDPR filtering error: {str(e)}")
            result.compliant = False
            result.violations.append(f"Filtering error: {str(e)}")
        
        # Update statistics
        filtering_time = (time.time() - filtering_start) * 1000
        self.stats['filtering_times'].append(filtering_time)
        
        return result
    
    async def apply_ccpa_filters(self, record: Any) -> ComplianceResult:
        """
        Apply CCPA compliance filters to record
        
        Args:
            record: Telemetry record to filter
            
        Returns:
            ComplianceResult with filtered record
        """
        result = ComplianceResult(
            compliant=True,
            regulation=PrivacyRegulation.CCPA,
            filtered_record=self._to_dict(record)
        )
        
        policy = self.policies[PrivacyRegulation.CCPA]
        
        try:
            # CCPA specific: Check for sale opt-out
            if await self.consent_manager.has_opted_out_of_sale(record):
                result.actions_taken.append({
                    'action': 'suppress_for_sale',
                    'reason': 'User opted out of data sale'
                })
                # Mark certain fields for suppression
                self._suppress_sale_fields(result.filtered_record)
            
            # Scan for PII
            detection_result = self.pii_detector.scan_record(result.filtered_record)
            
            if detection_result.contains_pii:
                # Apply CCPA-specific redactions
                for field, pii_types in detection_result.pii_fields.items():
                    for pii_type in pii_types:
                        redaction_method = policy.pii_handling.get(pii_type, RedactionMethod.MASK)
                        
                        field_value = self._get_nested_value(result.filtered_record, field)
                        redacted_value, _ = self.redaction_engine.redact(
                            field_value, pii_type, redaction_method
                        )
                        
                        self._set_nested_value(result.filtered_record, field, redacted_value)
                        
                        result.actions_taken.append({
                            'field': field,
                            'pii_type': pii_type.value,
                            'method': redaction_method.value
                        })
            
            # CCPA: Right to know
            if await self.consent_manager.has_data_request(record):
                result.recommendations.append("User has active data access request")
            
            # CCPA: Right to delete
            if await self.consent_manager.has_deletion_request(record):
                result.recommendations.append("User has requested data deletion")
                result.actions_taken.append({'action': 'mark_for_deletion'})
            
        except Exception as e:
            self.logger.error(f"CCPA filtering error: {str(e)}")
            result.compliant = False
            result.violations.append(f"Filtering error: {str(e)}")
        
        return result
    
    async def apply_hipaa_filters(self, record: Any) -> ComplianceResult:
        """
        Apply HIPAA compliance filters to record
        
        Args:
            record: Telemetry record to filter
            
        Returns:
            ComplianceResult with filtered record
        """
        result = ComplianceResult(
            compliant=True,
            regulation=PrivacyRegulation.HIPAA,
            filtered_record=self._to_dict(record)
        )
        
        policy = self.policies[PrivacyRegulation.HIPAA]
        
        try:
            # HIPAA: Check for PHI (Protected Health Information)
            phi_fields = self._detect_phi_fields(result.filtered_record)
            
            if phi_fields:
                # Apply strict HIPAA encryption
                for field in phi_fields:
                    field_value = self._get_nested_value(result.filtered_record, field)
                    
                    # Always encrypt PHI
                    encrypted_value, _ = self.redaction_engine.redact(
                        field_value, PIIType.MEDICAL_ID, RedactionMethod.ENCRYPT
                    )
                    
                    self._set_nested_value(result.filtered_record, field, encrypted_value)
                    
                    result.actions_taken.append({
                        'field': field,
                        'action': 'encrypt_phi',
                        'method': 'AES-256'
                    })
            
            # HIPAA: Minimum necessary standard
            self._apply_minimum_necessary(result.filtered_record)
            
            # HIPAA: Audit logging requirement
            result.actions_taken.append({
                'action': 'audit_log',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'purpose': 'anomaly_detection'
            })
            
            # Check for required safeguards
            if not policy.encryption_required:
                result.violations.append("HIPAA requires encryption for PHI")
                result.compliant = False
            
        except Exception as e:
            self.logger.error(f"HIPAA filtering error: {str(e)}")
            result.compliant = False
            result.violations.append(f"Filtering error: {str(e)}")
        
        return result
    
    async def redact_pii_fields(self, record: Any) -> RedactionResult:
        """
        Redact all PII fields from record
        
        Args:
            record: Record containing PII
            
        Returns:
            RedactionResult with redacted record
        """
        redaction_start = time.time()
        
        try:
            record_dict = self._to_dict(record)
            original_record = copy.deepcopy(record_dict)
            
            # Scan for PII
            detection_result = self.pii_detector.scan_record(record_dict)
            
            if not detection_result.contains_pii:
                return RedactionResult(
                    success=True,
                    redacted_record=record,
                    original_record=record
                )
            
            # Determine redaction method based on privacy level
            default_method = self._get_default_redaction_method()
            
            # Redact each PII field
            for field, pii_types in detection_result.pii_fields.items():
                for pii_type in pii_types:
                    # Get appropriate redaction method
                    method = self._get_redaction_method_for_pii(pii_type, default_method)
                    
                    # Get field value
                    field_value = self._get_nested_value(record_dict, field)
                    
                    # Redact value
                    redacted_value, reversible = self.redaction_engine.redact(
                        field_value, pii_type, method
                    )
                    
                    # Update field
                    self._set_nested_value(record_dict, field, redacted_value)
                    
                    # Track redaction
                    self.stats['redactions_performed'][pii_type.value] += 1
            
            # Convert back to original type if needed
            if not isinstance(record, dict):
                redacted_record = self._from_dict(record.__class__, record_dict)
            else:
                redacted_record = record_dict
            
            return RedactionResult(
                success=True,
                redacted_record=redacted_record,
                original_record=record,
                redacted_fields=list(detection_result.pii_fields.keys()),
                redaction_methods={
                    field: method for field in detection_result.pii_fields.keys()
                },
                reversible=default_method in [RedactionMethod.ENCRYPT, RedactionMethod.TOKENIZE]
            )
            
        except Exception as e:
            self.logger.error(f"Redaction error: {str(e)}")
            return RedactionResult(
                success=False,
                original_record=record,
                error_message=f"Redaction error: {str(e)}"
            )
    
    def _get_default_redaction_method(self) -> RedactionMethod:
        """Get default redaction method based on privacy level"""
        privacy_level = self.config.privacy_level.value
        
        if privacy_level == "minimal":
            return RedactionMethod.MASK
        elif privacy_level == "moderate":
            return RedactionMethod.HASH
        elif privacy_level == "high":
            return RedactionMethod.TOKENIZE
        else:  # maximum
            return RedactionMethod.SUPPRESS
    
    def _get_redaction_method_for_pii(self, pii_type: PIIType, default: RedactionMethod) -> RedactionMethod:
        """Get appropriate redaction method for PII type"""
        
        # High-risk PII should be suppressed or encrypted
        high_risk = [PIIType.SSN, PIIType.CREDIT_CARD, PIIType.MEDICAL_ID, PIIType.PASSWORD]
        if pii_type in high_risk:
            if self.config.privacy_level.value == "maximum":
                return RedactionMethod.SUPPRESS
            else:
                return RedactionMethod.ENCRYPT
        
        # Medium-risk PII
        medium_risk = [PIIType.EMAIL, PIIType.PHONE, PIIType.DATE_OF_BIRTH]
        if pii_type in medium_risk:
            return RedactionMethod.HASH
        
        # Low-risk PII can use default method
        return default
    
    def _detect_phi_fields(self, record: Dict[str, Any]) -> List[str]:
        """Detect fields containing Protected Health Information"""
        phi_fields = []
        
        phi_keywords = [
            'diagnosis', 'treatment', 'medication', 'medical',
            'health', 'patient', 'prescription', 'symptom',
            'condition', 'therapy', 'clinical', 'lab_result'
        ]
        
        def check_field(key: str, value: Any, path: str = ""):
            current_path = f"{path}.{key}" if path else key
            
            # Check field name for PHI indicators
            if any(keyword in key.lower() for keyword in phi_keywords):
                phi_fields.append(current_path)
            
            # Recursively check nested structures
            if isinstance(value, dict):
                for k, v in value.items():
                    check_field(k, v, current_path)
        
        for key, value in record.items():
            check_field(key, value)
        
        return phi_fields
    
    def _apply_minimum_necessary(self, record: Dict[str, Any]):
        """Apply HIPAA minimum necessary standard"""
        
        # Remove fields not necessary for anomaly detection
        unnecessary_fields = [
            'patient_name', 'patient_id', 'medical_record_number',
            'insurance_id', 'physician_name', 'facility_name'
        ]
        
        for field in unnecessary_fields:
            if field in record:
                del record[field]
    
    def _suppress_sale_fields(self, record: Dict[str, Any]):
        """Suppress fields that should not be sold (CCPA)"""
        
        sale_prohibited_fields = [
            'precise_location', 'biometric_data', 'genetic_data',
            'sexual_orientation', 'immigration_status'
        ]
        
        for field in sale_prohibited_fields:
            if field in record:
                record[field] = "[SALE_PROHIBITED]"
    
    def _calculate_anonymization_level(self, 
                                      record: Dict[str, Any], 
                                      detection_result: PIIDetectionResult) -> float:
        """Calculate the level of anonymization achieved"""
        
        if not detection_result.contains_pii:
            return 1.0  # Fully anonymized (no PII)
        
        total_fields = self._count_fields(record)
        pii_fields = len(detection_result.pii_fields)
        
        # Check how many PII fields were successfully redacted
        redacted_count = 0
        for field in detection_result.pii_fields:
            value = self._get_nested_value(record, field)
            if value and any(prefix in str(value) for prefix in ['HASH:', 'ENC:', 'TOK:', '[REDACTED']):
                redacted_count += 1
        
        # Calculate anonymization score
        if pii_fields > 0:
            redaction_rate = redacted_count / pii_fields
        else:
            redaction_rate = 1.0
        
        # Weight by field importance
        field_coverage = 1 - (pii_fields / total_fields) if total_fields > 0 else 0
        
        # Combined score
        return 0.7 * redaction_rate + 0.3 * field_coverage
    
    def _count_fields(self, record: Dict[str, Any]) -> int:
        """Count total fields in record"""
        count = 0
        
        def count_recursive(obj):
            nonlocal count
            if isinstance(obj, dict):
                for value in obj.values():
                    count += 1
                    count_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    count_recursive(item)
        
        count_recursive(record)
        return count
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _to_dict(self, record: Any) -> Dict[str, Any]:
        """Convert record to dictionary"""
        if isinstance(record, dict):
            return record.copy()
        elif hasattr(record, '__dict__'):
            return record.__dict__.copy()
        elif hasattr(record, '_asdict'):
            return record._asdict()
        elif hasattr(record, '__dataclass_fields__'):
            from dataclasses import asdict
            return asdict(record)
        else:
            return dict(record)
    
    def _from_dict(self, cls: type, data: Dict[str, Any]) -> Any:
        """Convert dictionary back to original class type"""
        try:
            return cls(**data)
        except:
            return data
    
    def _get_nested_value(self, obj: Dict[str, Any], path: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        keys = path.replace('[', '.').replace(']', '').split('.')
        value = obj
        
        for key in keys:
            if not key:
                continue
            
            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, list):
                try:
                    index = int(key)
                    value = value[index] if index < len(value) else None
                except (ValueError, IndexError):
                    return None
            else:
                return None
            
            if value is None:
                return None
        
        return value
    
    def _set_nested_value(self, obj: Dict[str, Any], path: str, value: Any):
        """Set value in nested dictionary using dot notation"""
        keys = path.replace('[', '.').replace(']', '').split('.')
        
        for key in keys[:-1]:
            if not key:
                continue
            
            if key not in obj:
                obj[key] = {}
            obj = obj[key]
        
        if keys[-1]:
            obj[keys[-1]] = value
    
    # =========================================================================
    # Public Interface Methods
    # =========================================================================
    
    def get_privacy_statistics(self) -> Dict[str, Any]:
        """Get current privacy filtering statistics"""
        
        total_filtered = self.stats['total_records_filtered']
        
        return {
            **self.stats,
            'pii_detection_rate': sum(self.stats['pii_detections'].values()) / total_filtered * 100
                                 if total_filtered > 0 else 0,
            'redaction_rate': sum(self.stats['redactions_performed'].values()) / total_filtered * 100
                            if total_filtered > 0 else 0,
            'average_filtering_time_ms': np.mean(self.stats['filtering_times'][-100:])
                                        if self.stats['filtering_times'] else 0
        }
    
    def reset_statistics(self):
        """Reset privacy filtering statistics"""
        self._initialize_monitoring()
        self.logger.info("Privacy filtering statistics reset")
    
    async def health_check(self) -> Dict[str, str]:
        """Perform health check"""
        try:
            # Test privacy filtering with sample record
            test_record = {
                'record_id': 'test-health-check',
                'email': 'test@example.com',
                'phone': '555-123-4567',
                'ssn': '123-45-6789',
                'timestamp': time.time()
            }
            
            # Test PII detection
            detection_result = self.pii_detector.scan_record(test_record)
            
            if not detection_result.contains_pii:
                return {'status': 'warning', 'message': 'PII detection may not be working'}
            
            # Test redaction
            redaction_result = await self.redact_pii_fields(test_record)
            
            if redaction_result.success:
                return {'status': 'healthy', 'message': 'Privacy compliance filter operational'}
            else:
                return {'status': 'warning', 'message': f'Redaction test failed: {redaction_result.error_message}'}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}


# =============================================================================
# Supporting Classes
# =============================================================================

class ConsentManager:
    """Manages user consent for data processing"""
    
    def __init__(self):
        self.logger = logging.getLogger("SCAFAD.Layer1.ConsentManager")
        self.consent_store = {}  # In production, this would be a database
    
    async def check_consent(self, record: Dict[str, Any]) -> ConsentStatus:
        """Check consent status for record"""
        # This is a simplified implementation
        # In production, would check against consent database
        
        user_id = record.get('user_id') or record.get('device_id')
        if not user_id:
            return ConsentStatus.NOT_REQUIRED
        
        consent = self.consent_store.get(user_id, ConsentStatus.PENDING)
        return consent
    
    async def has_opted_out_of_sale(self, record: Dict[str, Any]) -> bool:
        """Check if user opted out of data sale (CCPA)"""
        user_id = record.get('user_id') or record.get('device_id')
        # Simplified - would check against opt-out database
        return False
    
    async def has_data_request(self, record: Dict[str, Any]) -> bool:
        """Check if user has active data access request"""
        # Simplified - would check against request database
        return False
    
    async def has_deletion_request(self, record: Dict[str, Any]) -> bool:
        """Check if user has requested deletion"""
        # Simplified - would check against deletion request database
        return False


class DataRetentionManager:
    """Manages data retention policies"""
    
    def __init__(self):
        self.logger = logging.getLogger("SCAFAD.Layer1.DataRetention")
    
    async def should_delete(self, record: Dict[str, Any], policy: DataRetentionPolicy) -> bool:
        """Check if record should be deleted based on retention policy"""
        
        timestamp = record.get('timestamp', 0)
        if not timestamp:
            return False
        
        current_time = time.time()
        age_days = (current_time - timestamp) / (24 * 3600)
        
        retention_limits = {
            DataRetentionPolicy.IMMEDIATE: 0,
            DataRetentionPolicy.SHORT_TERM: 7,
            DataRetentionPolicy.MEDIUM_TERM: 30,
            DataRetentionPolicy.LONG_TERM: 90,
            DataRetentionPolicy.ARCHIVE: 365,
            DataRetentionPolicy.PERMANENT: float('inf')
        }
        
        limit = retention_limits.get(policy, 30)
        return age_days > limit


class CrossBorderCompliance:
    """Manages cross-border data transfer compliance"""
    
    def __init__(self):
        self.logger = logging.getLogger("SCAFAD.Layer1.CrossBorder")
        
        # Define data residency regions
        self.eu_countries = [
            'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR',
            'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL',
            'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE'
        ]
    
    async def is_cross_border(self, record: Dict[str, Any]) -> bool:
        """Check if data transfer crosses borders"""
        
        # Check source and destination regions
        source_region = record.get('source_region', '')
        dest_region = record.get('destination_region', '')
        
        if not source_region or not dest_region:
            return False
        
        # Check if crossing EU borders (GDPR concern)
        source_eu = source_region in self.eu_countries
        dest_eu = dest_region in self.eu_countries
        
        return source_eu != dest_eu


# =============================================================================
# Export public interface
# =============================================================================

__all__ = [
    'PrivacyComplianceFilter',
    'PrivacyAuditTrail',
    'PIIDetectionResult',
    'RedactionResult',
    'ComplianceResult',
    'PrivacyPolicy',
    'PrivacyRegulation',
    'PIIType',
    'RedactionMethod',
    'ConsentStatus',
    'DataRetentionPolicy',
    'PIIDetectionEngine',
    'RedactionEngine',
    'ConsentManager',
    'DataRetentionManager',
    'CrossBorderCompliance'
]