#!/usr/bin/env python3
"""
SCAFAD Layer 1: Input Validation Gateway
========================================

The Input Validation Gateway serves as the first line of defense in Layer 1's processing pipeline.
It ensures that all incoming telemetry data from Layer 0 meets structural, semantic, and 
security requirements before entering the data conditioning pipeline.

Key Responsibilities:
- Structural validation of telemetry records
- Semantic validation of field values and relationships
- Security validation against injection attacks
- Malformed field sanitization while preserving anomaly semantics
- Type enforcement and schema compliance checking
- Range validation and boundary checking
- Contextual validation based on execution phase

Performance Targets:
- Validation latency: <0.3ms per record
- False rejection rate: <0.1%
- Anomaly preservation: 100% (no anomaly signatures lost during validation)
- Security validation: 100% (all injection attempts blocked)

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
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from enum import Enum, auto
from datetime import datetime, timezone
import ipaddress
import base64
import urllib.parse
from collections import defaultdict
import traceback

# Mathematical and statistical operations
import numpy as np
from scipy import stats

# Security and validation utilities
from jsonschema import validate, ValidationError as JsonSchemaValidationError, Draft7Validator
import cerberus
import validators

# Performance monitoring
import time
from functools import wraps
import psutil


# =============================================================================
# Validation Data Models and Enums
# =============================================================================

class ValidationLevel(Enum):
    """Validation strictness levels"""
    MINIMAL = "minimal"          # Basic structural validation only
    STANDARD = "standard"        # Standard validation with type checking
    STRICT = "strict"           # Comprehensive validation with semantic checks
    PARANOID = "paranoid"       # Maximum validation including security scans

class ValidationStatus(Enum):
    """Validation result status codes"""
    VALID = "valid"
    INVALID_STRUCTURE = "invalid_structure"
    INVALID_SCHEMA = "invalid_schema"
    INVALID_TYPE = "invalid_type"
    INVALID_RANGE = "invalid_range"
    INVALID_SEMANTIC = "invalid_semantic"
    SECURITY_VIOLATION = "security_violation"
    MALFORMED_FIELD = "malformed_field"
    MISSING_REQUIRED = "missing_required"
    PARTIAL_VALID = "partial_valid"

class FieldType(Enum):
    """Supported telemetry field types"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"
    IP_ADDRESS = "ip_address"
    URL = "url"
    EMAIL = "email"
    JSON_OBJECT = "json_object"
    BASE64 = "base64"
    UUID = "uuid"
    ENUM = "enum"
    ARRAY = "array"
    NESTED_OBJECT = "nested_object"

class SecurityThreatType(Enum):
    """Security threat categories"""
    SQL_INJECTION = "sql_injection"
    COMMAND_INJECTION = "command_injection"
    XXE_INJECTION = "xxe_injection"
    PATH_TRAVERSAL = "path_traversal"
    LDAP_INJECTION = "ldap_injection"
    SCRIPT_INJECTION = "script_injection"
    BUFFER_OVERFLOW = "buffer_overflow"
    FORMAT_STRING = "format_string"


@dataclass
class ValidationResult:
    """
    Comprehensive validation result for a telemetry record
    """
    is_valid: bool
    status: ValidationStatus
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    sanitized_fields: Dict[str, Any] = field(default_factory=dict)
    validation_metadata: Dict[str, Any] = field(default_factory=dict)
    security_assessment: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None
    
    def add_error(self, field: str, error_type: str, message: str, severity: str = "error"):
        """Add validation error"""
        self.errors.append({
            'field': field,
            'type': error_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        self.is_valid = False
    
    def add_warning(self, field: str, warning_type: str, message: str):
        """Add validation warning"""
        self.warnings.append({
            'field': field,
            'type': warning_type,
            'message': message,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })


@dataclass
class FieldValidationRule:
    """
    Validation rule for a specific field
    """
    field_name: str
    field_type: FieldType
    required: bool = True
    nullable: bool = False
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable] = None
    security_checks: List[SecurityThreatType] = field(default_factory=list)
    transformation: Optional[Callable] = None
    default_value: Any = None


@dataclass
class SchemaDefinition:
    """
    Schema definition for telemetry records
    """
    schema_version: str
    schema_name: str
    field_rules: Dict[str, FieldValidationRule]
    required_fields: List[str]
    conditional_rules: List[Dict[str, Any]] = field(default_factory=list)
    cross_field_validations: List[Callable] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Core Validation Engine
# =============================================================================

class ValidationEngine:
    """
    Core validation engine that performs multi-level validation
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """Initialize validation engine with specified validation level"""
        self.validation_level = validation_level
        self.logger = logging.getLogger("SCAFAD.Layer1.ValidationEngine")
        
        # Initialize validation components
        self._initialize_validators()
        self._initialize_security_patterns()
        self._initialize_performance_monitoring()
        
        # Validation statistics
        self.stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'security_violations': 0,
            'avg_validation_time_ms': 0.0
        }
    
    def _initialize_validators(self):
        """Initialize various validators"""
        
        # Type validators
        self.type_validators = {
            FieldType.STRING: self._validate_string,
            FieldType.INTEGER: self._validate_integer,
            FieldType.FLOAT: self._validate_float,
            FieldType.BOOLEAN: self._validate_boolean,
            FieldType.TIMESTAMP: self._validate_timestamp,
            FieldType.IP_ADDRESS: self._validate_ip_address,
            FieldType.URL: self._validate_url,
            FieldType.EMAIL: self._validate_email,
            FieldType.JSON_OBJECT: self._validate_json_object,
            FieldType.BASE64: self._validate_base64,
            FieldType.UUID: self._validate_uuid,
            FieldType.ARRAY: self._validate_array,
            FieldType.NESTED_OBJECT: self._validate_nested_object
        }
        
        # Range validators
        self.range_validators = {
            'numeric': self._validate_numeric_range,
            'string_length': self._validate_string_length,
            'array_size': self._validate_array_size,
            'timestamp': self._validate_timestamp_range
        }
    
    def _initialize_security_patterns(self):
        """Initialize security threat detection patterns"""
        
        self.security_patterns = {
            SecurityThreatType.SQL_INJECTION: [
                r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|FROM|WHERE)\b)",
                r"('|\"|;|--|\*/|/\*|\|\||&&)",
                r"(xp_|sp_|0x[0-9a-f]+)",
            ],
            SecurityThreatType.COMMAND_INJECTION: [
                r"(;|\||&&|\$\(|\`|>|<|\{|\})",
                r"(\b(sh|bash|cmd|powershell|python|perl|ruby|php)\b)",
                r"(/bin/|/usr/bin/|/etc/|c:\\|cmd\.exe)",
            ],
            SecurityThreatType.PATH_TRAVERSAL: [
                r"(\.\./|\.\.\\|%2e%2e%2f|%2e%2e/)",
                r"(\.\.;|\.\.%00|\.\.%01)",
                r"(/etc/passwd|/etc/shadow|boot\.ini|win\.ini)",
            ],
            SecurityThreatType.SCRIPT_INJECTION: [
                r"(<script[^>]*>|</script>|javascript:|onerror=|onload=)",
                r"(eval\(|setTimeout\(|setInterval\(|Function\()",
                r"(document\.|window\.|alert\(|prompt\(|confirm\()",
            ],
            SecurityThreatType.LDAP_INJECTION: [
                r"(\*|\(|\)|\\|NUL|%00|%2a|%28|%29|%5c|%00)",
                r"(objectClass=|cn=|ou=|dc=)",
            ],
            SecurityThreatType.XXE_INJECTION: [
                r"(<!DOCTYPE[^>]*>|<!ENTITY[^>]*>|SYSTEM)",
                r"(file://|http://|https://|ftp://|php://|expect://)",
            ],
            SecurityThreatType.FORMAT_STRING: [
                r"(%s|%d|%x|%n|%p|%hn|%hhn|%lln)",
            ],
        }
        
        # Compile regex patterns for performance
        self.compiled_patterns = {}
        for threat_type, patterns in self.security_patterns.items():
            self.compiled_patterns[threat_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def _initialize_performance_monitoring(self):
        """Initialize performance monitoring"""
        self.performance_metrics = {
            'validation_times': [],
            'field_validation_times': defaultdict(list),
            'security_scan_times': [],
            'memory_usage': []
        }
    
    # =========================================================================
    # Type Validators
    # =========================================================================
    
    def _validate_string(self, value: Any, rule: FieldValidationRule) -> Tuple[bool, Optional[str]]:
        """Validate string field"""
        if not isinstance(value, str):
            return False, f"Expected string, got {type(value).__name__}"
        
        if rule.min_length and len(value) < rule.min_length:
            return False, f"String length {len(value)} below minimum {rule.min_length}"
        
        if rule.max_length and len(value) > rule.max_length:
            return False, f"String length {len(value)} exceeds maximum {rule.max_length}"
        
        if rule.pattern:
            if not re.match(rule.pattern, value):
                return False, f"String does not match pattern {rule.pattern}"
        
        if rule.allowed_values and value not in rule.allowed_values:
            return False, f"Value '{value}' not in allowed values"
        
        return True, None
    
    def _validate_integer(self, value: Any, rule: FieldValidationRule) -> Tuple[bool, Optional[str]]:
        """Validate integer field"""
        try:
            int_value = int(value)
        except (ValueError, TypeError):
            return False, f"Cannot convert {value} to integer"
        
        if rule.min_value is not None and int_value < rule.min_value:
            return False, f"Value {int_value} below minimum {rule.min_value}"
        
        if rule.max_value is not None and int_value > rule.max_value:
            return False, f"Value {int_value} exceeds maximum {rule.max_value}"
        
        return True, None
    
    def _validate_float(self, value: Any, rule: FieldValidationRule) -> Tuple[bool, Optional[str]]:
        """Validate float field"""
        try:
            float_value = float(value)
        except (ValueError, TypeError):
            return False, f"Cannot convert {value} to float"
        
        if rule.min_value is not None and float_value < rule.min_value:
            return False, f"Value {float_value} below minimum {rule.min_value}"
        
        if rule.max_value is not None and float_value > rule.max_value:
            return False, f"Value {float_value} exceeds maximum {rule.max_value}"
        
        # Check for NaN and Inf
        if np.isnan(float_value):
            return False, "Value is NaN"
        if np.isinf(float_value):
            return False, "Value is infinite"
        
        return True, None
    
    def _validate_boolean(self, value: Any, rule: FieldValidationRule) -> Tuple[bool, Optional[str]]:
        """Validate boolean field"""
        if isinstance(value, bool):
            return True, None
        
        if isinstance(value, str):
            if value.lower() in ['true', 'false', '1', '0', 'yes', 'no']:
                return True, None
        
        if isinstance(value, (int, float)):
            if value in [0, 1]:
                return True, None
        
        return False, f"Cannot interpret {value} as boolean"
    
    def _validate_timestamp(self, value: Any, rule: FieldValidationRule) -> Tuple[bool, Optional[str]]:
        """Validate timestamp field"""
        try:
            if isinstance(value, (int, float)):
                # Unix timestamp
                if value < 0:
                    return False, "Negative timestamp not allowed"
                if value > 2147483647:  # Max 32-bit timestamp
                    # Could be milliseconds
                    if value > 2147483647000:
                        return False, "Timestamp too far in future"
            elif isinstance(value, str):
                # Try parsing ISO format
                datetime.fromisoformat(value.replace('Z', '+00:00'))
            else:
                return False, f"Invalid timestamp type {type(value).__name__}"
            
            return True, None
        except Exception as e:
            return False, f"Invalid timestamp format: {str(e)}"
    
    def _validate_ip_address(self, value: Any, rule: FieldValidationRule) -> Tuple[bool, Optional[str]]:
        """Validate IP address field"""
        if not isinstance(value, str):
            return False, "IP address must be string"
        
        try:
            # Try IPv4
            ipaddress.IPv4Address(value)
            return True, None
        except ipaddress.AddressValueError:
            pass
        
        try:
            # Try IPv6
            ipaddress.IPv6Address(value)
            return True, None
        except ipaddress.AddressValueError:
            return False, f"Invalid IP address: {value}"
    
    def _validate_url(self, value: Any, rule: FieldValidationRule) -> Tuple[bool, Optional[str]]:
        """Validate URL field"""
        if not isinstance(value, str):
            return False, "URL must be string"
        
        if not validators.url(value):
            return False, f"Invalid URL format: {value}"
        
        # Additional security checks for URLs
        parsed = urllib.parse.urlparse(value)
        
        # Check for suspicious schemes
        suspicious_schemes = ['file', 'gopher', 'data', 'javascript', 'vbscript']
        if parsed.scheme in suspicious_schemes:
            return False, f"Suspicious URL scheme: {parsed.scheme}"
        
        return True, None
    
    def _validate_email(self, value: Any, rule: FieldValidationRule) -> Tuple[bool, Optional[str]]:
        """Validate email field"""
        if not isinstance(value, str):
            return False, "Email must be string"
        
        if not validators.email(value):
            return False, f"Invalid email format: {value}"
        
        return True, None
    
    def _validate_json_object(self, value: Any, rule: FieldValidationRule) -> Tuple[bool, Optional[str]]:
        """Validate JSON object field"""
        if isinstance(value, dict):
            return True, None
        
        if isinstance(value, str):
            try:
                json.loads(value)
                return True, None
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON: {str(e)}"
        
        return False, f"Expected JSON object, got {type(value).__name__}"
    
    def _validate_base64(self, value: Any, rule: FieldValidationRule) -> Tuple[bool, Optional[str]]:
        """Validate base64 encoded field"""
        if not isinstance(value, str):
            return False, "Base64 must be string"
        
        try:
            # Check if it's valid base64
            base64.b64decode(value, validate=True)
            return True, None
        except Exception as e:
            return False, f"Invalid base64 encoding: {str(e)}"
    
    def _validate_uuid(self, value: Any, rule: FieldValidationRule) -> Tuple[bool, Optional[str]]:
        """Validate UUID field"""
        if not isinstance(value, str):
            return False, "UUID must be string"
        
        uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
            re.IGNORECASE
        )
        
        if not uuid_pattern.match(value):
            return False, f"Invalid UUID format: {value}"
        
        return True, None
    
    def _validate_array(self, value: Any, rule: FieldValidationRule) -> Tuple[bool, Optional[str]]:
        """Validate array field"""
        if not isinstance(value, (list, tuple)):
            return False, f"Expected array, got {type(value).__name__}"
        
        if rule.min_length and len(value) < rule.min_length:
            return False, f"Array size {len(value)} below minimum {rule.min_length}"
        
        if rule.max_length and len(value) > rule.max_length:
            return False, f"Array size {len(value)} exceeds maximum {rule.max_length}"
        
        return True, None
    
    def _validate_nested_object(self, value: Any, rule: FieldValidationRule) -> Tuple[bool, Optional[str]]:
        """Validate nested object field"""
        if not isinstance(value, dict):
            return False, f"Expected nested object, got {type(value).__name__}"
        
        # Validate nested structure if schema provided
        if rule.custom_validator:
            return rule.custom_validator(value)
        
        return True, None
    
    # =========================================================================
    # Range Validators
    # =========================================================================
    
    def _validate_numeric_range(self, value: Union[int, float], 
                               min_val: Optional[Union[int, float]] = None,
                               max_val: Optional[Union[int, float]] = None) -> bool:
        """Validate numeric value is within range"""
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True
    
    def _validate_string_length(self, value: str, 
                               min_len: Optional[int] = None,
                               max_len: Optional[int] = None) -> bool:
        """Validate string length is within range"""
        length = len(value)
        if min_len is not None and length < min_len:
            return False
        if max_len is not None and length > max_len:
            return False
        return True
    
    def _validate_array_size(self, value: List[Any], 
                            min_size: Optional[int] = None,
                            max_size: Optional[int] = None) -> bool:
        """Validate array size is within range"""
        size = len(value)
        if min_size is not None and size < min_size:
            return False
        if max_size is not None and size > max_size:
            return False
        return True
    
    def _validate_timestamp_range(self, value: Union[int, float],
                                 min_time: Optional[Union[int, float]] = None,
                                 max_time: Optional[Union[int, float]] = None) -> bool:
        """Validate timestamp is within range"""
        if min_time is not None and value < min_time:
            return False
        if max_time is not None and value > max_time:
            return False
        return True
    
    # =========================================================================
    # Security Validators
    # =========================================================================
    
    def perform_security_scan(self, value: Any, threat_types: List[SecurityThreatType]) -> Dict[str, Any]:
        """
        Perform security scan for specified threat types
        """
        scan_start = time.time()
        
        security_assessment = {
            'threats_detected': [],
            'scan_time_ms': 0,
            'is_secure': True
        }
        
        # Convert value to string for pattern matching
        str_value = str(value) if not isinstance(value, str) else value
        
        for threat_type in threat_types:
            if threat_type in self.compiled_patterns:
                for pattern in self.compiled_patterns[threat_type]:
                    if pattern.search(str_value):
                        security_assessment['threats_detected'].append({
                            'type': threat_type.value,
                            'pattern_matched': pattern.pattern[:50] + '...' if len(pattern.pattern) > 50 else pattern.pattern,
                            'severity': 'high'
                        })
                        security_assessment['is_secure'] = False
                        break
        
        security_assessment['scan_time_ms'] = (time.time() - scan_start) * 1000
        self.performance_metrics['security_scan_times'].append(security_assessment['scan_time_ms'])
        
        return security_assessment
    
    def detect_anomalous_patterns(self, value: Any, field_name: str) -> List[Dict[str, Any]]:
        """
        Detect anomalous patterns that might indicate attacks
        """
        anomalies = []
        
        if isinstance(value, str):
            # Check for excessive special characters
            special_char_ratio = len([c for c in value if not c.isalnum()]) / len(value) if value else 0
            if special_char_ratio > 0.7:
                anomalies.append({
                    'type': 'excessive_special_chars',
                    'field': field_name,
                    'ratio': special_char_ratio,
                    'severity': 'medium'
                })
            
            # Check for unusual encoding
            if '%' in value:
                encoded_count = value.count('%')
                if encoded_count > len(value) * 0.3:
                    anomalies.append({
                        'type': 'excessive_url_encoding',
                        'field': field_name,
                        'encoded_ratio': encoded_count / len(value),
                        'severity': 'medium'
                    })
            
            # Check for binary data in text field
            try:
                value.encode('ascii')
            except UnicodeEncodeError:
                non_ascii_ratio = len([c for c in value if ord(c) > 127]) / len(value)
                if non_ascii_ratio > 0.5:
                    anomalies.append({
                        'type': 'binary_in_text',
                        'field': field_name,
                        'non_ascii_ratio': non_ascii_ratio,
                        'severity': 'low'
                    })
        
        elif isinstance(value, (int, float)):
            # Check for suspicious numeric values
            if value == float('inf') or value == float('-inf'):
                anomalies.append({
                    'type': 'infinite_value',
                    'field': field_name,
                    'severity': 'high'
                })
            
            # Check for very large numbers that might cause overflow
            if abs(value) > 2**53:  # JavaScript safe integer limit
                anomalies.append({
                    'type': 'potential_overflow',
                    'field': field_name,
                    'value': value,
                    'severity': 'medium'
                })
        
        return anomalies


# =============================================================================
# Input Validation Gateway
# =============================================================================

class InputValidationGateway:
    """
    Main Input Validation Gateway for Layer 1
    
    This class orchestrates all validation activities and provides the main
    interface for validating telemetry records from Layer 0.
    """
    
    def __init__(self, config: Any):
        """
        Initialize Input Validation Gateway
        
        Args:
            config: Layer 1 configuration object
        """
        self.config = config
        self.logger = logging.getLogger("SCAFAD.Layer1.ValidationGateway")
        
        # Determine validation level based on config
        self.validation_level = self._determine_validation_level()
        
        # Initialize validation engine
        self.validation_engine = ValidationEngine(self.validation_level)
        
        # Initialize schema definitions
        self._initialize_schemas()
        
        # Initialize sanitization rules
        self._initialize_sanitization_rules()
        
        # Initialize performance monitoring
        self._initialize_monitoring()
        
        # Validation statistics
        self.stats = {
            'total_records_validated': 0,
            'valid_records': 0,
            'invalid_records': 0,
            'sanitized_records': 0,
            'security_violations': 0,
            'average_validation_time_ms': 0.0
        }
        
        self.logger.info(f"Input Validation Gateway initialized with level: {self.validation_level.value}")
    
    def _determine_validation_level(self) -> ValidationLevel:
        """Determine validation level from configuration"""
        if self.config.test_mode:
            return ValidationLevel.MINIMAL
        
        if self.config.processing_mode.value == "production":
            return ValidationLevel.STRICT
        
        return ValidationLevel.STANDARD
    
    def _initialize_schemas(self):
        """Initialize telemetry record schemas"""
        
        # Define schema for v2.1 telemetry records
        self.schema_v2_1 = SchemaDefinition(
            schema_version="v2.1",
            schema_name="telemetry_record",
            field_rules={
                'record_id': FieldValidationRule(
                    field_name='record_id',
                    field_type=FieldType.UUID,
                    required=True,
                    nullable=False
                ),
                'timestamp': FieldValidationRule(
                    field_name='timestamp',
                    field_type=FieldType.TIMESTAMP,
                    required=True,
                    nullable=False,
                    min_value=0,
                    max_value=2147483647000  # Max millisecond timestamp
                ),
                'function_name': FieldValidationRule(
                    field_name='function_name',
                    field_type=FieldType.STRING,
                    required=True,
                    nullable=False,
                    min_length=1,
                    max_length=256,
                    pattern=r'^[a-zA-Z0-9_\-\.]+$'
                ),
                'execution_phase': FieldValidationRule(
                    field_name='execution_phase',
                    field_type=FieldType.ENUM,
                    required=True,
                    nullable=False,
                    allowed_values=['initialization', 'execution', 'completion', 'error', 'timeout']
                ),
                'anomaly_type': FieldValidationRule(
                    field_name='anomaly_type',
                    field_type=FieldType.STRING,
                    required=True,
                    nullable=False,
                    allowed_values=['benign', 'suspicious', 'malicious', 'unknown']
                ),
                'telemetry_data': FieldValidationRule(
                    field_name='telemetry_data',
                    field_type=FieldType.JSON_OBJECT,
                    required=True,
                    nullable=False
                ),
                'provenance_chain': FieldValidationRule(
                    field_name='provenance_chain',
                    field_type=FieldType.JSON_OBJECT,
                    required=False,
                    nullable=True
                ),
                'context_metadata': FieldValidationRule(
                    field_name='context_metadata',
                    field_type=FieldType.JSON_OBJECT,
                    required=False,
                    nullable=True
                ),
                'schema_version': FieldValidationRule(
                    field_name='schema_version',
                    field_type=FieldType.STRING,
                    required=True,
                    nullable=False,
                    pattern=r'^v\d+\.\d+$'
                )
            },
            required_fields=['record_id', 'timestamp', 'function_name', 'execution_phase', 'anomaly_type', 'telemetry_data'],
            conditional_rules=[
                {
                    'condition': lambda r: r.get('execution_phase') == 'error',
                    'required_fields': ['error_message', 'error_stack']
                }
            ]
        )
        
        # Store schemas by version
        self.schemas = {
            'v2.1': self.schema_v2_1,
            'v2.0': self._create_legacy_schema_v2_0(),
            'v1.0': self._create_legacy_schema_v1_0()
        }
        
        # Set current schema
        self.current_schema = self.schemas.get(self.config.schema_version, self.schema_v2_1)
    
    def _create_legacy_schema_v2_0(self) -> SchemaDefinition:
        """Create schema definition for v2.0 (backward compatibility)"""
        return SchemaDefinition(
            schema_version="v2.0",
            schema_name="telemetry_record",
            field_rules={
                'record_id': FieldValidationRule(
                    field_name='record_id',
                    field_type=FieldType.STRING,  # v2.0 used string IDs
                    required=True,
                    nullable=False
                ),
                'timestamp': FieldValidationRule(
                    field_name='timestamp',
                    field_type=FieldType.TIMESTAMP,
                    required=True,
                    nullable=False
                ),
                'function_name': FieldValidationRule(
                    field_name='function_name',
                    field_type=FieldType.STRING,
                    required=True,
                    nullable=False,
                    min_length=1,
                    max_length=256
                ),
                'execution_phase': FieldValidationRule(
                    field_name='execution_phase',
                    field_type=FieldType.STRING,  # v2.0 didn't enforce enum
                    required=True,
                    nullable=False
                ),
                'anomaly_type': FieldValidationRule(
                    field_name='anomaly_type',
                    field_type=FieldType.STRING,
                    required=True,
                    nullable=False
                ),
                'telemetry_data': FieldValidationRule(
                    field_name='telemetry_data',
                    field_type=FieldType.JSON_OBJECT,
                    required=True,
                    nullable=False
                ),
                'schema_version': FieldValidationRule(
                    field_name='schema_version',
                    field_type=FieldType.STRING,
                    required=True,
                    nullable=False
                )
            },
            required_fields=['record_id', 'timestamp', 'function_name', 'execution_phase', 'anomaly_type', 'telemetry_data']
        )
    
    def _create_legacy_schema_v1_0(self) -> SchemaDefinition:
        """Create schema definition for v1.0 (backward compatibility)"""
        return SchemaDefinition(
            schema_version="v1.0",
            schema_name="telemetry_record",
            field_rules={
                'id': FieldValidationRule(  # v1.0 used 'id' instead of 'record_id'
                    field_name='id',
                    field_type=FieldType.STRING,
                    required=True,
                    nullable=False
                ),
                'timestamp': FieldValidationRule(
                    field_name='timestamp',
                    field_type=FieldType.INTEGER,  # v1.0 used integer timestamps
                    required=True,
                    nullable=False
                ),
                'function': FieldValidationRule(  # v1.0 used 'function' instead of 'function_name'
                    field_name='function',
                    field_type=FieldType.STRING,
                    required=True,
                    nullable=False
                ),
                'phase': FieldValidationRule(  # v1.0 used 'phase' instead of 'execution_phase'
                    field_name='phase',
                    field_type=FieldType.STRING,
                    required=True,
                    nullable=False
                ),
                'data': FieldValidationRule(  # v1.0 used 'data' instead of 'telemetry_data'
                    field_name='data',
                    field_type=FieldType.JSON_OBJECT,
                    required=True,
                    nullable=False
                )
            },
            required_fields=['id', 'timestamp', 'function', 'phase', 'data']
        )
    
    def _initialize_sanitization_rules(self):
        """Initialize field sanitization rules"""
        
        self.sanitization_rules = {
            'remove_null_bytes': lambda x: x.replace('\x00', '') if isinstance(x, str) else x,
            'trim_whitespace': lambda x: x.strip() if isinstance(x, str) else x,
            'normalize_unicode': lambda x: x.encode('utf-8', 'ignore').decode('utf-8') if isinstance(x, str) else x,
            'escape_html': lambda x: self._escape_html_entities(x) if isinstance(x, str) else x,
            'normalize_paths': lambda x: self._normalize_path(x) if isinstance(x, str) and ('/' in x or '\\' in x) else x,
            'sanitize_urls': lambda x: self._sanitize_url(x) if isinstance(x, str) and x.startswith(('http://', 'https://')) else x,
            'limit_string_length': lambda x: x[:1024] if isinstance(x, str) and len(x) > 1024 else x,
            'round_floats': lambda x: round(x, 6) if isinstance(x, float) else x,
            'clamp_integers': lambda x: max(-2**31, min(2**31-1, x)) if isinstance(x, int) else x
        }
        
        # Define field-specific sanitization mappings
        self.field_sanitization_map = {
            'function_name': ['remove_null_bytes', 'trim_whitespace', 'normalize_unicode'],
            'execution_phase': ['trim_whitespace', 'normalize_unicode'],
            'anomaly_type': ['trim_whitespace', 'normalize_unicode'],
            'error_message': ['remove_null_bytes', 'trim_whitespace', 'escape_html', 'limit_string_length'],
            'url': ['sanitize_urls', 'trim_whitespace'],
            'file_path': ['normalize_paths', 'remove_null_bytes'],
            'memory_usage_mb': ['round_floats'],
            'execution_time_ms': ['round_floats'],
            'request_count': ['clamp_integers']
        }
    
    def _initialize_monitoring(self):
        """Initialize performance monitoring"""
        self.performance_monitor = {
            'validation_latencies': [],
            'sanitization_latencies': [],
            'security_scan_latencies': [],
            'memory_snapshots': [],
            'last_reset': time.time()
        }
    
    # =========================================================================
    # Main Validation Methods
    # =========================================================================
    
    async def validate_telemetry_record(self, record: Any) -> ValidationResult:
        """
        Validate a single telemetry record
        
        Args:
            record: Telemetry record to validate
            
        Returns:
            ValidationResult with validation status and details
        """
        validation_start = time.time()
        
        # Initialize validation result
        result = ValidationResult(
            is_valid=True,
            status=ValidationStatus.VALID,
            performance_metrics={}
        )
        
        try:
            # Phase 1: Structural validation
            structural_valid = await self._validate_structure(record, result)
            if not structural_valid:
                result.status = ValidationStatus.INVALID_STRUCTURE
                return self._finalize_validation_result(result, validation_start)
            
            # Phase 2: Schema validation
            schema_valid = await self._validate_schema(record, result)
            if not schema_valid:
                result.status = ValidationStatus.INVALID_SCHEMA
                return self._finalize_validation_result(result, validation_start)
            
            # Phase 3: Type validation
            type_valid = await self._validate_types(record, result)
            if not type_valid:
                result.status = ValidationStatus.INVALID_TYPE
                return self._finalize_validation_result(result, validation_start)
            
            # Phase 4: Range validation
            range_valid = await self._validate_ranges(record, result)
            if not range_valid:
                result.status = ValidationStatus.INVALID_RANGE
                return self._finalize_validation_result(result, validation_start)
            
            # Phase 5: Semantic validation
            semantic_valid = await self._validate_semantics(record, result)
            if not semantic_valid:
                result.status = ValidationStatus.INVALID_SEMANTIC
                return self._finalize_validation_result(result, validation_start)
            
            # Phase 6: Security validation (if enabled)
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                security_valid = await self._validate_security(record, result)
                if not security_valid:
                    result.status = ValidationStatus.SECURITY_VIOLATION
                    self.stats['security_violations'] += 1
                    return self._finalize_validation_result(result, validation_start)
            
            # Record is valid
            result.is_valid = True
            result.status = ValidationStatus.VALID
            self.stats['valid_records'] += 1
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            result.is_valid = False
            result.status = ValidationStatus.INVALID_STRUCTURE
            result.add_error('_system', 'validation_error', str(e))
            self.stats['invalid_records'] += 1
        
        return self._finalize_validation_result(result, validation_start)
    
    async def _validate_structure(self, record: Any, result: ValidationResult) -> bool:
        """Validate record structure"""
        
        # Check if record is a dictionary or has required attributes
        if not isinstance(record, dict):
            if not hasattr(record, '__dict__'):
                result.add_error('_structure', 'invalid_type', 'Record must be a dictionary or object')
                return False
            # Convert object to dictionary
            record_dict = record.__dict__ if hasattr(record, '__dict__') else asdict(record)
        else:
            record_dict = record
        
        # Check for required top-level fields
        schema = self.current_schema
        for field in schema.required_fields:
            if field not in record_dict:
                result.add_error(field, 'missing_required', f'Required field {field} is missing')
                return False
        
        return True
    
    async def _validate_schema(self, record: Any, result: ValidationResult) -> bool:
        """Validate record against schema"""
        
        record_dict = self._to_dict(record)
        schema = self.current_schema
        
        # Validate each field against schema rules
        for field_name, field_rule in schema.field_rules.items():
            if field_name in record_dict:
                value = record_dict[field_name]
                
                # Check nullable
                if value is None and not field_rule.nullable:
                    result.add_error(field_name, 'null_value', f'Field {field_name} cannot be null')
                    return False
                
                # Skip further validation if null and nullable
                if value is None and field_rule.nullable:
                    continue
                
                # Store field for type validation
                result.validation_metadata[field_name] = field_rule
        
        # Check conditional rules
        for conditional_rule in schema.conditional_rules:
            if conditional_rule['condition'](record_dict):
                for required_field in conditional_rule.get('required_fields', []):
                    if required_field not in record_dict:
                        result.add_error(
                            required_field, 
                            'conditional_required', 
                            f'Field {required_field} is required based on conditional rule'
                        )
                        return False
        
        return True
    
    async def _validate_types(self, record: Any, result: ValidationResult) -> bool:
        """Validate field types"""
        
        record_dict = self._to_dict(record)
        all_valid = True
        
        for field_name, field_rule in result.validation_metadata.items():
            if field_name in record_dict:
                value = record_dict[field_name]
                
                if value is not None:  # Skip null values (already validated)
                    # Get appropriate validator
                    validator = self.validation_engine.type_validators.get(field_rule.field_type)
                    
                    if validator:
                        is_valid, error_msg = validator(value, field_rule)
                        
                        if not is_valid:
                            result.add_error(field_name, 'invalid_type', error_msg)
                            all_valid = False
        
        return all_valid
    
    async def _validate_ranges(self, record: Any, result: ValidationResult) -> bool:
        """Validate field ranges"""
        
        record_dict = self._to_dict(record)
        all_valid = True
        
        for field_name, field_rule in result.validation_metadata.items():
            if field_name in record_dict:
                value = record_dict[field_name]
                
                if value is not None:
                    # Numeric range validation
                    if field_rule.field_type in [FieldType.INTEGER, FieldType.FLOAT]:
                        if not self.validation_engine._validate_numeric_range(
                            value, field_rule.min_value, field_rule.max_value
                        ):
                            result.add_error(
                                field_name, 
                                'out_of_range', 
                                f'Value {value} is out of allowed range [{field_rule.min_value}, {field_rule.max_value}]'
                            )
                            all_valid = False
                    
                    # String length validation
                    elif field_rule.field_type == FieldType.STRING:
                        if not self.validation_engine._validate_string_length(
                            value, field_rule.min_length, field_rule.max_length
                        ):
                            result.add_error(
                                field_name,
                                'invalid_length',
                                f'String length {len(value)} is out of allowed range [{field_rule.min_length}, {field_rule.max_length}]'
                            )
                            all_valid = False
        
        return all_valid
    
    async def _validate_semantics(self, record: Any, result: ValidationResult) -> bool:
        """Validate semantic correctness"""
        
        record_dict = self._to_dict(record)
        all_valid = True
        
        # Validate timestamp is not in future
        if 'timestamp' in record_dict:
            current_time = time.time()
            timestamp = record_dict['timestamp']
            
            # Allow 5 minutes clock skew
            if timestamp > current_time + 300:
                result.add_warning('timestamp', 'future_timestamp', 'Timestamp is in the future')
            
            # Check if timestamp is too old (more than 30 days)
            if timestamp < current_time - (30 * 24 * 3600):
                result.add_warning('timestamp', 'old_timestamp', 'Timestamp is more than 30 days old')
        
        # Validate telemetry_data structure
        if 'telemetry_data' in record_dict:
            telemetry_data = record_dict['telemetry_data']
            
            # Check for minimum expected fields based on execution phase
            execution_phase = record_dict.get('execution_phase')
            
            if execution_phase == 'execution':
                expected_fields = ['memory_usage_mb', 'execution_time_ms']
                for field in expected_fields:
                    if field not in telemetry_data:
                        result.add_warning(
                            'telemetry_data',
                            'missing_expected_field',
                            f'Expected field {field} not found in telemetry_data for execution phase'
                        )
            
            elif execution_phase == 'error':
                if 'error_message' not in telemetry_data and 'error_code' not in telemetry_data:
                    result.add_error(
                        'telemetry_data',
                        'missing_error_info',
                        'Error phase must include error_message or error_code in telemetry_data'
                    )
                    all_valid = False
        
        # Cross-field validation
        if 'anomaly_type' in record_dict and 'execution_phase' in record_dict:
            anomaly_type = record_dict['anomaly_type']
            execution_phase = record_dict['execution_phase']
            
            # Suspicious anomaly in initialization phase is unusual
            if anomaly_type in ['suspicious', 'malicious'] and execution_phase == 'initialization':
                result.add_warning(
                    'anomaly_type',
                    'unusual_combination',
                    f'Anomaly type {anomaly_type} in {execution_phase} phase is unusual'
                )
        
        return all_valid
    
    async def _validate_security(self, record: Any, result: ValidationResult) -> bool:
        """Validate security aspects"""
        
        record_dict = self._to_dict(record)
        all_secure = True
        
        # Define which fields need security scanning
        fields_to_scan = ['function_name', 'error_message', 'stack_trace']
        threat_types = [
            SecurityThreatType.SQL_INJECTION,
            SecurityThreatType.COMMAND_INJECTION,
            SecurityThreatType.SCRIPT_INJECTION,
            SecurityThreatType.PATH_TRAVERSAL
        ]
        
        # Scan string fields for security threats
        for field_name in fields_to_scan:
            value = self._get_nested_value(record_dict, field_name)
            
            if value and isinstance(value, str):
                security_assessment = self.validation_engine.perform_security_scan(value, threat_types)
                
                if not security_assessment['is_secure']:
                    for threat in security_assessment['threats_detected']:
                        result.add_error(
                            field_name,
                            'security_threat',
                            f"Potential {threat['type']} detected",
                            severity='critical'
                        )
                    all_secure = False
                    
                # Store security assessment in metadata
                if 'security_assessments' not in result.validation_metadata:
                    result.validation_metadata['security_assessments'] = {}
                result.validation_metadata['security_assessments'][field_name] = security_assessment
        
        # Check for anomalous patterns
        for field_name, value in record_dict.items():
            anomalies = self.validation_engine.detect_anomalous_patterns(value, field_name)
            
            for anomaly in anomalies:
                if anomaly['severity'] == 'high':
                    result.add_error(
                        field_name,
                        'anomalous_pattern',
                        f"Anomalous pattern detected: {anomaly['type']}"
                    )
                    all_secure = False
                else:
                    result.add_warning(
                        field_name,
                        'anomalous_pattern',
                        f"Unusual pattern detected: {anomaly['type']}"
                    )
        
        # Set security assessment in result
        result.security_assessment = {
            'scan_performed': True,
            'is_secure': all_secure,
            'threats_found': len([e for e in result.errors if e['type'] == 'security_threat']),
            'anomalies_found': len([e for e in result.errors + result.warnings if e['type'] == 'anomalous_pattern'])
        }
        
        return all_secure
    
    # =========================================================================
    # Sanitization Methods
    # =========================================================================
    
    async def sanitize_malformed_fields(self, record: Any) -> Any:
        """
        Sanitize malformed fields while preserving anomaly semantics
        
        Args:
            record: Record with potentially malformed fields
            
        Returns:
            Sanitized record
        """
        sanitization_start = time.time()
        
        record_dict = self._to_dict(record)
        sanitized_dict = record_dict.copy()
        
        # Apply general sanitization rules
        for field_name, value in record_dict.items():
            if value is not None:
                # Apply field-specific sanitization if defined
                if field_name in self.field_sanitization_map:
                    for rule_name in self.field_sanitization_map[field_name]:
                        if rule_name in self.sanitization_rules:
                            sanitized_dict[field_name] = self.sanitization_rules[rule_name](sanitized_dict[field_name])
                
                # Apply general sanitization for all string fields
                elif isinstance(value, str):
                    sanitized_dict[field_name] = self.sanitization_rules['remove_null_bytes'](value)
                    sanitized_dict[field_name] = self.sanitization_rules['trim_whitespace'](sanitized_dict[field_name])
                
                # Recursively sanitize nested objects
                elif isinstance(value, dict):
                    sanitized_dict[field_name] = await self._sanitize_nested_object(value)
                
                # Sanitize arrays
                elif isinstance(value, list):
                    sanitized_dict[field_name] = await self._sanitize_array(value)
        
        # Update sanitization statistics
        self.stats['sanitized_records'] += 1
        sanitization_time = (time.time() - sanitization_start) * 1000
        self.performance_monitor['sanitization_latencies'].append(sanitization_time)
        
        # Convert back to original type if needed
        if not isinstance(record, dict):
            return self._from_dict(record.__class__, sanitized_dict)
        
        return sanitized_dict
    
    async def _sanitize_nested_object(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize nested objects"""
        sanitized = {}
        
        for key, value in obj.items():
            if isinstance(value, str):
                sanitized[key] = self.sanitization_rules['remove_null_bytes'](value)
                sanitized[key] = self.sanitization_rules['trim_whitespace'](sanitized[key])
            elif isinstance(value, dict):
                sanitized[key] = await self._sanitize_nested_object(value)
            elif isinstance(value, list):
                sanitized[key] = await self._sanitize_array(value)
            else:
                sanitized[key] = value
        
        return sanitized
    
    async def _sanitize_array(self, arr: List[Any]) -> List[Any]:
        """Sanitize array elements"""
        sanitized = []
        
        for item in arr:
            if isinstance(item, str):
                sanitized_item = self.sanitization_rules['remove_null_bytes'](item)
                sanitized_item = self.sanitization_rules['trim_whitespace'](sanitized_item)
                sanitized.append(sanitized_item)
            elif isinstance(item, dict):
                sanitized.append(await self._sanitize_nested_object(item))
            elif isinstance(item, list):
                sanitized.append(await self._sanitize_array(item))
            else:
                sanitized.append(item)
        
        return sanitized
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _to_dict(self, record: Any) -> Dict[str, Any]:
        """Convert record to dictionary"""
        if isinstance(record, dict):
            return record
        elif hasattr(record, '__dict__'):
            return record.__dict__
        elif hasattr(record, '_asdict'):
            return record._asdict()
        else:
            return asdict(record)
    
    def _from_dict(self, cls: type, data: Dict[str, Any]) -> Any:
        """Convert dictionary back to original class type"""
        try:
            return cls(**data)
        except:
            # Fallback to returning dictionary if conversion fails
            return data
    
    def _get_nested_value(self, obj: Dict[str, Any], path: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        keys = path.split('.')
        value = obj
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def _escape_html_entities(self, text: str) -> str:
        """Escape HTML entities"""
        html_escape_table = {
            "&": "&amp;",
            '"': "&quot;",
            "'": "&apos;",
            ">": "&gt;",
            "<": "&lt;",
        }
        return "".join(html_escape_table.get(c, c) for c in text)
    
    def _normalize_path(self, path: str) -> str:
        """Normalize file paths"""
        # Remove path traversal attempts
        path = path.replace('../', '').replace('..\\', '')
        # Normalize slashes
        path = path.replace('\\', '/')
        # Remove double slashes
        while '//' in path:
            path = path.replace('//', '/')
        return path
    
    def _sanitize_url(self, url: str) -> str:
        """Sanitize URLs"""
        try:
            parsed = urllib.parse.urlparse(url)
            
            # Remove dangerous schemes
            if parsed.scheme in ['javascript', 'data', 'vbscript', 'file']:
                return ''
            
            # Rebuild URL with only safe components
            safe_url = urllib.parse.urlunparse((
                parsed.scheme,
                parsed.netloc,
                self._normalize_path(parsed.path) if parsed.path else '',
                '',  # Remove params
                parsed.query,
                ''   # Remove fragment
            ))
            
            return safe_url
        except:
            return url
    
    def _finalize_validation_result(self, result: ValidationResult, start_time: float) -> ValidationResult:
        """Finalize validation result with metrics"""
        
        # Calculate validation time
        validation_time = (time.time() - start_time) * 1000
        result.performance_metrics['validation_time_ms'] = validation_time
        
        # Update statistics
        self.stats['total_records_validated'] += 1
        self.performance_monitor['validation_latencies'].append(validation_time)
        
        # Calculate average validation time
        if self.performance_monitor['validation_latencies']:
            self.stats['average_validation_time_ms'] = np.mean(self.performance_monitor['validation_latencies'][-100:])
        
        # Add memory usage if monitoring
        if self.validation_level == ValidationLevel.PARANOID:
            process = psutil.Process()
            result.performance_metrics['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
        
        return result
    
    # =========================================================================
    # Public Interface Methods
    # =========================================================================
    
    async def validate_batch(self, records: List[Any]) -> List[ValidationResult]:
        """
        Validate a batch of telemetry records
        
        Args:
            records: List of telemetry records
            
        Returns:
            List of validation results
        """
        results = []
        
        for record in records:
            result = await self.validate_telemetry_record(record)
            results.append(result)
        
        return results
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get current validation statistics"""
        
        return {
            **self.stats,
            'validation_level': self.validation_level.value,
            'performance_metrics': {
                'avg_validation_latency_ms': np.mean(self.performance_monitor['validation_latencies'][-100:]) if self.performance_monitor['validation_latencies'] else 0,
                'p95_validation_latency_ms': np.percentile(self.performance_monitor['validation_latencies'][-100:], 95) if self.performance_monitor['validation_latencies'] else 0,
                'p99_validation_latency_ms': np.percentile(self.performance_monitor['validation_latencies'][-100:], 99) if self.performance_monitor['validation_latencies'] else 0,
            }
        }
    
    def reset_statistics(self):
        """Reset validation statistics"""
        
        self.stats = {
            'total_records_validated': 0,
            'valid_records': 0,
            'invalid_records': 0,
            'sanitized_records': 0,
            'security_violations': 0,
            'average_validation_time_ms': 0.0
        }
        
        self.performance_monitor = {
            'validation_latencies': [],
            'sanitization_latencies': [],
            'security_scan_latencies': [],
            'memory_snapshots': [],
            'last_reset': time.time()
        }
    
    async def health_check(self) -> Dict[str, str]:
        """Perform health check"""
        
        try:
            # Test validation with minimal record
            test_record = {
                'record_id': 'test-health-check',
                'timestamp': time.time(),
                'function_name': 'health_check',
                'execution_phase': 'execution',
                'anomaly_type': 'benign',
                'telemetry_data': {},
                'schema_version': 'v2.1'
            }
            
            result = await self.validate_telemetry_record(test_record)
            
            if result.is_valid:
                return {'status': 'healthy', 'message': 'Validation gateway operational'}
            else:
                return {'status': 'warning', 'message': 'Validation test failed'}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}


# =============================================================================
# Export public interface
# =============================================================================

__all__ = [
    'InputValidationGateway',
    'ValidationResult',
    'ValidationLevel',
    'ValidationStatus',
    'FieldType',
    'SecurityThreatType',
    'FieldValidationRule',
    'SchemaDefinition',
    'ValidationEngine'
]