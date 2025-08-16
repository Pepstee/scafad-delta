#!/usr/bin/env python3
"""
SCAFAD Layer 1: Sanitization Processor
=======================================

The Sanitization Processor cleanses and normalizes telemetry data while preserving
critical anomaly semantics. It removes malformed data, normalizes formats, and ensures
data consistency without destroying behavioral patterns needed for anomaly detection.

Key Responsibilities:
- Data cleansing and normalization
- Format standardization across fields
- Malformed data correction
- Encoding normalization (UTF-8, Base64, etc.)
- Whitespace and special character handling
- Numeric precision standardization
- Timestamp normalization
- Path and URL canonicalization
- Semantic-preserving transformations
- Anomaly signature preservation

Performance Targets:
- Sanitization latency: <0.3ms per record
- Anomaly preservation: 99.8%+
- Data integrity: 100%
- Format consistency: 100%
- Zero false positives from sanitization artifacts

Author: SCAFAD Research Team
Institution: Birmingham Newman University
License: MIT
Version: 1.0.0
"""

import re
import json
import base64
import hashlib
import logging
import asyncio
import html
import urllib.parse
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from enum import Enum, auto
from datetime import datetime, timezone
from collections import defaultdict
import traceback
import copy

# String and text processing
import string
import textwrap
from difflib import SequenceMatcher

# Numeric and statistical operations
import numpy as np
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation

# Date and time handling
from dateutil import parser as date_parser
import pytz

# Performance monitoring
import time
from functools import wraps

# Character encoding detection
import chardet

# IP address handling
import ipaddress

# Path handling
from pathlib import Path, PurePosixPath, PureWindowsPath


# =============================================================================
# Sanitization Data Models and Enums
# =============================================================================

class SanitizationLevel(Enum):
    """Sanitization intensity levels"""
    MINIMAL = "minimal"          # Basic cleaning only
    STANDARD = "standard"        # Standard sanitization
    AGGRESSIVE = "aggressive"    # Aggressive cleaning
    PARANOID = "paranoid"        # Maximum sanitization

class SanitizationType(Enum):
    """Types of sanitization operations"""
    WHITESPACE = "whitespace"
    ENCODING = "encoding"
    SPECIAL_CHARS = "special_chars"
    NUMERIC = "numeric"
    TIMESTAMP = "timestamp"
    PATH = "path"
    URL = "url"
    HTML = "html"
    SQL = "sql"
    COMMAND = "command"
    UNICODE = "unicode"
    CASE = "case"
    TRUNCATION = "truncation"

class DataIntegrityLevel(Enum):
    """Data integrity after sanitization"""
    INTACT = "intact"            # No data loss
    MINIMAL_LOSS = "minimal_loss"    # <1% information loss
    MODERATE_LOSS = "moderate_loss"  # 1-5% information loss
    SIGNIFICANT_LOSS = "significant_loss"  # >5% information loss

class AnomalyPreservationStatus(Enum):
    """Status of anomaly preservation during sanitization"""
    FULLY_PRESERVED = "fully_preserved"
    MOSTLY_PRESERVED = "mostly_preserved"
    PARTIALLY_PRESERVED = "partially_preserved"
    NOT_PRESERVED = "not_preserved"


@dataclass
class SanitizationResult:
    """
    Result of sanitization operation
    """
    success: bool
    sanitized_record: Optional[Any] = None
    original_record: Optional[Any] = None
    operations_applied: List[str] = field(default_factory=list)
    fields_sanitized: List[str] = field(default_factory=list)
    data_integrity: DataIntegrityLevel = DataIntegrityLevel.INTACT
    anomaly_preservation: AnomalyPreservationStatus = AnomalyPreservationStatus.FULLY_PRESERVED
    error_message: Optional[str] = None
    sanitization_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SanitizationRule:
    """
    Definition of a sanitization rule
    """
    rule_name: str
    rule_type: SanitizationType
    target_fields: List[str]
    sanitization_function: Callable
    preserve_anomaly: bool = True
    priority: int = 0  # Higher priority rules execute first
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Compare rules by priority"""
        return self.priority > other.priority  # Higher priority first


@dataclass
class FieldSanitizationProfile:
    """
    Sanitization profile for a specific field
    """
    field_name: str
    field_type: str
    sanitization_rules: List[SanitizationRule]
    max_length: Optional[int] = None
    allowed_chars: Optional[str] = None
    preserve_case: bool = True
    preserve_special_chars: bool = False
    normalize_whitespace: bool = True
    encoding: str = "utf-8"


@dataclass
class AnomalyContext:
    """
    Context for preserving anomaly semantics during sanitization
    """
    anomaly_type: str
    anomaly_confidence: float
    critical_fields: List[str]
    behavioral_patterns: Dict[str, Any]
    sanitization_constraints: Dict[str, Any]
    preservation_priority: int  # 0-100, higher means more important to preserve


# =============================================================================
# Core Sanitization Engine
# =============================================================================

class SanitizationEngine:
    """
    Core engine for data sanitization operations
    """
    
    def __init__(self, sanitization_level: SanitizationLevel = SanitizationLevel.STANDARD):
        """Initialize sanitization engine"""
        self.sanitization_level = sanitization_level
        self.logger = logging.getLogger("SCAFAD.Layer1.SanitizationEngine")
        
        # Initialize sanitization functions
        self._initialize_sanitizers()
        
        # Initialize preservation strategies
        self._initialize_preservation_strategies()
        
        # Performance monitoring
        self.performance_metrics = defaultdict(list)
        
        # Sanitization cache for repeated patterns
        self.sanitization_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _initialize_sanitizers(self):
        """Initialize sanitization functions"""
        
        self.sanitizers = {
            SanitizationType.WHITESPACE: self._sanitize_whitespace,
            SanitizationType.ENCODING: self._sanitize_encoding,
            SanitizationType.SPECIAL_CHARS: self._sanitize_special_chars,
            SanitizationType.NUMERIC: self._sanitize_numeric,
            SanitizationType.TIMESTAMP: self._sanitize_timestamp,
            SanitizationType.PATH: self._sanitize_path,
            SanitizationType.URL: self._sanitize_url,
            SanitizationType.HTML: self._sanitize_html,
            SanitizationType.SQL: self._sanitize_sql,
            SanitizationType.COMMAND: self._sanitize_command,
            SanitizationType.UNICODE: self._sanitize_unicode,
            SanitizationType.CASE: self._sanitize_case,
            SanitizationType.TRUNCATION: self._sanitize_truncation
        }
    
    def _initialize_preservation_strategies(self):
        """Initialize anomaly preservation strategies"""
        
        self.preservation_strategies = {
            'pattern_matching': self._preserve_pattern_anomalies,
            'statistical': self._preserve_statistical_anomalies,
            'structural': self._preserve_structural_anomalies,
            'temporal': self._preserve_temporal_anomalies,
            'semantic': self._preserve_semantic_anomalies
        }
    
    # =========================================================================
    # Sanitization Functions
    # =========================================================================
    
    def _sanitize_whitespace(self, value: Any, config: Dict[str, Any] = None) -> Any:
        """Sanitize whitespace in strings"""
        if not isinstance(value, str):
            return value
        
        config = config or {}
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Normalize line endings
        value = value.replace('\r\n', '\n').replace('\r', '\n')
        
        # Handle various whitespace characters
        if config.get('normalize_spaces', True):
            # Replace multiple spaces with single space
            value = re.sub(r' +', ' ', value)
            
            # Replace tabs with spaces
            if config.get('tabs_to_spaces', True):
                value = value.replace('\t', ' ')
        
        # Trim leading/trailing whitespace
        if config.get('trim', True):
            value = value.strip()
        
        # Remove zero-width characters
        if config.get('remove_zero_width', True):
            zero_width_chars = [
                '\u200b',  # Zero-width space
                '\u200c',  # Zero-width non-joiner
                '\u200d',  # Zero-width joiner
                '\ufeff',  # Zero-width no-break space
                '\u2060',  # Word joiner
            ]
            for char in zero_width_chars:
                value = value.replace(char, '')
        
        return value
    
    def _sanitize_encoding(self, value: Any, config: Dict[str, Any] = None) -> Any:
        """Sanitize character encoding"""
        if not isinstance(value, (str, bytes)):
            return value
        
        config = config or {}
        target_encoding = config.get('target_encoding', 'utf-8')
        
        if isinstance(value, bytes):
            # Detect encoding if not specified
            detected = chardet.detect(value)
            source_encoding = detected['encoding'] or 'utf-8'
            
            try:
                # Decode and re-encode
                decoded = value.decode(source_encoding, errors='ignore')
                return decoded.encode(target_encoding, errors='ignore').decode(target_encoding)
            except Exception:
                # Fallback to replacing invalid characters
                return value.decode('utf-8', errors='replace')
        
        else:  # string
            try:
                # Ensure valid encoding
                return value.encode(target_encoding, errors='ignore').decode(target_encoding)
            except Exception:
                return value
    
    def _sanitize_special_chars(self, value: Any, config: Dict[str, Any] = None) -> Any:
        """Sanitize special characters"""
        if not isinstance(value, str):
            return value
        
        config = config or {}
        
        # Define allowed characters based on sanitization level
        if self.sanitization_level == SanitizationLevel.MINIMAL:
            # Allow most characters
            allowed_chars = string.printable
        elif self.sanitization_level == SanitizationLevel.STANDARD:
            # Allow alphanumeric and common punctuation
            allowed_chars = string.ascii_letters + string.digits + ' .,;:!?-_/\\()[]{}@#$%&*+=~`"\''
        elif self.sanitization_level == SanitizationLevel.AGGRESSIVE:
            # Only alphanumeric and basic punctuation
            allowed_chars = string.ascii_letters + string.digits + ' .,;:!?-_()'
        else:  # PARANOID
            # Only alphanumeric and spaces
            allowed_chars = string.ascii_letters + string.digits + ' '
        
        # Override with custom allowed chars if provided
        if config.get('allowed_chars'):
            allowed_chars = config['allowed_chars']
        
        # Filter characters
        if config.get('remove_disallowed', False):
            value = ''.join(c for c in value if c in allowed_chars)
        else:
            # Replace disallowed with placeholder
            placeholder = config.get('placeholder', '')
            value = ''.join(c if c in allowed_chars else placeholder for c in value)
        
        return value
    
    def _sanitize_numeric(self, value: Any, config: Dict[str, Any] = None) -> Any:
        """Sanitize numeric values"""
        if not isinstance(value, (int, float, str)):
            return value
        
        config = config or {}
        
        try:
            # Convert to Decimal for precision handling
            if isinstance(value, str):
                # Remove non-numeric characters except . and -
                cleaned = re.sub(r'[^\d\.\-]', '', value)
                if not cleaned:
                    return 0
                decimal_value = Decimal(cleaned)
            else:
                decimal_value = Decimal(str(value))
            
            # Check for special values
            if decimal_value.is_nan():
                return config.get('nan_replacement', 0)
            if decimal_value.is_infinite():
                if decimal_value > 0:
                    return config.get('inf_replacement', float('inf'))
                else:
                    return config.get('neg_inf_replacement', float('-inf'))
            
            # Apply precision limits
            if config.get('max_precision'):
                decimal_value = decimal_value.quantize(
                    Decimal(10) ** -config['max_precision'],
                    rounding=ROUND_HALF_UP
                )
            
            # Apply range limits
            if config.get('min_value') is not None:
                decimal_value = max(decimal_value, Decimal(str(config['min_value'])))
            if config.get('max_value') is not None:
                decimal_value = min(decimal_value, Decimal(str(config['max_value'])))
            
            # Convert back to appropriate type
            if isinstance(value, int) or config.get('force_integer', False):
                return int(decimal_value)
            else:
                return float(decimal_value)
            
        except (InvalidOperation, ValueError):
            return config.get('default_value', 0)
    
    def _sanitize_timestamp(self, value: Any, config: Dict[str, Any] = None) -> Any:
        """Sanitize timestamp values"""
        if not value:
            return value
        
        config = config or {}
        target_format = config.get('target_format', 'unix')  # unix, iso, or custom format
        
        try:
            # Parse various timestamp formats
            if isinstance(value, (int, float)):
                # Assume Unix timestamp
                if value > 10**10:  # Likely milliseconds
                    dt = datetime.fromtimestamp(value / 1000, tz=timezone.utc)
                else:
                    dt = datetime.fromtimestamp(value, tz=timezone.utc)
            elif isinstance(value, str):
                # Parse string timestamp
                dt = date_parser.parse(value)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            elif isinstance(value, datetime):
                dt = value
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            else:
                return value
            
            # Apply range limits
            min_date = config.get('min_date')
            max_date = config.get('max_date')
            
            if min_date:
                min_dt = date_parser.parse(min_date) if isinstance(min_date, str) else min_date
                if dt < min_dt:
                    dt = min_dt
            
            if max_date:
                max_dt = date_parser.parse(max_date) if isinstance(max_date, str) else max_date
                if dt > max_dt:
                    dt = max_dt
            
            # Convert to target format
            if target_format == 'unix':
                return dt.timestamp()
            elif target_format == 'unix_ms':
                return int(dt.timestamp() * 1000)
            elif target_format == 'iso':
                return dt.isoformat()
            elif target_format:
                return dt.strftime(target_format)
            else:
                return dt
            
        except Exception:
            return config.get('default_value', 0)
    
    def _sanitize_path(self, value: Any, config: Dict[str, Any] = None) -> str:
        """Sanitize file system paths"""
        if not isinstance(value, str):
            return value
        
        config = config or {}
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Detect path type
        if '\\' in value and '/' not in value:
            path_cls = PureWindowsPath
        else:
            path_cls = PurePosixPath
        
        try:
            path = path_cls(value)
            
            # Remove path traversal attempts
            parts = []
            for part in path.parts:
                if part not in ['.', '..', '']:
                    # Remove dangerous characters from path components
                    cleaned_part = re.sub(r'[<>:"|?*]', '', part) if path_cls == PureWindowsPath else part
                    parts.append(cleaned_part)
            
            # Reconstruct path
            if parts:
                sanitized = str(path_cls(*parts))
            else:
                sanitized = ''
            
            # Normalize separators
            if config.get('normalize_separators', True):
                if path_cls == PureWindowsPath:
                    sanitized = sanitized.replace('/', '\\')
                else:
                    sanitized = sanitized.replace('\\', '/')
            
            # Apply length limit
            max_length = config.get('max_length', 4096)
            if len(sanitized) > max_length:
                # Preserve extension if possible
                if '.' in sanitized:
                    base, ext = sanitized.rsplit('.', 1)
                    sanitized = base[:max_length - len(ext) - 1] + '.' + ext
                else:
                    sanitized = sanitized[:max_length]
            
            return sanitized
            
        except Exception:
            return config.get('default_value', '')
    
    def _sanitize_url(self, value: Any, config: Dict[str, Any] = None) -> str:
        """Sanitize URLs"""
        if not isinstance(value, str):
            return value
        
        config = config or {}
        
        try:
            # Parse URL
            parsed = urllib.parse.urlparse(value)
            
            # Check for dangerous schemes
            dangerous_schemes = config.get('dangerous_schemes', [
                'javascript', 'data', 'vbscript', 'file', 'about', 'chrome'
            ])
            
            if parsed.scheme.lower() in dangerous_schemes:
                return config.get('default_value', '')
            
            # Sanitize components
            scheme = parsed.scheme.lower() if parsed.scheme else 'http'
            netloc = self._sanitize_domain(parsed.netloc)
            path = urllib.parse.quote(parsed.path, safe='/-_.~')
            params = urllib.parse.quote(parsed.params, safe='')
            query = urllib.parse.quote(parsed.query, safe='&=')
            fragment = urllib.parse.quote(parsed.fragment, safe='')
            
            # Reconstruct URL
            sanitized = urllib.parse.urlunparse((
                scheme, netloc, path, params, query, fragment
            ))
            
            # Apply length limit
            max_length = config.get('max_length', 2048)
            if len(sanitized) > max_length:
                sanitized = sanitized[:max_length]
            
            return sanitized
            
        except Exception:
            return config.get('default_value', '')
    
    def _sanitize_domain(self, domain: str) -> str:
        """Sanitize domain names"""
        # Remove userinfo if present
        if '@' in domain:
            domain = domain.split('@', 1)[1]
        
        # Handle port
        if ':' in domain:
            host, port = domain.rsplit(':', 1)
            try:
                port = int(port)
                if 0 <= port <= 65535:
                    domain = f"{host}:{port}"
                else:
                    domain = host
            except ValueError:
                domain = host
        
        # Basic domain validation
        domain = domain.lower()
        domain = re.sub(r'[^a-z0-9\.\-:]', '', domain)
        
        return domain
    
    def _sanitize_html(self, value: Any, config: Dict[str, Any] = None) -> str:
        """Sanitize HTML content"""
        if not isinstance(value, str):
            return value
        
        config = config or {}
        
        if config.get('strip_tags', False):
            # Remove all HTML tags
            value = re.sub(r'<[^>]+>', '', value)
        else:
            # Escape HTML entities
            value = html.escape(value, quote=True)
        
        return value
    
    def _sanitize_sql(self, value: Any, config: Dict[str, Any] = None) -> str:
        """Sanitize SQL-injectable content"""
        if not isinstance(value, str):
            return value
        
        config = config or {}
        
        # SQL keywords to escape
        sql_keywords = [
            'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE',
            'ALTER', 'EXEC', 'EXECUTE', 'UNION', 'FROM', 'WHERE',
            'ORDER BY', 'GROUP BY', 'HAVING', 'JOIN', 'INNER', 'OUTER',
            'LEFT', 'RIGHT', 'AND', 'OR', 'NOT', 'NULL', 'LIKE', 'IN'
        ]
        
        # Escape single quotes
        value = value.replace("'", "''")
        
        # Handle SQL comments
        value = re.sub(r'--.*$', '', value, flags=re.MULTILINE)
        value = re.sub(r'/\*.*?\*/', '', value, flags=re.DOTALL)
        
        # Optionally escape keywords
        if config.get('escape_keywords', False):
            for keyword in sql_keywords:
                pattern = r'\b' + keyword + r'\b'
                value = re.sub(pattern, f'[{keyword}]', value, flags=re.IGNORECASE)
        
        return value
    
    def _sanitize_command(self, value: Any, config: Dict[str, Any] = None) -> str:
        """Sanitize command injection attempts"""
        if not isinstance(value, str):
            return value
        
        config = config or {}
        
        # Dangerous characters for command injection
        dangerous_chars = [
            ';', '|', '&', '$', '`', '\\', '\n', '\r',
            '$(', '${', '<', '>', '>>', '&&', '||'
        ]
        
        # Escape or remove dangerous characters
        if config.get('escape_mode', 'remove') == 'escape':
            for char in dangerous_chars:
                value = value.replace(char, '\\' + char)
        else:
            for char in dangerous_chars:
                value = value.replace(char, '')
        
        return value
    
    def _sanitize_unicode(self, value: Any, config: Dict[str, Any] = None) -> str:
        """Sanitize Unicode characters"""
        if not isinstance(value, str):
            return value
        
        config = config or {}
        
        # Normalize Unicode
        normalization_form = config.get('normalization', 'NFC')  # NFC, NFD, NFKC, NFKD
        value = unicodedata.normalize(normalization_form, value)
        
        # Remove non-printable characters
        if config.get('remove_non_printable', True):
            value = ''.join(c for c in value if c.isprintable() or c in '\n\r\t')
        
        # Handle bidirectional text attacks
        if config.get('remove_bidi', True):
            bidi_chars = [
                '\u202a',  # Left-to-right embedding
                '\u202b',  # Right-to-left embedding
                '\u202c',  # Pop directional formatting
                '\u202d',  # Left-to-right override
                '\u202e',  # Right-to-left override
            ]
            for char in bidi_chars:
                value = value.replace(char, '')
        
        # Remove homoglyphs (lookalike characters)
        if config.get('remove_homoglyphs', False):
            # Simple homoglyph replacement (expand as needed)
            homoglyphs = {
                'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c',
                'у': 'y', 'х': 'x', 'А': 'A', 'В': 'B', 'Е': 'E',
                'К': 'K', 'М': 'M', 'Н': 'H', 'О': 'O', 'Р': 'P',
                'С': 'C', 'Т': 'T', 'Х': 'X'
            }
            for homoglyph, replacement in homoglyphs.items():
                value = value.replace(homoglyph, replacement)
        
        return value
    
    def _sanitize_case(self, value: Any, config: Dict[str, Any] = None) -> str:
        """Sanitize string case"""
        if not isinstance(value, str):
            return value
        
        config = config or {}
        case_mode = config.get('case_mode', 'preserve')
        
        if case_mode == 'lower':
            return value.lower()
        elif case_mode == 'upper':
            return value.upper()
        elif case_mode == 'title':
            return value.title()
        elif case_mode == 'capitalize':
            return value.capitalize()
        else:  # preserve
            return value
    
    def _sanitize_truncation(self, value: Any, config: Dict[str, Any] = None) -> Any:
        """Truncate values to maximum length"""
        if not value:
            return value
        
        config = config or {}
        max_length = config.get('max_length', 1024)
        
        if isinstance(value, str):
            if len(value) > max_length:
                # Try to truncate at word boundary
                if config.get('word_boundary', True):
                    truncated = value[:max_length]
                    last_space = truncated.rfind(' ')
                    if last_space > max_length * 0.8:  # If we can break at word near the end
                        return truncated[:last_space] + '...'
                
                return value[:max_length - 3] + '...'
        
        elif isinstance(value, (list, tuple)):
            max_items = config.get('max_items', 100)
            if len(value) > max_items:
                return value[:max_items]
        
        elif isinstance(value, dict):
            max_keys = config.get('max_keys', 100)
            if len(value) > max_keys:
                keys = list(value.keys())[:max_keys]
                return {k: value[k] for k in keys}
        
        return value
    
    # =========================================================================
    # Anomaly Preservation Functions
    # =========================================================================
    
    def _preserve_pattern_anomalies(self, value: Any, context: AnomalyContext) -> Any:
        """Preserve pattern-based anomalies during sanitization"""
        if not context.behavioral_patterns:
            return value
        
        # Check if value matches known anomaly patterns
        for pattern_name, pattern_data in context.behavioral_patterns.items():
            if self._matches_pattern(value, pattern_data):
                # Apply minimal sanitization to preserve pattern
                return self._minimal_sanitize(value)
        
        return value
    
    def _preserve_statistical_anomalies(self, value: Any, context: AnomalyContext) -> Any:
        """Preserve statistical anomalies"""
        if isinstance(value, (int, float)):
            # Check if value is statistical outlier
            if 'mean' in context.behavioral_patterns and 'std' in context.behavioral_patterns:
                mean = context.behavioral_patterns['mean']
                std = context.behavioral_patterns['std']
                
                z_score = abs((value - mean) / std) if std > 0 else 0
                
                if z_score > 3:  # Outlier
                    # Preserve exact value for anomaly detection
                    return value
        
        return value
    
    def _preserve_structural_anomalies(self, value: Any, context: AnomalyContext) -> Any:
        """Preserve structural anomalies in data"""
        if isinstance(value, dict):
            # Check for unusual key patterns
            expected_keys = context.behavioral_patterns.get('expected_keys', set())
            actual_keys = set(value.keys())
            
            unexpected_keys = actual_keys - expected_keys
            if unexpected_keys and context.anomaly_confidence > 0.7:
                # Preserve unexpected structure
                return value
        
        elif isinstance(value, list):
            # Check for unusual list lengths
            expected_length = context.behavioral_patterns.get('expected_length', 0)
            if abs(len(value) - expected_length) > expected_length * 0.5:
                # Significant deviation in length
                return value
        
        return value
    
    def _preserve_temporal_anomalies(self, value: Any, context: AnomalyContext) -> Any:
        """Preserve temporal anomalies"""
        if 'timestamp' in str(context.critical_fields):
            # Check for temporal anomalies
            if isinstance(value, (int, float)):
                current_time = time.time()
                
                # Check for future timestamps
                if value > current_time + 3600:  # More than 1 hour in future
                    return value
                
                # Check for very old timestamps
                if value < current_time - (365 * 24 * 3600):  # More than 1 year old
                    return value
        
        return value
    
    def _preserve_semantic_anomalies(self, value: Any, context: AnomalyContext) -> Any:
        """Preserve semantic anomalies in text"""
        if isinstance(value, str) and context.anomaly_type in ['suspicious', 'malicious']:
            # Preserve potentially malicious patterns for detection
            suspicious_patterns = [
                r'<script.*?>.*?</script>',
                r'javascript:',
                r'on\w+\s*=',
                r'\.\./\.\./\.\.',
                r'cmd\.exe|/bin/sh|/bin/bash',
                r'SELECT.*FROM.*WHERE',
                r'INSERT.*INTO.*VALUES'
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    # Apply minimal sanitization to preserve detection capability
                    return self._minimal_sanitize(value)