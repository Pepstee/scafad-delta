# ============================================================================
# Privacy-Preserving Sanitizer
# ============================================================================

import re 

class PrivacyPreservingSanitizer:
    """Privacy-preserving sanitization with semantic preservation"""
    
    def __init__(self, config: Layer1Config):
        self.config = config
        self.pii_patterns = self._compile_pii_patterns()
        self.sanitization_cache = {}
        self.stats = {
            'records_processed': 0,
            'pii_detections': 0,
            'sanitizations_applied': 0
        }
    
    def _compile_pii_patterns(self) -> Dict[str, Any]:
        """Compile PII detection patterns"""
        import re
        
        patterns = {
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'credit_card': re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            'ip_address': re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
            'aws_access_key': re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
            'jwt_token': re.compile(r'\beyJ[A-Za-z0-9_/+=-]+\.[A-Za-z0-9_/+=-]+\.[A-Za-z0-9_/+=-]*\b')
        }
        
        return patterns
    
    async def sanitize_telemetry_record(self, record: FusedTelemetryRecord) -> FusedTelemetryRecord:
        """Sanitize fused telemetry record while preserving semantics"""
        self.stats['records_processed'] += 1
        
        # Sanitize log data
        sanitized_logs = []
        for log_entry in record.log_data:
            sanitized_log = await self._sanitize_log_entry(log_entry)
            sanitized_logs.append(sanitized_log)
        
        # Sanitize trace data if present
        sanitized_trace = None
        if record.trace_data:
            sanitized_trace = await self._sanitize_trace_correlation(record.trace_data)
        
        # Calculate privacy score
        privacy_score = self._calculate_privacy_score(record, sanitized_logs, sanitized_trace)
        
        # Create sanitized record
        sanitized_record = FusedTelemetryRecord(
            event_id=record.event_id,
            timestamp=record.timestamp,
            function_id=record.function_id,
            session_id=record.session_id,
            l0_telemetry=record.l0_telemetry,  # Layer 0 data assumed clean
            log_data=sanitized_logs,
            metric_data=record.metric_data,  # Metrics typically safe
            trace_data=sanitized_trace,
            unified_embedding=record.unified_embedding,
            anomaly_indicators=record.anomaly_indicators,
            quality_assessment=record.quality_assessment,
            privacy_score=privacy_score,
            sanitization_applied=any(log.pii_detected for log in sanitized_logs),
            audit_trail=self._create_audit_trail(record, sanitized_logs),
            execution_graph=record.execution_graph,
            graph_features=record.graph_features
        )
        
        return sanitized_record
    
    async def _sanitize_log_entry(self, log_entry: ProcessedLogEntry) -> ProcessedLogEntry:
        """Sanitize individual log entry"""
        sanitized_message = log_entry.message
        sanitized_fields = []
        pii_detected = False
        
        # Check for PII in message
        for pii_type, pattern in self.pii_patterns.items():
            if pattern.search(sanitized_message):
                pii_detected = True
                self.stats['pii_detections'] += 1
                
                if self.config.sanitization_mode == "PRESERVE_SEMANTIC":
                    sanitized_message = pattern.sub(f'<{pii_type.upper()}_REDACTED>', sanitized_message)
                elif self.config.sanitization_mode == "STRICT":
                    sanitized_message = f"<REDACTED_{pii_type}>"
                else:  # BALANCED
                    sanitized_message = pattern.sub('<REDACTED>', sanitized_message)
                
                sanitized_fields.append(pii_type)
        
        # Check parameters for PII
        sanitized_parameters = {}
        for key, value in log_entry.parameters.items():
            str_value = str(value)
            value_pii_detected = False
            
            for pii_type, pattern in self.pii_patterns.items():
                if pattern.search(str_value):
                    value_pii_detected = True
                    pii_detected = True
                    sanitized_fields.append(f'param_{key}')
                    break
            
            if value_pii_detected:
                if self.config.sanitization_mode == "PRESERVE_SEMANTIC":
                    sanitized_parameters[key] = '<REDACTED>'
                else:
                    sanitized_parameters[key] = '<REDACTED>'
            else:
                sanitized_parameters[key] = value
        
        if pii_detected:
            self.stats['sanitizations_applied'] += 1
        
        # Create sanitized log entry
        return ProcessedLogEntry(
            timestamp=log_entry.timestamp,
            log_level=log_entry.log_level,
            message=sanitized_message,
            template=log_entry.template,  # Templates typically safe
            parameters=sanitized_parameters,
            semantic_embedding=log_entry.semantic_embedding,
            semantic_cluster=log_entry.semantic_cluster,
            content_entropy=log_entry.content_entropy,
            pattern_id=log_entry.pattern_id,
            pattern_frequency=log_entry.pattern_frequency,
            pattern_anomaly_score=log_entry.pattern_anomaly_score,
            known_anomaly_match=log_entry.known_anomaly_match,
            anomaly_confidence=log_entry.anomaly_confidence,
            function_id=log_entry.function_id,
            request_id=log_entry.request_id,
            source=log_entry."""
SCAFAD Layer 1: Multi-Modal Fusion Intake
==========================================

This implementation builds upon Layer 0's adaptive telemetry output and performs
intelligent fusion of logs, metrics, and traces into unified representations
suitable for downstream anomaly detection in serverless environments.

Key Features:
- GLAD-inspired content-aware log processing  
- Efficient metric aggregation with temporal alignment
- Privacy-preserving sanitization with semantic preservation
- Graph-based session correlation
- Multi-modal fusion with quality assessment
- Resilient processing under telemetry starvation

Literature Integration:
- GLAD (Li et al.) - Content-aware log parsing
- EGNN (Zhang et al.) - Energy-efficient aggregation  
- FaaSRCA (Kumar et al.) - Trace correlation
- SSLA (Sefati et al.) - Semi-supervised analysis
- LO2 Dataset (Zhao et al.) - Microservice patterns
"""

import asyncio
import json
import time
import hashlib
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Any, Tuple, Iterator, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import logging
from abc import ABC, abstractmethod

# Import Layer 0 structures
from layer0_structures import TelemetryRecord, AnomalyType, ExecutionPhase

# ============================================================================
# Core Data Structures
# ============================================================================

@dataclass
class ProcessedLogEntry:
    """Structured representation of processed log data"""
    timestamp: float
    log_level: str
    message: str
    template: str
    parameters: Dict[str, Any]
    
    # Semantic features
    semantic_embedding: np.ndarray
    semantic_cluster: int
    content_entropy: float
    
    # Pattern features  
    pattern_id: str
    pattern_frequency: int
    pattern_anomaly_score: float
    
    # Anomaly indicators
    known_anomaly_match: bool
    anomaly_confidence: float
    
    # Metadata
    function_id: str
    request_id: str
    source: str
    parser_used: str
    
    # Privacy preservation
    pii_detected: bool = False
    sanitized_fields: List[str] = field(default_factory=list)

@dataclass  
class MetricAggregation:
    """Aggregated metric data with temporal alignment"""
    timestamp: float
    metric_type: str
    function_id: str
    
    # Core metrics from Layer 0
    duration_stats: Dict[str, float]
    memory_stats: Dict[str, float] 
    cpu_stats: Dict[str, float]
    io_stats: Dict[str, float]
    
    # Derived metrics
    concurrency_level: int
    request_rate: float
    error_rate: float
    
    # Quality indicators
    completeness_score: float
    reliability_score: float
    drift_score: float

@dataclass
class TraceCorrelation:
    """Correlated trace data across function boundaries"""
    trace_id: str
    session_id: str
    root_span_id: str
    
    # Span hierarchy
    spans: List[Dict[str, Any]]
    dependency_graph: nx.DiGraph
    
    # Timing analysis
    total_duration: float
    critical_path: List[str]
    bottleneck_spans: List[str]
    
    # Anomaly indicators
    timing_anomalies: List[Dict[str, Any]]
    dependency_anomalies: List[Dict[str, Any]]
    
    # Quality metrics
    trace_completeness: float
    correlation_confidence: float

@dataclass
class FusedTelemetryRecord:
    """Unified multi-modal telemetry representation"""
    # Core identification
    event_id: str
    timestamp: float
    function_id: str
    session_id: str
    
    # Original Layer 0 data
    l0_telemetry: TelemetryRecord
    
    # Processed modalities
    log_data: List[ProcessedLogEntry]
    metric_data: MetricAggregation
    trace_data: Optional[TraceCorrelation]
    
    # Fusion results
    unified_embedding: np.ndarray
    anomaly_indicators: Dict[str, float]
    quality_assessment: Dict[str, float]
    
    # Privacy and compliance
    privacy_score: float
    sanitization_applied: bool
    audit_trail: Dict[str, Any]
    
    # Graph representation
    execution_graph: nx.DiGraph
    graph_features: Dict[str, Any]

# ============================================================================
# Layer 1 Configuration
# ============================================================================

@dataclass
class Layer1Config:
    """Configuration for Layer 1 processing"""
    # Processing modes
    enable_semantic_analysis: bool = True
    enable_graph_correlation: bool = True
    enable_privacy_sanitization: bool = True
    enable_temporal_alignment: bool = True
    
    # Model configurations
    bert_model_name: str = "bert-base-uncased"
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    
    # Processing windows
    correlation_window_seconds: int = 300
    aggregation_window_seconds: int = 60
    session_timeout_seconds: int = 1800
    
    # Quality thresholds
    min_completeness_score: float = 0.7
    min_correlation_confidence: float = 0.6
    max_processing_latency_ms: int = 50
    
    # Privacy settings
    pii_detection_enabled: bool = True
    sanitization_mode: str = "PRESERVE_SEMANTIC"  # PRESERVE_SEMANTIC, STRICT, BALANCED
    redaction_patterns: List[str] = field(default_factory=lambda: [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b',  # Credit card
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
    ])
    
    # Performance tuning
    batch_size: int = 32
    max_concurrent_sessions: int = 1000
    embedding_cache_size: int = 10000

# ============================================================================
# Content-Aware Log Processor
# ============================================================================

class LogParser(ABC):
    """Abstract base class for log parsing strategies"""
    
    @abstractmethod
    def parse(self, log_entry: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod  
    def get_template(self, log_entry: str) -> str:
        pass

class DrainParser(LogParser):
    """DRAIN algorithm implementation for online log parsing"""
    
    def __init__(self, max_depth: int = 4, max_children: int = 100):
        self.max_depth = max_depth
        self.max_children = max_children
        self.templates = {}
        self.parse_tree = {}
        
    def parse(self, log_entry: str) -> Dict[str, Any]:
        """Parse log entry using DRAIN algorithm"""
        tokens = log_entry.split()
        template = self._get_template(tokens)
        parameters = self._extract_parameters(tokens, template)
        
        return {
            'template': template,
            'parameters': parameters,
            'token_count': len(tokens),
            'template_id': hashlib.md5(template.encode()).hexdigest()[:8]
        }
    
    def get_template(self, log_entry: str) -> str:
        tokens = log_entry.split()
        return self._get_template(tokens)
    
    def _get_template(self, tokens: List[str]) -> str:
        """Extract template using DRAIN clustering"""
        # Simplified DRAIN implementation
        # In production, use more sophisticated clustering
        template_tokens = []
        for token in tokens:
            if self._is_variable(token):
                template_tokens.append('<*>')
            else:
                template_tokens.append(token)
        return ' '.join(template_tokens)
    
    def _is_variable(self, token: str) -> bool:
        """Determine if token is a variable parameter"""
        # Check for numeric values
        try:
            float(token)
            return True
        except ValueError:
            pass
        
        # Check for timestamp patterns
        if len(token) > 8 and any(c.isdigit() for c in token):
            return True
            
        # Check for IDs (alphanumeric strings > 6 chars)
        if len(token) > 6 and any(c.isdigit() for c in token) and any(c.isalpha() for c in token):
            return True
            
        return False
    
    def _extract_parameters(self, tokens: List[str], template: str) -> Dict[str, Any]:
        """Extract parameter values from tokens"""
        template_tokens = template.split()
        parameters = {}
        param_count = 0
        
        for i, (token, template_token) in enumerate(zip(tokens, template_tokens)):
            if template_token == '<*>':
                param_count += 1
                parameters[f'param_{param_count}'] = token
                
        return parameters

class ServerlessLogParser(LogParser):
    """Specialized parser for serverless/Lambda logs"""
    
    def __init__(self):
        self.serverless_patterns = {
            'cold_start': r'INIT_START|Cold Start|container.*start',
            'request_start': r'START RequestId|Request\s+\w+\s+started',
            'request_end': r'END RequestId|Request\s+\w+\s+ended',
            'billing': r'REPORT RequestId|Duration:\s+[\d.]+\s+ms',
            'error': r'ERROR|Exception|Traceback|Failed',
            'timeout': r'Task timed out|TIMEOUT|deadline exceeded'
        }
        
    def parse(self, log_entry: str) -> Dict[str, Any]:
        """Parse serverless-specific log patterns"""
        # Detect serverless event type
        event_type = self._detect_event_type(log_entry)
        
        # Extract standard fields
        fields = self._extract_standard_fields(log_entry)
        
        # Extract serverless-specific fields
        serverless_fields = self._extract_serverless_fields(log_entry, event_type)
        
        return {
            'event_type': event_type,
            'template': self._generate_template(log_entry, event_type),
            'parameters': {**fields, **serverless_fields},
            'serverless_specific': True
        }
    
    def get_template(self, log_entry: str) -> str:
        event_type = self._detect_event_type(log_entry)
        return self._generate_template(log_entry, event_type)
    
    def _detect_event_type(self, log_entry: str) -> str:
        """Detect type of serverless event"""
        for event_type, pattern in self.serverless_patterns.items():
            if re.search(pattern, log_entry, re.IGNORECASE):
                return event_type
        return 'application'
    
    def _extract_standard_fields(self, log_entry: str) -> Dict[str, Any]:
        """Extract standard log fields"""
        fields = {}
        
        # Extract timestamp
        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', log_entry)
        if timestamp_match:
            fields['timestamp'] = timestamp_match.group(1)
        
        # Extract request ID
        request_id_match = re.search(r'RequestId:\s*([a-f0-9-]+)', log_entry)
        if request_id_match:
            fields['request_id'] = request_id_match.group(1)
        
        # Extract log level
        level_match = re.search(r'\b(DEBUG|INFO|WARN|ERROR|FATAL)\b', log_entry)
        if level_match:
            fields['level'] = level_match.group(1)
        
        return fields
    
    def _extract_serverless_fields(self, log_entry: str, event_type: str) -> Dict[str, Any]:
        """Extract serverless-specific fields"""
        fields = {}
        
        if event_type == 'billing':
            # Extract duration
            duration_match = re.search(r'Duration:\s*([\d.]+)\s*ms', log_entry)
            if duration_match:
                fields['duration_ms'] = float(duration_match.group(1))
            
            # Extract billed duration
            billed_match = re.search(r'Billed Duration:\s*(\d+)\s*ms', log_entry)
            if billed_match:
                fields['billed_duration_ms'] = int(billed_match.group(1))
            
            # Extract memory usage
            memory_match = re.search(r'Max Memory Used:\s*(\d+)\s*MB', log_entry)
            if memory_match:
                fields['memory_used_mb'] = int(memory_match.group(1))
        
        return fields
    
    def _generate_template(self, log_entry: str, event_type: str) -> str:
        """Generate template based on event type"""
        if event_type == 'cold_start':
            return 'INIT_START Runtime Version: <*> Runtime Version ARN: <*>'
        elif event_type == 'request_start':
            return 'START RequestId: <*> Version: <*>'
        elif event_type == 'request_end':
            return 'END RequestId: <*>'
        elif event_type == 'billing':
            return 'REPORT RequestId: <*> Duration: <*> ms Billed Duration: <*> ms Memory Size: <*> MB Max Memory Used: <*> MB'
        else:
            # Use DRAIN-like approach for application logs
            tokens = log_entry.split()
            template_tokens = []
            for token in tokens:
                if self._is_variable_token(token):
                    template_tokens.append('<*>')
                else:
                    template_tokens.append(token)
            return ' '.join(template_tokens)
    
    def _is_variable_token(self, token: str) -> bool:
        """Check if token is variable in serverless context"""
        # Request IDs, timestamps, durations, memory values, etc.
        if re.match(r'^[a-f0-9-]{36}$', token):  # UUID
            return True
        if re.match(r'^\d+(\.\d+)?$', token):  # Numbers
            return True
        if re.match(r'^\d{4}-\d{2}-\d{2}', token):  # Dates
            return True
        return False

class LogPatternExtractor:
    """Extract and analyze patterns from parsed logs"""
    
    def __init__(self):
        self.pattern_cache = {}
        self.frequency_tracker = defaultdict(int)
        self.anomaly_scorer = LogAnomalyScorer()
        
    def extract(self, parsed_log: Dict[str, Any]) -> Dict[str, Any]:
        """Extract patterns from parsed log"""
        template = parsed_log.get('template', '')
        
        # Generate pattern ID
        pattern_id = self._generate_pattern_id(template)
        
        # Update frequency tracking
        self.frequency_tracker[pattern_id] += 1
        
        # Calculate anomaly score
        anomaly_score = self.anomaly_scorer.score_pattern(template, parsed_log)
        
        return {
            'pattern_id': pattern_id,
            'frequency': self.frequency_tracker[pattern_id],
            'anomaly_score': anomaly_score,
            'template': template
        }
    
    def _generate_pattern_id(self, template: str) -> str:
        """Generate unique pattern ID from template"""
        return hashlib.md5(template.encode()).hexdigest()[:12]

class LogAnomalyScorer:
    """Score log patterns for anomaly likelihood"""
    
    def __init__(self):
        self.baseline_patterns = set()
        self.error_indicators = ['error', 'exception', 'failed', 'timeout', 'abort']
        
    def score_pattern(self, template: str, parsed_log: Dict[str, Any]) -> float:
        """Score pattern for anomaly likelihood"""
        score = 0.0
        
        # Check for error indicators
        if any(indicator in template.lower() for indicator in self.error_indicators):
            score += 0.7
        
        # Check for unusual parameter counts
        param_count = len(parsed_log.get('parameters', {}))
        if param_count > 10:  # Unusually high parameter count
            score += 0.3
        
        # Check for novel patterns
        pattern_id = hashlib.md5(template.encode()).hexdigest()[:12]
        if pattern_id not in self.baseline_patterns:
            score += 0.4
        
        # Check for timing anomalies in serverless logs
        if 'timeout' in template.lower() or 'deadline' in template.lower():
            score += 0.8
        
        return min(score, 1.0)

class AnomalyPatternLibrary:
    """Library of known anomaly patterns"""
    
    def __init__(self):
        self.patterns = {
            'injection_attempt': {
                'patterns': [r'SELECT.*FROM', r'UNION.*SELECT', r'<script>', r'javascript:'],
                'confidence': 0.9
            },
            'path_traversal': {
                'patterns': [r'\.\./', r'%2e%2e%2f', r'..\\'],
                'confidence': 0.8
            },
            'resource_exhaustion': {
                'patterns': [r'OutOfMemoryError', r'timeout', r'resource.*limit'],
                'confidence': 0.7
            },
            'privilege_escalation': {
                'patterns': [r'sudo', r'admin', r'root', r'privilege'],
                'confidence': 0.6
            }
        }
    
    def match(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Match against known anomaly patterns"""
        template = patterns.get('template', '')
        
        for anomaly_type, config in self.patterns.items():
            for pattern in config['patterns']:
                if re.search(pattern, template, re.IGNORECASE):
                    return {
                        'match': True,
                        'type': anomaly_type,
                        'confidence': config['confidence'],
                        'pattern': pattern
                    }
        
        return {'match': False, 'confidence': 0.0}

class ContentAwareLogProcessor:
    """Main log processing component with GLAD-inspired architecture"""
    
    def __init__(self, config: Layer1Config):
        self.config = config
        
        # Initialize parsers
        self.parsers = {
            'drain': DrainParser(),
            'serverless': ServerlessLogParser()
        }
        
        # Initialize semantic models
        if config.enable_semantic_analysis:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name)
            self.bert_model = AutoModel.from_pretrained(config.bert_model_name)
            self.sentence_transformer = SentenceTransformer(config.sentence_transformer_model)
            self.semantic_clusterer = SemanticClusterer()
        
        # Initialize pattern components
        self.pattern_extractor = LogPatternExtractor()
        self.anomaly_patterns = AnomalyPatternLibrary()
        
        # Processing statistics
        self.stats = {
            'logs_processed': 0,
            'parse_errors': 0,
            'semantic_analysis_time': 0.0,
            'privacy_violations_detected': 0
        }
    
    async def process_log_stream(self, log_stream: AsyncIterator[str]) -> Dict[str, Any]:
        """Process streaming logs with content awareness"""
        processed_logs = []
        session_graphs = []
        
        async for log_entry in log_stream:
            try:
                processed_log = await self.process_single_log(log_entry)
                processed_logs.append(processed_log)
                self.stats['logs_processed'] += 1
                
            except Exception as e:
                logging.error(f"Error processing log: {e}")
                self.stats['parse_errors'] += 1
                continue
        
        # Build session graphs
        if processed_logs:
            session_graphs = self.build_session_graphs(processed_logs)
        
        return {
            'processed_logs': processed_logs,
            'session_graphs': session_graphs,
            'statistics': self.compute_statistics(processed_logs)
        }
    
    async def process_single_log(self, log_entry: str) -> ProcessedLogEntry:
        """Process single log entry"""
        start_time = time.time()
        
        # 1. Parse log structure
        parsed = self.parse_log(log_entry)
        
        # 2. Privacy screening
        pii_detected, sanitized_entry = self.screen_for_pii(log_entry)
        if pii_detected:
            self.stats['privacy_violations_detected'] += 1
        
        # 3. Extract semantic features
        semantic_features = {}
        if self.config.enable_semantic_analysis:
            semantic_features = await self.extract_semantic_features(sanitized_entry, parsed)
        
        # 4. Extract patterns
        patterns = self.pattern_extractor.extract(parsed)
        
        # 5. Check anomaly patterns
        anomaly_indicators = self.anomaly_patterns.match(patterns)
        
        # 6. Build structured representation
        processed_log = ProcessedLogEntry(
            timestamp=parsed.get('timestamp', time.time()),
            log_level=parsed.get('level', 'INFO'),
            message=sanitized_entry,
            template=parsed.get('template', ''),
            parameters=parsed.get('parameters', {}),
            
            # Semantic features
            semantic_embedding=semantic_features.get('embedding', np.array([])),
            semantic_cluster=semantic_features.get('cluster_id', -1),
            content_entropy=self.calculate_entropy(sanitized_entry),
            
            # Pattern features
            pattern_id=patterns['pattern_id'],
            pattern_frequency=patterns['frequency'],
            pattern_anomaly_score=patterns['anomaly_score'],
            
            # Anomaly indicators
            known_anomaly_match=anomaly_indicators['match'],
            anomaly_confidence=anomaly_indicators['confidence'],
            
            # Metadata
            function_id=parsed.get('function_id', 'unknown'),
            request_id=parsed.get('request_id', ''),
            source=parsed.get('source', 'unknown'),
            parser_used=parsed.get('parser_used', 'unknown'),
            
            # Privacy
            pii_detected=pii_detected,
            sanitized_fields=[] if not pii_detected else ['message']
        )
        
        # Update timing statistics
        processing_time = time.time() - start_time
        self.stats['semantic_analysis_time'] += processing_time
        
        return processed_log
    
    def parse_log(self, log_entry: str) -> Dict[str, Any]:
        """Multi-strategy log parsing with fallback"""
        
        # Try serverless parser first for Lambda logs
        if self._is_serverless_log(log_entry):
            try:
                parsed = self.parsers['serverless'].parse(log_entry)
                if self.validate_parsing(parsed):
                    parsed['parser_used'] = 'serverless'
                    return parsed
            except Exception:
                pass
        
        # Try DRAIN parser
        try:
            parsed = self.parsers['drain'].parse(log_entry)
            if self.validate_parsing(parsed):
                parsed['parser_used'] = 'drain'
                return parsed
        except Exception:
            pass
        
        # Fallback: basic parsing
        return self.fallback_parse(log_entry)
    
    def _is_serverless_log(self, log_entry: str) -> bool:
        """Check if log entry is from serverless environment"""
        serverless_indicators = ['RequestId:', 'INIT_START', 'START RequestId', 'END RequestId', 'REPORT RequestId']
        return any(indicator in log_entry for indicator in serverless_indicators)
    
    def validate_parsing(self, parsed: Dict[str, Any]) -> bool:
        """Validate parsing results"""
        required_fields = ['template']
        return all(field in parsed for field in required_fields)
    
    def fallback_parse(self, log_entry: str) -> Dict[str, Any]:
        """Fallback parsing using regex"""
        return {
            'template': log_entry,
            'parameters': {},
            'parser_used': 'fallback',
            'timestamp': time.time(),
            'level': 'INFO'
        }
    
    def screen_for_pii(self, log_entry: str) -> Tuple[bool, str]:
        """Screen for PII and apply sanitization"""
        if not self.config.pii_detection_enabled:
            return False, log_entry
        
        pii_detected = False
        sanitized_entry = log_entry
        
        for pattern in self.config.redaction_patterns:
            if re.search(pattern, log_entry):
                pii_detected = True
                if self.config.sanitization_mode == "PRESERVE_SEMANTIC":
                    # Replace with semantic placeholder
                    sanitized_entry = re.sub(pattern, '<REDACTED>', sanitized_entry)
                elif self.config.sanitization_mode == "STRICT":
                    # Hash the entry
                    sanitized_entry = hashlib.sha256(log_entry.encode()).hexdigest()[:16]
                
        return pii_detected, sanitized_entry
    
    async def extract_semantic_features(self, log_entry: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Extract semantic features using BERT and sentence transformers"""
        
        # Get sentence-level embedding
        sentence_embedding = self.sentence_transformer.encode(
            log_entry,
            convert_to_tensor=True
        )
        
        # Get BERT embeddings for deeper analysis
        tokens = self.bert_tokenizer(
            log_entry,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            bert_output = self.bert_model(**tokens)
            bert_embedding = bert_output.last_hidden_state.mean(dim=1)
        
        # Cluster assignment
        cluster_id = self.semantic_clusterer.predict(sentence_embedding.cpu().numpy())
        
        # Calculate semantic anomaly score
        semantic_anomaly = self.calculate_semantic_anomaly(
            sentence_embedding.cpu().numpy(),
            cluster_id
        )
        
        return {
            'embedding': sentence_embedding.cpu().numpy(),
            'bert_embedding': bert_embedding.cpu().numpy(),
            'cluster_id': int(cluster_id),
            'semantic_anomaly_score': float(semantic_anomaly),
            'keywords': self.extract_keywords(log_entry)
        }
    
    def calculate_entropy(self, text: str) -> float:
        """Calculate content entropy"""
        if not text:
            return 0.0
        
        # Character frequency analysis
        char_counts = defaultdict(int)
        for char in text:
            char_counts[char] += 1
        
        # Calculate entropy
        text_length = len(text)
        entropy = 0.0
        for count in char_counts.values():
            probability = count / text_length
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def calculate_semantic_anomaly(self, embedding: np.ndarray, cluster_id: int) -> float:
        """Calculate semantic anomaly score"""
        # Get cluster centroid
        centroid = self.semantic_clusterer.get_centroid(cluster_id)
        
        # Calculate distance to centroid
        if centroid is not None:
            distance = np.linalg.norm(embedding - centroid)
            # Normalize to [0, 1] range
            return min(distance / 2.0, 1.0)
        
        return 0.5  # Default moderate anomaly score
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction
        words = text.lower().split()
        # Filter out common words and keep meaningful terms
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if len(word) > 3 and word not in stopwords]
        return keywords[:10]  # Return top 10 keywords
    
    def build_session_graphs(self, processed_logs: List[ProcessedLogEntry]) -> List[Dict[str, Any]]:
        """Build session-based graphs from logs (LogGD approach)"""
        
        # Group logs by session
        sessions = self.group_into_sessions(processed_logs)
        
        graphs = []
        for session_id, session_logs in sessions.items():
            # Create directed graph for session
            G = nx.DiGraph()
            
            # Add nodes for each log entry
            for i, log in enumerate(session_logs):
                G.add_node(i, **{
                    'template': log.template,
                    'embedding': log.semantic_embedding.tolist() if log.semantic_embedding.size > 0 else [],
                    'timestamp': log.timestamp,
                    'anomaly_score': log.pattern_anomaly_score,
                    'log_level': log.log_level
                })
            
            # Add edges based on temporal sequence and semantic similarity
            for i in range(len(session_logs) - 1):
                for j in range(i + 1, min(i + 5, len(session_logs))):  # Window of 5
                    # Temporal edge
                    time_diff = session_logs[j].timestamp - session_logs[i].timestamp
                    if time_diff < 60:  # Within 1 minute
                        G.add_edge(i, j, weight=1.0/time_diff, type='temporal')
                    
                    # Semantic similarity edge
                    if (session_logs[i].semantic_embedding.size > 0 and 
                        session_logs[j].semantic_embedding.size > 0):
                        similarity = self.compute_similarity(
                            session_logs[i].semantic_embedding,
                            session_logs[j].semantic_embedding
                        )
                        if similarity > 0.7:
                            G.add_edge(i, j, weight=similarity, type='semantic')
            
            graphs.append({
                'session_id': session_id,
                'graph': G,
                'size': len(session_logs),
                'duration': session_logs[-1].timestamp - session_logs[0].timestamp,
                'anomaly_nodes': [i for i, log in enumerate(session_logs) 
                                if log.pattern_anomaly_score > 0.5]
            })
        
        return graphs
    
    def group_into_sessions(self, logs: List[ProcessedLogEntry]) -> Dict[str, List[ProcessedLogEntry]]:
        """Group logs into sessions based on request_id or time windows"""
        sessions = defaultdict(list)
        
        for log in logs:
            if log.request_id:
                sessions[log.request_id].append(log)
            else:
                # Group by time window if no request_id
                time_window = int(log.timestamp // self.config.aggregation_window_seconds)
                session_key = f"time_window_{time_window}_{log.function_id}"
                sessions[session_key].append(log)
        
        return dict(sessions)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        if embedding1.size == 0 or embedding2.size == 0:
            return 0.0
        
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    def compute_statistics(self, logs: List[ProcessedLogEntry]) -> Dict[str, Any]:
        """Compute processing statistics"""
        if not logs:
            return {}
        
        return {
            'total_logs': len(logs),
            'unique_patterns': len(set(log.pattern_id for log in logs)),
            'anomaly_logs': sum(1 for log in logs if log.pattern_anomaly_score > 0.5),
            'pii_detected_logs': sum(1 for log in logs if log.pii_detected),
            'avg_entropy': np.mean([log.content_entropy for log in logs]),
            'processing_stats': self.stats
        }

class SemanticClusterer:
    """Semantic clustering for log embeddings"""
    
    def __init__(self, n_clusters: int = 50):
        self.n_clusters = n_clusters
        self.centroids = {}
        self.cluster_assignments = {}
        self.is_fitted = False
    
    def predict(self, embedding: np.ndarray) -> int:
        """Predict cluster assignment for embedding"""
        if not self.is_fitted:
            # Initialize with simple hashing-based clustering
            return hash(str(embedding[:10])) % self.n_clusters
        
        # Find closest centroid
        min_distance = float('inf')
        closest_cluster = 0
        
        for cluster_id, centroid in self.centroids.items():
            distance = np.linalg.norm(embedding - centroid)
            if distance < min_distance:
                min_distance = distance
                closest_cluster = cluster_id
        
        return closest_cluster
    
    def get_centroid(self, cluster_id: int) -> Optional[np.ndarray]:
        """Get centroid for cluster"""
        return self.centroids.get(cluster_id)

# ============================================================================
# Efficient Metric Aggregator
# ============================================================================

class EfficientMetricAggregator:
    """EGNN-inspired efficient metric aggregation"""
    
    def __init__(self, config: Layer1Config):
        self.config = config
        self.metric_buffers = defaultdict(lambda: deque(maxlen=1000))
        self.aggregation_cache = {}
        self.stats = {
            'metrics_processed': 0,
            'aggregations_computed': 0,
            'cache_hits': 0
        }
    
    async def process_metric_stream(self, telemetry_stream: AsyncIterator[TelemetryRecord]) -> AsyncIterator[MetricAggregation]:
        """Process streaming telemetry into aggregated metrics"""
        
        async for telemetry in telemetry_stream:
            # Buffer the telemetry record
            function_id = telemetry.function_id
            self.metric_buffers[function_id].append(telemetry)
            self.stats['metrics_processed'] += 1
            
            # Check if aggregation window is complete
            if self._should_aggregate(function_id):
                aggregation = await self.compute_aggregation(function_id)
                if aggregation:
                    yield aggregation
    
    def _should_aggregate(self, function_id: str) -> bool:
        """Determine if aggregation should be computed"""
        buffer = self.metric_buffers[function_id]
        if len(buffer) < 2:
            return False
        
        # Check time window
        latest_time = buffer[-1].timestamp
        oldest_time = buffer[0].timestamp
        return (latest_time - oldest_time) >= self.config.aggregation_window_seconds
    
    async def compute_aggregation(self, function_id: str) -> Optional[MetricAggregation]:
        """Compute metric aggregation for function"""
        buffer = self.metric_buffers[function_id]
        if not buffer:
            return None
        
        # Check cache first
        cache_key = self._generate_cache_key(function_id, buffer)
        if cache_key in self.aggregation_cache:
            self.stats['cache_hits'] += 1
            return self.aggregation_cache[cache_key]
        
        # Compute aggregation
        records = list(buffer)
        
        # Duration statistics
        durations = [r.duration for r in records]
        duration_stats = {
            'mean': np.mean(durations),
            'median': np.median(durations),
            'std': np.std(durations),
            'min': np.min(durations),
            'max': np.max(durations),
            'p95': np.percentile(durations, 95),
            'p99': np.percentile(durations, 99)
        }
        
        # Memory statistics
        memory_values = [r.memory_spike_kb for r in records]
        memory_stats = {
            'mean': np.mean(memory_values),
            'median': np.median(memory_values),
            'std': np.std(memory_values),
            'min': np.min(memory_values),
            'max': np.max(memory_values),
            'p95': np.percentile(memory_values, 95),
            'p99': np.percentile(memory_values, 99)
        }
        
        # CPU statistics
        cpu_values = [r.cpu_utilization for r in records]
        cpu_stats = {
            'mean': np.mean(cpu_values),
            'median': np.median(cpu_values),
            'std': np.std(cpu_values),
            'min': np.min(cpu_values),
            'max': np.max(cpu_values)
        }
        
        # I/O statistics
        io_values = [r.network_io_bytes for r in records]
        io_stats = {
            'mean': np.mean(io_values),
            'median': np.median(io_values),
            'std': np.std(io_values),
            'total': np.sum(io_values)
        }
        
        # Derived metrics
        concurrency_level = len(set(r.concurrency_id for r in records))
        request_rate = len(records) / (records[-1].timestamp - records[0].timestamp)
        error_rate = sum(1 for r in records if r.anomaly_type != AnomalyType.BENIGN) / len(records)
        
        # Quality metrics
        completeness_score = np.mean([r.completeness_score for r in records])
        reliability_score = 1.0 - sum(1 for r in records if r.fallback_mode) / len(records)
        drift_score = self._calculate_drift_score(records)
        
        aggregation = MetricAggregation(
            timestamp=records[-1].timestamp,
            metric_type='telemetry_aggregation',
            function_id=function_id,
            duration_stats=duration_stats,
            memory_stats=memory_stats,
            cpu_stats=cpu_stats,
            io_stats=io_stats,
            concurrency_level=concurrency_level,
            request_rate=request_rate,
            error_rate=error_rate,
            completeness_score=completeness_score,
            reliability_score=reliability_score,
            drift_score=drift_score
        )
        
        # Cache the result
        self.aggregation_cache[cache_key] = aggregation
        self.stats['aggregations_computed'] += 1
        
        return aggregation
    
    def _generate_cache_key(self, function_id: str, buffer: deque) -> str:
        """Generate cache key for aggregation"""
        if not buffer:
            return ""
        
        # Use first and last timestamps + buffer size
        first_ts = buffer[0].timestamp
        last_ts = buffer[-1].timestamp
        size = len(buffer)
        
        return f"{function_id}_{first_ts}_{last_ts}_{size}"
    
    def _calculate_drift_score(self, records: List[TelemetryRecord]) -> float:
        """Calculate metric drift score"""
        if len(records) < 10:
            return 0.0
        
        # Split into first and second half
        mid = len(records) // 2
        first_half = records[:mid]
        second_half = records[mid:]
        
        # Compare duration distributions
        first_durations = [r.duration for r in first_half]
        second_durations = [r.duration for r in second_half]
        
        first_mean = np.mean(first_durations)
        second_mean = np.mean(second_durations)
        
        # Calculate relative change
        if first_mean > 0:
            duration_drift = abs(second_mean - first_mean) / first_mean
        else:
            duration_drift = 0.0
        
        # Similar for memory
        first_memory = np.mean([r.memory_spike_kb for r in first_half])
        second_memory = np.mean([r.memory_spike_kb for r in second_half])
        
        if first_memory > 0:
            memory_drift = abs(second_memory - first_memory) / first_memory
        else:
            memory_drift = 0.0
        
        # Combined drift score
        return min((duration_drift + memory_drift) / 2, 1.0)

# ============================================================================
# Trace Correlator
# ============================================================================

class TraceCorrelator:
    """FaaSRCA-inspired trace correlation across function boundaries"""
    
    def __init__(self, config: Layer1Config):
        self.config = config
        self.active_traces = {}
        self.completed_traces = {}
        self.correlation_graph = nx.DiGraph()
        self.stats = {
            'traces_processed': 0,
            'correlations_found': 0,
            'orphaned_spans': 0
        }
    
    async def process_telemetry_for_correlation(self, telemetry: TelemetryRecord) -> Optional[TraceCorrelation]:
        """Process telemetry record for trace correlation"""
        
        # Extract trace information
        trace_id = self._extract_trace_id(telemetry)
        span_id = self._generate_span_id(telemetry)
        
        # Create span from telemetry
        span = self._create_span_from_telemetry(telemetry, span_id)
        
        # Add to active traces
        if trace_id not in self.active_traces:
            self.active_traces[trace_id] = {
                'spans': [],
                'start_time': telemetry.timestamp,
                'last_activity': telemetry.timestamp
            }
        
        self.active_traces[trace_id]['spans'].append(span)
        self.active_traces[trace_id]['last_activity'] = telemetry.timestamp
        
        # Check if trace is complete
        if self._is_trace_complete(trace_id):
            correlation = await self._build_trace_correlation(trace_id)
            self.completed_traces[trace_id] = correlation
            del self.active_traces[trace_id]
            return correlation
        
        return None
    
    def _extract_trace_id(self, telemetry: TelemetryRecord) -> str:
        """Extract trace ID from telemetry"""
        # Use provenance chain or generate from event context
        if hasattr(telemetry, 'provenance_chain') and telemetry.provenance_chain:
            return telemetry.provenance_chain
        
        # Fall back to request-based grouping
        return f"trace_{telemetry.function_id}_{int(telemetry.timestamp // 300)}"
    
    def _generate_span_id(self, telemetry: TelemetryRecord) -> str:
        """Generate span ID for telemetry record"""
        return f"span_{telemetry.event_id}"
    
    def _create_span_from_telemetry(self, telemetry: TelemetryRecord, span_id: str) -> Dict[str, Any]:
        """Create span representation from telemetry"""
        return {
            'span_id': span_id,
            'function_id': telemetry.function_id,
            'start_time': telemetry.timestamp,
            'duration': telemetry.duration,
            'execution_phase': telemetry.execution_phase.value,
            'anomaly_type': telemetry.anomaly_type.value,
            'memory_usage': telemetry.memory_spike_kb,
            'cpu_utilization': telemetry.cpu_utilization,
            'network_io': telemetry.network_io_bytes,
            'fallback_mode': telemetry.fallback_mode,
            'economic_risk_score': getattr(telemetry, 'economic_risk_score', 0.0),
            'tags': {
                'concurrency_id': getattr(telemetry, 'concurrency_id', ''),
                'source': telemetry.source
            }
        }
    
    def _is_trace_complete(self, trace_id: str) -> bool:
        """Determine if trace is complete"""
        trace = self.active_traces.get(trace_id)
        if not trace:
            return False
        
        # Check timeout
        if time.time() - trace['last_activity'] > self.config.session_timeout_seconds:
            return True
        
        # Check for end markers
        spans = trace['spans']
        has_end_span = any(span.get('execution_phase') == 'shutdown' for span in spans)
        
        return has_end_span
    
    async def _build_trace_correlation(self, trace_id: str) -> TraceCorrelation:
        """Build correlation from completed trace"""
        trace = self.active_traces[trace_id]
        spans = trace['spans']
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(spans)
        
        # Calculate timing metrics
        total_duration = max(span['start_time'] + span['duration'] for span in spans) - min(span['start_time'] for span in spans)
        
        # Find critical path
        critical_path = self._find_critical_path(dependency_graph, spans)
        
        # Identify bottlenecks
        bottleneck_spans = self._identify_bottlenecks(spans)
        
        # Detect anomalies
        timing_anomalies = self._detect_timing_anomalies(spans)
        dependency_anomalies = self._detect_dependency_anomalies(dependency_graph)
        
        # Calculate quality metrics
        trace_completeness = len(spans) / max(len(spans), 1)  # Simplified
        correlation_confidence = self._calculate_correlation_confidence(spans)
        
        correlation = TraceCorrelation(
            trace_id=trace_id,
            session_id=trace_id,  # Same for simplicity
            root_span_id=spans[0]['span_id'] if spans else '',
            spans=spans,
            dependency_graph=dependency_graph,
            total_duration=total_duration,
            critical_path=critical_path,
            bottleneck_spans=bottleneck_spans,
            timing_anomalies=timing_anomalies,
            dependency_anomalies=dependency_anomalies,
            trace_completeness=trace_completeness,
            correlation_confidence=correlation_confidence
        )
        
        self.stats['traces_processed'] += 1
        if correlation_confidence > self.config.min_correlation_confidence:
            self.stats['correlations_found'] += 1
        
        return correlation
    
    def _build_dependency_graph(self, spans: List[Dict[str, Any]]) -> nx.DiGraph:
        """Build dependency graph from spans"""
        G = nx.DiGraph()
        
        # Add nodes
        for span in spans:
            G.add_node(span['span_id'], **span)
        
        # Add edges based on temporal ordering and function relationships
        sorted_spans = sorted(spans, key=lambda x: x['start_time'])
        
        for i, span in enumerate(sorted_spans):
            for j in range(i + 1, min(i + 3, len(sorted_spans))):  # Look ahead 2 spans
                next_span = sorted_spans[j]
                
                # Check if spans could be related
                if self._spans_could_be_related(span, next_span):
                    G.add_edge(span['span_id'], next_span['span_id'], 
                             weight=1.0, relationship='temporal')
        
        return G
    
    def _spans_could_be_related(self, span1: Dict[str, Any], span2: Dict[str, Any]) -> bool:
        """Check if two spans could be related"""
        # Time-based relationship
        time_gap = span2['start_time'] - (span1['start_time'] + span1['duration'])
        if time_gap < 0.1:  # Within 100ms
            return True
        
        # Function relationship
        if span1['function_id'] == span2['function_id']:
            return True
        
        # Same concurrency group
        if span1['tags'].get('concurrency_id') == span2['tags'].get('concurrency_id'):
            return True
        
        return False
    
    def _find_critical_path(self, graph: nx.DiGraph, spans: List[Dict[str, Any]]) -> List[str]:
        """Find critical path through dependency graph"""
        if not graph.nodes():
            return []
        
        # Find longest path (critical path)
        try:
            # Simple approach: find path with maximum total duration
            best_path = []
            max_duration = 0
            
            for source in graph.nodes():
                for target in graph.nodes():
                    if source != target:
                        try:
                            paths = list(nx.all_simple_paths(graph, source, target))
                            for path in paths:
                                path_duration = sum(
                                    next(s['duration'] for s in spans if s['span_id'] == span_id)
                                    for span_id in path
                                )
                                if path_duration > max_duration:
                                    max_duration = path_duration
                                    best_path = path
                        except nx.NetworkXNoPath:
                            continue
            
            return best_path
        except:
            return []
    
    def _identify_bottlenecks(self, spans: List[Dict[str, Any]]) -> List[str]:
        """Identify bottleneck spans"""
        if not spans:
            return []
        
        # Find spans with duration > 95th percentile
        durations = [span['duration'] for span in spans]
        threshold = np.percentile(durations, 95)
        
        bottlenecks = [
            span['span_id'] for span in spans 
            if span['duration'] > threshold
        ]
        
        return bottlenecks
    
    def _detect_timing_anomalies(self, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect timing anomalies in spans"""
        anomalies = []
        
        for span in spans:
            anomaly_score = 0.0
            reasons = []
            
            # Check for unusually long duration
            if span['duration'] > 30.0:  # > 30 seconds
                anomaly_score += 0.8
                reasons.append('excessive_duration')
            
            # Check for timing inconsistencies
            if span['execution_phase'] == 'init' and span['duration'] > 5.0:
                anomaly_score += 0.6
                reasons.append('slow_initialization')
            
            # Check for resource anomalies
            if span['memory_usage'] > 500000:  # > 500MB
                anomaly_score += 0.4
                reasons.append('high_memory_usage')
            
            if anomaly_score > 0.5:
                anomalies.append({
                    'span_id': span['span_id'],
                    'anomaly_score': min(anomaly_score, 1.0),
                    'reasons': reasons,
                    'span_data': span
                })
        
        return anomalies
    
    def _detect_dependency_anomalies(self, graph: nx.DiGraph) -> List[Dict[str, Any]]:
        """Detect dependency anomalies"""
        anomalies = []
        
        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(graph))
            for cycle in cycles:
                anomalies.append({
                    'type': 'dependency_cycle',
                    'nodes': cycle,
                    'severity': 'high'
                })
        except:
            pass
        
        # Check for isolated nodes
        isolated = list(nx.isolates(graph))
        if isolated:
            anomalies.append({
                'type': 'isolated_spans',
                'nodes': isolated,
                'severity': 'medium'
            })
        
        return anomalies
    
    def _calculate_correlation_confidence(self, spans: List[Dict[str, Any]]) -> float:
        """Calculate confidence in trace correlation"""
        if not spans:
            return 0.0
        
        confidence = 0.0
        
        # Reward complete traces
        has_init = any(span['execution_phase'] == 'init' for span in spans)
        has_invoke = any(span['execution_phase'] == 'invoke' for span in spans)
        has_shutdown = any(span['execution_phase'] == 'shutdown' for span in spans)
        
        if has_init and has_invoke:
            confidence += 0.4
        if has_shutdown:
            confidence += 0.2
        
        # Reward temporal consistency
        sorted_spans = sorted(spans, key=lambda x: x['start_time'])
        temporal_consistency = 1.0
        for i in range(len(sorted_spans) - 1):
            gap = sorted_spans[i+1]['start_time'] - (sorted_spans[i]['start_time'] + sorted_spans[i]['duration'])
            if gap > 10.0:  # > 10 second gap
                temporal_consistency *= 0.8
        
        confidence += 0.3 * temporal_consistency
        
        # Reward function diversity (cross-function traces)
        unique_functions = len(set(span['function_id'] for span in spans))
        if unique_functions > 1:
            confidence += 0.1
        
        return min(confidence, 1.0)
source,
           parser_used=log_entry.parser_used,
           pii_detected=pii_detected,
           sanitized_fields=sanitized_fields
       )
   
   async def _sanitize_trace_correlation(self, trace: TraceCorrelation) -> TraceCorrelation:
       """Sanitize trace correlation data"""
       sanitized_spans = []
       
       for span in trace.spans:
           sanitized_span = span.copy()
           
           # Sanitize span tags
           if 'tags' in sanitized_span:
               for key, value in sanitized_span['tags'].items():
                   str_value = str(value)
                   for pii_type, pattern in self.pii_patterns.items():
                       if pattern.search(str_value):
                           sanitized_span['tags'][key] = '<REDACTED>'
                           break
           
           sanitized_spans.append(sanitized_span)
       
       return TraceCorrelation(
           trace_id=trace.trace_id,
           session_id=trace.session_id,
           root_span_id=trace.root_span_id,
           spans=sanitized_spans,
           dependency_graph=trace.dependency_graph,
           total_duration=trace.total_duration,
           critical_path=trace.critical_path,
           bottleneck_spans=trace.bottleneck_spans,
           timing_anomalies=trace.timing_anomalies,
           dependency_anomalies=trace.dependency_anomalies,
           trace_completeness=trace.trace_completeness,
           correlation_confidence=trace.correlation_confidence
       )
   
   def _calculate_privacy_score(self, original: FusedTelemetryRecord, 
                              sanitized_logs: List[ProcessedLogEntry],
                              sanitized_trace: Optional[TraceCorrelation]) -> float:
       """Calculate privacy compliance score"""
       score = 1.0
       
       # Penalize for PII detections
       pii_count = sum(1 for log in sanitized_logs if log.pii_detected)
       if pii_count > 0:
           score -= 0.1 * pii_count
       
       # Reward for successful sanitization
       if any(log.sanitized_fields for log in sanitized_logs):
           score += 0.2
       
       return max(score, 0.0)
   
   def _create_audit_trail(self, original: FusedTelemetryRecord,
                         sanitized_logs: List[ProcessedLogEntry]) -> Dict[str, Any]:
       """Create audit trail for sanitization process"""
       return {
           'sanitization_timestamp': time.time(),
           'sanitization_mode': self.config.sanitization_mode,
           'pii_types_detected': list(set(
               pii_type for log in sanitized_logs 
               for pii_type in log.sanitized_fields
           )),
           'logs_sanitized': sum(1 for log in sanitized_logs if log.pii_detected),
           'total_logs': len(sanitized_logs)
       }

# ============================================================================
# Semantic Enricher
# ============================================================================

class SemanticEnricher:
   """Enhance telemetry with semantic understanding"""
   
   def __init__(self, config: Layer1Config):
       self.config = config
       self.domain_knowledge = DomainKnowledgeBase()
       self.context_analyzer = ContextAnalyzer()
       
   async def enrich_telemetry(self, record: FusedTelemetryRecord) -> FusedTelemetryRecord:
       """Enrich telemetry record with semantic understanding"""
       
       # Extract semantic context
       semantic_context = await self._extract_semantic_context(record)
       
       # Enhance anomaly indicators with semantic understanding
       enhanced_indicators = await self._enhance_anomaly_indicators(
           record.anomaly_indicators, semantic_context
       )
       
       # Generate semantic features for graph
       graph_features = await self._generate_semantic_graph_features(record)
       
       # Update record with enrichments
       enriched_record = FusedTelemetryRecord(
           event_id=record.event_id,
           timestamp=record.timestamp,
           function_id=record.function_id,
           session_id=record.session_id,
           l0_telemetry=record.l0_telemetry,
           log_data=record.log_data,
           metric_data=record.metric_data,
           trace_data=record.trace_data,
           unified_embedding=record.unified_embedding,
           anomaly_indicators=enhanced_indicators,
           quality_assessment=record.quality_assessment,
           privacy_score=record.privacy_score,
           sanitization_applied=record.sanitization_applied,
           audit_trail=record.audit_trail,
           execution_graph=record.execution_graph,
           graph_features={**record.graph_features, **graph_features}
       )
       
       return enriched_record
   
   async def _extract_semantic_context(self, record: FusedTelemetryRecord) -> Dict[str, Any]:
       """Extract semantic context from telemetry record"""
       context = {}
       
       # Extract context from logs
       if record.log_data:
           log_keywords = []
           for log in record.log_data:
               if hasattr(log, 'semantic_embedding') and log.semantic_embedding.size > 0:
                   log_keywords.extend(getattr(log, 'keywords', []))
           
           context['log_keywords'] = list(set(log_keywords))
       
       # Extract context from function behavior
       if record.l0_telemetry:
           context['execution_phase'] = record.l0_telemetry.execution_phase.value
           context['anomaly_type'] = record.l0_telemetry.anomaly_type.value
           context['performance_profile'] = self._classify_performance_profile(record.l0_telemetry)
       
       # Extract context from traces
       if record.trace_data:
           context['trace_complexity'] = len(record.trace_data.spans)
           context['has_dependencies'] = len(record.trace_data.dependency_graph.edges()) > 0
       
       return context
   
   def _classify_performance_profile(self, telemetry: TelemetryRecord) -> str:
       """Classify performance profile of execution"""
       if telemetry.duration > 10.0:
           return "long_running"
       elif telemetry.memory_spike_kb > 100000:  # >100MB
           return "memory_intensive"
       elif telemetry.cpu_utilization > 80.0:
           return "cpu_intensive"
       elif telemetry.network_io_bytes > 1000000:  # >1MB
           return "io_intensive"
       else:
           return "normal"
   
   async def _enhance_anomaly_indicators(self, indicators: Dict[str, float],
                                       context: Dict[str, Any]) -> Dict[str, Any]:
       """Enhance anomaly indicators with semantic understanding"""
       enhanced = indicators.copy()
       
       # Add semantic anomaly scores
       enhanced['semantic_anomaly'] = self._calculate_semantic_anomaly_score(context)
       enhanced['behavioral_anomaly'] = self._calculate_behavioral_anomaly_score(context)
       enhanced['contextual_anomaly'] = self._calculate_contextual_anomaly_score(context)
       
       return enhanced
   
   def _calculate_semantic_anomaly_score(self, context: Dict[str, Any]) -> float:
       """Calculate semantic-based anomaly score"""
       score = 0.0
       
       # Check for suspicious keywords
       keywords = context.get('log_keywords', [])
       suspicious_keywords = ['attack', 'breach', 'unauthorized', 'exploit', 'malware']
       
       for keyword in keywords:
           if any(sus in keyword.lower() for sus in suspicious_keywords):
               score += 0.3
       
       return min(score, 1.0)
   
   def _calculate_behavioral_anomaly_score(self, context: Dict[str, Any]) -> float:
       """Calculate behavior-based anomaly score"""
       score = 0.0
       
       # Check performance profile anomalies
       profile = context.get('performance_profile', 'normal')
       if profile in ['long_running', 'memory_intensive']:
           score += 0.4
       
       # Check execution phase anomalies
       phase = context.get('execution_phase', 'invoke')
       anomaly_type = context.get('anomaly_type', 'benign')
       
       if phase == 'init' and anomaly_type != 'cold_start':
           score += 0.3
       
       return min(score, 1.0)
   
   def _calculate_contextual_anomaly_score(self, context: Dict[str, Any]) -> float:
       """Calculate context-based anomaly score"""
       score = 0.0
       
       # Check trace complexity anomalies
       complexity = context.get('trace_complexity', 1)
       if complexity > 10:  # Very complex trace
           score += 0.2
       
       # Check dependency anomalies
       has_deps = context.get('has_dependencies', False)
       if has_deps:
           score += 0.1  # Dependencies add complexity
       
       return min(score, 1.0)
   
   async def _generate_semantic_graph_features(self, record: FusedTelemetryRecord) -> Dict[str, Any]:
       """Generate semantic features for graph representation"""
       features = {}
       
       # Log-based features
       if record.log_data:
           features['log_diversity'] = len(set(log.pattern_id for log in record.log_data))
           features['avg_log_anomaly'] = np.mean([log.pattern_anomaly_score for log in record.log_data])
           features['pii_risk'] = sum(1 for log in record.log_data if log.pii_detected)
       
       # Metric-based features
       if record.metric_data:
           features['performance_variance'] = record.metric_data.drift_score
           features['reliability_score'] = record.metric_data.reliability_score
           features['concurrency_level'] = record.metric_data.concurrency_level
       
       # Trace-based features
       if record.trace_data:
           features['trace_completeness'] = record.trace_data.trace_completeness
           features['correlation_confidence'] = record.trace_data.correlation_confidence
           features['has_timing_anomalies'] = len(record.trace_data.timing_anomalies) > 0
       
       return features

class DomainKnowledgeBase:
   """Domain knowledge for serverless security"""
   
   def __init__(self):
       self.attack_patterns = {
           'injection': ['sql', 'script', 'command', 'ldap'],
           'traversal': ['../../../', '%2e%2e%2f', '..\\'],
           'enumeration': ['admin', 'test', 'debug', 'config'],
           'resource_abuse': ['timeout', 'memory', 'cpu', 'exhaust']
       }
       
       self.normal_patterns = {
           'initialization': ['cold_start', 'init', 'bootstrap'],
           'processing': ['invoke', 'execute', 'process'],
           'cleanup': ['shutdown', 'cleanup', 'teardown']
       }

class ContextAnalyzer:
   """Analyze execution context for anomalies"""
   
   def __init__(self):
       self.baseline_contexts = {}
   
   def analyze_context(self, context: Dict[str, Any]) -> Dict[str, float]:
       """Analyze context for anomalies"""
       return {
           'context_deviation': 0.0,  # Placeholder
           'temporal_anomaly': 0.0,   # Placeholder
           'behavioral_shift': 0.0    # Placeholder
       }

# ============================================================================
# Temporal Aligner
# ============================================================================

class TemporalAligner:
   """Align multi-modal data temporally"""
   
   def __init__(self, config: Layer1Config):
       self.config = config
       self.alignment_window = config.correlation_window_seconds
       self.alignment_tolerance = 1.0  # 1 second tolerance
       
   async def align_telemetry_data(self, 
                                logs: List[ProcessedLogEntry],
                                metrics: List[MetricAggregation],
                                traces: List[TraceCorrelation]) -> List[Dict[str, Any]]:
       """Align multi-modal telemetry data temporally"""
       
       # Create time-ordered events
       all_events = []
       
       # Add log events
       for log in logs:
           all_events.append({
               'timestamp': log.timestamp,
               'type': 'log',
               'data': log
           })
       
       # Add metric events
       for metric in metrics:
           all_events.append({
               'timestamp': metric.timestamp,
               'type': 'metric',
               'data': metric
           })
       
       # Add trace events
       for trace in traces:
           # Use the latest span timestamp as trace timestamp
           if trace.spans:
               latest_span_time = max(span['start_time'] + span['duration'] for span in trace.spans)
               all_events.append({
                   'timestamp': latest_span_time,
                   'type': 'trace',
                   'data': trace
               })
       
       # Sort by timestamp
       all_events.sort(key=lambda x: x['timestamp'])
       
       # Group events into alignment windows
       aligned_groups = []
       current_group = []
       current_window_start = None
       
       for event in all_events:
           if current_window_start is None:
               current_window_start = event['timestamp']
               current_group = [event]
           elif event['timestamp'] - current_window_start <= self.alignment_window:
               current_group.append(event)
           else:
               # Finalize current group
               if current_group:
                   aligned_groups.append({
                       'window_start': current_window_start,
                       'window_end': current_window_start + self.alignment_window,
                       'events': current_group,
                       'event_count': len(current_group)
                   })
               
               # Start new group
               current_window_start = event['timestamp']
               current_group = [event]
       
       # Add final group
       if current_group:
           aligned_groups.append({
               'window_start': current_window_start,
               'window_end': current_window_start + self.alignment_window,
               'events': current_group,
               'event_count': len(current_group)
           })
       
       return aligned_groups

# ============================================================================
# Modality Fusion Engine
# ============================================================================

class ModalityFusionEngine:
   """Fuse different telemetry modalities into unified representation"""
   
   def __init__(self, config: Layer1Config):
       self.config = config
       self.fusion_weights = {
           'logs': 0.4,
           'metrics': 0.3,
           'traces': 0.3
       }
       
   async def fuse_telemetry_modalities(self,
                                     logs: List[ProcessedLogEntry],
                                     metrics: Optional[MetricAggregation],
                                     traces: Optional[TraceCorrelation],
                                     l0_telemetry: TelemetryRecord) -> FusedTelemetryRecord:
       """Fuse multiple telemetry modalities into unified record"""
       
       # Generate unified embedding
       unified_embedding = await self._create_unified_embedding(logs, metrics, traces)
       
       # Calculate anomaly indicators
       anomaly_indicators = await self._fuse_anomaly_indicators(logs, metrics, traces, l0_telemetry)
       
       # Assess data quality
       quality_assessment = await self._assess_fusion_quality(logs, metrics, traces)
       
       # Build execution graph
       execution_graph = await self._build_execution_graph(logs, metrics, traces, l0_telemetry)
       
       # Extract graph features
       graph_features = await self._extract_graph_features(execution_graph)
       
       # Generate session ID
       session_id = self._generate_session_id(l0_telemetry, logs)
       
       fused_record = FusedTelemetryRecord(
           event_id=l0_telemetry.event_id,
           timestamp=l0_telemetry.timestamp,
           function_id=l0_telemetry.function_id,
           session_id=session_id,
           l0_telemetry=l0_telemetry,
           log_data=logs,
           metric_data=metrics,
           trace_data=traces,
           unified_embedding=unified_embedding,
           anomaly_indicators=anomaly_indicators,
           quality_assessment=quality_assessment,
           privacy_score=1.0,  # Will be updated by sanitizer
           sanitization_applied=False,  # Will be updated by sanitizer
           audit_trail={},  # Will be updated by sanitizer
           execution_graph=execution_graph,
           graph_features=graph_features
       )
       
       return fused_record
   
   async def _create_unified_embedding(self,
                                     logs: List[ProcessedLogEntry],
                                     metrics: Optional[MetricAggregation],
                                     traces: Optional[TraceCorrelation]) -> np.ndarray:
       """Create unified embedding from all modalities"""
       
       embeddings = []
       
       # Log embeddings
       if logs:
           log_embeddings = [
               log.semantic_embedding for log in logs 
               if log.semantic_embedding.size > 0
           ]
           if log_embeddings:
               avg_log_embedding = np.mean(log_embeddings, axis=0)
               embeddings.append(avg_log_embedding * self.fusion_weights['logs'])
       
       # Metric embeddings (create from numeric features)
       if metrics:
           metric_features = np.array([
               metrics.duration_stats['mean'],
               metrics.memory_stats['mean'],
               metrics.cpu_stats['mean'],
               metrics.concurrency_level,
               metrics.request_rate,
               metrics.error_rate,
               metrics.completeness_score,
               metrics.reliability_score,
               metrics.drift_score
           ])
           
           # Normalize to embedding space
           metric_embedding = metric_features / np.linalg.norm(metric_features) if np.linalg.norm(metric_features) > 0 else metric_features
           embeddings.append(metric_embedding * self.fusion_weights['metrics'])
       
       # Trace embeddings (create from trace features)
       if traces:
           trace_features = np.array([
               traces.total_duration,
               len(traces.spans),
               len(traces.dependency_graph.nodes()),
               len(traces.dependency_graph.edges()),
               traces.trace_completeness,
               traces.correlation_confidence,
               len(traces.timing_anomalies),
               len(traces.dependency_anomalies)
           ])
           
           # Normalize to embedding space
           trace_embedding = trace_features / np.linalg.norm(trace_features) if np.linalg.norm(trace_features) > 0 else trace_features
           embeddings.append(trace_embedding * self.fusion_weights['traces'])
       
       # Combine embeddings
       if embeddings:
           # Pad to same dimension
           max_dim = max(emb.shape[0] for emb in embeddings)
           padded_embeddings = []
           for emb in embeddings:
               if emb.shape[0] < max_dim:
                   padded = np.pad(emb, (0, max_dim - emb.shape[0]), mode='constant')
                   padded_embeddings.append(padded)
               else:
                   padded_embeddings.append(emb[:max_dim])
           
           unified = np.mean(padded_embeddings, axis=0)
           return unified
       else:
           return np.array([])
   
   async def _fuse_anomaly_indicators(self,
                                    logs: List[ProcessedLogEntry],
                                    metrics: Optional[MetricAggregation],
                                    traces: Optional[TraceCorrelation],
                                    l0_telemetry: TelemetryRecord) -> Dict[str, float]:
       """Fuse anomaly indicators from all modalities"""
       
       indicators = {}
       
       # Layer 0 anomaly indicator
       indicators['l0_anomaly'] = 1.0 if l0_telemetry.anomaly_type != AnomalyType.BENIGN else 0.0
       
       # Log-based indicators
       if logs:
           indicators['log_anomaly'] = np.mean([log.pattern_anomaly_score for log in logs])
           indicators['known_pattern_match'] = any(log.known_anomaly_match for log in logs)
           indicators['pii_risk'] = any(log.pii_detected for log in logs)
       
       # Metric-based indicators
       if metrics:
           indicators['performance_drift'] = metrics.drift_score
           indicators['reliability_degradation'] = 1.0 - metrics.reliability_score
           indicators['high_error_rate'] = 1.0 if metrics.error_rate > 0.1 else 0.0
       
       # Trace-based indicators
       if traces:
           indicators['timing_anomaly'] = 1.0 if traces.timing_anomalies else 0.0
           indicators['dependency_anomaly'] = 1.0 if traces.dependency_anomalies else 0.0
           indicators['correlation_uncertainty'] = 1.0 - traces.correlation_confidence
       
       # Composite indicators
       indicators['overall_anomaly_score'] = self._calculate_composite_anomaly_score(indicators)
       
       return indicators
   
   def _calculate_composite_anomaly_score(self, indicators: Dict[str, float]) -> float:
       """Calculate composite anomaly score"""
       # Weighted combination of indicators
       weights = {
           'l0_anomaly': 0.3,
           'log_anomaly': 0.2,
           'performance_drift': 0.15,
           'timing_anomaly': 0.15,
           'known_pattern_match': 0.1,
           'pii_risk': 0.1
       }
       
       score = 0.0
       for indicator, weight in weights.items():
           if indicator in indicators:
               score += indicators[indicator] * weight
       
       return min(score, 1.0)
   
   async def _assess_fusion_quality(self,
                                  logs: List[ProcessedLogEntry],
                                  metrics: Optional[MetricAggregation],
                                  traces: Optional[TraceCorrelation]) -> Dict[str, float]:
       """Assess quality of data fusion"""
       
       quality = {}
       
       # Data completeness
       modalities_present = 0
       if logs:
           modalities_present += 1
       if metrics:
           modalities_present += 1
       if traces:
           modalities_present += 1
       
       quality['completeness'] = modalities_present / 3.0
       
       # Data consistency (simplified)
       quality['consistency'] = 1.0  # Placeholder - would check temporal alignment
       
       # Data freshness
       if logs:
           newest_log_time = max(log.timestamp for log in logs)
           freshness = max(0, 1.0 - (time.time() - newest_log_time) / 300.0)  # 5 min window
           quality['freshness'] = freshness
       else:
           quality['freshness'] = 0.0
       
       # Overall quality score
       quality['overall'] = np.mean(list(quality.values()))
       
       return quality
   
   async def _build_execution_graph(self,
                                  logs: List[ProcessedLogEntry],
                                  metrics: Optional[MetricAggregation],
                                  traces: Optional[TraceCorrelation],
                                  l0_telemetry: TelemetryRecord) -> nx.DiGraph:
       """Build unified execution graph"""
       
       G = nx.DiGraph()
       
       # Add L0 telemetry as root node
       G.add_node('l0_root', 
                 type='telemetry',
                 timestamp=l0_telemetry.timestamp,
                 anomaly_type=l0_telemetry.anomaly_type.value,
                 execution_phase=l0_telemetry.execution_phase.value)
       
       # Add log nodes
       for i, log in enumerate(logs):
           node_id = f'log_{i}'
           G.add_node(node_id,
                     type='log',
                     timestamp=log.timestamp,
                     pattern_id=log.pattern_id,
                     anomaly_score=log.pattern_anomaly_score)
           
           # Connect to root if within time window
           if abs(log.timestamp - l0_telemetry.timestamp) < 60:
               G.add_edge('l0_root', node_id, weight=1.0, type='temporal')
       
       # Add metric node
       if metrics:
           G.add_node('metrics',
                     type='metrics',
                     timestamp=metrics.timestamp,
                     error_rate=metrics.error_rate,
                     drift_score=metrics.drift_score)
           G.add_edge('l0_root', 'metrics', weight=1.0, type='aggregation')
       
       # Add trace nodes
       if traces:
           for i, span in enumerate(traces.spans):
               span_id = f'span_{i}'
               G.add_node(span_id,
                         type='span',
                         timestamp=span['start_time'],
                         duration=span['duration'],
                         function_id=span['function_id'])
               G.add_edge('l0_root', span_id, weight=1.0, type='trace')
       
       return G
   
   async def _extract_graph_features(self, graph: nx.DiGraph) -> Dict[str, Any]:
       """Extract features from execution graph"""
       
       features = {}
       
       # Basic graph properties
       features['node_count'] = graph.number_of_nodes()
       features['edge_count'] = graph.number_of_edges()
       features['density'] = nx.density(graph)
       
       # Centrality measures
       if graph.number_of_nodes() > 0:
           try:
               betweenness = nx.betweenness_centrality(graph)
               features['max_betweenness'] = max(betweenness.values()) if betweenness else 0.0
               features['avg_betweenness'] = np.mean(list(betweenness.values())) if betweenness else 0.0
           except:
               features['max_betweenness'] = 0.0
               features['avg_betweenness'] = 0.0
       
       # Path characteristics
       try:
           if nx.is_weakly_connected(graph):
               features['diameter'] = nx.diameter(graph.to_undirected())
           else:
               features['diameter'] = 0
       except:
           features['diameter'] = 0
       
       # Node type distribution
       node_types = [graph.nodes[node].get('type', 'unknown') for node in graph.nodes()]
       type_counts = {node_type: node_types.count(node_type) for node_type in set(node_types)}
       features['node_type_distribution'] = type_counts
       
       return features
   
   def _generate_session_id(self, l0_telemetry: TelemetryRecord, logs: List[ProcessedLogEntry]) -> str:
       """Generate session ID for grouping related events"""
       
       # Try to use request ID from logs
       for log in logs:
           if log.request_id:
               return log.request_id
       
       # Fall back to time-based grouping
       time_window = int(l0_telemetry.timestamp // 300)  # 5-minute windows
       return f"session_{l0_telemetry.function_id}_{time_window}"

# ============================================================================
# Data Quality Assessor
# ============================================================================

class DataQualityAssessor:
   """Assess quality of fused telemetry data"""
   
   def __init__(self, config: Layer1Config):
       self.config = config
       self.quality_thresholds = {
           'completeness': config.min_completeness_score,
           'consistency': 0.8,
           'accuracy': 0.9,
           'timeliness': 0.7
       }
   
   async def assess_quality(self, record: FusedTelemetryRecord) -> Dict[str, Any]:
       """Comprehensive quality assessment"""
       
       assessment = {}
       
       # Completeness assessment
       assessment['completeness'] = self._assess_completeness(record)
       
       # Consistency assessment
       assessment['consistency'] = self._assess_consistency(record)
       
       # Accuracy assessment
       assessment['accuracy'] = self._assess_accuracy(record)
       
       # Timeliness assessment
       assessment['timeliness'] = self._assess_timeliness(record)
       
       # Overall quality score
       assessment['overall_score'] = np.mean([
           assessment['completeness'],
           assessment['consistency'],
           assessment['accuracy'],
           assessment['timeliness']
       ])
       
       # Quality flags
       assessment['quality_flags'] = self._generate_quality_flags(assessment)
       
       # Recommendations
       assessment['recommendations'] = self._generate_recommendations(assessment)
       
       return assessment
   
   def _assess_completeness(self, record: FusedTelemetryRecord) -> float:
       """Assess data completeness"""
       score = 0.0
       
       # L0 telemetry (required)
       if record.l0_telemetry:
           score += 0.4
       
       # Log data
       if record.log_data:
           score += 0.3
       
       # Metric data
       if record.metric_data:
           score += 0.2
       
       # Trace data
       if record.trace_data:
           score += 0.1
       
       return score
   
   def _assess_consistency(self, record: FusedTelemetryRecord) -> float:
       """Assess data consistency across modalities"""
       score = 1.0
       
       # Check timestamp consistency
       timestamps = [record.l0_telemetry.timestamp]
       
       if record.log_data:
           timestamps.extend([log.timestamp for log in record.log_data])
       
       if record.metric_data:
           timestamps.append(record.metric_data.timestamp)
       
       # Check for large timestamp deviations
       if len(timestamps) > 1:
           timestamp_std = np.std(timestamps)
           if timestamp_std > 60:  # > 1 minute deviation
               score -= 0.3
       
       # Check function ID consistency
       function_ids = [record.l0_telemetry.function_id]
       
       if record.log_data:
           function_ids.extend([log.function_id for log in record.log_data])
       
       if record.metric_data:
           function_ids.append(record.metric_data.function_id)
       
       unique_function_ids = set(function_ids)
       if len(unique_function_ids) > 1:
           score -= 0.2
       
       return max(score, 0.0)
   
   def _assess_accuracy(self, record: FusedTelemetryRecord) -> float:
       """Assess data accuracy"""
       score = 1.0
       
       # Check for fallback mode usage (indicates potential accuracy issues)
       if record.l0_telemetry.fallback_mode:
           score -= 0.2
       
       # Check log parsing accuracy
       if record.log_data:
           fallback_parsers = sum(1 for log in record.log_data if log.parser_
           parser_used == 'fallback')
           if fallback_parsers > 0:
               score -= 0.1 * (fallback_parsers / len(record.log_data))
       
       # Check trace correlation confidence
       if record.trace_data:
           if record.trace_data.correlation_confidence < 0.7:
               score -= 0.2
       
       # Check for data quality indicators
       if record.quality_assessment:
           if record.quality_assessment.get('overall', 1.0) < 0.8:
               score -= 0.1
       
       return max(score, 0.0)
   
   def _assess_timeliness(self, record: FusedTelemetryRecord) -> float:
       """Assess data timeliness"""
       current_time = time.time()
       record_age = current_time - record.timestamp
       
       # Score based on age (fresher is better)
       if record_age < 60:  # < 1 minute
           return 1.0
       elif record_age < 300:  # < 5 minutes
           return 0.8
       elif record_age < 900:  # < 15 minutes
           return 0.6
       elif record_age < 1800:  # < 30 minutes
           return 0.4
       else:
           return 0.2
   
   def _generate_quality_flags(self, assessment: Dict[str, float]) -> List[str]:
       """Generate quality warning flags"""
       flags = []
       
       for dimension, score in assessment.items():
           if dimension in self.quality_thresholds:
               threshold = self.quality_thresholds[dimension]
               if score < threshold:
                   flags.append(f"low_{dimension}")
       
       if assessment.get('overall_score', 1.0) < 0.7:
           flags.append("overall_quality_concern")
       
       return flags
   
   def _generate_recommendations(self, assessment: Dict[str, float]) -> List[str]:
       """Generate quality improvement recommendations"""
       recommendations = []
       
       if assessment.get('completeness', 1.0) < 0.8:
           recommendations.append("Enable additional telemetry sources")
       
       if assessment.get('consistency', 1.0) < 0.8:
           recommendations.append("Check temporal alignment configuration")
       
       if assessment.get('accuracy', 1.0) < 0.9:
           recommendations.append("Review parsing strategies and correlation logic")
       
       if assessment.get('timeliness', 1.0) < 0.7:
           recommendations.append("Reduce processing latency and buffering delays")
       
       return recommendations

# ============================================================================
# Main Layer 1 Controller
# ============================================================================

class Layer1_MultiModalFusionIntake:
   """
   Main Layer 1 controller implementing multi-modal fusion intake
   
   This class orchestrates all Layer 1 components to process telemetry from
   Layer 0 and produce unified, privacy-preserving, semantically-enriched
   representations suitable for downstream anomaly detection.
   """
   
   def __init__(self, config: Optional[Layer1Config] = None):
       self.config = config or Layer1Config()
       
       # Initialize core components
       self.log_processor = ContentAwareLogProcessor(self.config)
       self.metric_aggregator = EfficientMetricAggregator(self.config)
       self.trace_correlator = TraceCorrelator(self.config)
       self.privacy_sanitizer = PrivacyPreservingSanitizer(self.config)
       self.semantic_enricher = SemanticEnricher(self.config)
       self.temporal_aligner = TemporalAligner(self.config)
       self.modality_fusion = ModalityFusionEngine(self.config)
       self.quality_assessor = DataQualityAssessor(self.config)
       
       # Processing state
       self.processing_stats = {
           'records_processed': 0,
           'errors_encountered': 0,
           'average_processing_time': 0.0,
           'quality_failures': 0
       }
       
       # Buffers for streaming processing
       self.log_buffer = deque(maxlen=1000)
       self.metric_buffer = deque(maxlen=100)
       self.trace_buffer = deque(maxlen=100)
       
       # Background tasks
       self.background_tasks = []
       
       logging.info("Layer 1 Multi-Modal Fusion Intake initialized")
   
   async def process_l0_telemetry_stream(self, 
                                       telemetry_stream: AsyncIterator[TelemetryRecord]) -> AsyncIterator[FusedTelemetryRecord]:
       """
       Main processing pipeline: convert L0 telemetry stream to fused records
       """
       
       async for l0_telemetry in telemetry_stream:
           try:
               start_time = time.time()
               
               # Process the L0 telemetry record
               fused_record = await self.process_single_telemetry(l0_telemetry)
               
               # Update statistics
               processing_time = time.time() - start_time
               self._update_processing_stats(processing_time)
               
               yield fused_record
               
           except Exception as e:
               logging.error(f"Error processing L0 telemetry {l0_telemetry.event_id}: {e}")
               self.processing_stats['errors_encountered'] += 1
               
               # Generate fallback record
               fallback_record = await self._create_fallback_record(l0_telemetry, e)
               yield fallback_record
   
   async def process_single_telemetry(self, l0_telemetry: TelemetryRecord) -> FusedTelemetryRecord:
       """Process a single L0 telemetry record through the full pipeline"""
       
       # Step 1: Simulate log extraction (in production, this would come from actual log streams)
       simulated_logs = await self._simulate_logs_from_telemetry(l0_telemetry)
       
       # Step 2: Process logs through content-aware processor
       processed_logs = []
       if simulated_logs:
           log_processing_result = await self.log_processor.process_log_stream(
               self._async_iterator_from_list(simulated_logs)
           )
           processed_logs = log_processing_result['processed_logs']
       
       # Step 3: Generate metric aggregation
       metric_aggregation = None
       if self.config.enable_temporal_alignment:
           # In production, this would aggregate over time windows
           metric_aggregation = await self._create_metric_aggregation_from_telemetry(l0_telemetry)
       
       # Step 4: Generate trace correlation
       trace_correlation = None
       if self.config.enable_graph_correlation:
           trace_correlation = await self.trace_correlator.process_telemetry_for_correlation(l0_telemetry)
       
       # Step 5: Fuse modalities
       fused_record = await self.modality_fusion.fuse_telemetry_modalities(
           processed_logs, metric_aggregation, trace_correlation, l0_telemetry
       )
       
       # Step 6: Apply privacy sanitization
       if self.config.enable_privacy_sanitization:
           fused_record = await self.privacy_sanitizer.sanitize_telemetry_record(fused_record)
       
       # Step 7: Semantic enrichment
       if self.config.enable_semantic_analysis:
           fused_record = await self.semantic_enricher.enrich_telemetry(fused_record)
       
       # Step 8: Quality assessment
       quality_assessment = await self.quality_assessor.assess_quality(fused_record)
       
       # Update record with quality assessment
       fused_record.quality_assessment.update(quality_assessment)
       
       # Check quality thresholds
       if quality_assessment['overall_score'] < self.config.min_completeness_score:
           self.processing_stats['quality_failures'] += 1
           logging.warning(f"Quality failure for record {fused_record.event_id}: {quality_assessment['overall_score']}")
       
       self.processing_stats['records_processed'] += 1
       
       return fused_record
   
   async def _simulate_logs_from_telemetry(self, telemetry: TelemetryRecord) -> List[str]:
       """
       Simulate log entries from telemetry (in production, logs would come from actual streams)
       """
       logs = []
       
       # Generate initialization log
       if telemetry.execution_phase == ExecutionPhase.INIT:
           logs.append(f"INIT_START Runtime Version: 3.9 Runtime Version ARN: arn:aws:lambda:us-east-1::runtime:python3.9")
           logs.append(f"START RequestId: {telemetry.event_id} Version: $LATEST")
       
       # Generate application logs based on anomaly type
       if telemetry.anomaly_type == AnomalyType.CPU_BURST:
           logs.append(f"INFO High CPU utilization detected: {telemetry.cpu_utilization}%")
           logs.append(f"WARN CPU intensive operation taking longer than expected")
       elif telemetry.anomaly_type == AnomalyType.MEMORY_SPIKE:
           logs.append(f"INFO Memory usage spike: {telemetry.memory_spike_kb}KB")
           logs.append(f"WARN Memory allocation pattern unusual")
       elif telemetry.anomaly_type == AnomalyType.IO_INTENSIVE:
           logs.append(f"INFO Network I/O operation: {telemetry.network_io_bytes} bytes")
           logs.append(f"DEBUG Processing large data transfer")
       
       # Generate completion logs
       if telemetry.execution_phase == ExecutionPhase.INVOKE:
           logs.append(f"END RequestId: {telemetry.event_id}")
           logs.append(f"REPORT RequestId: {telemetry.event_id} Duration: {telemetry.duration*1000:.2f} ms "
                      f"Billed Duration: {int(telemetry.duration*1000)} ms Memory Size: 512 MB "
                      f"Max Memory Used: {telemetry.memory_spike_kb//1024} MB")
       
       return logs
   
   async def _create_metric_aggregation_from_telemetry(self, telemetry: TelemetryRecord) -> MetricAggregation:
       """Create metric aggregation from single telemetry record (simplified for demo)"""
       
       return MetricAggregation(
           timestamp=telemetry.timestamp,
           metric_type='single_record_aggregation',
           function_id=telemetry.function_id,
           duration_stats={
               'mean': telemetry.duration,
               'median': telemetry.duration,
               'std': 0.0,
               'min': telemetry.duration,
               'max': telemetry.duration,
               'p95': telemetry.duration,
               'p99': telemetry.duration
           },
           memory_stats={
               'mean': float(telemetry.memory_spike_kb),
               'median': float(telemetry.memory_spike_kb),
               'std': 0.0,
               'min': float(telemetry.memory_spike_kb),
               'max': float(telemetry.memory_spike_kb),
               'p95': float(telemetry.memory_spike_kb),
               'p99': float(telemetry.memory_spike_kb)
           },
           cpu_stats={
               'mean': telemetry.cpu_utilization,
               'median': telemetry.cpu_utilization,
               'std': 0.0,
               'min': telemetry.cpu_utilization,
               'max': telemetry.cpu_utilization
           },
           io_stats={
               'mean': float(telemetry.network_io_bytes),
               'median': float(telemetry.network_io_bytes),
               'std': 0.0,
               'total': float(telemetry.network_io_bytes)
           },
           concurrency_level=1,
           request_rate=1.0,
           error_rate=0.0 if telemetry.anomaly_type == AnomalyType.BENIGN else 0.1,
           completeness_score=telemetry.completeness_score,
           reliability_score=0.0 if telemetry.fallback_mode else 1.0,
           drift_score=0.0
       )
   
   async def _async_iterator_from_list(self, items: List[Any]) -> AsyncIterator[Any]:
       """Convert list to async iterator"""
       for item in items:
           yield item
   
   async def _create_fallback_record(self, l0_telemetry: TelemetryRecord, error: Exception) -> FusedTelemetryRecord:
       """Create fallback fused record when processing fails"""
       
       return FusedTelemetryRecord(
           event_id=l0_telemetry.event_id,
           timestamp=l0_telemetry.timestamp,
           function_id=l0_telemetry.function_id,
           session_id=f"fallback_{l0_telemetry.event_id}",
           l0_telemetry=l0_telemetry,
           log_data=[],
           metric_data=None,
           trace_data=None,
           unified_embedding=np.array([]),
           anomaly_indicators={'processing_error': 1.0},
           quality_assessment={'overall_score': 0.0, 'error': str(error)},
           privacy_score=1.0,
           sanitization_applied=False,
           audit_trail={'fallback_reason': str(error)},
           execution_graph=nx.DiGraph(),
           graph_features={'fallback_mode': True}
       )
   
   def _update_processing_stats(self, processing_time: float):
       """Update processing statistics"""
       self.processing_stats['average_processing_time'] = (
           (self.processing_stats['average_processing_time'] * self.processing_stats['records_processed'] + processing_time) /
           (self.processing_stats['records_processed'] + 1)
       )
   
   def get_processing_statistics(self) -> Dict[str, Any]:
       """Get current processing statistics"""
       return {
           **self.processing_stats,
           'log_processor_stats': self.log_processor.stats,
           'metric_aggregator_stats': self.metric_aggregator.stats,
           'trace_correlator_stats': self.trace_correlator.stats,
           'privacy_sanitizer_stats': self.privacy_sanitizer.stats
       }
   
   async def health_check(self) -> Dict[str, Any]:
       """Perform health check of all components"""
       health = {
           'status': 'healthy',
           'components': {},
           'timestamp': time.time()
       }
       
       # Check component health
       components = [
           ('log_processor', self.log_processor),
           ('metric_aggregator', self.metric_aggregator),
           ('trace_correlator', self.trace_correlator),
           ('privacy_sanitizer', self.privacy_sanitizer),
           ('semantic_enricher', self.semantic_enricher),
           ('modality_fusion', self.modality_fusion),
           ('quality_assessor', self.quality_assessor)
       ]
       
       for name, component in components:
           try:
               # Basic health check - ensure component is initialized
               if hasattr(component, 'config'):
                   health['components'][name] = 'healthy'
               else:
                   health['components'][name] = 'unhealthy'
                   health['status'] = 'degraded'
           except Exception as e:
               health['components'][name] = f'error: {str(e)}'
               health['status'] = 'degraded'
       
       return health
   
   async def shutdown(self):
       """Graceful shutdown of Layer 1 processing"""
       logging.info("Shutting down Layer 1 Multi-Modal Fusion Intake")
       
       # Cancel background tasks
       for task in self.background_tasks:
           task.cancel()
       
       # Clear buffers
       self.log_buffer.clear()
       self.metric_buffer.clear()
       self.trace_buffer.clear()
       
       logging.info("Layer 1 shutdown complete")

# ============================================================================
# Integration Interface for Layer 2
# ============================================================================

class Layer1Interface:
   """Standardized interface for Layer 2 integration"""
   
   def __init__(self, layer1_controller: Layer1_MultiModalFusionIntake):
       self.controller = layer1_controller
   
   async def get_fused_telemetry_stream(self, 
                                      l0_stream: AsyncIterator[TelemetryRecord]) -> AsyncIterator[FusedTelemetryRecord]:
       """Get stream of fused telemetry records for Layer 2"""
       async for fused_record in self.controller.process_l0_telemetry_stream(l0_stream):
           yield fused_record
   
   def get_processing_metrics(self) -> Dict[str, Any]:
       """Get processing metrics for monitoring"""
       return self.controller.get_processing_statistics()
   
   async def get_health_status(self) -> Dict[str, Any]:
       """Get health status for monitoring"""
       return await self.controller.health_check()

# ============================================================================
# Example Usage and Testing
# ============================================================================

async def main():
   """Example usage of Layer 1 Multi-Modal Fusion Intake"""
   
   # Initialize configuration
   config = Layer1Config(
       enable_semantic_analysis=True,
       enable_graph_correlation=True,
       enable_privacy_sanitization=True,
       enable_temporal_alignment=True
   )
   
   # Initialize Layer 1
   layer1 = Layer1_MultiModalFusionIntake(config)
   
   # Example L0 telemetry records
   sample_telemetry = [
       TelemetryRecord(
           event_id="test_001",
           timestamp=time.time(),
           function_id="example_function",
           execution_phase=ExecutionPhase.INVOKE,
           anomaly_type=AnomalyType.CPU_BURST,
           duration=2.5,
           memory_spike_kb=150000,
           cpu_utilization=85.0,
           network_io_bytes=50000,
           fallback_mode=False,
           source="layer0",
           concurrency_id="TEST",
           completeness_score=0.95
       )
   ]
   
   # Process telemetry
   async def telemetry_generator():
       for record in sample_telemetry:
           yield record
   
   print("Processing sample telemetry through Layer 1...")
   
   async for fused_record in layer1.process_l0_telemetry_stream(telemetry_generator()):
       print(f"Processed record {fused_record.event_id}")
       print(f"  - Log entries: {len(fused_record.log_data)}")
       print(f"  - Anomaly indicators: {fused_record.anomaly_indicators}")
       print(f"  - Quality score: {fused_record.quality_assessment.get('overall_score', 'N/A')}")
       print(f"  - Privacy score: {fused_record.privacy_score}")
       print()
   
   # Get statistics
   stats = layer1.get_processing_statistics()
   print("Processing Statistics:")
   for key, value in stats.items():
       print(f"  {key}: {value}")
   
   # Health check
   health = await layer1.health_check()
   print(f"\nHealth Status: {health['status']}")
   
   # Shutdown
   await layer1.shutdown()

if __name__ == "__main__":
   asyncio.run(main())