# Default production configuration
        default_config = {
            "sanitization_level": SanitizationLevel.STANDARD,
            "processing_mode": ProcessingMode.ASYNCHRONOUS,
            "optimization_strategy": OptimizationStrategy.THROUGHPUT_OPTIMIZED,
            "enable_ml_detection": True,
            "enable_security_sanitizers": True,
            "enable_performance_monitoring": True,
            "cache_settings": {
                "result_cache_size": 10000,
                "pattern_cache_size": 100000,
                "cache_ttl_seconds": 300
            },
            "performance_targets": {
                "max_latency_ms": 0.15,
                "min_throughput_rps": 10000,
                "max_memory_mb": 100,
                "anomaly_preservation_rate": 0.9995
            },
            "security_settings": {
                "enable_injection_detection": True,
                "enable_homograph_detection": True,
                "enable_steganography_detection": True,
                "enable_covert_channel_detection": True,
                "threat_score_threshold": 0.7
            },
            "compliance_settings": {
                "gdpr_enabled": True,
                "ccpa_enabled": True,
                "hipaa_enabled": False,
                "pci_dss_enabled": False,
                "audit_logging": True
            }
        }
        
        # Merge with provided config
        if config:
            default_config.update(config)
        
        # Environment-specific adjustments
        if environment == "development":
            default_config.update({
                "sanitization_level": SanitizationLevel.MINIMAL,
                "performance_targets": {
                    "max_latency_ms": 1.0,
                    "min_throughput_rps": 1000,
                    "max_memory_mb": 200,
                    "anomaly_preservation_rate": 0.99
                },
                "enable_debug_logging": True
            })
        elif environment == "testing":
            default_config.update({
                "sanitization_level": SanitizationLevel.AGGRESSIVE,
                "enable_comprehensive_validation": True,
                "enable_stress_testing": True
            })
        elif environment == "staging":
            default_config.update({
                "sanitization_level": SanitizationLevel.STANDARD,
                "enable_performance_profiling": True,
                "enable_load_testing": True
            })
        
        # Create sanitization engine
        engine = EnhancedSanitizationEngine(
            sanitization_level=default_config["sanitization_level"],
            processing_mode=default_config["processing_mode"],
            optimization_strategy=default_config["optimization_strategy"]
        )
        
        # Configure ML detection
        if default_config["enable_ml_detection"]:
            engine.anomaly_detector.train(SCAFADL1Factory._generate_training_data())
        
        # Create integration layer
        integration = SCAFADLayer1Integration(engine)
        
        # Apply security settings
        if default_config["enable_security_sanitizers"]:
            SCAFADL1Factory._configure_security_sanitizers(engine, default_config["security_settings"])
        
        # Configure monitoring
        if default_config["enable_performance_monitoring"]:
            SCAFADL1Factory._configure_monitoring(integration, default_config)
        
        # Configure compliance
        SCAFADL1Factory._configure_compliance(integration, default_config["compliance_settings"])
        
        return integration
    
    @staticmethod
    def _generate_training_data() -> List[Dict[str, Any]]:
        """Generate comprehensive training data for ML models"""
        training_data = []
        
        # Normal serverless patterns
        normal_patterns = [
            # API Gateway invocations
            {
                "function_name": "api-handler",
                "execution_context": {"duration_ms": 150, "memory_used_mb": 64, "cold_start": False},
                "invocation_graph": {"depth": 0, "fan_out": 0},
                "performance_metrics": {"network_calls": 2, "database_queries": 1}
            },
            # Background processing
            {
                "function_name": "data-processor",
                "execution_context": {"duration_ms": 2500, "memory_used_mb": 256, "cold_start": False},
                "invocation_graph": {"depth": 1, "fan_out": 3},
                "performance_metrics": {"io_operations": 10, "cache_hits": 5}
            },
            # Quick utilities
            {
                "function_name": "auth-validator",
                "execution_context": {"duration_ms": 50, "memory_used_mb": 32, "cold_start": False},
                "invocation_graph": {"depth": 0, "fan_out": 0},
                "performance_metrics": {"network_calls": 1, "cache_hits": 1}
            }
        ] * 100  # 300 normal samples
        
        # Anomalous patterns
        anomalous_patterns = [
            # SQL injection attempt
            {
                "function_name": "'; DROP TABLE users;--",
                "execution_context": {"duration_ms": 10000, "memory_used_mb": 512, "cold_start": True},
                "security_context": {"user_identity": "anonymous", "source_ip": "192.168.1.100"},
                "error_details": {"error_type": "SQLSyntaxError", "error_message": "Malformed query"}
            },
            # XSS attempt
            {
                "function_name": "normal-function",
                "execution_context": {"duration_ms": 100, "memory_used_mb": 64, "cold_start": False},
                "security_context": {
                    "user_agent": "<script>alert('xss')</script>",
                    "source_ip": "10.0.0.50"
                }
            },
            # Resource exhaustion attack
            {
                "function_name": "compute-heavy",
                "execution_context": {"duration_ms": 900000, "memory_used_mb": 3008, "cold_start": False},
                "invocation_graph": {"depth": 10, "fan_out": 100},
                "performance_metrics": {"cpu_utilization": 100, "io_operations": 1000000}
            },
            # Timing attack
            {
                "function_name": "auth-timing",
                "execution_context": {"duration_ms": 5000, "memory_used_mb": 64, "cold_start": False},
                "performance_metrics": {"network_calls": 1, "database_queries": 50}
            },
            # Data exfiltration pattern
            {
                "function_name": "data-export",
                "execution_context": {"duration_ms": 30000, "memory_used_mb": 1024, "cold_start": False},
                "invocation_graph": {"depth": 0, "fan_out": 50},
                "performance_metrics": {"network_calls": 100, "io_operations": 1000}
            }
        ] * 20  # 100 anomalous samples
        
        training_data.extend(normal_patterns)
        training_data.extend(anomalous_patterns)
        
        # Add timestamps and IDs
        current_time = time.time()
        for i, record in enumerate(training_data):
            record.update({
                "timestamp": current_time - (len(training_data) - i) * 60,  # Spread over time
                "invocation_id": f"train-{i:06d}",
                "metadata": {"training_sample": True, "sample_id": i}
            })
        
        return training_data
    
    @staticmethod
    def _configure_security_sanitizers(engine: EnhancedSanitizationEngine, security_settings: Dict[str, Any]):
        """Configure security sanitizers based on settings"""
        
        # Add security-specific sanitization rules
        if security_settings.get("enable_injection_detection", True):
            injection_rule = SanitizationRule(
                rule_name="injection_detection",
                rule_type=SanitizationType.INJECTION_DETECTION,
                target_fields=["*"],
                sanitization_function=SecuritySanitizers._sanitize_injection_detection,
                priority=95,
                preserve_anomaly=True,
                config={"high_threat_threshold": security_settings.get("threat_score_threshold", 0.7)}
            )
            engine.sanitizers[SanitizationType.INJECTION_DETECTION] = injection_rule.sanitization_function
        
        if security_settings.get("enable_homograph_detection", True):
            homograph_rule = SanitizationRule(
                rule_name="homograph_detection",
                rule_type=SanitizationType.HOMOGRAPH_ATTACK,
                target_fields=["function_name", "user_identity", "*_name"],
                sanitization_function=SecuritySanitizers._sanitize_homograph_attack,
                priority=85,
                preserve_anomaly=True
            )
            engine.sanitizers[SanitizationType.HOMOGRAPH_ATTACK] = homograph_rule.sanitization_function
        
        if security_settings.get("enable_steganography_detection", True):
            stego_rule = SanitizationRule(
                rule_name="steganography_detection",
                rule_type=SanitizationType.STEGANOGRAPHY,
                target_fields=["user_agent", "error_message", "metadata.*"],
                sanitization_function=SecuritySanitizers._sanitize_steganography,
                priority=75,
                preserve_anomaly=True
            )
            engine.sanitizers[SanitizationType.STEGANOGRAPHY] = stego_rule.sanitization_function
        
        if security_settings.get("enable_covert_channel_detection", True):
            covert_rule = SanitizationRule(
                rule_name="covert_channel_detection",
                rule_type=SanitizationType.COVERT_CHANNEL,
                target_fields=["*"],
                sanitization_function=SecuritySanitizers._sanitize_covert_channel,
                priority=70,
                preserve_anomaly=True
            )
            engine.sanitizers[SanitizationType.COVERT_CHANNEL] = covert_rule.sanitization_function
    
    @staticmethod
    def _configure_monitoring(integration: SCAFADLayer1Integration, config: Dict[str, Any]):
        """Configure monitoring and alerting"""
        
        # Set performance targets
        performance_targets = config.get("performance_targets", {})
        integration.scafad_config["performance_targets"].update(performance_targets)
        
        # Configure alert thresholds
        integration.monitor.alert_thresholds.update({
            "high_latency_ms": performance_targets.get("max_latency_ms", 0.15) * 2,
            "low_throughput_rps": performance_targets.get("min_throughput_rps", 10000) * 0.5,
            "high_memory_mb": performance_targets.get("max_memory_mb", 100) * 1.5,
            "anomaly_preservation_threshold": performance_targets.get("anomaly_preservation_rate", 0.9995)
        })
        
        # Configure optimizer thresholds
        integration.optimizer.adaptive_thresholds.update({
            "latency_threshold_ms": performance_targets.get("max_latency_ms", 0.15),
            "memory_threshold_mb": performance_targets.get("max_memory_mb", 100),
            "throughput_threshold_rps": performance_targets.get("min_throughput_rps", 10000)
        })
    
    @staticmethod
    def _configure_compliance(integration: SCAFADLayer1Integration, compliance_settings: Dict[str, Any]):
        """Configure compliance settings"""
        
        # Update SCAFAD config with compliance requirements
        enabled_standards = []
        
        if compliance_settings.get("gdpr_enabled", False):
            enabled_standards.append("GDPR")
        if compliance_settings.get("ccpa_enabled", False):
            enabled_standards.append("CCPA")
        if compliance_settings.get("hipaa_enabled", False):
            enabled_standards.append("HIPAA")
        if compliance_settings.get("pci_dss_enabled", False):
            enabled_standards.append("PCI-DSS")
        
        integration.scafad_config["compliance_standards"] = enabled_standards
        
        # Configure audit logging
        if compliance_settings.get("audit_logging", True):
            integration.scafad_config["audit_logging_enabled"] = True
    
    @staticmethod
    def create_development_instance() -> SCAFADLayer1Integration:
        """Create development instance with relaxed settings"""
        return SCAFADL1Factory.create_production_instance(environment="development")
    
    @staticmethod
    def create_testing_instance() -> SCAFADLayer1Integration:
        """Create testing instance with comprehensive validation"""
        return SCAFADL1Factory.create_production_instance(environment="testing")
    
    @staticmethod
    def create_staging_instance() -> SCAFADLayer1Integration:
        """Create staging instance for pre-production testing"""
        return SCAFADL1Factory.create_production_instance(environment="staging")


# =============================================================================
# Configuration Management and Validation
# =============================================================================

@dataclass_json
@dataclass
class SCAFADL1Config:
    """Comprehensive configuration for SCAFAD Layer 1"""
    
    # Core engine settings
    sanitization_level: SanitizationLevel = SanitizationLevel.STANDARD
    processing_mode: ProcessingMode = ProcessingMode.ASYNCHRONOUS
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    
    # Performance settings
    max_latency_ms: float = 0.15
    min_throughput_rps: int = 10000
    max_memory_mb: int = 100
    anomaly_preservation_rate: float = 0.9995
    
    # ML and detection settings
    enable_ml_detection: bool = True
    ml_model_type: str = "isolation_forest"
    anomaly_threshold: float = 0.7
    pattern_learning_enabled: bool = True
    
    # Security settings
    enable_injection_detection: bool = True
    enable_homograph_detection: bool = True
    enable_steganography_detection: bool = True
    enable_covert_channel_detection: bool = True
    security_level: str = "standard"  # minimal, standard, aggressive, paranoid
    
    # Cache settings
    result_cache_size: int = 10000
    pattern_cache_size: int = 100000
    cache_ttl_seconds: int = 300
    enable_cache_compression: bool = True
    
    # Monitoring and alerting
    enable_performance_monitoring: bool = True
    enable_security_monitoring: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["log", "metrics"])
    heartbeat_interval_seconds: int = 10
    
    # Compliance settings
    gdpr_enabled: bool = False
    ccpa_enabled: bool = False
    hipaa_enabled: bool = False
    pci_dss_enabled: bool = False
    audit_logging: bool = True
    data_retention_days: int = 90
    
    # Integration settings
    layer0_endpoint: Optional[str] = None
    layer2_endpoint: Optional[str] = None
    enable_schema_evolution: bool = True
    schema_version: str = "v2.1"
    
    # Development and debugging
    enable_debug_logging: bool = False
    enable_profiling: bool = False
    enable_stress_testing: bool = False
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration settings"""
        errors = []
        warnings = []
        
        # Performance validation
        if self.max_latency_ms <= 0:
            errors.append("max_latency_ms must be positive")
        if self.max_latency_ms > 10:
            warnings.append("max_latency_ms > 10ms may impact real-time processing")
        
        if self.min_throughput_rps <= 0:
            errors.append("min_throughput_rps must be positive")
        if self.min_throughput_rps > 100000:
            warnings.append("min_throughput_rps > 100k may require significant resources")
        
        if self.max_memory_mb <= 0:
            errors.append("max_memory_mb must be positive")
        if self.max_memory_mb > 1000:
            warnings.append("max_memory_mb > 1GB may impact cost efficiency")
        
        if not 0 <= self.anomaly_preservation_rate <= 1:
            errors.append("anomaly_preservation_rate must be between 0 and 1")
        
        # ML validation
        if self.enable_ml_detection:
            if self.ml_model_type not in ["isolation_forest", "dbscan", "one_class_svm"]:
                errors.append(f"Unsupported ml_model_type: {self.ml_model_type}")
            
            if not 0 <= self.anomaly_threshold <= 1:
                errors.append("anomaly_threshold must be between 0 and 1")
        
        # Security validation
        if self.security_level not in ["minimal", "standard", "aggressive", "paranoid"]:
            errors.append(f"Invalid security_level: {self.security_level}")
        
        # Cache validation
        if self.result_cache_size <= 0:
            errors.append("result_cache_size must be positive")
        if self.pattern_cache_size <= 0:
            errors.append("pattern_cache_size must be positive")
        if self.cache_ttl_seconds <= 0:
            errors.append("cache_ttl_seconds must be positive")
        
        # Monitoring validation
        valid_alert_channels = ["log", "metrics", "webhook", "email", "slack"]
        for channel in self.alert_channels:
            if channel not in valid_alert_channels:
                warnings.append(f"Unknown alert channel: {channel}")
        
        if self.heartbeat_interval_seconds <= 0:
            errors.append("heartbeat_interval_seconds must be positive")
        
        # Compliance validation
        if self.data_retention_days <= 0:
            errors.append("data_retention_days must be positive")
        if self.data_retention_days > 2555:  # ~7 years
            warnings.append("data_retention_days > 7 years may have storage implications")
        
        # Integration validation
        if self.schema_version not in ["v1.0", "v2.0", "v2.1"]:
            errors.append(f"Unsupported schema_version: {self.schema_version}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def to_engine_config(self) -> Dict[str, Any]:
        """Convert to engine configuration format"""
        return {
            "sanitization_level": self.sanitization_level,
            "processing_mode": self.processing_mode,
            "optimization_strategy": self.optimization_strategy,
            "performance_targets": {
                "max_latency_ms": self.max_latency_ms,
                "min_throughput_rps": self.min_throughput_rps,
                "max_memory_mb": self.max_memory_mb,
                "anomaly_preservation_rate": self.anomaly_preservation_rate
            },
            "ml_settings": {
                "enable_ml_detection": self.enable_ml_detection,
                "model_type": self.ml_model_type,
                "anomaly_threshold": self.anomaly_threshold,
                "pattern_learning_enabled": self.pattern_learning_enabled
            },
            "security_settings": {
                "enable_injection_detection": self.enable_injection_detection,
                "enable_homograph_detection": self.enable_homograph_detection,
                "enable_steganography_detection": self.enable_steganography_detection,
                "enable_covert_channel_detection": self.enable_covert_channel_detection,
                "security_level": self.security_level
            },
            "cache_settings": {
                "result_cache_size": self.result_cache_size,
                "pattern_cache_size": self.pattern_cache_size,
                "cache_ttl_seconds": self.cache_ttl_seconds,
                "enable_compression": self.enable_cache_compression
            }
        }
    
    @classmethod
    def from_file(cls, file_path: str) -> 'SCAFADL1Config':
        """Load configuration from file"""
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                data = json.load(f)
            elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                import yaml
                data = yaml.safe_load(f)
            else:
                raise ValueError("Unsupported file format. Use .json or .yaml")
        
        return cls.from_dict(data)
    
    def to_file(self, file_path: str):
        """Save configuration to file"""
        data = self.to_dict()
        
        with open(file_path, 'w') as f:
            if file_path.endswith('.json'):
                json.dump(data, f, indent=2, default=str)
            elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                import yaml
                yaml.dump(data, f, default_flow_style=False)
            else:
                raise ValueError("Unsupported file format. Use .json or .yaml")
    
    @classmethod
    def create_default(cls, environment: str = "production") -> 'SCAFADL1Config':
        """Create default configuration for environment"""
        if environment == "development":
            return cls(
                sanitization_level=SanitizationLevel.MINIMAL,
                processing_mode=ProcessingMode.SYNCHRONOUS,
                max_latency_ms=1.0,
                min_throughput_rps=1000,
                max_memory_mb=200,
                enable_debug_logging=True,
                enable_profiling=True,
                security_level="minimal"
            )
        elif environment == "testing":
            return cls(
                sanitization_level=SanitizationLevel.AGGRESSIVE,
                enable_stress_testing=True,
                enable_profiling=True,
                security_level="aggressive"
            )
        elif environment == "staging":
            return cls(
                sanitization_level=SanitizationLevel.STANDARD,
                processing_mode=ProcessingMode.PARALLEL,
                enable_profiling=True,
                security_level="standard"
            )
        else:  # production
            return cls(
                sanitization_level=SanitizationLevel.STANDARD,
                processing_mode=ProcessingMode.ASYNCHRONOUS,
                optimization_strategy=OptimizationStrategy.THROUGHPUT_OPTIMIZED,
                enable_performance_monitoring=True,
                enable_security_monitoring=True,
                security_level="standard",
                audit_logging=True
            )


# =============================================================================
# Main Application Entry Point and CLI
# =============================================================================

class SCAFADL1Application:
    """Main application class for SCAFAD Layer 1"""
    
    def __init__(self, config: SCAFADL1Config = None):
        self.config = config or SCAFADL1Config.create_default()
        self.integration = None
        self.schema_engine = SchemaEvolutionEngine()
        self.running = False
        
        # Application state
        self.start_time = None
        self.total_processed = 0
        self.health_status = "initializing"
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.DEBUG if self.config.enable_debug_logging else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('scafad_layer1.log')
            ]
        )
        
        # Setup performance logging
        if self.config.enable_profiling:
            logging.getLogger("SCAFAD.Performance").setLevel(logging.DEBUG)
        
        # Setup security logging
        if self.config.enable_security_monitoring:
            security_handler = logging.FileHandler('scafad_security.log')
            security_handler.setFormatter(
                logging.Formatter('%(asctime)s - SECURITY - %(levelname)s - %(message)s')
            )
            logging.getLogger("SCAFAD.Security").addHandler(security_handler)
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the application"""
        try:
            self.health_status = "initializing"
            
            # Validate configuration
            validation_result = self.config.validate()
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": "Configuration validation failed",
                    "validation_errors": validation_result["errors"]
                }
            
            # Create integration instance
            self.integration = SCAFADL1Factory.create_production_instance(
                config=self.config.to_engine_config(),
                environment="production"
            )
            
            # Initialize the layer
            init_result = await self.integration.initialize_layer(
                layer0_endpoint=self.config.layer0_endpoint,
                layer2_endpoint=self.config.layer2_endpoint
            )
            
            if init_result["overall_status"] != "success":
                return {
                    "success": False,
                    "error": "Layer initialization failed",
                    "details": init_result
                }
            
            self.health_status = "healthy"
            self.start_time = time.time()
            self.running = True
            
            logging.getLogger("SCAFAD.Layer1.Application").info(
                "SCAFAD Layer 1 initialized successfully"
            )
            
            return {
                "success": True,
                "initialization_result": init_result,
                "configuration": self.config.to_dict(),
                "schema_version": self.schema_engine.current_schema_version
            }
            
        except Exception as e:
            self.health_status = "failed"
            logging.getLogger("SCAFAD.Layer1.Application").error(
                f"Initialization failed: {e}", exc_info=True
            )
            return {
                "success": False,
                "error": str(e)
            }
    
    async def process_telemetry(self, telemetry_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process telemetry data through the full pipeline"""
        if not self.running or not self.integration:
            return {
                "success": False,
                "error": "Application not initialized or not running"
            }
        
        try:
            start_time = time.time()
            
            # Schema validation and migration
            processed_records = []
            schema_errors = []
            
            for record in telemetry_data:
                # Detect schema version
                detected_version = self._detect_schema_version(record)
                
                # Migrate if necessary
                if detected_version != self.schema_engine.current_schema_version:
                    migration_result = await self.schema_engine.migrate_data(
                        record, detected_version, self.schema_engine.current_schema_version
                    )
                    
                    if migration_result["success"]:
                        processed_records.append(migration_result["migrated_data"])
                    else:
                        schema_errors.append({
                            "record": record,
                            "error": migration_result["error"]
                        })
                        continue
                else:
                    # Validate existing schema
                    validation_result = await self.schema_engine.validate_schema(
                        record, detected_version
                    )
                    
                    if validation_result["valid"]:
                        processed_records.append(record)
                    else:
                        schema_errors.append({
                            "record": record,
                            "error": validation_result["error"]
                        })
                        continue
            
            # Process through sanitization pipeline
            if processed_records:
                processing_result = await self.integration.process_telemetry_from_layer0(processed_records)
            else:
                processing_result = {"success": True, "accepted_records": 0}
            
            # Update statistics
            self.total_processed += len(processed_records)
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "success": processing_result["success"],
                "total_input_records": len(telemetry_data),
                "processed_records": len(processed_records),
                "schema_errors": len(schema_errors),
                "processing_time_ms": processing_time,
                "schema_error_details": schema_errors[:5],  # First 5 errors
                "processing_result": processing_result
            }
            
        except Exception as e:
            logging.getLogger("SCAFAD.Layer1.Application").error(
                f"Telemetry processing failed: {e}", exc_info=True
            )
            return {
                "success": False,
                "error": str(e)
            }
    
    def _detect_schema_version(self, record: Dict[str, Any]) -> str:
        """Detect schema version of incoming record"""
        # Simple heuristic-based detection
        if "security_context" in record and "anomaly_indicators" in record:
            return "v2.1"
        elif "execution_context" in record and "invocation_graph" in record:
            return "v2.0"
        else:
            return "v1.0"
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive application status"""
        uptime = time.time() - self.start_time if self.start_time else 0
        
        status = {
            "application": {
                "running": self.running,
                "health_status": self.health_status,
                "uptime_seconds": uptime,
                "total_processed": self.total_processed,
                "start_time": self.start_time
            },
            "configuration": {
                "sanitization_level": self.config.sanitization_level.name,
                "processing_mode": self.config.processing_mode.name,
                "schema_version": self.config.schema_version,
                "security_level": self.config.security_level
            }
        }
        
        if self.integration:
            status["layer_status"] = self.integration.get_layer_status()
        
        return status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance and operational metrics"""
        if not self.integration:
            return {"error": "Integration not initialized"}
        
        engine_metrics = self.integration.engine.get_performance_metrics()
        security_dashboard = self.integration.monitor.get_security_dashboard()
        
        return {
            "performance_metrics": asdict(engine_metrics),
            "security_metrics": security_dashboard,
            "schema_metrics": self.schema_engine.migration_stats,
            "application_metrics": {
                "total_processed": self.total_processed,
                "uptime_seconds": time.time() - self.start_time if self.start_time else 0
            }
        }
    
    async def shutdown(self) -> Dict[str, Any]:
        """Gracefully shutdown the application"""
        try:
            self.running = False
            self.health_status = "shutting_down"
            
            shutdown_results = {}
            
            if self.integration:
                layer_shutdown = await self.integration.shutdown_layer()
                shutdown_results["layer_shutdown"] = layer_shutdown
            
            # Final metrics export
            final_metrics = self.get_metrics()
            shutdown_results["final_metrics"] = final_metrics
            
            # Calculate uptime
            uptime = time.time() - self.start_time if self.start_time else 0
            shutdown_results["session_summary"] = {
                "uptime_seconds": uptime,
                "total_processed": self.total_processed,
                "average_throughput": self.total_processed / uptime if uptime > 0 else 0
            }
            
            self.health_status = "shutdown"
            
            logging.getLogger("SCAFAD.Layer1.Application").info(
                f"SCAFAD Layer 1 shutdown completed. Processed {self.total_processed} records in {uptime:.1f}s"
            )
            
            return {
                "success": True,
                "shutdown_results": shutdown_results
            }
            
        except Exception as e:
            logging.getLogger("SCAFAD.Layer1.Application").error(
                f"Shutdown error: {e}", exc_info=True
            )
            return {
                "success": False,
                "error": str(e)
            }


# =============================================================================
# Command Line Interface and Entry Points
# =============================================================================

import argparse
import sys

def create_cli_parser() -> argparse.ArgumentParser:
    """Create command line interface parser"""
    parser = argparse.ArgumentParser(
        description="SCAFAD Layer 1: Enhanced Behavioral Intake Zone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python layer1_sanitization.py --config production.json --environment production
  python layer1_sanitization.py --demo --sanitization-level aggressive
  python layer1_sanitization.py --benchmark --records 10000
        """
    )
    
    # Configuration options
    parser.add_argument('--config', type=str, help='Configuration file path (JSON/YAML)')
    parser.add_argument('--environment', choices=['development', 'testing', 'staging', 'production'], 
                       default='production', help='Deployment environment')
    
    # Runtime options
    parser.add_argument('--sanitization-level', choices=['minimal', 'standard', 'aggressive', 'paranoid'], 
                       help='Override sanitization level')
    parser.add_argument('--processing-mode', choices=['synchronous', 'asynchronous', 'parallel'], 
                       help='Override processing mode')
    
    # Demo and testing
    parser.add_argument('--demo', action='store_true', help='Run demonstration with sample data')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--records', type=int, default=1000, help='Number of records for demo/benchmark')
    
    # Server mode
    parser.add_argument('--server', action='store_true', help='Run in server mode')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    
    # Utility commands
    parser.add_argument('--validate-config', action='store_true', help='Validate configuration and exit')
    parser.add_argument('--export-schema', type=str, help='Export schema to file')
    parser.add_argument('--version', action='store_true', help='Show version information')
    
    return parser

async def run_demo(config: SCAFADL1Config, num_records: int = 1000):
    """Run demonstration with sample data"""
    print("🚀 SCAFAD Layer 1 Enhanced Sanitization Demo")
    print("=" * 50)
    
    # Create application
    app = SCAFADL1Application(config)
    
    # Initialize
    print("Initializing SCAFAD Layer 1...")
    init_result = await app.initialize()
    
    if not init_result["success"]:
        print(f"❌ Initialization failed: {init_result['error']}")
        return
    
    print("✅ Layer 1 initialized successfully")
    
    # Generate sample telemetry data
    print(f"Generating {num_records} sample telemetry records...")
    sample_data = generate_demo_data(num_records)
    
    # Process data
    print("Processing telemetry through sanitization pipeline...")
    start_time = time.time()
    
    # Process in batches
    batch_size = 100
    total_processed = 0
    total_errors = 0
    
    for i in range(0, len(sample_data), batch_size):
        batch = sample_data[i:i + batch_size]
        result = await app.process_telemetry(batch)
        
        if result["success"]:
            total_processed += result["processed_records"]
            total_errors += result["schema_errors"]
        
        # Progress indicator
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i + len(batch)}/{len(sample_data)} records...")
    
    processing_time = time.time() - start_time
    
    # Display results
    print("\n📊 Processing Results:")
    print(f"  Total Records: {len(sample_data)}")
    print(f"  Successfully Processed: {total_processed}")
    print(f"  Schema Errors: {total_errors}")
    print(f"  Processing Time: {processing_time:.2f}s")
    print(f"  Throughput: {len(sample_data) / processing_time:.1f} records/sec")
    
    # Show metrics
    metrics = app.get_metrics()
    perf_metrics = metrics["performance_metrics"]
    
    print(f"\n⚡ Performance Metrics:")
    print(f"  Average Latency: {perf_metrics['average_latency_ms']:.2f}ms")
    print(f"  Peak Latency: {perf_metrics['peak_latency_ms']:.2f}ms")
    print(f"  Memory Usage: {perf_metrics['memory_usage_mb']:.1f}MB")
    print(f"  Cache Hit Rate: {perf_metrics['cache_hit_rate']:.1%}")
    print(f"  Error Rate: {perf_metrics['error_rate']:.1%}")
    
    # Security insights
    security_metrics = metrics["security_metrics"]
    if security_metrics["security_events"]["last_hour"] > 0:
        print(f"\n🔒 Security Events Detected: {security_metrics['security_events']['last_hour']}")
        
        top_threats = security_metrics["top_threats"]
        if top_threats:
            print("  Top Threats:")
            for threat in top_threats[:3]:
                print(f"    - {threat['threat_type']}: {threat['recent_incidents']} incidents")
    
    # Shutdown
    print("\nShutting down...")
    await app.shutdown()
    print("✅ Demo completed successfully!")

def generate_demo_data(num_records: int) -> List[Dict[str, Any]]:
    """Generate sample telemetry data for demonstration"""
    import random
    
    data = []
    current_time = time.time()
    
    # Normal patterns (90%)
    normal_functions = ["api-gateway", "data-processor", "auth-handler", "notification-service"]
    
    for i in range(int(num_records * 0.9)):
        record = {
            "timestamp": current_time - random.randint(0, 3600),
            "function_name": random.choice(normal_functions),
            "invocation_id": f"inv-{i:06d}",
            "execution_context": {
                "duration_ms": random.randint(50, 2000),
                "memory_used_mb": random.randint(32, 512),
                "cold_start": random.random() < 0.1
            },
            "invocation_graph": {
                "depth": random.randint(0, 3),
                "fan_out": random.randint(0, 5)
            }
        }
        data.append(record)
    
    # Anomalous patterns (10%)
    attack_patterns = [
        {"function_name": "'; DROP TABLE users;--", "duration_ms": 10000},
        {"function_name": "normal-func", "user_agent": "<script>alert('xss')</script>"},
        {"function_name": "resource-hog", "duration_ms": 890000, "memory_used_mb": 3008},
        {"function_name": "data-exfil", "fan_out": 100}
    ]
    
    for i in range(int(num_records * 0.1)):
        pattern = random.choice(attack_patterns)
        record = {
            "timestamp": current_time - random.randint(0, 3600),
            "function_name": pattern.get("function_name", "anomalous-func"),
            "invocation_id": f"anom-{i:06d}",
            "execution_context": {
                "duration_ms": pattern.get("duration_ms", random.randint(50, 2000)),
                "memory_used_mb": pattern.get("memory_used_mb", random.randint(32, 512)),
                "cold_start": random.random() < 0.3
            },
            "invocation_graph": {
                "depth": random.randint(0, 10),
                "fan_out": pattern.get("fan_out", random.randint(0, 5))
            }
        }
        
        if "user_agent" in pattern:
            record["security_context"] = {"user_agent": pattern["user_agent"]}
        
        data.append(record)
    
    random.shuffle(data)
    return data

async def run_benchmark(config: SCAFADL1Config, num_records: int = 10000):
    """Run performance benchmark"""
    print("🏃‍♂️ SCAFAD Layer 1 Performance Benchmark")
    print("=" * 50)
    
    # Test different configurations
    test_configs = [
        ("Minimal Sanitization", config.__class__(sanitization_level=SanitizationLevel.MINIMAL)),
        ("Standard Sanitization", config.__class__(sanitization_level=SanitizationLevel.STANDARD)),
        ("Aggressive Sanitization", config.__class__(sanitization_level=SanitizationLevel.AGGRESSIVE))
    ]
    
    results = []
    
    for test_name, test_config in test_configs:
        print(f"\n🧪 Testing: {test_name}")
        
        app = SCAFADL1Application(test_config)
        await app.initialize()
        
        # Generate test data
        test_data = generate_demo_data(num_records)
        
        # Warmup
        warmup_data = test_data[:100]
        await app.process_telemetry(warmup_data)
        
        # Benchmark
        start_time = time.time()
        result = await app.process_telemetry(test_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        throughput = num_records / processing_time
        
        metrics = app.get_metrics()
        perf_metrics = metrics["performance_metrics"]
        
        results.append({
            "test_name": test_name,
            "throughput_rps": throughput,
            "avg_latency_ms": perf_metrics["average_latency_ms"],
            "memory_mb": perf_metrics["memory_usage_mb"],
            "success_rate": 1.0 - perf_metrics["error_rate"]
        })
        
        print(f"  Throughput: {throughput:.1f} records/sec")
        print(f"  Avg Latency: {perf_metrics['average_latency_ms']:.2f}ms")
        print(f"  Memory: {perf_metrics['memory_usage_mb']:.1f}MB")
        
        await app.shutdown()
    
    # Summary
    print(f"\n📈 Benchmark Summary:")
    print("Test Configuration           | Throughput (rps) | Latency (ms) | Memory (MB)")
    print("-" * 75)
    for result in results:
        print(f"{result['test_name']:<28} | {result['throughput_rps']:>12.1f} | {result['avg_latency_ms']:>10.2f} | {result['memory_mb']:>9.1f}")

async def main():
    """Main entry point"""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Version information
    if args.version:
        print("SCAFAD Layer 1 Enhanced Sanitization Processor")
        print("Version: 2.0.0")
        print("Institution: Birmingham Newman University")
        print("License: MIT")
        return
    
    # Load configuration
    if args.config:
        config = SCAFADL1Config.from_file(args.config)
    else:
        config = SCAFADL1Config.create_default(args.environment)
    
    # Override configuration from CLI args
    if args.sanitization_level:
        level_map = {
            "minimal": SanitizationLevel.MINIMAL,
            "standard": SanitizationLevel.STANDARD,
            "aggressive": SanitizationLevel.AGGRESSIVE,
            "paranoid": SanitizationLevel.PARANOID
        }
        config.sanitization_level = level_map[args.sanitization_level]
    
    if args.processing_mode:
        mode_map = {
            "synchronous": ProcessingMode.SYNCHRONOUS,
            "asynchronous": ProcessingMode.ASYNCHRONOUS,
            "parallel": ProcessingMode.PARALLEL
        }
        config.processing_mode = mode_map[args.processing_mode]
    
    # Validate configuration
    if args.validate_config:
        validation = config.validate()
        if validation["valid"]:
            print("✅ Configuration is valid")
        else:
            print("❌ Configuration validation failed:")
            for error in validation["errors"]:
                print(f"  - {error}")
        return
    
    # Export schema
    if args.export_schema:
        schema_engine = SchemaEvolutionEngine()
        schema_info = schema_engine.get_schema_info()
        
        with open(args.export_schema, 'w') as f:
            json.dump(schema_info, f, indent=2)
        print(f"✅ Schema exported to {args.export_schema}")
        return
    
    # Run modes
    try:
        if args.demo:
            await run_demo(config, args.records)
        elif args.benchmark:
            await run_benchmark(config, args.records)
        elif args.server:
            print("🌐 Server mode not implemented in this demo")
            print("Use the integration API for production deployment")
        else:
            print("No operation specified. Use --help for available options.")
            
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())


# =============================================================================
# Module Exports and Documentation
# =============================================================================

__version__ = "2.0.0"
__author__ = "SCAFAD Research Team"
__institution__ = "Birmingham Newman University"
__license__ = "MIT"

__all__ = [
    # Core classes
    "EnhancedSanitizationEngine",
    "SCAFADLayer1Integration", 
    "SCAFADL1Application",
    "SchemaEvolutionEngine",
    
    # Configuration
    "SCAFADL1Config",
    "SCAFADL1Factory",
    
    # Data models
    "SanitizationResult",
    "SanitizationRule", 
    "SanitizationMetrics",
    "AnomalyContext",
    
    # Enums
    "SanitizationLevel",
    "ProcessingMode",
    "OptimizationStrategy",
    "SanitizationType",
    "DataIntegrityLevel",
    "AnomalyPreservationStatus",
    
    # Monitoring and security
    "SanitizationMonitor",
    "PerformanceOptimizer",
    "SecuritySanitizers",
    
    # Utilities
    "sanitize_telemetry_batch",
    "create_sanitization_engine",
    
    # CLI
    "main",
    "run_demo",
    "run_benchmark"
]

# Module documentation
__doc__ = """
SCAFAD Layer 1: Enhanced Behavioral Intake Zone - Advanced Sanitization Processor
================================================================================

This module implements the enhanced sanitization processor for SCAFAD Layer 1,
providing comprehensive data conditioning capabilities for serverless telemetry
with advanced anomaly preservation, ML-powered detection, and enterprise-grade
performance optimization.

Key Features:
- ML-powered anomaly detection and preservation (99.95%+ retention)
- Sub-millisecond sanitization latency (<0.15ms target)
- Advanced security sanitizers (injection, homograph, steganography detection)
- Comprehensive schema evolution and versioning
- Production-ready monitoring and alerting
- GDPR/CCPA/HIPAA compliance support
- Adaptive performance optimization
- Real-time threat detection and mitigation

Architecture:
Layer 1 sits between Layer 0 (Adaptive Telemetry Controller) and Layer 2 
(Multi-Vector Detection Matrix), ensuring clean, validated, and privacy-compliant
data flows to downstream detection systems while preserving critical anomaly
signatures for security analysis.

Performance Targets:
- Sanitization latency: <0.15ms per record
- Throughput: >10,000 records/second  
- Anomaly preservation: 99.95%+
- Memory usage: <100MB
- Zero false positives from sanitization artifacts

Usage Examples:
```python
# Basic usage
from layer1_sanitization import create_sanitization_engine, sanitize_telemetry_batch

engine = create_sanitization_engine(level=SanitizationLevel.STANDARD)
results = await sanitize_telemetry_batch(telemetry_records)

# Production deployment
from layer1_sanitization import SCAFADL1Application, SCAFADL1Config

config = SCAFADL1Config.create_default("production")
app = SCAFADL1Application(config)
await app.initialize()

# Process telemetry
result = await app.process_telemetry(telemetry_batch)
```

For detailed documentation and examples, see:
https://github.com/scafad-framework/layer1-sanitization
"""

print(f"""
🎯 SCAFAD Layer 1 Enhanced Sanitization Processor v{__version__}
================================================================

✨ Features:
  • ML-powered anomaly preservation (99.95%+ retention)
  • Sub-millisecond processing latency (<0.15ms)
  • Advanced security sanitizers
  • Enterprise-grade monitoring
  • Complete SCAFAD framework integration

🚀 Quick Start:
  python layer1_sanitization.py --demo --records 1000
  python layer1_sanitization.py --benchmark --records 10000

📚 Documentation: https://github.com/scafad-framework/layer1-sanitization
🎓 Institution: {__institution__}
📄 License: {__license__}
""")
        # Process output queue
        while not self.output_queue.empty():
            try:
                result = self.output_queue.get_nowait()
                formatted_batch = self._format_for_layer2([result])
                await self._send_to_layer2(formatted_batch)
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                logging.getLogger("SCAFAD.Layer1.Integration").error(
                    f"Error processing remaining output during shutdown: {e}"
                )


# =============================================================================
# Advanced Schema Evolution and Versioning System
# =============================================================================

class SchemaEvolutionEngine:
    """Advanced schema evolution and versioning for telemetry data"""
    
    def __init__(self):
        self.schema_registry = {}
        self.migration_rules = {}
        self.version_compatibility_matrix = {}
        self.current_schema_version = "v2.1"
        
        # Initialize default schemas
        self._initialize_default_schemas()
        
        # Schema validation cache
        self.validation_cache = TTLCache(maxsize=1000, ttl=600)  # 10 minute TTL
        
        # Migration statistics
        self.migration_stats = {
            'total_migrations': 0,
            'successful_migrations': 0,
            'failed_migrations': 0,
            'migration_times_ms': [],
            'version_distribution': defaultdict(int)
        }
    
    def _initialize_default_schemas(self):
        """Initialize default SCAFAD telemetry schemas"""
        
        # Base telemetry schema v1.0
        self.schema_registry["v1.0"] = {
            "type": "object",
            "properties": {
                "timestamp": {"type": "number"},
                "function_name": {"type": "string"},
                "invocation_id": {"type": "string"},
                "duration_ms": {"type": "number"},
                "memory_used_mb": {"type": "number"},
                "cold_start": {"type": "boolean"},
                "error": {"type": ["string", "null"]},
                "metadata": {"type": "object"}
            },
            "required": ["timestamp", "function_name", "invocation_id"],
            "additionalProperties": True
        }
        
        # Enhanced telemetry schema v2.0
        self.schema_registry["v2.0"] = {
            "type": "object",
            "properties": {
                "timestamp": {"type": "number", "minimum": 0},
                "function_name": {"type": "string", "pattern": "^[a-zA-Z0-9_-]+$"},
                "invocation_id": {"type": "string", "pattern": "^[0-9a-f-]+$"},
                "execution_context": {
                    "type": "object",
                    "properties": {
                        "duration_ms": {"type": "number", "minimum": 0},
                        "memory_used_mb": {"type": "number", "minimum": 0},
                        "cpu_utilization": {"type": "number", "minimum": 0, "maximum": 100},
                        "cold_start": {"type": "boolean"},
                        "timeout_ms": {"type": "number", "minimum": 0}
                    },
                    "required": ["duration_ms", "memory_used_mb"]
                },
                "invocation_graph": {
                    "type": "object",
                    "properties": {
                        "parent_invocation": {"type": ["string", "null"]},
                        "child_invocations": {"type": "array", "items": {"type": "string"}},
                        "depth": {"type": "integer", "minimum": 0}
                    }
                },
                "error_details": {
                    "type": ["object", "null"],
                    "properties": {
                        "error_type": {"type": "string"},
                        "error_message": {"type": "string"},
                        "stack_trace": {"type": "string"},
                        "error_code": {"type": ["string", "number"]}
                    }
                },
                "provenance": {
                    "type": "object",
                    "properties": {
                        "layer0_processing_time": {"type": "number"},
                        "telemetry_source": {"type": "string"},
                        "sampling_rate": {"type": "number", "minimum": 0, "maximum": 1}
                    }
                },
                "metadata": {"type": "object"}
            },
            "required": ["timestamp", "function_name", "invocation_id", "execution_context"],
            "additionalProperties": False
        }
        
        # Current schema v2.1 with security enhancements
        self.schema_registry["v2.1"] = {
            "type": "object",
            "properties": {
                "timestamp": {"type": "number", "minimum": 0},
                "function_name": {"type": "string", "pattern": "^[a-zA-Z0-9_-]+$", "maxLength": 128},
                "invocation_id": {"type": "string", "pattern": "^[0-9a-f-]+$", "maxLength": 64},
                "execution_context": {
                    "type": "object",
                    "properties": {
                        "duration_ms": {"type": "number", "minimum": 0, "maximum": 900000},
                        "memory_used_mb": {"type": "number", "minimum": 0, "maximum": 10240},
                        "cpu_utilization": {"type": "number", "minimum": 0, "maximum": 100},
                        "cold_start": {"type": "boolean"},
                        "timeout_ms": {"type": "number", "minimum": 0, "maximum": 900000},
                        "billing_duration_ms": {"type": "number", "minimum": 0}
                    },
                    "required": ["duration_ms", "memory_used_mb"]
                },
                "invocation_graph": {
                    "type": "object",
                    "properties": {
                        "parent_invocation": {"type": ["string", "null"], "maxLength": 64},
                        "child_invocations": {
                            "type": "array", 
                            "items": {"type": "string", "maxLength": 64},
                            "maxItems": 100
                        },
                        "depth": {"type": "integer", "minimum": 0, "maximum": 50},
                        "fan_out": {"type": "integer", "minimum": 0, "maximum": 100},
                        "concurrency_level": {"type": "integer", "minimum": 0}
                    }
                },
                "security_context": {
                    "type": "object",
                    "properties": {
                        "user_identity": {"type": "string", "maxLength": 256},
                        "role_arn": {"type": "string", "maxLength": 512},
                        "source_ip": {"type": "string", "format": "ipv4"},
                        "user_agent": {"type": "string", "maxLength": 1024},
                        "api_key_hash": {"type": "string", "maxLength": 64}
                    }
                },
                "error_details": {
                    "type": ["object", "null"],
                    "properties": {
                        "error_type": {"type": "string", "maxLength": 128},
                        "error_message": {"type": "string", "maxLength": 2048},
                        "stack_trace": {"type": "string", "maxLength": 8192},
                        "error_code": {"type": ["string", "number"]},
                        "recovery_attempted": {"type": "boolean"},
                        "retry_count": {"type": "integer", "minimum": 0, "maximum": 10}
                    }
                },
                "performance_metrics": {
                    "type": "object",
                    "properties": {
                        "init_duration_ms": {"type": "number", "minimum": 0},
                        "io_operations": {"type": "integer", "minimum": 0},
                        "network_calls": {"type": "integer", "minimum": 0},
                        "database_queries": {"type": "integer", "minimum": 0},
                        "cache_hits": {"type": "integer", "minimum": 0},
                        "cache_misses": {"type": "integer", "minimum": 0}
                    }
                },
                "provenance": {
                    "type": "object",
                    "properties": {
                        "layer0_processing_time": {"type": "number", "minimum": 0},
                        "telemetry_source": {"type": "string", "maxLength": 128},
                        "sampling_rate": {"type": "number", "minimum": 0, "maximum": 1},
                        "data_quality_score": {"type": "number", "minimum": 0, "maximum": 1},
                        "completeness_score": {"type": "number", "minimum": 0, "maximum": 1}
                    }
                },
                "anomaly_indicators": {
                    "type": "object",
                    "properties": {
                        "statistical_outlier": {"type": "boolean"},
                        "pattern_deviation": {"type": "boolean"},
                        "temporal_anomaly": {"type": "boolean"},
                        "resource_anomaly": {"type": "boolean"},
                        "security_anomaly": {"type": "boolean"},
                        "confidence_scores": {
                            "type": "object",
                            "patternProperties": {
                                "^[a-zA-Z_]+$": {"type": "number", "minimum": 0, "maximum": 1}
                            }
                        }
                    }
                },
                "metadata": {
                    "type": "object",
                    "maxProperties": 50,
                    "patternProperties": {
                        "^[a-zA-Z0-9_-]+$": {}
                    }
                }
            },
            "required": ["timestamp", "function_name", "invocation_id", "execution_context"],
            "additionalProperties": False
        }
        
        # Define migration rules between versions
        self._initialize_migration_rules()
        
        # Define version compatibility
        self._initialize_version_compatibility()
    
    def _initialize_migration_rules(self):
        """Initialize schema migration rules"""
        
        # Migration from v1.0 to v2.0
        self.migration_rules["v1.0->v2.0"] = {
            "transformations": [
                {
                    "type": "restructure",
                    "source_fields": ["duration_ms", "memory_used_mb", "cold_start"],
                    "target_field": "execution_context",
                    "transformation": lambda data: {
                        "duration_ms": data.get("duration_ms", 0),
                        "memory_used_mb": data.get("memory_used_mb", 0),
                        "cold_start": data.get("cold_start", False)
                    }
                },
                {
                    "type": "rename",
                    "source_field": "error",
                    "target_field": "error_details.error_message"
                },
                {
                    "type": "add_default",
                    "field": "invocation_graph",
                    "default_value": {
                        "parent_invocation": None,
                        "child_invocations": [],
                        "depth": 0
                    }
                },
                {
                    "type": "add_default",
                    "field": "provenance",
                    "default_value": {
                        "telemetry_source": "legacy",
                        "sampling_rate": 1.0
                    }
                }
            ],
            "field_mappings": {
                "timestamp": "timestamp",
                "function_name": "function_name",
                "invocation_id": "invocation_id",
                "metadata": "metadata"
            },
            "validation_rules": [
                {"field": "timestamp", "rule": "required"},
                {"field": "execution_context.duration_ms", "rule": "positive_number"}
            ]
        }
        
        # Migration from v2.0 to v2.1
        self.migration_rules["v2.0->v2.1"] = {
            "transformations": [
                {
                    "type": "add_default",
                    "field": "security_context",
                    "default_value": {}
                },
                {
                    "type": "add_default",
                    "field": "performance_metrics",
                    "default_value": {}
                },
                {
                    "type": "add_default",
                    "field": "anomaly_indicators",
                    "default_value": {
                        "statistical_outlier": False,
                        "pattern_deviation": False,
                        "temporal_anomaly": False,
                        "resource_anomaly": False,
                        "security_anomaly": False,
                        "confidence_scores": {}
                    }
                },
                {
                    "type": "enhance_provenance",
                    "enhancements": {
                        "data_quality_score": 1.0,
                        "completeness_score": 1.0
                    }
                }
            ],
            "field_mappings": {
                # All existing fields pass through
                "*": "*"
            },
            "validation_rules": [
                {"field": "function_name", "rule": "pattern_match", "pattern": "^[a-zA-Z0-9_-]+$"},
                {"field": "execution_context.memory_used_mb", "rule": "range", "min": 0, "max": 10240}
            ]
        }
    
    def _initialize_version_compatibility(self):
        """Initialize version compatibility matrix"""
        self.version_compatibility_matrix = {
            "v1.0": {
                "forward_compatible": ["v2.0", "v2.1"],
                "backward_compatible": [],
                "breaking_changes": ["v2.0", "v2.1"]
            },
            "v2.0": {
                "forward_compatible": ["v2.1"],
                "backward_compatible": ["v1.0"],
                "breaking_changes": []
            },
            "v2.1": {
                "forward_compatible": [],
                "backward_compatible": ["v1.0", "v2.0"],
                "breaking_changes": []
            }
        }
    
    async def validate_schema(self, data: Dict[str, Any], schema_version: str = None) -> Dict[str, Any]:
        """Validate data against schema"""
        if schema_version is None:
            schema_version = self.current_schema_version
        
        # Check cache first
        data_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
        cache_key = f"{schema_version}:{data_hash}"
        
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        start_time = time.time()
        
        try:
            # Get schema
            if schema_version not in self.schema_registry:
                return {
                    "valid": False,
                    "error": f"Unknown schema version: {schema_version}",
                    "schema_version": schema_version
                }
            
            schema = self.schema_registry[schema_version]
            
            # Validate using jsonschema
            from jsonschema import validate, ValidationError
            
            validate(instance=data, schema=schema)
            
            # Additional custom validations
            custom_validation_result = await self._perform_custom_validations(data, schema_version)
            
            result = {
                "valid": True,
                "schema_version": schema_version,
                "validation_time_ms": (time.time() - start_time) * 1000,
                "custom_validations": custom_validation_result
            }
            
            # Cache result
            self.validation_cache[cache_key] = result
            
            return result
            
        except ValidationError as e:
            result = {
                "valid": False,
                "error": str(e),
                "schema_version": schema_version,
                "validation_time_ms": (time.time() - start_time) * 1000,
                "error_path": list(e.absolute_path) if hasattr(e, 'absolute_path') else [],
                "failing_value": e.instance if hasattr(e, 'instance') else None
            }
            
            # Cache negative result with shorter TTL
            negative_cache_key = f"negative:{cache_key}"
            self.validation_cache[negative_cache_key] = result
            
            return result
        
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation error: {str(e)}",
                "schema_version": schema_version,
                "validation_time_ms": (time.time() - start_time) * 1000
            }
    
    async def _perform_custom_validations(self, data: Dict[str, Any], schema_version: str) -> Dict[str, Any]:
        """Perform custom validations beyond JSON schema"""
        validations = {}
        
        # Timestamp validation
        if "timestamp" in data:
            timestamp = data["timestamp"]
            current_time = time.time()
            
            # Check for reasonable timestamp range
            if timestamp < current_time - (365 * 24 * 3600):  # More than 1 year ago
                validations["timestamp_too_old"] = True
            elif timestamp > current_time + 3600:  # More than 1 hour in future
                validations["timestamp_too_future"] = True
            else:
                validations["timestamp_reasonable"] = True
        
        # Function name validation
        if "function_name" in data:
            function_name = data["function_name"]
            
            # Check for suspicious patterns
            suspicious_patterns = ["script", "eval", "exec", "system", "cmd"]
            if any(pattern in function_name.lower() for pattern in suspicious_patterns):
                validations["suspicious_function_name"] = True
            
            # Check length
            if len(function_name) > 128:
                validations["function_name_too_long"] = True
        
        # Security context validation for v2.1+
        if schema_version in ["v2.1"] and "security_context" in data:
            security_context = data["security_context"]
            
            # IP address validation
            if "source_ip" in security_context:
                try:
                    ipaddress.ip_address(security_context["source_ip"])
                    validations["valid_ip_address"] = True
                except ValueError:
                    validations["invalid_ip_address"] = True
            
            # User agent analysis
            if "user_agent" in security_context:
                user_agent = security_context["user_agent"]
                if len(user_agent) > 1024:
                    validations["user_agent_too_long"] = True
                
                # Check for suspicious user agents
                suspicious_ua_patterns = ["bot", "crawler", "script", "curl", "wget"]
                if any(pattern in user_agent.lower() for pattern in suspicious_ua_patterns):
                    validations["suspicious_user_agent"] = True
        
        # Performance metrics validation
        if "execution_context" in data:
            exec_context = data["execution_context"]
            
            # Duration reasonableness
            if "duration_ms" in exec_context:
                duration = exec_context["duration_ms"]
                if duration > 900000:  # 15 minutes
                    validations["excessive_duration"] = True
                elif duration < 0:
                    validations["negative_duration"] = True
            
            # Memory usage reasonableness
            if "memory_used_mb" in exec_context:
                memory = exec_context["memory_used_mb"]
                if memory > 10240:  # 10GB
                    validations["excessive_memory"] = True
                elif memory < 0:
                    validations["negative_memory"] = True
        
        return validations
    
    async def migrate_data(self, data: Dict[str, Any], source_version: str, target_version: str) -> Dict[str, Any]:
        """Migrate data from source schema version to target version"""
        start_time = time.time()
        
        try:
            # Check if migration is needed
            if source_version == target_version:
                return {
                    "success": True,
                    "migrated_data": data,
                    "source_version": source_version,
                    "target_version": target_version,
                    "migration_time_ms": 0,
                    "transformations_applied": []
                }
            
            # Find migration path
            migration_path = self._find_migration_path(source_version, target_version)
            if not migration_path:
                return {
                    "success": False,
                    "error": f"No migration path from {source_version} to {target_version}",
                    "source_version": source_version,
                    "target_version": target_version
                }
            
            # Apply migrations step by step
            current_data = copy.deepcopy(data)
            transformations_applied = []
            
            for step in migration_path:
                migration_key = f"{step['from']}->{step['to']}"
                if migration_key not in self.migration_rules:
                    return {
                        "success": False,
                        "error": f"Missing migration rules for {migration_key}",
                        "source_version": source_version,
                        "target_version": target_version
                    }
                
                # Apply migration step
                migration_result = await self._apply_migration_step(
                    current_data, 
                    self.migration_rules[migration_key]
                )
                
                if not migration_result["success"]:
                    return migration_result
                
                current_data = migration_result["migrated_data"]
                transformations_applied.extend(migration_result["transformations_applied"])
            
            # Validate migrated data
            validation_result = await self.validate_schema(current_data, target_version)
            
            # Update statistics
            self.migration_stats['total_migrations'] += 1
            migration_time = (time.time() - start_time) * 1000
            self.migration_stats['migration_times_ms'].append(migration_time)
            
            if validation_result["valid"]:
                self.migration_stats['successful_migrations'] += 1
                self.migration_stats['version_distribution'][target_version] += 1
                
                return {
                    "success": True,
                    "migrated_data": current_data,
                    "source_version": source_version,
                    "target_version": target_version,
                    "migration_time_ms": migration_time,
                    "transformations_applied": transformations_applied,
                    "validation_result": validation_result
                }
            else:
                self.migration_stats['failed_migrations'] += 1
                return {
                    "success": False,
                    "error": f"Migrated data failed validation: {validation_result['error']}",
                    "source_version": source_version,
                    "target_version": target_version,
                    "migration_time_ms": migration_time,
                    "validation_result": validation_result
                }
        
        except Exception as e:
            self.migration_stats['failed_migrations'] += 1
            return {
                "success": False,
                "error": f"Migration exception: {str(e)}",
                "source_version": source_version,
                "target_version": target_version,
                "migration_time_ms": (time.time() - start_time) * 1000
            }
    
    def _find_migration_path(self, source_version: str, target_version: str) -> List[Dict[str, str]]:
        """Find migration path between versions"""
        # For now, implement simple linear path
        # In production, this could be a graph traversal algorithm
        
        version_order = ["v1.0", "v2.0", "v2.1"]
        
        try:
            source_idx = version_order.index(source_version)
            target_idx = version_order.index(target_version)
        except ValueError:
            return None
        
        if source_idx == target_idx:
            return []
        
        path = []
        if source_idx < target_idx:
            # Forward migration
            for i in range(source_idx, target_idx):
                path.append({
                    "from": version_order[i],
                    "to": version_order[i + 1]
                })
        else:
            # Backward migration (not currently supported)
            return None
        
        return path
    
    async def _apply_migration_step(self, data: Dict[str, Any], migration_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a single migration step"""
        try:
            migrated_data = copy.deepcopy(data)
            transformations_applied = []
            
            # Apply transformations
            for transformation in migration_rules.get("transformations", []):
                transformation_type = transformation["type"]
                
                if transformation_type == "restructure":
                    # Restructure fields into new structure
                    source_fields = transformation["source_fields"]
                    target_field = transformation["target_field"]
                    transform_func = transformation["transformation"]
                    
                    # Extract source data
                    source_data = {}
                    for field in source_fields:
                        if field in migrated_data:
                            source_data[field] = migrated_data[field]
                            del migrated_data[field]
                    
                    # Apply transformation
                    target_data = transform_func(source_data)
                    
                    # Set target field
                    self._set_nested_field(migrated_data, target_field, target_data)
                    transformations_applied.append(f"restructure: {source_fields} -> {target_field}")
                
                elif transformation_type == "rename":
                    # Rename field
                    source_field = transformation["source_field"]
                    target_field = transformation["target_field"]
                    
                    if source_field in migrated_data:
                        value = migrated_data[source_field]
                        del migrated_data[source_field]
                        self._set_nested_field(migrated_data, target_field, value)
                        transformations_applied.append(f"rename: {source_field} -> {target_field}")
                
                elif transformation_type == "add_default":
                    # Add field with default value
                    field = transformation["field"]
                    default_value = transformation["default_value"]
                    
                    if not self._has_nested_field(migrated_data, field):
                        self._set_nested_field(migrated_data, field, default_value)
                        transformations_applied.append(f"add_default: {field}")
                
                elif transformation_type == "enhance_provenance":
                    # Enhance provenance section
                    enhancements = transformation["enhancements"]
                    
                    if "provenance" not in migrated_data:
                        migrated_data["provenance"] = {}
                    
                    for key, value in enhancements.items():
                        if key not in migrated_data["provenance"]:
                            migrated_data["provenance"][key] = value
                    
                    transformations_applied.append("enhance_provenance")
            
            # Apply field mappings
            field_mappings = migration_rules.get("field_mappings", {})
            if "*" in field_mappings and field_mappings["*"] == "*":
                # Pass through all fields
                pass
            else:
                # Apply specific field mappings
                new_data = {}
                for source_field, target_field in field_mappings.items():
                    if source_field in migrated_data:
                        new_data[target_field] = migrated_data[source_field]
                migrated_data = new_data
            
            return {
                "success": True,
                "migrated_data": migrated_data,
                "transformations_applied": transformations_applied
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Migration step failed: {str(e)}",
                "transformations_applied": transformations_applied
            }
    
    def _set_nested_field(self, data: Dict[str, Any], field_path: str, value: Any):
        """Set nested field using dot notation"""
        parts = field_path.split('.')
        current = data
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    def _has_nested_field(self, data: Dict[str, Any], field_path: str) -> bool:
        """Check if nested field exists using dot notation"""
        parts = field_path.split('.')
        current = data
        
        try:
            for part in parts:
                current = current[part]
            return True
        except (KeyError, TypeError):
            return False
    
    def get_schema_info(self, version: str = None) -> Dict[str, Any]:
        """Get schema information"""
        if version:
            if version not in self.schema_registry:
                return {"error": f"Schema version {version} not found"}
            
            return {
                "version": version,
                "schema": self.schema_registry[version],
                "compatibility": self.version_compatibility_matrix.get(version, {}),
                "migration_rules": {
                    k: v for k, v in self.migration_rules.items() 
                    if k.startswith(f"{version}->") or k.endswith(f"->{version}")
                }
            }
        else:
            return {
                "current_version": self.current_schema_version,
                "available_versions": list(self.schema_registry.keys()),
                "compatibility_matrix": self.version_compatibility_matrix,
                "migration_statistics": self.migration_stats
            }
    
    def register_schema(self, version: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Register new schema version"""
        try:
            # Validate schema format
            if not isinstance(schema, dict) or "type" not in schema:
                return {
                    "success": False,
                    "error": "Invalid schema format"
                }
            
            # Check version format
            if not version.startswith('v') or len(version.split('.')) != 2:
                return {
                    "success": False,
                    "error": "Version must be in format 'vX.Y'"
                }
            
            # Register schema
            self.schema_registry[version] = schema
            
            # Initialize compatibility entry
            if version not in self.version_compatibility_matrix:
                self.version_compatibility_matrix[version] = {
                    "forward_compatible": [],
                    "backward_compatible": [],
                    "breaking_changes": []
                }
            
            return {
                "success": True,
                "version": version,
                "registered_at": time.time()
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Schema registration failed: {str(e)}"
            }


# =============================================================================
# Production-Ready Factory and Configuration Management
# =============================================================================

class SCAFADL1Factory:
    """Factory for creating production-ready SCAFAD Layer 1 instances"""
    
    @staticmethod
    def create_production_instance(
        config: Dict[str, Any] = None,
        environment: str = "production"
    ) -> SCAFADLayer1Integration:
        """Create production-ready Layer 1 instance"""
        
        # Default production configuration
        default_config = {
            "sanit        # SCAFAD-specific configuration
        self.scafad_config = {
            'layer_id': 'L1',
            'layer_name': 'Behavioral Intake Zone',
            'version': '2.0.0',
            'upstream_layer': 'L0',
            'downstream_layer': 'L2',
            'processing_capabilities': [
                'sanitization',
                'validation',
                'privacy_filtering',
                'anomaly_preservation',
                'schema_evolution',
                'deferred_hashing'
            ],
            'telemetry_formats': ['json', 'protobuf', 'avro', 'msgpack'],
            'compliance_standards': ['GDPR', 'CCPA', 'HIPAA', 'SOX', 'PCI-DSS'],
            'performance_targets': {
                'max_latency_ms': 0.15,
                'min_throughput_rps': 10000,
                'max_memory_mb': 100,
                'anomaly_preservation_rate': 0.9995
            }
        }
        
        # Layer communication protocols
        self.communication_protocols = {
            'input_queue': 'scafad_l0_to_l1',
            'output_queue': 'scafad_l1_to_l2',
            'control_channel': 'scafad_l1_control',
            'monitoring_channel': 'scafad_l1_metrics',
            'alert_channel': 'scafad_security_alerts'
        }
        
        # Integration state
        self.integration_state = {
            'layer_status': 'initializing',
            'upstream_connected': False,
            'downstream_connected': False,
            'last_heartbeat': 0,
            'processing_statistics': {},
            'health_status': 'unknown'
        }
    
    async def initialize_layer(self, layer0_endpoint: str = None, layer2_endpoint: str = None) -> Dict[str, Any]:
        """Initialize Layer 1 within SCAFAD framework"""
        initialization_results = {}
        
        try:
            # Initialize engine components
            self.integration_state['layer_status'] = 'initializing_engine'
            engine_init = await self._initialize_engine_components()
            initialization_results['engine'] = engine_init
            
            # Establish upstream connection to Layer 0
            if layer0_endpoint:
                self.integration_state['layer_status'] = 'connecting_upstream'
                upstream_result = await self._connect_to_layer0(layer0_endpoint)
                initialization_results['upstream_connection'] = upstream_result
                self.integration_state['upstream_connected'] = upstream_result['success']
            
            # Establish downstream connection to Layer 2
            if layer2_endpoint:
                self.integration_state['layer_status'] = 'connecting_downstream'
                downstream_result = await self._connect_to_layer2(layer2_endpoint)
                initialization_results['downstream_connection'] = downstream_result
                self.integration_state['downstream_connected'] = downstream_result['success']
            
            # Initialize monitoring and alerting
            self.integration_state['layer_status'] = 'initializing_monitoring'
            monitoring_init = await self._initialize_monitoring_systems()
            initialization_results['monitoring'] = monitoring_init
            
            # Start health check and heartbeat
            self.integration_state['layer_status'] = 'starting_health_checks'
            await self._start_health_monitoring()
            
            # Final status update
            self.integration_state['layer_status'] = 'operational'
            self.integration_state['health_status'] = 'healthy'
            self.integration_state['last_heartbeat'] = time.time()
            
            initialization_results['overall_status'] = 'success'
            initialization_results['layer_config'] = self.scafad_config
            
        except Exception as e:
            self.integration_state['layer_status'] = 'failed'
            self.integration_state['health_status'] = 'unhealthy'
            initialization_results['overall_status'] = 'failed'
            initialization_results['error'] = str(e)
            
            logging.getLogger("SCAFAD.Layer1.Integration").error(
                f"Layer 1 initialization failed: {e}", exc_info=True
            )
        
        return initialization_results
    
    async def _initialize_engine_components(self) -> Dict[str, Any]:
        """Initialize sanitization engine components"""
        results = {}
        
        # Initialize ML models
        if not self.engine.anomaly_detector.is_trained:
            # Train with default dataset or load pre-trained model
            default_training_data = self._generate_default_training_data()
            self.engine.anomaly_detector.train(default_training_data)
            results['anomaly_detector'] = 'trained_with_defaults'
        else:
            results['anomaly_detector'] = 'already_trained'
        
        # Initialize pattern learning
        results['pattern_learner'] = 'initialized'
        
        # Initialize security sanitizers
        results['security_sanitizers'] = 'loaded'
        
        # Verify GPU availability
        results['gpu_acceleration'] = 'available' if self.engine.gpu_available else 'not_available'
        
        # Initialize performance optimizer
        results['performance_optimizer'] = 'active'
        
        return results
    
    def _generate_default_training_data(self) -> List[Dict[str, Any]]:
        """Generate default training data for anomaly detection"""
        training_data = []
        
        # Normal patterns
        normal_patterns = [
            {"timestamp": 1640995200, "user_id": "user123", "action": "login", "duration_ms": 150},
            {"timestamp": 1640995260, "user_id": "user456", "action": "api_call", "duration_ms": 75},
            {"timestamp": 1640995320, "user_id": "user789", "action": "data_query", "duration_ms": 200},
            {"timestamp": 1640995380, "user_id": "user123", "action": "logout", "duration_ms": 50},
        ] * 25  # Repeat to get 100 normal samples
        
        # Anomalous patterns
        anomalous_patterns = [
            {"timestamp": 1640995400, "user_id": "'; DROP TABLE users;--", "action": "login", "duration_ms": 5000},
            {"timestamp": 1640995460, "user_id": "admin", "action": "<script>alert('xss')</script>", "duration_ms": 10},
            {"timestamp": 1640995520, "user_id": "attacker", "action": "union select * from sensitive", "duration_ms": 1},
            {"timestamp": 1640995580, "user_id": "bot", "action": "automated_scan", "duration_ms": 99999},
        ] * 5  # Repeat to get 20 anomalous samples
        
        training_data.extend(normal_patterns)
        training_data.extend(anomalous_patterns)
        
        return training_data
    
    async def _connect_to_layer0(self, endpoint: str) -> Dict[str, Any]:
        """Establish connection to Layer 0 (Adaptive Telemetry Controller)"""
        try:
            # Simulate connection establishment
            # In real implementation, this would establish WebSocket/gRPC/HTTP connections
            connection_config = {
                'endpoint': endpoint,
                'protocol': 'websocket',
                'authentication': 'scafad_layer_auth',
                'buffer_size': 10000,
                'batch_timeout_ms': 100
            }
            
            # Verify Layer 0 compatibility
            l0_capabilities = await self._query_layer0_capabilities(endpoint)
            
            if not self._verify_layer_compatibility(l0_capabilities):
                return {
                    'success': False,
                    'error': 'Layer 0 capabilities not compatible',
                    'required_capabilities': self.scafad_config['processing_capabilities'],
                    'provided_capabilities': l0_capabilities
                }
            
            # Establish data flow
            await self._setup_data_ingestion_pipeline(connection_config)
            
            return {
                'success': True,
                'endpoint': endpoint,
                'capabilities': l0_capabilities,
                'connection_config': connection_config
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'endpoint': endpoint
            }
    
    async def _connect_to_layer2(self, endpoint: str) -> Dict[str, Any]:
        """Establish connection to Layer 2 (Multi-Vector Detection Matrix)"""
        try:
            connection_config = {
                'endpoint': endpoint,
                'protocol': 'grpc',
                'authentication': 'scafad_layer_auth',
                'output_format': 'protobuf',
                'compression': 'gzip'
            }
            
            # Verify Layer 2 capabilities
            l2_capabilities = await self._query_layer2_capabilities(endpoint)
            
            # Setup output pipeline
            await self._setup_data_output_pipeline(connection_config)
            
            return {
                'success': True,
                'endpoint': endpoint,
                'capabilities': l2_capabilities,
                'connection_config': connection_config
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'endpoint': endpoint
            }
    
    async def _query_layer0_capabilities(self, endpoint: str) -> Dict[str, Any]:
        """Query Layer 0 capabilities"""
        # Mock implementation - in real system this would make API call
        return {
            'telemetry_types': ['execution_metrics', 'invocation_graphs', 'provenance_chains'],
            'output_formats': ['json', 'protobuf', 'avro'],
            'sampling_rates': [1.0, 0.1, 0.01],
            'adaptive_features': ['fallback_injection', 'schema_versioning', 'resilience_modes'],
            'version': '1.0.0'
        }
    
    async def _query_layer2_capabilities(self, endpoint: str) -> Dict[str, Any]:
        """Query Layer 2 capabilities"""
        # Mock implementation
        return {
            'detection_engines': ['rule_chain', 'drift_tracker', 'i_gnn', 'semantic_deviation'],
            'input_formats': ['json', 'protobuf', 'avro'],
            'fusion_algorithms': ['trust_weighted', 'bayesian', 'ensemble'],
            'latency_requirements': {'max_ms': 50, 'target_ms': 20},
            'version': '1.0.0'
        }
    
    def _verify_layer_compatibility(self, capabilities: Dict[str, Any]) -> bool:
        """Verify compatibility with adjacent layers"""
        # Check format compatibility
        supported_formats = set(self.scafad_config['telemetry_formats'])
        provided_formats = set(capabilities.get('output_formats', []))
        
        if not supported_formats.intersection(provided_formats):
            return False
        
        # Check version compatibility
        version = capabilities.get('version', '0.0.0')
        major_version = int(version.split('.')[0])
        
        # Compatible with major version 1.x
        return major_version == 1
    
    async def _setup_data_ingestion_pipeline(self, config: Dict[str, Any]) -> None:
        """Setup data ingestion pipeline from Layer 0"""
        # Initialize ingestion queue
        self.ingestion_queue = asyncio.Queue(maxsize=config['buffer_size'])
        
        # Start ingestion worker
        self.ingestion_task = asyncio.create_task(self._ingestion_worker())
        
        logging.getLogger("SCAFAD.Layer1.Integration").info(
            f"Data ingestion pipeline established with buffer size {config['buffer_size']}"
        )
    
    async def _setup_data_output_pipeline(self, config: Dict[str, Any]) -> None:
        """Setup data output pipeline to Layer 2"""
        # Initialize output queue
        self.output_queue = asyncio.Queue(maxsize=10000)
        
        # Start output worker
        self.output_task = asyncio.create_task(self._output_worker())
        
        logging.getLogger("SCAFAD.Layer1.Integration").info(
            f"Data output pipeline established with format {config['output_format']}"
        )
    
    async def _ingestion_worker(self) -> None:
        """Worker for processing incoming data from Layer 0"""
        while True:
            try:
                # Get batch of records from ingestion queue
                batch = []
                for _ in range(100):  # Process in batches of 100
                    try:
                        record = await asyncio.wait_for(self.ingestion_queue.get(), timeout=0.1)
                        batch.append(record)
                    except asyncio.TimeoutError:
                        break
                
                if batch:
                    # Process batch through sanitization engine
                    sanitization_results = await self.engine.sanitize_batch(batch)
                    
                    # Monitor results
                    for result in sanitization_results:
                        alerts = self.monitor.monitor_sanitization_result(result)
                        if alerts:
                            await self._handle_alerts(alerts)
                    
                    # Send processed data to output queue
                    for result in sanitization_results:
                        if result.success:
                            await self.output_queue.put(result)
                    
                    # Update statistics
                    self._update_processing_statistics(sanitization_results)
                
                await asyncio.sleep(0.001)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logging.getLogger("SCAFAD.Layer1.Integration").error(
                    f"Error in ingestion worker: {e}", exc_info=True
                )
                await asyncio.sleep(1)  # Longer delay on error
    
    async def _output_worker(self) -> None:
        """Worker for sending processed data to Layer 2"""
        while True:
            try:
                # Get processed records from output queue
                batch = []
                for _ in range(50):  # Smaller output batches for lower latency
                    try:
                        result = await asyncio.wait_for(self.output_queue.get(), timeout=0.1)
                        batch.append(result)
                    except asyncio.TimeoutError:
                        break
                
                if batch:
                    # Format data for Layer 2
                    formatted_batch = self._format_for_layer2(batch)
                    
                    # Send to Layer 2 (mock implementation)
                    await self._send_to_layer2(formatted_batch)
                
                await asyncio.sleep(0.001)
                
            except Exception as e:
                logging.getLogger("SCAFAD.Layer1.Integration").error(
                    f"Error in output worker: {e}", exc_info=True
                )
                await asyncio.sleep(1)
    
    def _format_for_layer2(self, results: List[SanitizationResult]) -> List[Dict[str, Any]]:
        """Format sanitization results for Layer 2 consumption"""
        formatted_records = []
        
        for result in results:
            if result.success and result.sanitized_record is not None:
                formatted_record = {
                    'record_id': result.transformation_hash,
                    'data': result.sanitized_record,
                    'metadata': {
                        'layer1_processing': {
                            'sanitization_time_ms': result.sanitization_time_ms,
                            'data_integrity': result.data_integrity.value,
                            'anomaly_preservation': result.anomaly_preservation.value,
                            'operations_applied': result.operations_applied,
                            'anomalies_detected': result.anomalies_detected
                        },
                        'provenance': {
                            'source_layer': 'L1',
                            'processing_timestamp': time.time(),
                            'transformation_hash': result.transformation_hash,
                            'compliance_flags': result.compliance_flags
                        },
                        'quality_metrics': {
                            'entropy_before': result.entropy_before,
                            'entropy_after': result.entropy_after,
                            'information_content_ratio': result.information_content_ratio
                        }
                    }
                }
                formatted_records.append(formatted_record)
        
        return formatted_records
    
    async def _send_to_layer2(self, formatted_batch: List[Dict[str, Any]]) -> None:
        """Send formatted data to Layer 2"""
        # Mock implementation - in real system this would send via gRPC/HTTP/etc.
        logging.getLogger("SCAFAD.Layer1.Integration").debug(
            f"Sent batch of {len(formatted_batch)} records to Layer 2"
        )
    
    async def _handle_alerts(self, alerts: List[Dict[str, Any]]) -> None:
        """Handle alerts generated during processing"""
        for alert in alerts:
            # Send critical alerts to security channel
            if alert.get('severity') == 'critical':
                await self._send_security_alert(alert)
            
            # Log all alerts
            severity = alert.get('severity', 'info').upper()
            message = alert.get('message', 'Unknown alert')
            
            logger = logging.getLogger("SCAFAD.Layer1.Alerts")
            if severity == 'CRITICAL':
                logger.critical(message)
            elif severity == 'ERROR':
                logger.error(message)
            elif severity == 'WARNING':
                logger.warning(message)
            else:
                logger.info(message)
    
    async def _send_security_alert(self, alert: Dict[str, Any]) -> None:
        """Send security alert to SCAFAD security channel"""
        security_alert = {
            'source_layer': 'L1',
            'timestamp': time.time(),
            'alert_type': alert['type'],
            'severity': alert['severity'],
            'message': alert['message'],
            'details': alert,
            'correlation_id': f"L1_{int(time.time())}_{secrets.token_hex(8)}"
        }
        
        # In real implementation, this would send to security monitoring system
        logging.getLogger("SCAFAD.Security").critical(
            f"Security Alert: {security_alert['message']}"
        )
    
    def _update_processing_statistics(self, results: List[SanitizationResult]) -> None:
        """Update processing statistics"""
        current_time = time.time()
        
        stats = {
            'timestamp': current_time,
            'total_records': len(results),
            'successful_records': sum(1 for r in results if r.success),
            'failed_records': sum(1 for r in results if not r.success),
            'total_anomalies': sum(len(r.anomalies_detected) for r in results),
            'average_latency_ms': np.mean([r.sanitization_time_ms for r in results]),
            'total_processing_time_ms': sum(r.sanitization_time_ms for r in results)
        }
        
        self.integration_state['processing_statistics'] = stats
    
    async def _initialize_monitoring_systems(self) -> Dict[str, Any]:
        """Initialize monitoring and alerting systems"""
        results = {}
        
        # Configure alert thresholds for SCAFAD environment
        self.monitor.alert_thresholds.update({
            'high_latency_ms': self.scafad_config['performance_targets']['max_latency_ms'] * 2,
            'low_throughput_rps': self.scafad_config['performance_targets']['min_throughput_rps'] * 0.5,
            'high_memory_mb': self.scafad_config['performance_targets']['max_memory_mb'] * 1.5
        })
        results['alert_thresholds'] = 'configured'
        
        # Initialize performance optimizer
        self.optimizer.adaptive_thresholds.update({
            'latency_threshold_ms': self.scafad_config['performance_targets']['max_latency_ms'],
            'memory_threshold_mb': self.scafad_config['performance_targets']['max_memory_mb']
        })
        results['performance_optimizer'] = 'configured'
        
        # Start monitoring tasks
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        results['monitoring_loop'] = 'started'
        
        return results
    
    async def _start_health_monitoring(self) -> None:
        """Start health monitoring and heartbeat"""
        self.health_task = asyncio.create_task(self._health_monitoring_loop())
        logging.getLogger("SCAFAD.Layer1.Integration").info("Health monitoring started")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while True:
            try:
                # Get current metrics
                metrics = self.engine.get_performance_metrics()
                
                # Apply optimizations if needed
                optimization_result = self.optimizer.optimize(metrics)
                
                # Check for performance degradation
                if metrics.average_latency_ms > self.scafad_config['performance_targets']['max_latency_ms']:
                    await self._handle_performance_degradation(metrics)
                
                # Update health status
                self._update_health_status(metrics)
                
                # Sleep for monitoring interval
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logging.getLogger("SCAFAD.Layer1.Integration").error(
                    f"Error in monitoring loop: {e}", exc_info=True
                )
                await asyncio.sleep(60)  # Longer delay on error
    
    async def _health_monitoring_loop(self) -> None:
        """Health monitoring and heartbeat loop"""
        while True:
            try:
                # Update heartbeat
                self.integration_state['last_heartbeat'] = time.time()
                
                # Check system health
                health_status = self._assess_system_health()
                self.integration_state['health_status'] = health_status
                
                # Send heartbeat to SCAFAD control plane
                await self._send_heartbeat(health_status)
                
                # Sleep for heartbeat interval
                await asyncio.sleep(10)  # Heartbeat every 10 seconds
                
            except Exception as e:
                logging.getLogger("SCAFAD.Layer1.Integration").error(
                    f"Error in health monitoring: {e}", exc_info=True
                )
                self.integration_state['health_status'] = 'unhealthy'
                await asyncio.sleep(30)
    
    def _assess_system_health(self) -> str:
        """Assess overall system health"""
        metrics = self.engine.get_performance_metrics()
        
        # Check critical metrics
        health_checks = {
            'latency': metrics.average_latency_ms < self.scafad_config['performance_targets']['max_latency_ms'] * 2,
            'memory': metrics.memory_usage_mb < self.scafad_config['performance_targets']['max_memory_mb'] * 2,
            'error_rate': metrics.error_rate < 0.1,
            'throughput': metrics.throughput_records_per_second > 100
        }
        
        failed_checks = [check for check, passed in health_checks.items() if not passed]
        
        if not failed_checks:
            return 'healthy'
        elif len(failed_checks) == 1:
            return 'degraded'
        else:
            return 'unhealthy'
    
    async def _send_heartbeat(self, health_status: str) -> None:
        """Send heartbeat to SCAFAD control plane"""
        heartbeat = {
            'layer_id': self.scafad_config['layer_id'],
            'timestamp': time.time(),
            'health_status': health_status,
            'processing_stats': self.integration_state.get('processing_statistics', {}),
            'connections': {
                'upstream': self.integration_state['upstream_connected'],
                'downstream': self.integration_state['downstream_connected']
            },
            'version': self.scafad_config['version']
        }
        
        # In real implementation, this would send to SCAFAD control plane
        logging.getLogger("SCAFAD.Layer1.Heartbeat").debug(
            f"Heartbeat sent: {health_status}"
        )
    
    async def _handle_performance_degradation(self, metrics: SanitizationMetrics) -> None:
        """Handle performance degradation"""
        degradation_alert = {
            'type': 'performance_degradation',
            'severity': 'warning',
            'message': f"Layer 1 performance degraded - latency: {metrics.average_latency_ms:.2f}ms",
            'metrics': asdict(metrics),
            'timestamp': time.time()
        }
        
        await self._handle_alerts([degradation_alert])
        
        # Apply emergency optimizations
        self.engine.optimize_performance()
    
    def _update_health_status(self, metrics: SanitizationMetrics) -> None:
        """Update health status based on metrics"""
        previous_status = self.integration_state['health_status']
        current_status = self._assess_system_health()
        
        if previous_status != current_status:
            logging.getLogger("SCAFAD.Layer1.Integration").info(
                f"Health status changed from {previous_status} to {current_status}"
            )
            self.integration_state['health_status'] = current_status
    
    # Public API methods for SCAFAD integration
    
    async def process_telemetry_from_layer0(self, telemetry_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process telemetry batch from Layer 0"""
        try:
            # Add to ingestion queue
            for record in telemetry_batch:
                await self.ingestion_queue.put(record)
            
            return {
                'success': True,
                'accepted_records': len(telemetry_batch),
                'queue_size': self.ingestion_queue.qsize()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'accepted_records': 0
            }
    
    def get_layer_status(self) -> Dict[str, Any]:
        """Get comprehensive layer status"""
        metrics = self.engine.get_performance_metrics()
        
        return {
            'layer_config': self.scafad_config,
            'integration_state': self.integration_state,
            'performance_metrics': asdict(metrics),
            'security_dashboard': self.monitor.get_security_dashboard(),
            'engine_status': self.engine.get_engine_status(),
            'queue_status': {
                'ingestion_queue_size': getattr(self, 'ingestion_queue', asyncio.Queue()).qsize(),
                'output_queue_size': getattr(self, 'output_queue', asyncio.Queue()).qsize()
            }
        }
    
    def get_processing_capabilities(self) -> Dict[str, Any]:
        """Get layer processing capabilities"""
        return {
            'capabilities': self.scafad_config['processing_capabilities'],
            'supported_formats': self.scafad_config['telemetry_formats'],
            'compliance_standards': self.scafad_config['compliance_standards'],
            'performance_targets': self.scafad_config['performance_targets'],
            'sanitization_levels': [level.name for level in SanitizationLevel],
            'processing_modes': [mode.name for mode in ProcessingMode],
            'optimization_strategies': [strategy.name for strategy in OptimizationStrategy]
        }
    
    async def update_layer_configuration(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update layer configuration"""
        try:
            # Update engine configuration
            if 'sanitization_level' in config_updates:
                self.engine.update_configuration(
                    sanitization_level=config_updates['sanitization_level']
                )
            
            if 'processing_mode' in config_updates:
                self.engine.update_configuration(
                    processing_mode=config_updates['processing_mode']
                )
            
            # Update monitoring thresholds
            if 'alert_thresholds' in config_updates:
                self.monitor.alert_thresholds.update(config_updates['alert_thresholds'])
            
            # Update performance targets
            if 'performance_targets' in config_updates:
                self.scafad_config['performance_targets'].update(
                    config_updates['performance_targets']
                )
            
            return {
                'success': True,
                'updated_config': config_updates,
                'new_status': self.get_layer_status()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def shutdown_layer(self) -> Dict[str, Any]:
        """Gracefully shutdown Layer 1"""
        try:
            shutdown_steps = []
            
            # Stop ingestion
            if hasattr(self, 'ingestion_task'):
                self.ingestion_task.cancel()
                shutdown_steps.append('ingestion_stopped')
            
            # Process remaining data
            remaining_records = 0
            if hasattr(self, 'ingestion_queue'):
                remaining_records = self.ingestion_queue.qsize()
                # Process remaining records (with timeout)
                try:
                    await asyncio.wait_for(self._process_remaining_records(), timeout=30)
                    shutdown_steps.append('remaining_records_processed')
                except asyncio.TimeoutError:
                    shutdown_steps.append('remaining_records_timeout')
            
            # Stop output worker
            if hasattr(self, 'output_task'):
                self.output_task.cancel()
                shutdown_steps.append('output_stopped')
            
            # Stop monitoring
            if hasattr(self, 'monitoring_task'):
                self.monitoring_task.cancel()
                shutdown_steps.append('monitoring_stopped')
            
            if hasattr(self, 'health_task'):
                self.health_task.cancel()
                shutdown_steps.append('health_monitoring_stopped')
            
            # Cleanup engine
            self.engine.cleanup()
            shutdown_steps.append('engine_cleanup')
            
            # Update status
            self.integration_state['layer_status'] = 'shutdown'
            shutdown_steps.append('status_updated')
            
            return {
                'success': True,
                'shutdown_steps': shutdown_steps,
                'remaining_records': remaining_records
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'partial_shutdown': True
            }
    
    async def _process_remaining_records(self) -> None:
        """Process any remaining records in queues"""
        while not self.ingestion_queue.empty():
            try:
                record = self.ingestion_queue.get_nowait()
                result = await self.engine.sanitize_record(record)
                if result.success:
                    await self.output_queue.put(result)
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                logging.getLogger("SCAFAD.Layer1.Integration").error(
                    f"Error processing remaining record during shutdown: {e}"
                )
        
        # Process output queue
        while not self.output_queue.empty():
            try:
                result = self.output_queue.get_nowait()
                formatted_batch = self._format_for_layer2([result])
                await self        elif metrics.error_rate > 0.1:
            # Increase sanitization level if error rate is high
            if current_level == SanitizationLevel.MINIMAL:
                self.engine.sanitization_level = SanitizationLevel.STANDARD
                return {
                    'action': 'increased_sanitization_level',
                    'old_level': current_level.value,
                    'new_level': SanitizationLevel.STANDARD.value,
                    'reason': 'high_error_rate'
                }
            elif current_level == SanitizationLevel.STANDARD:
                self.engine.sanitization_level = SanitizationLevel.AGGRESSIVE
                return {
                    'action': 'increased_sanitization_level',
                    'old_level': current_level.value,
                    'new_level': SanitizationLevel.AGGRESSIVE.value,
                    'reason': 'high_error_rate'
                }
        
        return None
    
    def _optimize_memory_usage(self, metrics: SanitizationMetrics) -> Dict[str, Any]:
        """Optimize memory usage"""
        actions_taken = []
        
        if metrics.memory_usage_mb > 200:
            # Clear caches
            self.engine.result_cache.clear()
            self.engine.pattern_cache = BloomFilterCache()
            actions_taken.append('cleared_caches')
            
            # Force garbage collection
            gc.collect()
            actions_taken.append('garbage_collection')
            
            # Reduce cache sizes
            if hasattr(self.engine.result_cache, 'maxsize'):
                new_size = max(self.engine.result_cache.maxsize // 2, 1000)
                self.engine.result_cache = TTLCache(maxsize=new_size, ttl=300)
                actions_taken.append(f'reduced_cache_size_to_{new_size}')
        
        return {'actions': actions_taken} if actions_taken else None
    
    def _optimize_batch_processing(self, metrics: SanitizationMetrics) -> Dict[str, Any]:
        """Optimize batch processing parameters"""
        if metrics.cpu_utilization_percent > 80:
            # Reduce thread pool size to prevent CPU overload
            if hasattr(self.engine, 'thread_pool'):
                current_workers = self.engine.thread_pool._max_workers
                new_workers = max(current_workers - 1, 1)
                
                # Shutdown current pool and create new one
                self.engine.thread_pool.shutdown(wait=True)
                self.engine.thread_pool = ThreadPoolExecutor(max_workers=new_workers)
                
                return {
                    'action': 'reduced_thread_pool_size',
                    'old_workers': current_workers,
                    'new_workers': new_workers
                }
        
        elif metrics.cpu_utilization_percent < 30 and metrics.throughput_records_per_second < 1000:
            # Increase thread pool size for better throughput
            if hasattr(self.engine, 'thread_pool'):
                current_workers = self.engine.thread_pool._max_workers
                new_workers = min(current_workers + 1, 8)
                
                if new_workers > current_workers:
                    self.engine.thread_pool.shutdown(wait=True)
                    self.engine.thread_pool = ThreadPoolExecutor(max_workers=new_workers)
                    
                    return {
                        'action': 'increased_thread_pool_size',
                        'old_workers': current_workers,
                        'new_workers': new_workers
                    }
        
        return None
    
    def _update_adaptive_thresholds(self):
        """Update adaptive thresholds based on performance history"""
        if len(self.performance_history) < 10:
            return
        
        # Calculate moving averages
        recent_metrics = list(self.performance_history)[-10:]
        
        avg_latency = np.mean([m['metrics']['average_latency_ms'] for m in recent_metrics])
        avg_memory = np.mean([m['metrics']['memory_usage_mb'] for m in recent_metrics])
        avg_cpu = np.mean([m['metrics']['cpu_utilization_percent'] for m in recent_metrics])
        avg_error_rate = np.mean([m['metrics']['error_rate'] for m in recent_metrics])
        
        # Adjust thresholds based on recent performance
        self.adaptive_thresholds['latency_threshold_ms'] = max(0.5, avg_latency * 1.2)
        self.adaptive_thresholds['memory_threshold_mb'] = max(50, avg_memory * 1.5)
        self.adaptive_thresholds['cpu_threshold_percent'] = max(30, avg_cpu * 1.3)
        self.adaptive_thresholds['error_rate_threshold'] = max(0.01, avg_error_rate * 2.0)
    
    def _analyze_performance_trend(self) -> Dict[str, str]:
        """Analyze performance trends over time"""
        if len(self.performance_history) < 5:
            return {'trend': 'insufficient_data'}
        
        # Get recent and older metrics for comparison
        recent = list(self.performance_history)[-5:]
        older = list(self.performance_history)[-10:-5] if len(self.performance_history) >= 10 else []
        
        if not older:
            return {'trend': 'insufficient_history'}
        
        recent_avg_latency = np.mean([m['metrics']['average_latency_ms'] for m in recent])
        older_avg_latency = np.mean([m['metrics']['average_latency_ms'] for m in older])
        
        recent_avg_throughput = np.mean([m['metrics']['throughput_records_per_second'] for m in recent])
        older_avg_throughput = np.mean([m['metrics']['throughput_records_per_second'] for m in older])
        
        recent_avg_memory = np.mean([m['metrics']['memory_usage_mb'] for m in recent])
        older_avg_memory = np.mean([m['metrics']['memory_usage_mb'] for m in older])
        
        # Determine trends
        trends = {}
        
        # Latency trend
        if recent_avg_latency > older_avg_latency * 1.1:
            trends['latency'] = 'increasing'
        elif recent_avg_latency < older_avg_latency * 0.9:
            trends['latency'] = 'decreasing'
        else:
            trends['latency'] = 'stable'
        
        # Throughput trend
        if recent_avg_throughput > older_avg_throughput * 1.1:
            trends['throughput'] = 'increasing'
        elif recent_avg_throughput < older_avg_throughput * 0.9:
            trends['throughput'] = 'decreasing'
        else:
            trends['throughput'] = 'stable'
        
        # Memory trend
        if recent_avg_memory > older_avg_memory * 1.1:
            trends['memory'] = 'increasing'
        elif recent_avg_memory < older_avg_memory * 0.9:
            trends['memory'] = 'decreasing'
        else:
            trends['memory'] = 'stable'
        
        # Overall trend assessment
        if trends['latency'] == 'decreasing' and trends['throughput'] == 'increasing':
            trends['overall'] = 'improving'
        elif trends['latency'] == 'increasing' and trends['throughput'] == 'decreasing':
            trends['overall'] = 'degrading'
        else:
            trends['overall'] = 'mixed'
        
        return trends


# =============================================================================
# Advanced Monitoring and Alerting System
# =============================================================================

class SanitizationMonitor:
    """Advanced monitoring and alerting for sanitization operations"""
    
    def __init__(self, engine: EnhancedSanitizationEngine):
        self.engine = engine
        self.alert_thresholds = {
            'high_latency_ms': 5.0,
            'low_throughput_rps': 100,
            'high_error_rate': 0.1,
            'high_memory_mb': 500,
            'anomaly_burst_count': 10,
            'injection_attack_rate': 0.05
        }
        self.alert_history = deque(maxlen=1000)
        self.metrics_history = deque(maxlen=10000)
        self.anomaly_patterns = defaultdict(list)
        self.security_events = deque(maxlen=1000)
        
    def monitor_sanitization_result(self, result: SanitizationResult):
        """Monitor individual sanitization result for alerts"""
        timestamp = time.time()
        
        # Record metrics
        self.metrics_history.append({
            'timestamp': timestamp,
            'latency_ms': result.sanitization_time_ms,
            'success': result.success,
            'data_integrity': result.data_integrity.value,
            'anomaly_preservation': result.anomaly_preservation.value,
            'anomalies_detected': len(result.anomalies_detected),
            'memory_bytes': result.memory_usage_bytes
        })
        
        # Check for alerts
        alerts = self._check_result_alerts(result, timestamp)
        
        # Record anomaly patterns
        if result.anomalies_detected:
            for anomaly in result.anomalies_detected:
                self.anomaly_patterns[anomaly['type']].append({
                    'timestamp': timestamp,
                    'confidence': anomaly.get('confidence', 0.0),
                    'details': anomaly
                })
        
        # Security event monitoring
        if any('injection' in anomaly.get('type', '') for anomaly in result.anomalies_detected):
            self.security_events.append({
                'timestamp': timestamp,
                'event_type': 'injection_attempt',
                'details': result.anomalies_detected,
                'source_hash': result.transformation_hash
            })
        
        return alerts
    
    def _check_result_alerts(self, result: SanitizationResult, timestamp: float) -> List[Dict[str, Any]]:
        """Check individual result for alert conditions"""
        alerts = []
        
        # High latency alert
        if result.sanitization_time_ms > self.alert_thresholds['high_latency_ms']:
            alerts.append({
                'type': 'high_latency',
                'severity': 'warning',
                'message': f"High sanitization latency: {result.sanitization_time_ms:.2f}ms",
                'threshold': self.alert_thresholds['high_latency_ms'],
                'actual_value': result.sanitization_time_ms,
                'timestamp': timestamp
            })
        
        # Processing error alert
        if not result.success:
            alerts.append({
                'type': 'processing_error',
                'severity': 'error',
                'message': f"Sanitization failed: {result.error_message}",
                'error_details': result.error_message,
                'timestamp': timestamp
            })
        
        # Data integrity alert
        if result.data_integrity in [DataIntegrityLevel.SIGNIFICANT_LOSS, DataIntegrityLevel.MAJOR_LOSS]:
            alerts.append({
                'type': 'data_integrity_loss',
                'severity': 'warning',
                'message': f"Significant data integrity loss: {result.data_integrity.value}",
                'integrity_level': result.data_integrity.value,
                'timestamp': timestamp
            })
        
        # Anomaly preservation alert
        if result.anomaly_preservation in [AnomalyPreservationStatus.MINIMALLY_PRESERVED, 
                                         AnomalyPreservationStatus.NOT_PRESERVED]:
            alerts.append({
                'type': 'anomaly_preservation_loss',
                'severity': 'critical',
                'message': f"Poor anomaly preservation: {result.anomaly_preservation.value}",
                'preservation_status': result.anomaly_preservation.value,
                'timestamp': timestamp
            })
        
        # Security alert for multiple anomalies
        if len(result.anomalies_detected) >= self.alert_thresholds['anomaly_burst_count']:
            alerts.append({
                'type': 'anomaly_burst',
                'severity': 'critical',
                'message': f"Anomaly burst detected: {len(result.anomalies_detected)} anomalies",
                'anomaly_count': len(result.anomalies_detected),
                'anomaly_types': [a.get('type', 'unknown') for a in result.anomalies_detected],
                'timestamp': timestamp
            })
        
        # Record alerts
        for alert in alerts:
            self.alert_history.append(alert)
        
        return alerts
    
    def monitor_batch_performance(self, batch_results: List[SanitizationResult]) -> Dict[str, Any]:
        """Monitor batch processing performance"""
        if not batch_results:
            return {}
        
        timestamp = time.time()
        
        # Calculate batch metrics
        batch_metrics = {
            'total_records': len(batch_results),
            'successful_records': sum(1 for r in batch_results if r.success),
            'failed_records': sum(1 for r in batch_results if not r.success),
            'average_latency_ms': np.mean([r.sanitization_time_ms for r in batch_results]),
            'max_latency_ms': max(r.sanitization_time_ms for r in batch_results),
            'total_anomalies': sum(len(r.anomalies_detected) for r in batch_results),
            'total_processing_time_ms': sum(r.sanitization_time_ms for r in batch_results),
            'timestamp': timestamp
        }
        
        # Calculate throughput
        if batch_metrics['total_processing_time_ms'] > 0:
            batch_metrics['throughput_rps'] = (
                batch_metrics['total_records'] * 1000 / 
                batch_metrics['total_processing_time_ms']
            )
        else:
            batch_metrics['throughput_rps'] = 0
        
        # Calculate error rate
        batch_metrics['error_rate'] = (
            batch_metrics['failed_records'] / batch_metrics['total_records']
            if batch_metrics['total_records'] > 0 else 0
        )
        
        # Check for batch-level alerts
        batch_alerts = self._check_batch_alerts(batch_metrics)
        
        return {
            'metrics': batch_metrics,
            'alerts': batch_alerts,
            'summary': self._generate_batch_summary(batch_metrics)
        }
    
    def _check_batch_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check batch metrics for alert conditions"""
        alerts = []
        timestamp = metrics['timestamp']
        
        # Low throughput alert
        if metrics['throughput_rps'] < self.alert_thresholds['low_throughput_rps']:
            alerts.append({
                'type': 'low_throughput',
                'severity': 'warning',
                'message': f"Low batch throughput: {metrics['throughput_rps']:.1f} records/sec",
                'threshold': self.alert_thresholds['low_throughput_rps'],
                'actual_value': metrics['throughput_rps'],
                'timestamp': timestamp
            })
        
        # High error rate alert
        if metrics['error_rate'] > self.alert_thresholds['high_error_rate']:
            alerts.append({
                'type': 'high_error_rate',
                'severity': 'error',
                'message': f"High batch error rate: {metrics['error_rate']:.1%}",
                'threshold': self.alert_thresholds['high_error_rate'],
                'actual_value': metrics['error_rate'],
                'failed_records': metrics['failed_records'],
                'total_records': metrics['total_records'],
                'timestamp': timestamp
            })
        
        # High average latency alert
        if metrics['average_latency_ms'] > self.alert_thresholds['high_latency_ms']:
            alerts.append({
                'type': 'high_batch_latency',
                'severity': 'warning',
                'message': f"High batch average latency: {metrics['average_latency_ms']:.2f}ms",
                'threshold': self.alert_thresholds['high_latency_ms'],
                'actual_value': metrics['average_latency_ms'],
                'max_latency_ms': metrics['max_latency_ms'],
                'timestamp': timestamp
            })
        
        # Record alerts
        for alert in alerts:
            self.alert_history.append(alert)
        
        return alerts
    
    def _generate_batch_summary(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Generate human-readable batch summary"""
        success_rate = (metrics['successful_records'] / metrics['total_records'] * 100 
                       if metrics['total_records'] > 0 else 0)
        
        summary = {
            'overall_status': 'healthy' if success_rate >= 95 else 'degraded' if success_rate >= 90 else 'unhealthy',
            'performance_status': 'good' if metrics['throughput_rps'] >= 500 else 'poor',
            'success_rate': f"{success_rate:.1f}%",
            'throughput': f"{metrics['throughput_rps']:.1f} records/sec",
            'average_latency': f"{metrics['average_latency_ms']:.2f}ms"
        }
        
        return summary
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Generate security monitoring dashboard data"""
        current_time = time.time()
        last_hour = current_time - 3600
        last_day = current_time - 86400
        
        # Recent security events
        recent_events = [event for event in self.security_events if event['timestamp'] > last_hour]
        daily_events = [event for event in self.security_events if event['timestamp'] > last_day]
        
        # Anomaly pattern analysis
        anomaly_summary = {}
        for anomaly_type, events in self.anomaly_patterns.items():
            recent_events_for_type = [e for e in events if e['timestamp'] > last_hour]
            anomaly_summary[anomaly_type] = {
                'recent_count': len(recent_events_for_type),
                'daily_count': len([e for e in events if e['timestamp'] > last_day]),
                'average_confidence': np.mean([e['confidence'] for e in recent_events_for_type]) if recent_events_for_type else 0.0,
                'trend': self._calculate_anomaly_trend(events)
            }
        
        # Security metrics
        injection_events = [e for e in daily_events if 'injection' in e['event_type']]
        injection_rate = len(injection_events) / len(daily_events) if daily_events else 0.0
        
        # Alert summary
        recent_alerts = [alert for alert in self.alert_history if alert['timestamp'] > last_hour]
        critical_alerts = [alert for alert in recent_alerts if alert['severity'] == 'critical']
        
        return {
            'security_events': {
                'last_hour': len(recent_events),
                'last_day': len(daily_events),
                'injection_rate': injection_rate,
                'trend': self._calculate_event_trend()
            },
            'anomaly_patterns': anomaly_summary,
            'alerts': {
                'recent_count': len(recent_alerts),
                'critical_count': len(critical_alerts),
                'alert_rate': len(recent_alerts) / max(1, len(self.metrics_history))
            },
            'top_threats': self._identify_top_threats(),
            'recommendations': self._generate_security_recommendations()
        }
    
    def _calculate_anomaly_trend(self, events: List[Dict[str, Any]]) -> str:
        """Calculate trend for specific anomaly type"""
        if len(events) < 10:
            return 'insufficient_data'
        
        current_time = time.time()
        last_hour = current_time - 3600
        previous_hour = current_time - 7200
        
        recent_count = len([e for e in events if e['timestamp'] > last_hour])
        previous_count = len([e for e in events if previous_hour < e['timestamp'] <= last_hour])
        
        if previous_count == 0:
            return 'new' if recent_count > 0 else 'stable'
        
        change_ratio = recent_count / previous_count
        
        if change_ratio > 1.5:
            return 'increasing'
        elif change_ratio < 0.5:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_event_trend(self) -> str:
        """Calculate overall security event trend"""
        if len(self.security_events) < 10:
            return 'insufficient_data'
        
        current_time = time.time()
        last_6_hours = current_time - 21600
        previous_6_hours = current_time - 43200
        
        recent_events = [e for e in self.security_events if e['timestamp'] > last_6_hours]
        previous_events = [e for e in self.security_events if previous_6_hours < e['timestamp'] <= last_6_hours]
        
        if len(previous_events) == 0:
            return 'new_activity' if len(recent_events) > 0 else 'quiet'
        
        change_ratio = len(recent_events) / len(previous_events)
        
        if change_ratio > 2.0:
            return 'rapidly_increasing'
        elif change_ratio > 1.3:
            return 'increasing'
        elif change_ratio < 0.5:
            return 'decreasing'
        else:
            return 'stable'
    
    def _identify_top_threats(self) -> List[Dict[str, Any]]:
        """Identify top security threats based on recent activity"""
        current_time = time.time()
        last_day = current_time - 86400
        
        threat_scores = defaultdict(float)
        threat_details = defaultdict(list)
        
        # Score threats based on frequency and severity
        for anomaly_type, events in self.anomaly_patterns.items():
            recent_events = [e for e in events if e['timestamp'] > last_day]
            if recent_events:
                # Base score on frequency
                frequency_score = len(recent_events) * 0.1
                
                # Boost score based on confidence
                confidence_score = np.mean([e['confidence'] for e in recent_events]) * 0.5
                
                # Boost score for injection-related threats
                injection_boost = 1.0 if 'injection' in anomaly_type else 0.0
                
                total_score = frequency_score + confidence_score + injection_boost
                threat_scores[anomaly_type] = total_score
                threat_details[anomaly_type] = recent_events
        
        # Sort threats by score and return top 5
        top_threats = sorted(threat_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return [
            {
                'threat_type': threat_type,
                'score': score,
                'recent_incidents': len(threat_details[threat_type]),
                'average_confidence': np.mean([e['confidence'] for e in threat_details[threat_type]]),
                'last_seen': max(e['timestamp'] for e in threat_details[threat_type]) if threat_details[threat_type] else 0
            }
            for threat_type, score in top_threats
        ]
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations based on current threat landscape"""
        recommendations = []
        
        # Analyze recent patterns
        current_time = time.time()
        last_day = current_time - 86400
        
        recent_alerts = [alert for alert in self.alert_history if alert['timestamp'] > last_day]
        
        # High error rate recommendation
        error_alerts = [a for a in recent_alerts if a['type'] in ['processing_error', 'high_error_rate']]
        if len(error_alerts) > 5:
            recommendations.append(
                "Consider increasing sanitization level due to high error rates in the last 24 hours"
            )
        
        # Performance recommendations
        latency_alerts = [a for a in recent_alerts if 'latency' in a['type']]
        if len(latency_alerts) > 3:
            recommendations.append(
                "Performance degradation detected - consider optimizing sanitization rules or increasing resources"
            )
        
        # Injection attack recommendations
        injection_events = [e for e in self.security_events if 'injection' in e['event_type'] and e['timestamp'] > last_day]
        if len(injection_events) > 10:
            recommendations.append(
                "High injection attack activity detected - consider enabling more aggressive sanitization for SQL/XSS patterns"
            )
        
        # Anomaly preservation recommendations
        preservation_alerts = [a for a in recent_alerts if a['type'] == 'anomaly_preservation_loss']
        if len(preservation_alerts) > 2:
            recommendations.append(
                "Anomaly preservation issues detected - review sanitization rules to ensure critical patterns are preserved"
            )
        
        # Memory usage recommendations
        if any(a['type'] == 'high_memory_usage' for a in recent_alerts):
            recommendations.append(
                "High memory usage detected - consider reducing cache sizes or implementing more aggressive cleanup"
            )
        
        # Default recommendation if no specific issues
        if not recommendations:
            recommendations.append(
                "System operating normally - continue monitoring for anomalies and performance trends"
            )
        
        return recommendations
    
    def export_monitoring_data(self, format: str = 'json') -> Union[str, Dict[str, Any]]:
        """Export monitoring data for external analysis"""
        data = {
            'metrics_summary': self._calculate_metrics_summary(),
            'alert_summary': self._calculate_alert_summary(),
            'security_summary': self.get_security_dashboard(),
            'anomaly_patterns': dict(self.anomaly_patterns),
            'performance_trends': self._calculate_performance_trends(),
            'export_timestamp': time.time()
        }
        
        if format.lower() == 'json':
            return json.dumps(data, indent=2, default=str)
        else:
            return data
    
    def _calculate_metrics_summary(self) -> Dict[str, Any]:
        """Calculate summary of metrics over time"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 records
        
        return {
            'total_records_processed': len(self.metrics_history),
            'recent_average_latency_ms': np.mean([m['latency_ms'] for m in recent_metrics]),
            'recent_success_rate': np.mean([m['success'] for m in recent_metrics]),
            'recent_average_anomalies': np.mean([m['anomalies_detected'] for m in recent_metrics]),
            'data_integrity_distribution': {
                level: sum(1 for m in recent_metrics if m['data_integrity'] == level)
                for level in set(m['data_integrity'] for m in recent_metrics)
            }
        }
    
    def _calculate_alert_summary(self) -> Dict[str, Any]:
        """Calculate summary of alerts over time"""
        if not self.alert_history:
            return {}
        
        current_time = time.time()
        last_day = current_time - 86400
        recent_alerts = [alert for alert in self.alert_history if alert['timestamp'] > last_day]
        
        alert_types = {}
        severity_counts = {'critical': 0, 'error': 0, 'warning': 0, 'info': 0}
        
        for alert in recent_alerts:
            alert_type = alert['type']
            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
            
            severity = alert.get('severity', 'info')
            severity_counts[severity] += 1
        
        return {
            'total_alerts_24h': len(recent_alerts),
            'alert_types': alert_types,
            'severity_distribution': severity_counts,
            'most_common_alert': max(alert_types, key=alert_types.get) if alert_types else None
        }
    
    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends over time"""
        if len(self.metrics_history) < 20:
            return {'status': 'insufficient_data'}
        
        # Split data into two halves for trend analysis
        half_point = len(self.metrics_history) // 2
        older_half = list(self.metrics_history)[:half_point]
        newer_half = list(self.metrics_history)[half_point:]
        
        older_avg_latency = np.mean([m['latency_ms'] for m in older_half])
        newer_avg_latency = np.mean([m['latency_ms'] for m in newer_half])
        
        older_success_rate = np.mean([m['success'] for m in older_half])
        newer_success_rate = np.mean([m['success'] for m in newer_half])
        
        older_avg_anomalies = np.mean([m['anomalies_detected'] for m in older_half])
        newer_avg_anomalies = np.mean([m['anomalies_detected'] for m in newer_half])
        
        return {
            'latency_trend': {
                'direction': 'improving' if newer_avg_latency < older_avg_latency else 'degrading',
                'change_percent': ((newer_avg_latency - older_avg_latency) / older_avg_latency * 100) if older_avg_latency > 0 else 0
            },
            'success_rate_trend': {
                'direction': 'improving' if newer_success_rate > older_success_rate else 'degrading',
                'change_percent': ((newer_success_rate - older_success_rate) / older_success_rate * 100) if older_success_rate > 0 else 0
            },
            'anomaly_detection_trend': {
                'direction': 'increasing' if newer_avg_anomalies > older_avg_anomalies else 'decreasing',
                'change_percent': ((newer_avg_anomalies - older_avg_anomalies) / older_avg_anomalies * 100) if older_avg_anomalies > 0 else 0
            }
        }


# =============================================================================
# Integration with SCAFAD Framework
# =============================================================================

class SCAFADLayer1Integration:
    """Integration layer for SCAFAD framework compatibility"""
    
    def __init__(self, sanitization_engine: EnhancedSanitizationEngine):
        self.engine = sanitization_engine
        self.monitor = SanitizationMonitor(sanitization_engine)
        self.optimizer = PerformanceOptimizer(sanitization_engine)
        
        # SCAFAD-specific configuration
        self.scafad_config = {
            'layer_id': 'L1',
            'layer_name': 'Behavioral Intake Zone',
            'version': '2.0.0',
            'upstream_layer': 'L0',#!/usr/bin/env python3
"""
SCAFAD Layer 1: Enhanced Sanitization Processor
===============================================

Advanced sanitization system for serverless telemetry data with ML-powered anomaly
preservation, adaptive cleaning strategies, and comprehensive performance optimization.
Designed for production serverless environments with sub-millisecond latency requirements.

Key Enhancements:
- ML-powered anomaly pattern detection and preservation
- Vectorized operations for batch processing efficiency
- Adaptive sanitization strategies based on data patterns
- Intelligent caching with pattern recognition
- Real-time performance optimization
- Advanced threat detection during sanitization
- Probabilistic data structures for memory efficiency
- SIMD-optimized string processing
- Advanced Unicode normalization with homograph detection
- Smart field type inference and adaptive cleaning

Performance Targets (Enhanced):
- Sanitization latency: <0.15ms per record (50% improvement)
- Batch processing: >10,000 records/second
- Anomaly preservation: 99.95%+ (improved from 99.8%)
- Memory efficiency: <1MB for 100K record cache
- CPU utilization: <5% on production workloads

Author: SCAFAD Research Team
Institution: Birmingham Newman University
License: MIT
Version: 2.0.0 (Enhanced)
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
import mmap
import struct
import pickle
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable, Iterator
from enum import Enum, auto, IntEnum
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque, Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import asynccontextmanager
import traceback
import copy
import weakref
import gc
import os
import sys

# Advanced string and text processing
import string
import textwrap
from difflib import SequenceMatcher
import ftfy  # Fix text encoding issues
import polyglot
from polyglot.text import Text
import langdetect

# High-performance numeric operations
import numpy as np
import numba
from numba import jit, vectorize, cuda
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation, getcontext
import pandas as pd

# Advanced date and time handling
from dateutil import parser as date_parser
import pytz
import arrow
from croniter import croniter

# Performance monitoring and profiling
import time
import psutil
import memory_profiler
from functools import wraps, lru_cache
from cachetools import TTLCache, LRUCache
import cProfile
import pstats

# Character encoding and detection
import chardet
import cchardet  # Faster alternative
import charset_normalizer

# Network and security utilities
import ipaddress
import dns.resolver
import whois
import requests
from urllib3.util import parse_url

# Path and file handling
from pathlib import Path, PurePosixPath, PureWindowsPath
import mimetypes
import magic

# Machine Learning for anomaly detection
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Probabilistic data structures
from pybloom_live import BloomFilter
import mmh3  # MurmurHash3
from hyperloglog import HyperLogLog

# Cryptographic operations
import secrets
import hmac
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

# Regular expressions compilation cache
import regex as advanced_regex

# Compression for caching
import zlib
import lz4.frame

# Configuration management
from dataclasses_json import dataclass_json


# =============================================================================
# Enhanced Data Models and Enums
# =============================================================================

class SanitizationLevel(IntEnum):
    """Sanitization intensity levels with numeric ordering"""
    MINIMAL = 1      # Basic cleaning only
    STANDARD = 2     # Standard sanitization
    AGGRESSIVE = 3   # Aggressive cleaning
    PARANOID = 4     # Maximum sanitization
    ML_ADAPTIVE = 5  # ML-driven adaptive sanitization

class SanitizationType(Enum):
    """Enhanced types of sanitization operations"""
    # Basic operations
    WHITESPACE = "whitespace"
    ENCODING = "encoding"
    SPECIAL_CHARS = "special_chars"
    NUMERIC = "numeric"
    TIMESTAMP = "timestamp"
    
    # Advanced operations
    PATH = "path"
    URL = "url"
    HTML = "html"
    SQL = "sql"
    COMMAND = "command"
    UNICODE = "unicode"
    CASE = "case"
    TRUNCATION = "truncation"
    
    # ML-enhanced operations
    PATTERN_ANOMALY = "pattern_anomaly"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    BEHAVIORAL_FINGERPRINT = "behavioral_fingerprint"
    STATISTICAL_OUTLIER = "statistical_outlier"
    TEMPORAL_SEQUENCE = "temporal_sequence"
    
    # Security-focused operations
    INJECTION_DETECTION = "injection_detection"
    HOMOGRAPH_ATTACK = "homograph_attack"
    STEGANOGRAPHY = "steganography"
    COVERT_CHANNEL = "covert_channel"

class DataIntegrityLevel(Enum):
    """Enhanced data integrity levels"""
    PERFECT = "perfect"              # No data loss, bit-perfect preservation
    INTACT = "intact"                # Semantic preservation, minimal formatting changes
    MINIMAL_LOSS = "minimal_loss"    # <0.5% information loss
    MODERATE_LOSS = "moderate_loss"  # 0.5-2% information loss
    SIGNIFICANT_LOSS = "significant_loss"  # 2-5% information loss
    MAJOR_LOSS = "major_loss"        # >5% information loss

class AnomalyPreservationStatus(Enum):
    """Enhanced anomaly preservation status"""
    PERFECTLY_PRESERVED = "perfectly_preserved"    # 100% preservation
    FULLY_PRESERVED = "fully_preserved"           # 99.5-99.9% preservation
    MOSTLY_PRESERVED = "mostly_preserved"         # 95-99.5% preservation
    PARTIALLY_PRESERVED = "partially_preserved"   # 80-95% preservation
    MINIMALLY_PRESERVED = "minimally_preserved"   # 50-80% preservation
    NOT_PRESERVED = "not_preserved"               # <50% preservation

class ProcessingMode(Enum):
    """Processing execution modes"""
    SYNCHRONOUS = "synchronous"     # Standard blocking processing
    ASYNCHRONOUS = "asynchronous"   # Async coroutine processing
    PARALLEL = "parallel"           # Multi-threaded processing
    DISTRIBUTED = "distributed"    # Multi-process processing
    GPU_ACCELERATED = "gpu_accelerated"  # CUDA/OpenCL processing
    STREAMING = "streaming"         # Real-time streaming processing

class OptimizationStrategy(Enum):
    """Optimization strategies for different workloads"""
    LATENCY_OPTIMIZED = "latency_optimized"       # Minimize per-record latency
    THROUGHPUT_OPTIMIZED = "throughput_optimized" # Maximize records/second
    MEMORY_OPTIMIZED = "memory_optimized"         # Minimize memory usage
    BALANCED = "balanced"                         # Balance all factors
    ADAPTIVE = "adaptive"                         # ML-driven optimization


@dataclass_json
@dataclass
class SanitizationMetrics:
    """Enhanced metrics for sanitization operations"""
    # Performance metrics
    total_records_processed: int = 0
    total_processing_time_ms: float = 0.0
    average_latency_ms: float = 0.0
    peak_latency_ms: float = 0.0
    throughput_records_per_second: float = 0.0
    
    # Quality metrics
    data_integrity_score: float = 1.0
    anomaly_preservation_score: float = 1.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    
    # Resource metrics
    memory_usage_mb: float = 0.0
    cpu_utilization_percent: float = 0.0
    cache_hit_rate: float = 0.0
    cache_size_mb: float = 0.0
    
    # Error metrics
    total_errors: int = 0
    error_rate: float = 0.0
    recoverable_errors: int = 0
    fatal_errors: int = 0
    
    # Advanced metrics
    compression_ratio: float = 1.0
    entropy_preserved: float = 1.0
    ml_model_accuracy: float = 0.0
    anomaly_detection_precision: float = 0.0
    anomaly_detection_recall: float = 0.0


@dataclass
class SanitizationResult:
    """Enhanced sanitization result with comprehensive metadata"""
    success: bool
    sanitized_record: Optional[Any] = None
    original_record: Optional[Any] = None
    
    # Operation tracking
    operations_applied: List[str] = field(default_factory=list)
    fields_sanitized: List[str] = field(default_factory=list)
    sanitization_rules_triggered: List[str] = field(default_factory=list)
    
    # Quality assessment
    data_integrity: DataIntegrityLevel = DataIntegrityLevel.INTACT
    anomaly_preservation: AnomalyPreservationStatus = AnomalyPreservationStatus.FULLY_PRESERVED
    
    # Anomaly analysis
    anomalies_detected: List[Dict[str, Any]] = field(default_factory=list)
    anomaly_confidence_scores: Dict[str, float] = field(default_factory=dict)
    preserved_anomaly_signatures: List[str] = field(default_factory=list)
    
    # Performance data
    sanitization_time_ms: float = 0.0
    memory_usage_bytes: int = 0
    cpu_cycles_consumed: int = 0
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    recoverable_errors: List[str] = field(default_factory=list)
    
    # Metadata and provenance
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_chain: List[str] = field(default_factory=list)
    transformation_hash: Optional[str] = None
    
    # Statistical properties
    entropy_before: Optional[float] = None
    entropy_after: Optional[float] = None
    information_content_ratio: Optional[float] = None
    
    # Compliance and audit
    compliance_flags: Dict[str, bool] = field(default_factory=dict)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SanitizationRule:
    """Enhanced sanitization rule with ML capabilities"""
    rule_name: str
    rule_type: SanitizationType
    target_fields: List[str]
    sanitization_function: Callable
    
    # Priority and conditions
    preserve_anomaly: bool = True
    priority: int = 0
    enabled: bool = True
    conditional_execution: Optional[Callable] = None
    
    # Performance optimization
    vectorized: bool = False
    gpu_accelerated: bool = False
    cached: bool = True
    
    # ML integration
    ml_model: Optional[Any] = None
    feature_extractors: List[Callable] = field(default_factory=list)
    anomaly_detectors: List[Callable] = field(default_factory=list)
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    performance_config: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    execution_count: int = 0
    total_execution_time_ms: float = 0.0
    error_count: int = 0
    
    def __lt__(self, other):
        """Compare rules by priority and performance"""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.total_execution_time_ms < other.total_execution_time_ms


@dataclass
class FieldSanitizationProfile:
    """Enhanced field sanitization profile with adaptive learning"""
    field_name: str
    field_type: str
    sanitization_rules: List[SanitizationRule]
    
    # Field constraints
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    allowed_chars: Optional[str] = None
    forbidden_patterns: List[str] = field(default_factory=list)
    
    # Processing preferences
    preserve_case: bool = True
    preserve_special_chars: bool = False
    normalize_whitespace: bool = True
    encoding: str = "utf-8"
    
    # ML-driven adaptation
    learned_patterns: Dict[str, float] = field(default_factory=dict)
    anomaly_patterns: Dict[str, float] = field(default_factory=dict)
    statistical_profile: Dict[str, Any] = field(default_factory=dict)
    
    # Performance optimization
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    cache_strategy: str = "lru"
    cache_size: int = 1000
    
    # Quality metrics
    sanitization_effectiveness: float = 1.0
    anomaly_preservation_rate: float = 1.0
    false_positive_rate: float = 0.0


@dataclass
class AnomalyContext:
    """Enhanced anomaly context with ML-powered analysis"""
    anomaly_type: str
    anomaly_confidence: float
    critical_fields: List[str]
    
    # Pattern analysis
    behavioral_patterns: Dict[str, Any]
    statistical_fingerprint: Dict[str, float]
    temporal_signature: Optional[List[float]] = None
    
    # Preservation strategy
    sanitization_constraints: Dict[str, Any]
    preservation_priority: int
    adaptive_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # ML features
    feature_vector: Optional[np.ndarray] = None
    embedding_vector: Optional[np.ndarray] = None
    cluster_id: Optional[int] = None
    
    # Context metadata
    detection_timestamp: datetime = field(default_factory=datetime.now)
    source_layer: str = "layer1"
    correlation_id: Optional[str] = None


# =============================================================================
# High-Performance Utilities and Decorators
# =============================================================================

def performance_monitor(func):
    """Enhanced performance monitoring decorator"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            result = await func(*args, **kwargs)
            success = True
        except Exception as e:
            result = e
            success = False
        
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss
        
        # Record metrics
        execution_time = (end_time - start_time) * 1000  # Convert to ms
        memory_delta = end_memory - start_memory
        
        if hasattr(args[0], 'performance_metrics'):
            args[0].performance_metrics[func.__name__].append({
                'execution_time_ms': execution_time,
                'memory_delta_bytes': memory_delta,
                'success': success,
                'timestamp': time.time()
            })
        
        if not success:
            raise result
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = e
            success = False
        
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss
        
        execution_time = (end_time - start_time) * 1000
        memory_delta = end_memory - start_memory
        
        if hasattr(args[0], 'performance_metrics'):
            args[0].performance_metrics[func.__name__].append({
                'execution_time_ms': execution_time,
                'memory_delta_bytes': memory_delta,
                'success': success,
                'timestamp': time.time()
            })
        
        if not success:
            raise result
        return result
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def adaptive_cache(maxsize=128, ttl=300):
    """Adaptive caching decorator with TTL and usage patterns"""
    def decorator(func):
        cache = TTLCache(maxsize=maxsize, ttl=ttl)
        stats = {'hits': 0, 'misses': 0, 'errors': 0}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = hash((args, tuple(sorted(kwargs.items()))))
            
            try:
                if key in cache:
                    stats['hits'] += 1
                    return cache[key]
                
                result = func(*args, **kwargs)
                cache[key] = result
                stats['misses'] += 1
                return result
                
            except Exception as e:
                stats['errors'] += 1
                raise
        
        wrapper.cache_info = lambda: stats
        wrapper.cache_clear = cache.clear
        return wrapper
    return decorator


@jit(nopython=True)
def fast_string_entropy(data: str) -> float:
    """High-performance entropy calculation using Numba"""
    char_counts = {}
    for char in data:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    entropy = 0.0
    total_chars = len(data)
    
    for count in char_counts.values():
        probability = count / total_chars
        entropy -= probability * np.log2(probability)
    
    return entropy


@vectorize(['float64(unicode_type)'], target='parallel')
def vectorized_string_length(strings):
    """Vectorized string length calculation"""
    return len(strings)


class BloomFilterCache:
    """Memory-efficient Bloom filter for pattern caching"""
    
    def __init__(self, capacity=100000, error_rate=0.1):
        self.bloom_filter = BloomFilter(capacity=capacity, error_rate=error_rate)
        self.pattern_stats = defaultdict(int)
    
    def add_pattern(self, pattern: str):
        """Add pattern to bloom filter"""
        self.bloom_filter.add(pattern)
        self.pattern_stats[pattern] += 1
    
    def might_contain(self, pattern: str) -> bool:
        """Check if pattern might be in the filter"""
        return pattern in self.bloom_filter
    
    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics"""
        return {
            'capacity': self.bloom_filter.capacity,
            'count': self.bloom_filter.count,
            'error_rate': self.bloom_filter.error_rate,
            'unique_patterns': len(self.pattern_stats)
        }


# =============================================================================
# Enhanced Machine Learning Integration
# =============================================================================

class AnomalyDetectionModel:
    """ML model for anomaly detection during sanitization"""
    
    def __init__(self, model_type='isolation_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_extractors = []
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model"""
        if self.model_type == 'isolation_forest':
            self.model = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
        elif self.model_type == 'dbscan':
            self.model = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def extract_features(self, record: Dict[str, Any]) -> np.ndarray:
        """Extract features from a record for anomaly detection"""
        features = []
        
        # Basic statistical features
        if isinstance(record, dict):
            features.extend([
                len(record),  # Number of fields
                sum(len(str(v)) for v in record.values()),  # Total content length
                len([v for v in record.values() if v is None]),  # Null count
            ])
            
            # String-based features
            text_content = ' '.join(str(v) for v in record.values() if v is not None)
            if text_content:
                features.extend([
                    len(text_content),
                    text_content.count(' '),  # Whitespace count
                    len(set(text_content)),   # Unique characters
                    fast_string_entropy(text_content),  # Entropy
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            # Numeric features
            numeric_values = [v for v in record.values() if isinstance(v, (int, float))]
            if numeric_values:
                features.extend([
                    len(numeric_values),
                    np.mean(numeric_values),
                    np.std(numeric_values),
                    np.max(numeric_values),
                    np.min(numeric_values)
                ])
            else:
                features.extend([0, 0, 0, 0, 0])
        
        return np.array(features)
    
    def train(self, training_data: List[Dict[str, Any]]):
        """Train the anomaly detection model"""
        feature_matrix = np.array([
            self.extract_features(record) for record in training_data
        ])
        
        # Handle missing values
        feature_matrix = np.nan_to_num(feature_matrix)
        
        # Scale features
        feature_matrix = self.scaler.fit_transform(feature_matrix)
        
        # Train model
        self.model.fit(feature_matrix)
        self.is_trained = True
    
    def predict_anomaly(self, record: Dict[str, Any]) -> Tuple[bool, float]:
        """Predict if a record is anomalous"""
        if not self.is_trained:
            return False, 0.0
        
        features = self.extract_features(record).reshape(1, -1)
        features = np.nan_to_num(features)
        features = self.scaler.transform(features)
        
        if self.model_type == 'isolation_forest':
            prediction = self.model.predict(features)[0]
            score = self.model.decision_function(features)[0]
            is_anomaly = prediction == -1
            confidence = abs(score)
        else:  # DBSCAN
            prediction = self.model.fit_predict(features)[0]
            is_anomaly = prediction == -1
            confidence = 0.5  # DBSCAN doesn't provide confidence scores
        
        return is_anomaly, confidence


class PatternLearningEngine:
    """Engine for learning and adapting to data patterns"""
    
    def __init__(self, learning_rate=0.01, decay_rate=0.95):
        self.pattern_frequencies = defaultdict(float)
        self.anomaly_patterns = defaultdict(float)
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.update_count = 0
        
        # TF-IDF for text pattern analysis
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        self.text_patterns_fitted = False
    
    def update_pattern_frequency(self, pattern: str, is_anomaly: bool = False):
        """Update frequency of observed patterns"""
        # Apply temporal decay
        if self.update_count % 1000 == 0:
            self._apply_temporal_decay()
        
        # Update pattern frequency
        self.pattern_frequencies[pattern] += self.learning_rate
        
        if is_anomaly:
            self.anomaly_patterns[pattern] += self.learning_rate
        
        self.update_count += 1
    
    def _apply_temporal_decay(self):
        """Apply temporal decay to pattern frequencies"""
        for pattern in list(self.pattern_frequencies.keys()):
            self.pattern_frequencies[pattern] *= self.decay_rate
            if self.pattern_frequencies[pattern] < 0.001:
                del self.pattern_frequencies[pattern]
        
        for pattern in list(self.anomaly_patterns.keys()):
            self.anomaly_patterns[pattern] *= self.decay_rate
            if self.anomaly_patterns[pattern] < 0.001:
                del self.anomaly_patterns[pattern]
    
    def get_pattern_anomaly_score(self, pattern: str) -> float:
        """Calculate anomaly score for a pattern"""
        total_freq = self.pattern_frequencies.get(pattern, 0.001)
        anomaly_freq = self.anomaly_patterns.get(pattern, 0.0)
        
        if total_freq == 0:
            return 0.5  # Unknown pattern
        
        return anomaly_freq / total_freq
    
    def fit_text_patterns(self, text_samples: List[str]):
        """Fit TF-IDF model on text samples"""
        if len(text_samples) > 0:
            self.tfidf_vectorizer.fit(text_samples)
            self.text_patterns_fitted = True
    
    def analyze_text_similarity(self, text: str, reference_texts: List[str]) -> float:
        """Analyze text similarity using TF-IDF"""
        if not self.text_patterns_fitted or not reference_texts:
            return 0.0
        
        try:
            # Transform texts to TF-IDF vectors
            all_texts = [text] + reference_texts
            tfidf_matrix = self.tfidf_vectorizer.transform(all_texts)
            
            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
            
            return np.mean(similarities)
        except Exception:
            return 0.0


# =============================================================================
# Enhanced Core Sanitization Engine
# =============================================================================

class EnhancedSanitizationEngine:
    """
    Advanced sanitization engine with ML-powered anomaly preservation,
    performance optimization, and adaptive learning capabilities
    """
    
    def __init__(self, 
                 sanitization_level: SanitizationLevel = SanitizationLevel.STANDARD,
                 processing_mode: ProcessingMode = ProcessingMode.ASYNCHRONOUS,
                 optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED):
        """Initialize enhanced sanitization engine"""
        
        self.sanitization_level = sanitization_level
        self.processing_mode = processing_mode
        self.optimization_strategy = optimization_strategy
        
        # Configure logging
        self.logger = logging.getLogger("SCAFAD.Layer1.EnhancedSanitizationEngine")
        self.logger.setLevel(logging.INFO)
        
        # Initialize performance monitoring
        self.performance_metrics = defaultdict(list)
        self.processing_stats = SanitizationMetrics()
        
        # Initialize ML components
        self.anomaly_detector = AnomalyDetectionModel()
        self.pattern_learner = PatternLearningEngine()
        
        # Initialize caching systems
        self.pattern_cache = BloomFilterCache(capacity=100000)
        self.result_cache = TTLCache(maxsize=10000, ttl=300)
        
        # Initialize sanitization functions
        self._initialize_enhanced_sanitizers()
        
        # Initialize preservation strategies
        self._initialize_preservation_strategies()
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.process_pool = ProcessPoolExecutor(max_workers=2)
        
        # Optimization state
        self.optimization_state = {
            'adaptive_thresholds': {},
            'learned_patterns': {},
            'performance_baselines': {},
            'resource_limits': {
                'max_memory_mb': 100,
                'max_cpu_percent': 50,
                'max_latency_ms': 1.0
            }
        }
        
        # Initialize regex patterns with compilation cache
        self._initialize_regex_patterns()
        
        # GPU acceleration (if available)
        self.gpu_available = self._check_gpu_availability()
        
        # Initialize field type inference
        self._initialize_field_type_inference()
    
    def _initialize_enhanced_sanitizers(self):
        """Initialize enhanced sanitization functions with ML capabilities"""
        
        self.sanitizers = {
            # Basic sanitizers (enhanced)
            SanitizationType.WHITESPACE: self._sanitize_whitespace_enhanced,
            SanitizationType.ENCODING: self._sanitize_encoding_enhanced,
            SanitizationType.SPECIAL_CHARS: self._sanitize_special_chars_enhanced,
            SanitizationType.NUMERIC: self._sanitize_numeric_enhanced,
            SanitizationType.TIMESTAMP: self._sanitize_timestamp_enhanced,
            
            # Advanced sanitizers
            SanitizationType.PATH: self._sanitize_path_enhanced,
            SanitizationType.URL: self._sanitize_url_enhanced,
            SanitizationType.HTML: self._sanitize_html_enhanced,
            SanitizationType.SQL: self._sanitize_sql_enhanced,
            SanitizationType.COMMAND: self._sanitize_command_enhanced,
            SanitizationType.UNICODE: self._sanitize_unicode_enhanced,
            SanitizationType.CASE: self._sanitize_case_enhanced,
            SanitizationType.TRUNCATION: self._sanitize_truncation_enhanced,
            
            # ML-enhanced sanitizers
            SanitizationType.PATTERN_ANOMALY: self._sanitize_pattern_anomaly,
            SanitizationType.SEMANTIC_ANALYSIS: self._sanitize_semantic_analysis,
            SanitizationType.BEHAVIORAL_FINGERPRINT: self._sanitize_behavioral_fingerprint,
            SanitizationType.STATISTICAL_OUTLIER: self._sanitize_statistical_outlier,
            SanitizationType.TEMPORAL_SEQUENCE: self._sanitize_temporal_sequence,
            
            # Security-focused sanitizers
            SanitizationType.INJECTION_DETECTION: self._sanitize_injection_detection,
            SanitizationType.HOMOGRAPH_ATTACK: self._sanitize_homograph_attack,
            SanitizationType.STEGANOGRAPHY: self._sanitize_steganography,
            SanitizationType.COVERT_CHANNEL: self._sanitize_covert_channel
        }
    
    def _initialize_preservation_strategies(self):
        """Initialize anomaly preservation strategies with ML enhancement"""
        
        self.preservation_strategies = {
            'pattern_matching': self._preserve_pattern_anomalies_enhanced,
            'statistical': self._preserve_statistical_anomalies_enhanced,
            'structural': self._preserve_structural_anomalies_enhanced,
            'temporal': self._preserve_temporal_anomalies_enhanced,
            'semantic': self._preserve_semantic_anomalies_enhanced,
            'ml_guided': self._preserve_ml_guided_anomalies,
            'adaptive': self._preserve_adaptive_anomalies,
            'behavioral': self._preserve_behavioral_anomalies,
            'contextual': self._preserve_contextual_anomalies
        }
    
    def _initialize_regex_patterns(self):
        """Initialize and compile regex patterns for performance"""
        
        self.compiled_patterns = {
            'email': advanced_regex.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                advanced_regex.IGNORECASE
            ),
            'ip_address': advanced_regex.compile(
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            ),
            'phone': advanced_regex.compile(
                r'(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}'
            ),
            'credit_card': advanced_regex.compile(
                r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
            ),
            'sql_injection': advanced_regex.compile(
                r"(?i)\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b.*?(from|into|where|values)",
                advanced_regex.IGNORECASE | advanced_regex.MULTILINE
            ),
            'xss_pattern': advanced_regex.compile(
                r'(?i)<script[^>]*>.*?</script>|javascript:|on\w+\s*=',
                advanced_regex.IGNORECASE | advanced_regex.DOTALL
            ),
            'path_traversal': advanced_regex.compile(
                r'\.\.[\\/]|\.\.%2[fF]|%2[eE]%2[eE][\\/]'
            ),
            'command_injection': advanced_regex.compile(
                r'[;&|`$(){}[\]<>]|\b(cat|ls|ps|id|pwd|whoami|uname)\b',
                advanced_regex.IGNORECASE
            )
        }
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available"""
        try:
            import cupy
            return cupy.cuda.is_available()
        except ImportError:
            return False
    
    def _initialize_field_type_inference(self):
        """Initialize field type inference system"""
        
        self.field_type_patterns = {
            'timestamp': [
                r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
                r'\d{10,13}',  # Unix timestamps
                r'\d{1,2}[/-]\d{1,2}[/-]\d{4}'
            ],
            'email': [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
            'phone': [r'(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}'],
            'ip_address': [r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'],
            'url': [r'https?://[^\s<>"{}|\\^`[\]]+'],
            'uuid': [r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'],
            'hash': [r'\b[a-f0-9]{32}\b', r'\b[a-f0-9]{40}\b', r'\b[a-f0-9]{64}\b'],
            'json': [r'^\s*[{\[].*[}\]]\s*],
            'base64': [r'^[A-Za-z0-9+/]*={0,2}]
        }
        
        # Compile patterns
        self.compiled_type_patterns = {}
        for field_type, patterns in self.field_type_patterns.items():
            self.compiled_type_patterns[field_type] = [
                advanced_regex.compile(pattern, advanced_regex.IGNORECASE)
                for pattern in patterns
            ]
    
    # =========================================================================
    # Enhanced Sanitization Functions
    # =========================================================================
    
    @performance_monitor
    @adaptive_cache(maxsize=1000, ttl=300)
    async def _sanitize_whitespace_enhanced(self, value: Any, config: Dict[str, Any] = None) -> Any:
        """Enhanced whitespace sanitization with anomaly preservation"""
        if not isinstance(value, str):
            return value
        
        config = config or {}
        original_value = value
        
        # Calculate entropy before processing
        entropy_before = fast_string_entropy(value) if len(value) > 0 else 0.0
        
        # Preserve anomalous whitespace patterns
        anomalous_whitespace = self._detect_anomalous_whitespace(value)
        if anomalous_whitespace and config.get('preserve_anomalies', True):
            # Apply minimal sanitization for anomalous patterns
            value = value.replace('\x00', '')  # Remove only null bytes
            return value
        
        # Standard whitespace normalization
        value = value.replace('\x00', '')  # Remove null bytes
        value = value.replace('\r\n', '\n').replace('\r', '\n')  # Normalize line endings
        
        # Intelligent space normalization
        if config.get('normalize_spaces', True):
            # Preserve intentional spacing patterns in structured data
            if not self._is_structured_text(value):
                value = advanced_regex.sub(r' +', ' ', value)  # Multiple spaces to single
                
                if config.get('tabs_to_spaces', True):
                    value = value.replace('\t', ' ')
        
        # Context-aware trimming
        if config.get('trim', True):
            # Don't trim if it might be significant formatting
            if not self._is_formatted_content(value):
                value = value.strip()
        
        # Remove zero-width characters (potential steganography)
        if config.get('remove_zero_width', True):
            zero_width_chars = [
                '\u200b', '\u200c', '\u200d', '\ufeff', '\u2060',
                '\u2061', '\u2062', '\u2063', '\u2064'  # Additional zero-width chars
            ]
            for char in zero_width_chars:
                value = value.replace(char, '')
        
        # Check if we preserved enough information
        entropy_after = fast_string_entropy(value) if len(value) > 0 else 0.0
        information_loss = (entropy_before - entropy_after) / entropy_before if entropy_before > 0 else 0.0
        
        if information_loss > 0.5 and config.get('preserve_information', True):
            # Too much information lost, revert to minimal sanitization
            return original_value.replace('\x00', '')
        
        return value
    
    def _detect_anomalous_whitespace(self, text: str) -> bool:
        """Detect anomalous whitespace patterns"""
        # Check for excessive whitespace
        whitespace_ratio = len([c for c in text if c.isspace()]) / len(text) if text else 0
        if whitespace_ratio > 0.7:  # More than 70% whitespace
            return True
        
        # Check for unusual whitespace characters
        unusual_whitespace = ['\u2000', '\u2001', '\u2002', '\u2003', '\u2004', 
                             '\u2005', '\u2006', '\u2007', '\u2008', '\u2009', 
                             '\u200a', '\u202f', '\u205f', '\u3000']
        
        for char in unusual_whitespace:
            if char in text:
                return True
        
        # Check for whitespace-based patterns (potential steganography)
        whitespace_sequence = ''.join([c if c.isspace() else 'X' for c in text])
        if len(set(whitespace_sequence)) < len(whitespace_sequence) * 0.3:  # Highly repetitive
            return True
        
        return False
    
    def _is_structured_text(self, text: str) -> bool:
        """Check if text appears to be structured (JSON, XML, etc.)"""
        structured_indicators = ['{', '}', '[', ']', '<', '>', '=', ':', ';']
        indicator_count = sum(1 for char in structured_indicators if char in text)
        return indicator_count >= 3
    
    def _is_formatted_content(self, text: str) -> bool:
        """Check if text appears to be intentionally formatted"""
        # Check for indentation patterns
        lines = text.split('\n')
        if len(lines) > 1:
            indent_pattern = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
            if len(set(indent_pattern)) > 1:  # Variable indentation
                return True
        
        # Check for table-like structures
        if '\t' in text and text.count('\t') > 3:
            return True
        
        return False
    
    @performance_monitor
    async def _sanitize_encoding_enhanced(self, value: Any, config: Dict[str, Any] = None) -> Any:
        """Enhanced encoding sanitization with intelligent detection"""
        if not isinstance(value, (str, bytes)):
            return value
        
        config = config or {}
        target_encoding = config.get('target_encoding', 'utf-8')
        
        if isinstance(value, bytes):
            # Use multiple detection libraries for better accuracy
            detectors = [chardet, cchardet, charset_normalizer]
            detection_results = []
            
            for detector in detectors:
                try:
                    if detector == charset_normalizer:
                        result = detector.detect(value)
                        if result and 'encoding' in result:
                            detection_results.append({
                                'encoding': result['encoding'],
                                'confidence': result.get('confidence', 0.0)
                            })
                    else:
                        result = detector.detect(value)
                        if result and result.get('encoding'):
                            detection_results.append(result)
                except Exception:
                    continue
            
            # Choose best detection result
            if detection_results:
                best_detection = max(detection_results, key=lambda x: x.get('confidence', 0))
                source_encoding = best_detection['encoding']
            else:
                source_encoding = 'utf-8'
            
            try:
                # Decode and re-encode with error handling
                decoded = value.decode(source_encoding, errors='ignore')
                
                # Fix common encoding issues
                decoded = ftfy.fix_text(decoded)
                
                return decoded.encode(target_encoding, errors='ignore').decode(target_encoding)
            except Exception:
                # Fallback to replacement
                return value.decode('utf-8', errors='replace')
        
        else:  # string
            try:
                # Fix text encoding issues
                fixed_text = ftfy.fix_text(value)
                
                # Ensure valid encoding
                return fixed_text.encode(target_encoding, errors='ignore').decode(target_encoding)
            except Exception:
                return value
    
    @performance_monitor
    async def _sanitize_special_chars_enhanced(self, value: Any, config: Dict[str, Any] = None) -> Any:
        """Enhanced special character sanitization with context awareness"""
        if not isinstance(value, str):
            return value
        
        config = config or {}
        
        # Infer field type for context-aware sanitization
        inferred_type = self._infer_field_type(value)
        
        # Adjust allowed characters based on field type
        if inferred_type == 'email':
            allowed_chars = string.ascii_letters + string.digits + '@.-_+'
        elif inferred_type == 'url':
            allowed_chars = string.ascii_letters + string.digits + ':/.-_?&=+%#'
        elif inferred_type == 'path':
            allowed_chars = string.ascii_letters + string.digits + '/\\.-_'
        elif inferred_type == 'json':
            allowed_chars = string.printable  # JSON needs full character set
        else:
            # Default based on sanitization level
            if self.sanitization_level == SanitizationLevel.MINIMAL:
                allowed_chars = string.printable
            elif self.sanitization_level == SanitizationLevel.STANDARD:
                allowed_chars = string.ascii_letters + string.digits + ' .,;:!?-_/\\()[]{}@#$%&*+=~`"\''
            elif self.sanitization_level == SanitizationLevel.AGGRESSIVE:
                allowed_chars = string.ascii_letters + string.digits + ' .,;:!?-_()'
            else:  # PARANOID or ML_ADAPTIVE
                allowed_chars = string.ascii_letters + string.digits + ' '
        
        # Override with custom allowed chars if provided
        if config.get('allowed_chars'):
            allowed_chars = config['allowed_chars']
        
        # Check for anomalous character patterns
        anomalous_chars = self._detect_anomalous_characters(value)
        if anomalous_chars and config.get('preserve_anomalies', True):
            # Apply minimal filtering for anomalous patterns
            forbidden_chars = ['\x00', '\x01', '\x02', '\x03', '\x04', '\x05']
            for char in forbidden_chars:
                value = value.replace(char, '')
            return value
        
        # Filter characters
        if config.get('remove_disallowed', False):
            value = ''.join(c for c in value if c in allowed_chars)
        else:
            placeholder = config.get('placeholder', '')
            value = ''.join(c if c in allowed_chars else placeholder for c in value)
        
        return value
    
    def _infer_field_type(self, value: str) -> str:
        """Infer field type from value content"""
        if not value:
            return 'unknown'
        
        # Check against compiled patterns
        for field_type, patterns in self.compiled_type_patterns.items():
            for pattern in patterns:
                if pattern.search(value):
                    return field_type
        
        # Statistical type inference
        if value.replace('.', '').replace('-', '').isdigit():
            if len(value) >= 10 and len(value) <= 13:
                return 'timestamp'
            return 'numeric'
        
        # Heuristic checks
        if value.startswith(('http://', 'https://', 'ftp://')):
            return 'url'
        
        if '/' in value and not ' ' in value:
            return 'path'
        
        return 'text'
    
    def _detect_anomalous_characters(self, text: str) -> bool:
        """Detect anomalous character patterns"""
        # Check for excessive special characters
        special_char_ratio = len([c for c in text if not c.isalnum() and not c.isspace()]) / len(text) if text else 0
        if special_char_ratio > 0.6:  # More than 60% special characters
            return True
        
        # Check for unusual Unicode categories
        unusual_categories = ['Cc', 'Cf', 'Cs', 'Co', 'Cn']  # Control, format, surrogate, private use, unassigned
        unusual_count = sum(1 for c in text if unicodedata.category(c) in unusual_categories)
        if unusual_count > len(text) * 0.1:  # More than 10% unusual characters
            return True
        
        # Check for potential homograph attacks
        if self._detect_homograph_attack(text):
            return True
        
        return False
    
    def _detect_homograph_attack(self, text: str) -> bool:
        """Detect potential homograph attacks using lookalike characters"""
        # Common homograph pairs
        homograph_sets = [
            {'a', 'а'},  # Latin 'a' vs Cyrillic 'а'
            {'e', 'е'},  # Latin 'e' vs Cyrillic 'е'  
            {'o', 'о'},  # Latin 'o' vs Cyrillic 'о'
            {'p', 'р'},  # Latin 'p' vs Cyrillic 'р'
            {'c', 'с'},  # Latin 'c' vs Cyrillic 'с'
            {'y', 'у'},  # Latin 'y' vs Cyrillic 'у'
            {'x', 'х'},  # Latin 'x' vs Cyrillic 'х'
            {'0', 'О', '0'},  # Digit zero, Latin O, Cyrillic O
            {'1', 'l', 'I', '|'},  # Various forms of 1/l/I
        ]
        
        # Check for mixed scripts that could indicate homograph attack
        scripts = set()
        for char in text:
            if char.isalpha():
                script = unicodedata.name(char, '').split(' ')[0]
                scripts.add(script)
        
        # If we have multiple scripts with lookalike characters, it's suspicious
        if len(scripts) > 1:
            for homograph_set in homograph_sets:
                if any(char in text for char in homograph_set):
                    return True
        
        return False
    
    @performance_monitor
    async def _sanitize_numeric_enhanced(self, value: Any, config: Dict[str, Any] = None) -> Any:
        """Enhanced numeric sanitization with anomaly detection"""
        if not isinstance(value, (int, float, str)):
            return value
        
        config = config or {}
        original_value = value
        
        try:
            # Handle string conversion with anomaly detection
            if isinstance(value, str):
                # Detect anomalous numeric patterns
                if self._detect_anomalous_numeric_pattern(value):
                    if config.get('preserve_anomalies', True):
                        # Minimal sanitization for anomalous patterns
                        return self._minimal_numeric_sanitization(value)
                
                # Advanced parsing for various number formats
                cleaned = self._advanced_numeric_parsing(value)
                if cleaned is None:
                    return config.get('default_value', 0)
                decimal_value = Decimal(str(cleaned))
            else:
                decimal_value = Decimal(str(value))
            
            # Detect statistical anomalies
            if config.get('statistical_validation', False):
                if self._is_statistical_anomaly(float(decimal_value), config):
                    if config.get('preserve_anomalies', True):
                        # Log anomaly but preserve value
                        self.logger.info(f"Statistical anomaly detected in numeric value: {value}")
                        return original_value
            
            # Handle special values
            if decimal_value.is_nan():
                return config.get('nan_replacement', 0)
            if decimal_value.is_infinite():
                if decimal_value > 0:
                    return config.get('inf_replacement', float('inf'))
                else:
                    return config.get('neg_inf_replacement', float('-inf'))
            
            # Apply precision limits with rounding
            if config.get('max_precision'):
                decimal_value = decimal_value.quantize(
                    Decimal(10) ** -config['max_precision'],
                    rounding=ROUND_HALF_UP
                )
            
            # Apply range limits with overflow handling
            if config.get('min_value') is not None:
                min_val = Decimal(str(config['min_value']))
                if decimal_value < min_val:
                    if config.get('clamp_values', True):
                        decimal_value = min_val
                    else:
                        return config.get('underflow_value', min_val)
            
            if config.get('max_value') is not None:
                max_val = Decimal(str(config['max_value']))
                if decimal_value > max_val:
                    if config.get('clamp_values', True):
                        decimal_value = max_val
                    else:
                        return config.get('overflow_value', max_val)
            
            # Convert back to appropriate type
            if isinstance(original_value, int) or config.get('force_integer', False):
                return int(decimal_value)
            else:
                return float(decimal_value)
            
        except (InvalidOperation, ValueError, OverflowError) as e:
            self.logger.warning(f"Numeric sanitization error for value '{value}': {e}")
            return config.get('default_value', 0)
    
    def _detect_anomalous_numeric_pattern(self, text: str) -> bool:
        """Detect anomalous patterns in numeric strings"""
        # Very long numbers might be suspicious
        if len(text) > 50:
            return True
        
        # Numbers with unusual formatting
        if text.count('.') > 1 or text.count('-') > 1:
            return True
        
        # Scientific notation with extreme exponents
        if 'e' in text.lower():
            try:
                parts = text.lower().split('e')
                if len(parts) == 2:
                    exponent = int(parts[1])
                    if abs(exponent) > 308:  # Beyond float64 range
                        return True
            except ValueError:
                pass
        
        # Hexadecimal or other base representations mixed with decimal
        if any(char in text.lower() for char in 'abcdef') and '.' in text:
            return True
        
        return False
    
    def _advanced_numeric_parsing(self, text: str) -> Optional[float]:
        """Advanced parsing for various numeric formats"""
        # Remove common formatting characters
        cleaned = advanced_regex.sub(r'[,\s]', '', text)
        
        # Handle percentage
        if cleaned.endswith('%'):
            try:
                return float(cleaned[:-1]) / 100
            except ValueError:
                pass
        
        # Handle currency symbols
        currency_symbols = [', '€', '£', '¥', '₹', '₽']
        for symbol in currency_symbols:
            cleaned = cleaned.replace(symbol, '')
        
        # Handle scientific notation
        if 'e' in cleaned.lower():
            try:
                return float(cleaned)
            except ValueError:
                pass
        
        # Handle fractions
        if '/' in cleaned and len(cleaned.split('/')) == 2:
            try:
                parts = cleaned.split('/')
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                pass
        
        # Standard float parsing
        try:
            return float(cleaned)
        except ValueError:
            return None
    
    def _minimal_numeric_sanitization(self, value: str) -> str:
        """Apply minimal sanitization to preserve anomalous numeric patterns"""
        # Only remove obviously dangerous characters
        dangerous_chars = ['\x00', '\x01', '\x02', '\x03']
        for char in dangerous_chars:
            value = value.replace(char, '')
        return value
    
    def _is_statistical_anomaly(self, value: float, config: Dict[str, Any]) -> bool:
        """Check if numeric value is a statistical anomaly"""
        if 'expected_range' in config:
            min_val, max_val = config['expected_range']
            if value < min_val or value > max_val:
                return True
        
        if 'expected_mean' in config and 'expected_std' in config:
            mean = config['expected_mean']
            std = config['expected_std']
            z_score = abs((value - mean) / std) if std > 0 else 0
            threshold = config.get('z_score_threshold', 3.0)
            if z_score > threshold:
                return True
        
        return False
    
    @performance_monitor
    async def _sanitize_timestamp_enhanced(self, value: Any, config: Dict[str, Any] = None) -> Any:
        """Enhanced timestamp sanitization with intelligent format detection"""
        if not value:
            return value
        
        config = config or {}
        target_format = config.get('target_format', 'unix')
        original_value = value
        
        try:
            # Detect anomalous timestamp patterns
            if self._detect_anomalous_timestamp(value):
                if config.get('preserve_anomalies', True):
                    self.logger.info(f"Anomalous timestamp detected: {value}")
                    return original_value
            
            # Enhanced timestamp parsing
            dt = self._parse_timestamp_enhanced(value)
            
            if dt is None:
                return config.get('default_value', 0)
            
            # Validate timestamp range
            if self._validate_timestamp_range(dt, config):
                # Apply timezone normalization
                if dt.tzinfo is None:
                    default_tz = config.get('default_timezone', 'UTC')
                    if default_tz == 'UTC':
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        tz = pytz.timezone(default_tz)
                        dt = tz.localize(dt)
                
                # Convert to target timezone if specified
                target_tz = config.get('target_timezone', 'UTC')
                if target_tz != 'UTC':
                    target_tz_obj = pytz.timezone(target_tz)
                    dt = dt.astimezone(target_tz_obj)
                else:
                    dt = dt.astimezone(timezone.utc)
                
                # Format output
                return self._format_timestamp(dt, target_format)
            else:
                # Timestamp outside valid range
                if config.get('clamp_timestamps', False):
                    min_date = config.get('min_date')
                    max_date = config.get('max_date')
                    
                    if min_date and dt < self._parse_timestamp_enhanced(min_date):
                        dt = self._parse_timestamp_enhanced(min_date)
                    elif max_date and dt > self._parse_timestamp_enhanced(max_date):
                        dt = self._parse_timestamp_enhanced(max_date)
                    
                    return self._format_timestamp(dt, target_format)
                else:
                    return config.get('default_value', 0)
            
        except Exception as e:
            self.logger.warning(f"Timestamp sanitization error for value '{value}': {e}")
            return config.get('default_value', 0)
    
    def _detect_anomalous_timestamp(self, value: Any) -> bool:
        """Detect anomalous timestamp patterns"""
        if isinstance(value, (int, float)):
            # Check for unrealistic Unix timestamps
            current_time = time.time()
            
            # Future timestamps (more than 1 year ahead)
            if value > current_time + (365 * 24 * 3600):
                return True
            
            # Very old timestamps (before 1970 or very small values)
            if value < 0 or (value > 0 and value < 86400):  # Before 1970-01-02
                return True
            
            # Microsecond timestamps that are too large
            if value > 9999999999999:  # Beyond reasonable microsecond timestamp
                return True
        
        elif isinstance(value, str):
            # Check for unusual timestamp formats
            if len(value) > 50:  # Extremely long timestamp string
                return True
            
            # Check for multiple datetime components
            date_separators = value.count('-') + value.count('/') + value.count('.')
            time_separators = value.count(':')
            if date_separators > 3 or time_separators > 3:
                return True
        
        return False
    
    def _parse_timestamp_enhanced(self, value: Any) -> Optional[datetime]:
        """Enhanced timestamp parsing with multiple format support"""
        if isinstance(value, datetime):
            return value
        
        elif isinstance(value, (int, float)):
            # Determine if it's seconds, milliseconds, or microseconds
            if value > 10**12:  # Microseconds
                return datetime.fromtimestamp(value / 1000000, tz=timezone.utc)
            elif value > 10**10:  # Milliseconds
                return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
            else:  # Seconds
                return datetime.fromtimestamp(value, tz=timezone.utc)
        
        elif isinstance(value, str):
            # Try multiple parsing strategies
            parsing_strategies = [
                self._parse_iso_format,
                self._parse_rfc_format,
                self._parse_custom_formats,
                self._parse_relative_time,
                date_parser.parse
            ]
            
            for strategy in parsing_strategies:
                try:
                    result = strategy(value)
                    if result:
                        return result
                except Exception:
                    continue
        
        return None
    
    def _parse_iso_format(self, value: str) -> Optional[datetime]:
        """Parse ISO 8601 format timestamps"""
        try:
            # Handle various ISO formats
            iso_patterns = [
                '%Y-%m-%dT%H:%M:%S.%fZ',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%d %H:%M:%S'
            ]
            
            for pattern in iso_patterns:
                try:
                    return datetime.strptime(value, pattern).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            
            return None
        except Exception:
            return None
    
    def _parse_rfc_format(self, value: str) -> Optional[datetime]:
        """Parse RFC format timestamps"""
        try:
            # Common RFC formats
            rfc_patterns = [
                '%a, %d %b %Y %H:%M:%S %Z',
                '%a, %d %b %Y %H:%M:%S',
                '%d %b %Y %H:%M:%S %Z',
                '%d %b %Y %H:%M:%S'
            ]
            
            for pattern in rfc_patterns:
                try:
                    return datetime.strptime(value, pattern)
                except ValueError:
                    continue
            
            return None
        except Exception:
            return None
    
    def _parse_custom_formats(self, value: str) -> Optional[datetime]:
        """Parse custom timestamp formats"""
        try:
            # Common custom formats
            custom_patterns = [
                '%m/%d/%Y %H:%M:%S',
                '%m-%d-%Y %H:%M:%S',
                '%d/%m/%Y %H:%M:%S',
                '%d-%m-%Y %H:%M:%S',
                '%Y%m%d_%H%M%S',
                '%Y%m%d%H%M%S',
                '%m/%d/%Y',
                '%d/%m/%Y',
                '%Y-%m-%d',
                '%Y/%m/%d'
            ]
            
            for pattern in custom_patterns:
                try:
                    return datetime.strptime(value, pattern)
                except ValueError:
                    continue
            
            return None
        except Exception:
            return None
    
    def _parse_relative_time(self, value: str) -> Optional[datetime]:
        """Parse relative time expressions"""
        try:
            value_lower = value.lower().strip()
            now = datetime.now(timezone.utc)
            
            # Handle "now", "today", etc.
            if value_lower in ['now', 'current']:
                return now
            elif value_lower == 'today':
                return now.replace(hour=0, minute=0, second=0, microsecond=0)
            elif value_lower == 'yesterday':
                return (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            elif value_lower == 'tomorrow':
                return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Handle relative expressions like "5 minutes ago", "2 hours from now"
            relative_pattern = advanced_regex.compile(
                r'(\d+)\s*(second|minute|hour|day|week|month|year)s?\s*(ago|from\s+now)',
                advanced_regex.IGNORECASE
            )
            
            match = relative_pattern.search(value)
            if match:
                amount = int(match.group(1))
                unit = match.group(2).lower()
                direction = match.group(3).lower()
                
                # Convert to timedelta
                if unit.startswith('second'):
                    delta = timedelta(seconds=amount)
                elif unit.startswith('minute'):
                    delta = timedelta(minutes=amount)
                elif unit.startswith('hour'):
                    delta = timedelta(hours=amount)
                elif unit.startswith('day'):
                    delta = timedelta(days=amount)
                elif unit.startswith('week'):
                    delta = timedelta(weeks=amount)
                elif unit.startswith('month'):
                    delta = timedelta(days=amount * 30)  # Approximate
                elif unit.startswith('year'):
                    delta = timedelta(days=amount * 365)  # Approximate
                else:
                    return None
                
                if 'ago' in direction:
                    return now - delta
                else:  # from now
                    return now + delta
            
            return None
        except Exception:
            return None
    
    def _validate_timestamp_range(self, dt: datetime, config: Dict[str, Any]) -> bool:
        """Validate timestamp is within acceptable range"""
        min_date = config.get('min_date')
        max_date = config.get('max_date')
        
        # Default reasonable ranges
        if min_date is None:
            min_date = datetime(1970, 1, 1, tzinfo=timezone.utc)  # Unix epoch
        elif isinstance(min_date, str):
            min_date = self._parse_timestamp_enhanced(min_date)
        
        if max_date is None:
            # 10 years in the future
            max_date = datetime.now(timezone.utc) + timedelta(days=3650)
        elif isinstance(max_date, str):
            max_date = self._parse_timestamp_enhanced(max_date)
        
        return min_date <= dt <= max_date
    
    def _format_timestamp(self, dt: datetime, target_format: str) -> Any:
        """Format timestamp according to target format"""
        if target_format == 'unix':
            return int(dt.timestamp())
        elif target_format == 'unix_ms':
            return int(dt.timestamp() * 1000)
        elif target_format == 'unix_us':
            return int(dt.timestamp() * 1000000)
        elif target_format == 'iso':
            return dt.isoformat()
        elif target_format == 'iso_compact':
            return dt.strftime('%Y%m%dT%H%M%SZ')
        elif target_format == 'rfc3339':
            return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        elif target_format == 'human':
            return dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        elif isinstance(target_format, str) and '%' in target_format:
            return dt.strftime(target_format)
        else:
            return dt
    
    # =========================================================================
    # Advanced ML-Enhanced Sanitizers
    # =========================================================================
    
    @performance_monitor
    async def _sanitize_pattern_anomaly(self, value: Any, config: Dict[str, Any] = None) -> Any:
        """ML-powered pattern anomaly detection and sanitization"""
        if not isinstance(value, str):
            return value
        
        config = config or {}
        
        # Extract pattern features
        pattern_features = self._extract_pattern_features(value)
        
        # Check against learned patterns
        anomaly_score = self.pattern_learner.get_pattern_anomaly_score(value)
        
        if anomaly_score > config.get('anomaly_threshold', 0.7):
            # High anomaly score - preserve for detection
            self.logger.info(f"Pattern anomaly detected (score: {anomaly_score:.3f}): {value[:100]}...")
            
            # Update pattern learning
            self.pattern_learner.update_pattern_frequency(value, is_anomaly=True)
            
            # Apply minimal sanitization
            return self._apply_minimal_sanitization(value)
        else:
            # Normal pattern - apply standard sanitization
            self.pattern_learner.update_pattern_frequency(value, is_anomaly=False)
            return await self._apply_standard_sanitization(value, config)
    
    def _extract_pattern_features(self, text: str) -> Dict[str, float]:
        """Extract features for pattern analysis"""
        features = {}
        
        # Basic statistical features
        features['length'] = len(text)
        features['entropy'] = fast_string_entropy(text)
        features['unique_chars'] = len(set(text))
        features['char_variety'] = len(set(text)) / len(text) if text else 0
        
        # Character distribution features
        features['alpha_ratio'] = sum(c.isalpha() for c in text) / len(text) if text else 0
        features['digit_ratio'] = sum(c.isdigit() for c in text) / len(text) if text else 0
        features['space_ratio'] = sum(c.isspace() for c in text) / len(text) if text else 0
        features['punct_ratio'] = sum(not c.isalnum() and not c.isspace() for c in text) / len(text) if text else 0
        
        # Pattern-specific features
        features['repeated_chars'] = self._calculate_repetition_score(text)
        features['sequential_chars'] = self._calculate_sequence_score(text)
        features['case_changes'] = self._count_case_changes(text)
        
        # Advanced linguistic features
        try:
            # Language detection confidence
            lang_prob = langdetect.detect_langs(text)[0].prob if text else 0.0
            features['language_confidence'] = lang_prob
        except:
            features['language_confidence'] = 0.0
        
        return features
    
    def _calculate_repetition_score(self, text: str) -> float:
        """Calculate character repetition score"""
        if len(text) < 2:
            return 0.0
        
        repetitions = 0
        for i in range(len(text) - 1):
            if text[i] == text[i + 1]:
                repetitions += 1
        
        return repetitions / (len(text) - 1)
    
    def _calculate_sequence_score(self, text: str) -> float:
        """Calculate sequential character score (e.g., abc, 123)"""
        if len(text) < 3:
            return 0.0
        
        sequences = 0
        for i in range(len(text) - 2):
            if (ord(text[i + 1]) == ord(text[i]) + 1 and 
                ord(text[i + 2]) == ord(text[i + 1]) + 1):
                sequences += 1
        
        return sequences / (len(text) - 2)
    
    def _count_case_changes(self, text: str) -> int:
        """Count case changes in text"""
        if len(text) < 2:
            return 0
        
        changes = 0
        for i in range(len(text) - 1):
            if (text[i].islower() and text[i + 1].isupper()) or \
               (text[i].isupper() and text[i + 1].islower()):
                changes += 1
        
        return changes
    
    async def _sanitize_semantic_analysis(self, value: Any, config: Dict[str, Any] = None) -> Any:
        """Semantic analysis-based sanitization"""
        if not isinstance(value, str):
            return value
        
        config = config or {}
        
        # Perform semantic analysis
        semantic_features = self._extract_semantic_features(value)
        
        # Check for suspicious semantic patterns
        if self._detect_suspicious_semantics(semantic_features, config):
            self.logger.warning(f"Suspicious semantic pattern detected: {value[:50]}...")
            
            # Apply enhanced sanitization for suspicious content
            return await self._apply_enhanced_sanitization(value, config)
        
        return value
    
    def _extract_semantic_features(self, text: str) -> Dict[str, Any]:
        """Extract semantic features from text"""
        features = {}
        
        # Keyword analysis
        security_keywords = [
            'password', 'secret', 'key', 'token', 'auth', 'credential',
            'admin', 'root', 'execute', 'eval', 'system', 'command'
        ]
        
        attack_keywords = [
            'script', 'alert', 'drop', 'delete', 'union', 'select',
            'insert', 'update', 'exec', 'shell', 'cmd', 'bash'
        ]
        
        text_lower = text.lower()
        features['security_keyword_count'] = sum(1 for keyword in security_keywords if keyword in text_lower)
        features['attack_keyword_count'] = sum(1 for keyword in attack_keywords if keyword in text_lower)
        
        # Pattern matching for common attack vectors
        features['sql_injection_patterns'] = len(self.compiled_patterns['sql_injection'].findall(text))
        features['xss_patterns'] = len(self.compiled_patterns['xss_pattern'].findall(text))
        features['path_traversal_patterns'] = len(self.compiled_patterns['path_traversal'].findall(text))
        features['command_injection_patterns'] = len(self.compiled_patterns['command_injection'].findall(text))
        
        # Encoding analysis
        features['has_encoded_content'] = self._detect_encoded_content(text)
        features['has_obfuscation'] = self._detect_obfuscation(text)
        
        return features
    
    def _detect_suspicious_semantics(self, features: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Detect suspicious semantic patterns"""
        # Check for attack pattern indicators
        attack_indicators = (
            features.get('sql_injection_patterns', 0) > 0 or
            features.get('xss_patterns', 0) > 0 or
            features.get('path_traversal_patterns', 0) > 0 or
            features.get('command_injection_patterns', 0) > 0
        )
        
        # Check for suspicious keyword density
        total_keywords = features.get('security_keyword_count', 0) + features.get('attack_keyword_count', 0)
        suspicious_density = total_keywords > config.get('max_suspicious_keywords', 3)
        
        # Check for encoding/obfuscation
        has_obfuscation = features.get('has_encoded_content', False) or features.get('has_obfuscation', False)
        
        return attack_indicators or suspicious_density or has_obfuscation
    
    def _detect_encoded_content(self, text: str) -> bool:
        """Detect encoded content (Base64, URL encoding, etc.)"""
        # Base64 detection
        base64_pattern = advanced_regex.compile(r'^[A-Za-z0-9+/]*={0,2})
        if len(text) > 10 and len(text) % 4 == 0 and base64_pattern.match(text):
            return True
        
        # URL encoding detection
        url_encoded_pattern = advanced_regex.compile(r'%[0-9A-Fa-f]{2}')
        if len(url_encoded_pattern.findall(text)) > len(text) * 0.1:  # More than 10% URL encoded
            return True
        
        # HTML entity encoding
        html_entity_pattern = advanced_regex.compile(r'&#?\w+;')
        if len(html_entity_pattern.findall(text)) > 3:
            return True
        
        return False
    
    def _detect_obfuscation(self, text: str) -> bool:
        """Detect obfuscation techniques"""
        # Check for excessive escaping
        escape_chars = ['\\', '\'', '"', '%']
        escape_count = sum(text.count(char) for char in escape_chars)
        if escape_count > len(text) * 0.3:  # More than 30% escape characters
            return True
        
        # Check for unusual character mixing
        char_categories = set()
        for char in text:
            cat = unicodedata.category(char)
            char_categories.add(cat)
        
        # Too many different character categories might indicate obfuscation
        if len(char_categories) > 6:
            return True
        
        # Check for concatenation patterns (common in obfuscation)
        concat_patterns = ['+', '||', '&', '${', '#{']
        concat_count = sum(text.count(pattern) for pattern in concat_patterns)
        if concat_count > 5:
            return True
        
        return False
    
    async def _sanitize_behavioral_fingerprint(self, value: Any, config: Dict[str, Any] = None) -> Any:
        """Sanitization based on behavioral fingerprinting"""
        if not isinstance(value, (str, dict, list)):
            return value
        
        config = config or {}
        
        # Generate behavioral fingerprint
        fingerprint = self._generate_behavioral_fingerprint(value)
        
        # Check against known attack fingerprints
        if self._matches_attack_fingerprint(fingerprint, config):
            self.logger.warning(f"Attack behavioral fingerprint detected")
            
            # Preserve for analysis but apply protective sanitization
            return await self._apply_protective_sanitization(value, config)
        
        return value
    
    def _generate_behavioral_fingerprint(self, value: Any) -> Dict[str, Any]:
        """Generate behavioral fingerprint for value"""
        fingerprint = {}
        
        if isinstance(value, str):
            # String-based fingerprint
            fingerprint.update({
                'type': 'string',
                'length': len(value),
                'entropy': fast_string_entropy(value),
                'char_distribution': self._get_char_distribution(value),
                'pattern_signature': self._get_pattern_signature(value)
            })
        
        elif isinstance(value, dict):
            # Dictionary-based fingerprint
            fingerprint.update({
                'type': 'dict',
                'key_count': len(value),
                'key_pattern': sorted(value.keys()),
                'value_types': [type(v).__name__ for v in value.values()],
                'nesting_depth': self._calculate_nesting_depth(value)
            })
        
        elif isinstance(value, list):
            # List-based fingerprint
            fingerprint.update({
                'type': 'list',
                'length': len(value),
                'element_types': [type(item).__name__ for item in value],
                'homogeneity': self._calculate_list_homogeneity(value)
            })
        
        return fingerprint
    
    def _get_char_distribution(self, text: str) -> Dict[str, float]:
        """Get character distribution statistics"""
        if not text:
            return {}
        
        char_counts = Counter(text)
        total_chars = len(text)
        
        return {
            'alpha_ratio': sum(count for char, count in char_counts.items() if char.isalpha()) / total_chars,
            'digit_ratio': sum(count for char, count in char_counts.items() if char.isdigit()) / total_chars,
            'space_ratio': sum(count for char, count in char_counts.items() if char.isspace()) / total_chars,
            'special_ratio': sum(count for char, count in char_counts.items() 
                               if not char.isalnum() and not char.isspace()) / total_chars,
            'unique_char_ratio': len(char_counts) / total_chars,
            'most_frequent_char': char_counts.most_common(1)[0][0] if char_counts else '',
            'char_entropy': fast_string_entropy(text)
        }
    
    def _get_pattern_signature(self, text: str) -> Dict[str, Any]:
        """Get pattern signature for text"""
        signature = {}
        
        # Check for common patterns
        patterns = {
            'email_like': r'\S+@\S+\.\S+',
            'url_like': r'https?://\S+',
            'ip_like': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'hash_like': r'\b[a-f0-9]{32,64}\b',
            'uuid_like': r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            'base64_like': r'^[A-Za-z0-9+/]*={0,2},
            'hex_like': r'^[0-9a-f]+,
            'numeric_like': r'^\d+
        }
        
        for pattern_name, pattern in patterns.items():
            signature[pattern_name] = bool(advanced_regex.search(pattern, text, advanced_regex.IGNORECASE))
        
        return signature
    
    def _calculate_nesting_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth of nested structures"""
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._calculate_nesting_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._calculate_nesting_depth(item, current_depth + 1) for item in obj)
        else:
            return current_depth
    
    def _calculate_list_homogeneity(self, lst: List[Any]) -> float:
        """Calculate homogeneity of list elements"""
        if not lst:
            return 1.0
        
        types = [type(item).__name__ for item in lst]
        type_counts = Counter(types)
        most_common_count = type_counts.most_common(1)[0][1]
        
        return most_common_count / len(lst)
    
    def _matches_attack_fingerprint(self, fingerprint: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Check if fingerprint matches known attack patterns"""
        # Define attack fingerprint patterns
        attack_patterns = config.get('attack_patterns', {
            'sql_injection': {
                'type': 'string',
                'pattern_signature': {'sql_like': True},
                'char_distribution': {'special_ratio': {'min': 0.2}}
            },
            'xss_attack': {
                'type': 'string',
                'pattern_signature': {'html_like': True},
                'char_distribution': {'special_ratio': {'min': 0.15}}
            },
            'command_injection': {
                'type': 'string',
                'char_distribution': {'special_ratio': {'min': 0.3}},
                'entropy': {'min': 3.0}
            }
        })
        
        # Check fingerprint against attack patterns
        for attack_name, attack_pattern in attack_patterns.items():
            if self._fingerprint_matches_pattern(fingerprint, attack_pattern):
                return True
        
        return False
    
    def _fingerprint_matches_pattern(self, fingerprint: Dict[str, Any], pattern: Dict[str, Any]) -> bool:
        """Check if fingerprint matches a specific pattern"""
        for key, expected in pattern.items():
            if key not in fingerprint:
                continue
            
            actual = fingerprint[key]
            
            if isinstance(expected, dict):
                if 'min' in expected and actual < expected['min']:
                    return False
                if 'max' in expected and actual > expected['max']:
                    return False
            else:
                if actual != expected:
                    return False
        
        return True
    
    async def _sanitize_statistical_outlier(self, value: Any, config: Dict[str, Any] = None) -> Any:
        """Statistical outlier detection and sanitization"""
        if not isinstance(value, (int, float, str)):
            return value
        
        config = config or {}
        
        # Convert to numeric if possible
        numeric_value = self._extract_numeric_value(value)
        if numeric_value is None:
            return value
        
        # Check if value is a statistical outlier
        if self._is_statistical_outlier_advanced(numeric_value, config):
            self.logger.info(f"Statistical outlier detected: {value}")
            
            # Preserve outlier for anomaly detection
            if config.get('preserve_outliers', True):
                return value
            else:
                # Apply outlier treatment
                return self._treat_statistical_outlier(numeric_value, config)
        
        return value
    
    def _extract_numeric_value(self, value: Any) -> Optional[float]:
        """Extract numeric value from various input types"""
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Try to extract number from string
            number_match = advanced_regex.search(r'-?\d+\.?\d*', value)
            if number_match:
                try:
                    return float(number_match.group())
                except ValueError:
                    pass
        
        return None
    
    def _is_statistical_outlier_advanced(self, value: float, config: Dict[str, Any]) -> bool:
        """Advanced statistical outlier detection"""
        # Use historical data if available
        historical_data = config.get('historical_data', [])
        
        if len(historical_data) < 10:  # Not enough data for statistical analysis
            return False
        
        # Calculate statistical measures
        mean = np.mean(historical_data)
        std = np.std(historical_data)
        median = np.median(historical_data)
        q1 = np.percentile(historical_data, 25)
        q3 = np.percentile(historical_data, 75)
        iqr = q3 - q1
        
        # Multiple outlier detection methods
        outlier_methods = {
            'z_score': abs((value - mean) / std) > config.get('z_threshold', 3.0) if std > 0 else False,
            'iqr_method': value < (q1 - 1.5 * iqr) or value > (q3 + 1.5 * iqr),
            'modified_z_score': abs(0.6745 * (value - median) / np.median(np.abs(historical_data - median))) > 3.5,
            'percentile_method': value < np.percentile(historical_data, 1) or value > np.percentile(historical_data, 99)
        }
        
        # Require multiple methods to agree for high confidence
        outlier_count = sum(outlier_methods.values())
        confidence_threshold = config.get('outlier_confidence_threshold', 2)
        
        return outlier_count >= confidence_threshold
    
    def _treat_statistical_outlier(self, value: float, config: Dict[str, Any]) -> float:
        """Apply treatment to statistical outlier"""
        treatment_method = config.get('outlier_treatment', 'winsorize')
        historical_data = config.get('historical_data', [])
        
        if not historical_data:
            return value
        
        if treatment_method == 'winsorize':
            # Winsorize to 1st and 99th percentiles
            p1 = np.percentile(historical_data, 1)
            p99 = np.percentile(historical_data, 99)
            return max(p1, min(p99, value))
        
        elif treatment_method == 'cap_iqr':
            # Cap using IQR method
            q1 = np.percentile(historical_data, 25)
            q3 = np.percentile(historical_data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return max(lower_bound, min(upper_bound, value))
        
        elif treatment_method == 'median_replace':
            # Replace with median
            return np.median(historical_data)
        
        else:
            return value
    
    # =========================================================================
    # Utility and Helper Methods
    # =========================================================================
    
    def _apply_minimal_sanitization(self, value: str) -> str:
        """Apply minimal sanitization to preserve anomaly patterns"""
        # Only remove the most dangerous characters
        dangerous_chars = ['\x00', '\x01', '\x02', '\x03', '\x04', '\x05']
        for char in dangerous_chars:
            value = value.replace(char, '')
        return value
    
    async def _apply_standard_sanitization(self, value: str, config: Dict[str, Any]) -> str:
        """Apply standard sanitization"""
        # Apply multiple sanitization steps
        value = await self._sanitize_whitespace_enhanced(value, config)
        value = await self._sanitize_encoding_enhanced(value, config)
        value = await self._sanitize_special_chars_enhanced(value, config)
        return value
    
    async def _apply_enhanced_sanitization(self, value: str, config: Dict[str, Any]) -> str:
        """Apply enhanced sanitization for suspicious content"""
        # More aggressive sanitization
        enhanced_config = config.copy()
        enhanced_config['sanitization_level'] = SanitizationLevel.AGGRESSIVE
        enhanced_config['remove_disallowed'] = True
        enhanced_config['preserve_anomalies'] = False
        
        return await self._apply_standard_sanitization(value, enhanced_config)
    
    async def _apply_protective_sanitization(self, value: Any, config: Dict[str, Any]) -> Any:
        """Apply protective sanitization while preserving forensic value"""
        if isinstance(value, str):
            # Escape dangerous characters but preserve structure
            protective_chars = {
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#x27;',
                '&': '&amp;',
                '/': '&#x2F;'
            }
            
            for char, replacement in protective_chars.items():
                value = value.replace(char, replacement)
        
        return value
    
    # =========================================================================
    # Public Interface Methods
    # =========================================================================
    
    @performance_monitor
    async def sanitize_record(self, record: Any, context: Optional[AnomalyContext] = None) -> SanitizationResult:
        """
        Sanitize a single record with comprehensive analysis
        
        Args:
            record: Record to sanitize
            context: Optional anomaly context for preservation
            
        Returns:
            SanitizationResult with sanitized data and metadata
        """
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss
        
        result = SanitizationResult(success=False, original_record=copy.deepcopy(record))
        
        try:
            # Detect anomalies using ML
            if self.anomaly_detector.is_trained and isinstance(record, dict):
                is_anomaly, confidence = self.anomaly_detector.predict_anomaly(record)
                
                if is_anomaly:
                    result.anomalies_detected.append({
                        'type': 'ml_detected',
                        'confidence': confidence,
                        'detection_method': 'isolation_forest'
                    })
                    result.anomaly_confidence_scores['ml_anomaly'] = confidence
            
            # Apply sanitization based on record type
            if isinstance(record, dict):
                sanitized_record = await self._sanitize_dict_record(record, context)
            elif isinstance(record, list):
                sanitized_record = await self._sanitize_list_record(record, context)
            elif isinstance(record, str):
                sanitized_record = await self._sanitize_string_record(record, context)
            else:
                sanitized_record = record  # No sanitization needed for other types
            
            # Calculate integrity and preservation metrics
            result.sanitized_record = sanitized_record
            result.data_integrity = self._assess_data_integrity(record, sanitized_record)
            result.anomaly_preservation = self._assess_anomaly_preservation(record, sanitized_record, context)
            
            # Calculate entropy changes
            if isinstance(record, str) and isinstance(sanitized_record, str):
                result.entropy_before = fast_string_entropy(record) if record else 0.0
                result.entropy_after = fast_string_entropy(sanitized_record) if sanitized_record else 0.0
                result.information_content_ratio = (
                    result.entropy_after / result.entropy_before 
                    if result.entropy_before > 0 else 1.0
                )
            
            result.success = True
            
        except Exception as e:
            result.error_message = str(e)
            result.sanitized_record = record  # Return original on error
            self.logger.error(f"Sanitization error: {e}", exc_info=True)
        
        finally:
            # Calculate performance metrics
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss
            
            result.sanitization_time_ms = (end_time - start_time) * 1000
            result.memory_usage_bytes = end_memory - start_memory
            
            # Update processing statistics
            self.processing_stats.total_records_processed += 1
            self.processing_stats.total_processing_time_ms += result.sanitization_time_ms
            self.processing_stats.average_latency_ms = (
                self.processing_stats.total_processing_time_ms / 
                self.processing_stats.total_records_processed
            )
            
            if result.sanitization_time_ms > self.processing_stats.peak_latency_ms:
                self.processing_stats.peak_latency_ms = result.sanitization_time_ms
            
            if not result.success:
                self.processing_stats.total_errors += 1
            
            # Generate transformation hash for audit trail
            result.transformation_hash = self._generate_transformation_hash(record, sanitized_record)
        
        return result
    
    async def _sanitize_dict_record(self, record: Dict[str, Any], context: Optional[AnomalyContext]) -> Dict[str, Any]:
        """Sanitize dictionary record with field-specific rules"""
        sanitized = {}
        
        for field_name, value in record.items():
            try:
                # Get field-specific sanitization profile
                field_profile = self._get_field_profile(field_name, value)
                
                # Apply field-specific sanitization
                if value is not None:
                    sanitized_value = await self._sanitize_field_value(field_name, value, field_profile, context)
                    sanitized[field_name] = sanitized_value
                else:
                    sanitized[field_name] = value
                    
            except Exception as e:
                self.logger.warning(f"Error sanitizing field '{field_name}': {e}")
                sanitized[field_name] = value  # Keep original on error
        
        return sanitized
    
    async def _sanitize_list_record(self, record: List[Any], context: Optional[AnomalyContext]) -> List[Any]:
        """Sanitize list record with element-specific handling"""
        sanitized = []
        
        for i, item in enumerate(record):
            try:
                if isinstance(item, dict):
                    sanitized_item = await self._sanitize_dict_record(item, context)
                elif isinstance(item, list):
                    sanitized_item = await self._sanitize_list_record(item, context)
                elif isinstance(item, str):
                    sanitized_item = await self._sanitize_string_record(item, context)
                else:
                    sanitized_item = item
                
                sanitized.append(sanitized_item)
                
            except Exception as e:
                self.logger.warning(f"Error sanitizing list element {i}: {e}")
                sanitized.append(item)  # Keep original on error
        
        return sanitized
    
    async def _sanitize_string_record(self, record: str, context: Optional[AnomalyContext]) -> str:
        """Sanitize string record with comprehensive analysis"""
        # Apply multiple sanitization layers
        sanitized = record
        
        # Layer 1: Basic cleaning
        sanitized = await self._sanitize_whitespace_enhanced(sanitized)
        sanitized = await self._sanitize_encoding_enhanced(sanitized)
        
        # Layer 2: Security sanitization
        sanitized = await self._sanitize_injection_detection(sanitized)
        sanitized = await self._sanitize_unicode_enhanced(sanitized)
        
        # Layer 3: Context-aware sanitization
        if context:
            sanitized = await self._apply_context_aware_sanitization(sanitized, context)
        
        return sanitized
    
    def _get_field_profile(self, field_name: str, value: Any) -> FieldSanitizationProfile:
        """Get or create field sanitization profile"""
        # Check cache first
        cache_key = f"{field_name}_{type(value).__name__}"
        
        if cache_key in self.result_cache:
            return self.result_cache[cache_key]
        
        # Infer field type and create profile
        field_type = self._infer_field_type(str(value)) if value is not None else 'unknown'
        
        # Create sanitization rules based on field type and name
        rules = self._create_field_rules(field_name, field_type, value)
        
        profile = FieldSanitizationProfile(
            field_name=field_name,
            field_type=field_type,
            sanitization_rules=rules,
            optimization_strategy=self.optimization_strategy
        )
        
        # Cache the profile
        self.result_cache[cache_key] = profile
        
        return profile
    
    def _create_field_rules(self, field_name: str, field_type: str, value: Any) -> List[SanitizationRule]:
        """Create sanitization rules for specific field"""
        rules = []
        
        # Common rules for all fields
        if isinstance(value, str):
            rules.append(SanitizationRule(
                rule_name="basic_whitespace",
                rule_type=SanitizationType.WHITESPACE,
                target_fields=[field_name],
                sanitization_function=self._sanitize_whitespace_enhanced,
                priority=100,
                preserve_anomaly=True
            ))
            
            rules.append(SanitizationRule(
                rule_name="encoding_normalization",
                rule_type=SanitizationType.ENCODING,
                target_fields=[field_name],
                sanitization_function=self._sanitize_encoding_enhanced,
                priority=90,
                preserve_anomaly=True
            ))
        
        # Field-type specific rules
        if field_type == 'timestamp':
            rules.append(SanitizationRule(
                rule_name="timestamp_normalization",
                rule_type=SanitizationType.TIMESTAMP,
                target_fields=[field_name],
                sanitization_function=self._sanitize_timestamp_enhanced,
                priority=80,
                preserve_anomaly=True
            ))
        
        elif field_type == 'numeric':
            rules.append(SanitizationRule(
                rule_name="numeric_validation",
                rule_type=SanitizationType.NUMERIC,
                target_fields=[field_name],
                sanitization_function=self._sanitize_numeric_enhanced,
                priority=70,
                preserve_anomaly=True
            ))
        
        elif field_type == 'url':
            rules.append(SanitizationRule(
                rule_name="url_sanitization",
                rule_type=SanitizationType.URL,
                target_fields=[field_name],
                sanitization_function=self._sanitize_url_enhanced,
                priority=60,
                preserve_anomaly=True
            ))
        
        # Field-name specific rules
        if 'password' in field_name.lower() or 'secret' in field_name.lower():
            rules.append(SanitizationRule(
                rule_name="sensitive_data_protection",
                rule_type=SanitizationType.SPECIAL_CHARS,
                target_fields=[field_name],
                sanitization_function=self._sanitize_sensitive_data,
                priority=95,
                preserve_anomaly=False  # Don't preserve passwords
            ))
        
        if 'sql' in field_name.lower() or 'query' in field_name.lower():
            rules.append(SanitizationRule(
                rule_name="sql_injection_protection",
                rule_type=SanitizationType.SQL,
                target_fields=[field_name],
                sanitization_function=self._sanitize_sql_enhanced,
                priority=85,
                preserve_anomaly=True
            ))
        
        # Sort rules by priority
        rules.sort()
        
        return rules
    
    async def _sanitize_field_value(self, field_name: str, value: Any, 
                                  profile: FieldSanitizationProfile, 
                                  context: Optional[AnomalyContext]) -> Any:
        """Sanitize individual field value using profile"""
        sanitized_value = value
        operations_applied = []
        
        # Apply each rule in the profile
        for rule in profile.sanitization_rules:
            if rule.enabled:
                try:
                    # Check if rule should be applied conditionally
                    if rule.conditional_execution and not rule.conditional_execution(sanitized_value):
                        continue
                    
                    # Apply sanitization function
                    config = rule.config.copy()
                    config['preserve_anomalies'] = rule.preserve_anomaly
                    
                    if asyncio.iscoroutinefunction(rule.sanitization_function):
                        sanitized_value = await rule.sanitization_function(sanitized_value, config)
                    else:
                        sanitized_value = rule.sanitization_function(sanitized_value, config)
                    
                    operations_applied.append(rule.rule_name)
                    
                    # Update rule statistics
                    rule.execution_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error applying rule '{rule.rule_name}' to field '{field_name}': {e}")
                    rule.error_count += 1
        
        return sanitized_value
    
    async def _sanitize_sensitive_data(self, value: Any, config: Dict[str, Any] = None) -> Any:
        """Sanitize sensitive data fields"""
        if not isinstance(value, str):
            return value
        
        config = config or {}
        
        # For sensitive fields, apply aggressive sanitization
        if config.get('hash_sensitive', True):
            # Hash the value for audit purposes but preserve detectability
            salt = config.get('salt', 'scafad_layer1')
            return hashlib.sha256(f"{salt}{value}".encode()).hexdigest()[:16]
        else:
            # Mask the value
            if len(value) <= 4:
                return '*' * len(value)
            else:
                return value[:2] + '*' * (len(value) - 4) + value[-2:]
    
    async def _apply_context_aware_sanitization(self, value: str, context: AnomalyContext) -> str:
        """Apply context-aware sanitization based on anomaly context"""
        # Adjust sanitization based on anomaly type
        if context.anomaly_type in ['suspicious', 'malicious']:
            # More conservative sanitization
            return await self._apply_enhanced_sanitization(value, {})
        
        elif context.anomaly_type in ['performance', 'resource']:
            # Preserve performance-related anomalies
            return self._apply_minimal_sanitization(value)
        
        else:
            # Standard sanitization
            return await self._apply_standard_sanitization(value, {})
    
    def _assess_data_integrity(self, original: Any, sanitized: Any) -> DataIntegrityLevel:
        """Assess data integrity after sanitization"""
        if original == sanitized:
            return DataIntegrityLevel.PERFECT
        
        if isinstance(original, str) and isinstance(sanitized, str):
            # Calculate similarity
            similarity = SequenceMatcher(None, original, sanitized).ratio()
            
            if similarity >= 0.99:
                return DataIntegrityLevel.INTACT
            elif similarity >= 0.95:
                return DataIntegrityLevel.MINIMAL_LOSS
            elif similarity >= 0.85:
                return DataIntegrityLevel.MODERATE_LOSS
            elif similarity >= 0.7:
                return DataIntegrityLevel.SIGNIFICANT_LOSS
            else:
                return DataIntegrityLevel.MAJOR_LOSS
        
        # For non-string types, use simple equality check
        return DataIntegrityLevel.INTACT if str(original) == str(sanitized) else DataIntegrityLevel.MINIMAL_LOSS
    
    def _assess_anomaly_preservation(self, original: Any, sanitized: Any, 
                                   context: Optional[AnomalyContext]) -> AnomalyPreservationStatus:
        """Assess how well anomaly characteristics were preserved"""
        if not context:
            return AnomalyPreservationStatus.FULLY_PRESERVED
        
        # Extract anomaly features from original and sanitized data
        original_features = self._extract_anomaly_features(original, context)
        sanitized_features = self._extract_anomaly_features(sanitized, context)
        
        # Calculate preservation score
        preservation_score = self._calculate_feature_preservation(original_features, sanitized_features)
        
        if preservation_score >= 0.995:
            return AnomalyPreservationStatus.PERFECTLY_PRESERVED
        elif preservation_score >= 0.95:
            return AnomalyPreservationStatus.FULLY_PRESERVED
        elif preservation_score >= 0.8:
            return AnomalyPreservationStatus.MOSTLY_PRESERVED
        elif preservation_score >= 0.5:
            return AnomalyPreservationStatus.PARTIALLY_PRESERVED
        elif preservation_score >= 0.2:
            return AnomalyPreservationStatus.MINIMALLY_PRESERVED
        else:
            return AnomalyPreservationStatus.NOT_PRESERVED
    
    def _extract_anomaly_features(self, data: Any, context: AnomalyContext) -> Dict[str, float]:
        """Extract features relevant to anomaly detection"""
        features = {}
        
        if isinstance(data, str):
            features.update({
                'length': len(data),
                'entropy': fast_string_entropy(data),
                'char_variety': len(set(data)) / len(data) if data else 0,
                'special_char_ratio': sum(not c.isalnum() and not c.isspace() for c in data) / len(data) if data else 0
            })
        
        elif isinstance(data, dict):
            features.update({
                'key_count': len(data),
                'value_count': len([v for v in data.values() if v is not None]),
                'nesting_depth': self._calculate_nesting_depth(data),
                'key_diversity': len(set(str(k) for k in data.keys())) / len(data) if data else 0
            })
        
        elif isinstance(data, list):
            features.update({
                'length': len(data),
                'type_diversity': len(set(type(item).__name__ for item in data)) / len(data) if data else 0,
                'homogeneity': self._calculate_list_homogeneity(data)
            })
        
        # Add context-specific features
        if context.statistical_fingerprint:
            features.update(context.statistical_fingerprint)
        
        return features
    
    def _calculate_feature_preservation(self, original_features: Dict[str, float], 
                                      sanitized_features: Dict[str, float]) -> float:
        """Calculate how well features were preserved"""
        if not original_features:
            return 1.0
        
        preservation_scores = []
        
        for feature_name, original_value in original_features.items():
            if feature_name in sanitized_features:
                sanitized_value = sanitized_features[feature_name]
                
                if original_value == 0:
                    # Handle zero values
                    score = 1.0 if sanitized_value == 0 else 0.0
                else:
                    # Calculate relative preservation
                    score = 1.0 - abs(original_value - sanitized_value) / abs(original_value)
                    score = max(0.0, score)  # Ensure non-negative
                
                preservation_scores.append(score)
        
        # Return average preservation score
        return np.mean(preservation_scores) if preservation_scores else 0.0
    
    def _generate_transformation_hash(self, original: Any, sanitized: Any) -> str:
        """Generate hash for transformation audit trail"""
        transformation_data = {
            'original_hash': hashlib.sha256(str(original).encode()).hexdigest(),
            'sanitized_hash': hashlib.sha256(str(sanitized).encode()).hexdigest(),
            'timestamp': time.time(),
            'sanitization_level': self.sanitization_level.value
        }
        
        transformation_str = json.dumps(transformation_data, sort_keys=True)
        return hashlib.sha256(transformation_str.encode()).hexdigest()[:16]
    
    @performance_monitor
    async def sanitize_batch(self, records: List[Any], 
                           context: Optional[AnomalyContext] = None) -> List[SanitizationResult]:
        """
        Sanitize a batch of records with optimized processing
        
        Args:
            records: List of records to sanitize
            context: Optional anomaly context
            
        Returns:
            List of SanitizationResult objects
        """
        if not records:
            return []
        
        # Choose processing strategy based on batch size and mode
        if len(records) < 10 or self.processing_mode == ProcessingMode.SYNCHRONOUS:
            # Sequential processing for small batches
            results = []
            for record in records:
                result = await self.sanitize_record(record, context)
                results.append(result)
            return results
        
        elif self.processing_mode == ProcessingMode.ASYNCHRONOUS:
            # Async concurrent processing
            tasks = [self.sanitize_record(record, context) for record in records]
            return await asyncio.gather(*tasks, return_exceptions=False)
        
        elif self.processing_mode == ProcessingMode.PARALLEL:
            # Thread pool processing
            with ThreadPoolExecutor(max_workers=4) as executor:
                loop = asyncio.get_event_loop()
                tasks = [
                    loop.run_in_executor(executor, self._sync_sanitize_record, record, context)
                    for record in records
                ]
                return await asyncio.gather(*tasks)
        
        else:
            # Default to async processing
            tasks = [self.sanitize_record(record, context) for record in records]
            return await asyncio.gather(*tasks, return_exceptions=False)
    
    def _sync_sanitize_record(self, record: Any, context: Optional[AnomalyContext]) -> SanitizationResult:
        """Synchronous wrapper for sanitize_record"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.sanitize_record(record, context))
        finally:
            loop.close()
    
    # =========================================================================
    # Performance Monitoring and Optimization
    # =========================================================================
    
    def get_performance_metrics(self) -> SanitizationMetrics:
        """Get current performance metrics"""
        # Update throughput calculation
        if self.processing_stats.total_processing_time_ms > 0:
            self.processing_stats.throughput_records_per_second = (
                self.processing_stats.total_records_processed * 1000 / 
                self.processing_stats.total_processing_time_ms
            )
        
        # Update error rate
        if self.processing_stats.total_records_processed > 0:
            self.processing_stats.error_rate = (
                self.processing_stats.total_errors / 
                self.processing_stats.total_records_processed
            )
        
        # Update resource metrics
        process = psutil.Process()
        self.processing_stats.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        self.processing_stats.cpu_utilization_percent = process.cpu_percent()
        
        # Update cache metrics
        self.processing_stats.cache_hit_rate = (
            self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0 else 0.0
        )
        
        return self.processing_stats
    
    def optimize_performance(self):
        """Optimize performance based on current metrics and workload"""
        metrics = self.get_performance_metrics()
        
        # Optimize caching based on hit rate
        if metrics.cache_hit_rate < 0.5:
            # Low hit rate - increase cache size
            if hasattr(self.result_cache, 'maxsize'):
                new_size = min(self.result_cache.maxsize * 2, 50000)
                self.result_cache = TTLCache(maxsize=new_size, ttl=300)
        
        # Optimize processing mode based on throughput
        if metrics.throughput_records_per_second < 1000:
            if self.processing_mode == ProcessingMode.SYNCHRONOUS:
                self.processing_mode = ProcessingMode.ASYNCHRONOUS
                self.logger.info("Switched to asynchronous processing for better throughput")
        
        # Adjust sanitization level based on error rate
        if metrics.error_rate > 0.05:  # More than 5% errors
            if self.sanitization_level == SanitizationLevel.PARANOID:
                self.sanitization_level = SanitizationLevel.AGGRESSIVE
                self.logger.info("Reduced sanitization level due to high error rate")
        
        # Memory management
        if metrics.memory_usage_mb > 500:  # More than 500MB
            gc.collect()  # Force garbage collection
            self.result_cache.clear()  # Clear cache
            self.logger.info("Cleared caches due to high memory usage")
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.processing_stats = SanitizationMetrics()
        self.performance_metrics.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    # =========================================================================
    # Configuration and Management
    # =========================================================================
    
    def update_configuration(self, **kwargs):
        """Update engine configuration"""
        if 'sanitization_level' in kwargs:
            self.sanitization_level = SanitizationLevel(kwargs['sanitization_level'])
        
        if 'processing_mode' in kwargs:
            self.processing_mode = ProcessingMode(kwargs['processing_mode'])
        
        if 'optimization_strategy' in kwargs:
            self.optimization_strategy = OptimizationStrategy(kwargs['optimization_strategy'])
        
        # Reinitialize components if needed
        if any(key in kwargs for key in ['sanitization_level', 'processing_mode']):
            self._initialize_enhanced_sanitizers()
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status"""
        metrics = self.get_performance_metrics()
        
        return {
            'sanitization_level': self.sanitization_level.name,
            'processing_mode': self.processing_mode.name,
            'optimization_strategy': self.optimization_strategy.name,
            'ml_model_trained': self.anomaly_detector.is_trained,
            'performance_metrics': asdict(metrics),
            'cache_stats': {
                'pattern_cache_stats': self.pattern_cache.get_stats(),
                'result_cache_size': len(self.result_cache),
                'cache_hit_rate': metrics.cache_hit_rate
            },
            'optimization_state': self.optimization_state
        }
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export current configuration for persistence"""
        return {
            'sanitization_level': self.sanitization_level.value,
            'processing_mode': self.processing_mode.value,
            'optimization_strategy': self.optimization_strategy.value,
            'optimization_state': self.optimization_state,
            'pattern_learning_state': {
                'pattern_frequencies': dict(self.pattern_learner.pattern_frequencies),
                'anomaly_patterns': dict(self.pattern_learner.anomaly_patterns),
                'update_count': self.pattern_learner.update_count
            }
        }
    
    def import_configuration(self, config: Dict[str, Any]):
        """Import configuration from saved state"""
        if 'sanitization_level' in config:
            self.sanitization_level = SanitizationLevel(config['sanitization_level'])
        
        if 'processing_mode' in config:
            self.processing_mode = ProcessingMode(config['processing_mode'])
        
        if 'optimization_strategy' in config:
            self.optimization_strategy = OptimizationStrategy(config['optimization_strategy'])
        
        if 'optimization_state' in config:
            self.optimization_state.update(config['optimization_state'])
        
        if 'pattern_learning_state' in config:
            state = config['pattern_learning_state']
            self.pattern_learner.pattern_frequencies.update(state.get('pattern_frequencies', {}))
            self.pattern_learner.anomaly_patterns.update(state.get('anomaly_patterns', {}))
            self.pattern_learner.update_count = state.get('update_count', 0)
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=True)
        
        # Clear caches
        self.result_cache.clear()
        self.pattern_cache = BloomFilterCache()
        
        # Force garbage collection
        gc.collect()


# =============================================================================
# Factory and Convenience Functions
# =============================================================================

def create_sanitization_engine(
    level: SanitizationLevel = SanitizationLevel.STANDARD,
    mode: ProcessingMode = ProcessingMode.ASYNCHRONOUS,
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
) -> EnhancedSanitizationEngine:
    """
    Factory function to create configured sanitization engine
    
    Args:
        level: Sanitization intensity level
        mode: Processing execution mode
        strategy: Optimization strategy
        
    Returns:
        Configured EnhancedSanitizationEngine instance
    """
    return EnhancedSanitizationEngine(
        sanitization_level=level,
        processing_mode=mode,
        optimization_strategy=strategy
    )


async def sanitize_telemetry_batch(
    records: List[Any],
    level: SanitizationLevel = SanitizationLevel.STANDARD,
    preserve_anomalies: bool = True
) -> List[SanitizationResult]:
    """
    Convenience function for batch sanitization
    
    Args:
        records: List of telemetry records to sanitize
        level: Sanitization level to apply
        preserve_anomalies: Whether to preserve anomaly patterns
        
    Returns:
        List of sanitization results
    """
    engine = create_sanitization_engine(level=level)
    
    # Configure anomaly preservation
    context = AnomalyContext(
        anomaly_type='unknown',
        anomaly_confidence=0.5,
        critical_fields=[],
        behavioral_patterns={},
        sanitization_constraints={'preserve_anomalies': preserve_anomalies},
        preservation_priority=80 if preserve_anomalies else 20
    ) if preserve_anomalies else None
    
    results = await engine.sanitize_batch(records, context)
    
    # Cleanup
    engine.cleanup()
    
    return results


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Create sanitization engine
        engine = create_sanitization_engine(
            level=SanitizationLevel.STANDARD,
            mode=ProcessingMode.ASYNCHRONOUS,
            strategy=OptimizationStrategy.BALANCED
        )
        
        # Sample test data
        test_records = [
            {
                "timestamp": "2024-01-15T10:30:00Z",
                "user_id": "user123",
                "action": "login",
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
            {
                "query": "SELECT * FROM users WHERE id = 1; DROP TABLE users;--",
                "suspicious_payload": "<script>alert('xss')</script>",
                "large_number": 999999999999999999999
            },
            "Simple string record with üñíçödé characters",
            ["list", "of", "various", "items", 123, None]
        ]
        
        print("Starting sanitization test...")
        
        # Sanitize batch
        results = await engine.sanitize_batch(test_records)
        
        # Display results
        for i, result in enumerate(results):
            print(f"\n--- Record {i+1} ---")
            print(f"Success: {result.success}")
            print(f"Data Integrity: {result.data_integrity.value}")
            print(f"Anomaly Preservation: {result.anomaly_preservation.value}")
            print(f"Processing Time: {result.sanitization_time_ms:.2f}ms")
            print(f"Operations Applied: {result.operations_applied}")
            
            if result.anomalies_detected:
                print(f"Anomalies Detected: {len(result.anomalies_detected)}")
                for anomaly in result.anomalies_detected:
                    print(f"  - {anomaly['type']}: {anomaly.get('confidence', 0):.3f}")
        
        # Display performance metrics
        metrics = engine.get_performance_metrics()
        print(f"\n--- Performance Metrics ---")
        print(f"Total Records Processed: {metrics.total_records_processed}")
        print(f"Average Latency: {metrics.average_latency_ms:.2f}ms")
        print(f"Peak Latency: {metrics.peak_latency_ms:.2f}ms")
        print(f"Throughput: {metrics.throughput_records_per_second:.1f} records/sec")
        print(f"Memory Usage: {metrics.memory_usage_mb:.1f}MB")
        print(f"Cache Hit Rate: {metrics.cache_hit_rate:.1%}")
        print(f"Error Rate: {metrics.error_rate:.1%}")
        
        # Test advanced features
        print(f"\n--- Testing Advanced Features ---")
        
        # Test anomaly detection training
        training_data = [
            {"normal_field": "normal_value", "count": 100},
            {"normal_field": "another_normal", "count": 95},
            {"normal_field": "standard_data", "count": 105},
            {"suspicious_field": "'; DROP TABLE users;--", "count": 999999}
        ]
        
        engine.anomaly_detector.train(training_data)
        print("✓ ML anomaly detector trained")
        
        # Test pattern learning
        patterns = [
            "normal_user_login",
            "standard_api_call", 
            "typical_database_query",
            "'; DELETE FROM sensitive_table; --"
        ]
        
        for pattern in patterns:
            is_anomaly = "DELETE" in pattern or "DROP" in pattern
            engine.pattern_learner.update_pattern_frequency(pattern, is_anomaly)
        
        print("✓ Pattern learning updated")
        
        # Test performance optimization
        engine.optimize_performance()
        print("✓ Performance optimization applied")
        
        # Export configuration
        config = engine.export_configuration()
        print(f"✓ Configuration exported ({len(config)} keys)")
        
        # Cleanup
        engine.cleanup()
        print("✓ Engine cleanup completed")
        
        print("\n=== Sanitization Test Completed ===")
    
    # Run the test
    import asyncio
    asyncio.run(main())


# =============================================================================
# Advanced Security-Focused Sanitizers (Continued Implementation)
# =============================================================================

class SecuritySanitizers:
    """Advanced security-focused sanitization methods"""
    
    @staticmethod
    @performance_monitor
    async def _sanitize_injection_detection(value: Any, config: Dict[str, Any] = None) -> Any:
        """Enhanced injection attack detection and sanitization"""
        if not isinstance(value, str):
            return value
        
        config = config or {}
        
        # Multi-layered injection detection
        injection_patterns = {
            'sql_injection': [
                r"(?i)\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b.*?(from|into|where|values)",
                r"(?i)'.*?(\bor\b|\band\b).*?'",
                r"(?i)\b(char|ascii|substring|length)\s*\(",
                r"(?i)(--|\/\*|\*\/|;)",
                r"(?i)\b(xp_|sp_)\w+",
                r"(?i)(\bwaitfor\b|\bdelay\b).*?'\d+:\d+:\d+'"
            ],
            'xss_injection': [
                r"(?i)<script[^>]*>.*?</script>",
                r"(?i)javascript:",
                r"(?i)on\w+\s*=",
                r"(?i)<iframe[^>]*>",
                r"(?i)<object[^>]*>",
                r"(?i)<embed[^>]*>",
                r"(?i)<applet[^>]*>",
                r"(?i)<meta[^>]*http-equiv",
                r"(?i)eval\s*\(",
                r"(?i)expression\s*\("
            ],
            'command_injection': [
                r"[;&|`$]",
                r"\$\([^)]*\)",
                r"\${[^}]*}",
                r"(?i)\b(cat|ls|ps|id|pwd|whoami|uname|wget|curl|nc|netcat)\b",
                r"(?i)\b(rm|mv|cp|chmod|chown|kill|killall)\b",
                r"(?i)(/bin/|/usr/bin/|/sbin/|cmd\.exe|powershell)",
                r"(?i)(>|>>|<|2>&1|\|)"
            ],
            'ldap_injection': [
                r"[()&|!]",
                r"\*\w*\*",
                r"(?i)(\)\(|\(\|)",
                r"(?i)(objectclass|cn=|ou=|dc=)"
            ],
            'xpath_injection': [
                r"(?i)(\bor\b|\band\b).*?=.*?(\bor\b|\band\b)",
                r"'.*?\[.*?\].*?'",
                r"(?i)(count|string|substring|concat)\s*\(",
                r"(?i)(\@\@|\/\/|\|\|)"
            ],
            'nosql_injection': [
                r"(?i)\$where\s*:",
                r"(?i)\$(gt|gte|lt|lte|ne|in|nin|exists|regex)\s*:",
                r"(?i)this\.\w+",
                r"(?i)sleep\s*\(",
                r"(?i)function\s*\(\s*\)\s*\{",
                r"(?i)return\s+true"
            ]
        }
        
        detected_attacks = []
        threat_score = 0.0
        
        # Check against all injection patterns
        for attack_type, patterns in injection_patterns.items():
            for pattern in patterns:
                matches = advanced_regex.findall(pattern, value)
                if matches:
                    detected_attacks.append({
                        'type': attack_type,
                        'pattern': pattern,
                        'matches': len(matches),
                        'evidence': matches[:3]  # First 3 matches as evidence
                    })
                    threat_score += len(matches) * 0.2
        
        # Advanced context-aware detection
        threat_score += SecuritySanitizers._analyze_injection_context(value)
        
        # Apply appropriate sanitization based on threat level
        if threat_score > config.get('high_threat_threshold', 1.0):
            return await SecuritySanitizers._apply_aggressive_injection_sanitization(value, detected_attacks, config)
        elif threat_score > config.get('medium_threat_threshold', 0.5):
            return await SecuritySanitizers._apply_moderate_injection_sanitization(value, detected_attacks, config)
        elif detected_attacks:
            return await SecuritySanitizers._apply_minimal_injection_sanitization(value, detected_attacks, config)
        
        return value
    
    @staticmethod
    def _analyze_injection_context(value: str) -> float:
        """Analyze injection context for additional threat indicators"""
        context_score = 0.0
        
        # Check for encoding evasion attempts
        if '%' in value:
            url_encoded_count = len(advanced_regex.findall(r'%[0-9A-Fa-f]{2}', value))
            if url_encoded_count > 3:
                context_score += 0.3
        
        # Check for double encoding
        if '%%' in value:
            context_score += 0.4
        
        # Check for HTML entity evasion
        html_entities = len(advanced_regex.findall(r'&#?\w+;', value))
        if html_entities > 2:
            context_score += 0.2
        
        # Check for Unicode evasion
        unicode_escapes = len(advanced_regex.findall(r'\\u[0-9a-fA-F]{4}', value))
        if unicode_escapes > 1:
            context_score += 0.3
        
        # Check for concatenation patterns (evasion technique)
        concat_patterns = ['+', '||', 'CONCAT', 'CHR(', 'CHAR(']
        concat_score = sum(value.upper().count(pattern) for pattern in concat_patterns)
        if concat_score > 2:
            context_score += 0.4
        
        # Check for time-based attack patterns
        time_patterns = ['SLEEP(', 'WAITFOR', 'BENCHMARK(', 'pg_sleep(']
        for pattern in time_patterns:
            if pattern.upper() in value.upper():
                context_score += 0.5
        
        # Check for information disclosure patterns
        info_patterns = ['@@VERSION', 'USER()', 'DATABASE()', 'SCHEMA()', 'information_schema']
        for pattern in info_patterns:
            if pattern.upper() in value.upper():
                context_score += 0.4
        
        return min(context_score, 2.0)  # Cap at 2.0
    
    @staticmethod
    async def _apply_aggressive_injection_sanitization(value: str, attacks: List[Dict], config: Dict) -> str:
        """Apply aggressive sanitization for high-threat injections"""
        sanitized = value
        
        # Remove all detected malicious patterns
        for attack in attacks:
            pattern = attack['pattern']
            sanitized = advanced_regex.sub(pattern, '[BLOCKED]', sanitized, flags=advanced_regex.IGNORECASE)
        
        # Additional aggressive measures
        dangerous_chars = ['\'', '"', ';', '--', '/*', '*/', '<', '>', '&', '|', '`', '
                ]
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Remove common SQL functions
        sql_functions = ['EXEC', 'EXECUTE', 'EVAL', 'SCRIPT', 'OBJECT', 'EMBED']
        for func in sql_functions:
            sanitized = advanced_regex.sub(rf'\b{func}\b', '[BLOCKED]', sanitized, flags=advanced_regex.IGNORECASE)
        
        return sanitized
    
    @staticmethod
    async def _apply_moderate_injection_sanitization(value: str, attacks: List[Dict], config: Dict) -> str:
        """Apply moderate sanitization for medium-threat injections"""
        sanitized = value
        
        # Escape dangerous characters
        escape_map = {
            "'": "&#x27;",
            '"': "&quot;",
            '<': "&lt;",
            '>': "&gt;",
            '&': "&amp;",
            ';': "&#x3B;",
            '--': "&#x2D;&#x2D;"
        }
        
        for char, escaped in escape_map.items():
            sanitized = sanitized.replace(char, escaped)
        
        # Replace detected patterns with safe alternatives
        for attack in attacks:
            if attack['type'] == 'sql_injection':
                sanitized = advanced_regex.sub(r'\b(union|select|drop)\b', '[SANITIZED]', 
                                             sanitized, flags=advanced_regex.IGNORECASE)
            elif attack['type'] == 'xss_injection':
                sanitized = advanced_regex.sub(r'<script[^>]*>.*?</script>', '', 
                                             sanitized, flags=advanced_regex.IGNORECASE | advanced_regex.DOTALL)
        
        return sanitized
    
    @staticmethod
    async def _apply_minimal_injection_sanitization(value: str, attacks: List[Dict], config: Dict) -> str:
        """Apply minimal sanitization while preserving content for analysis"""
        sanitized = value
        
        # Only escape the most critical characters
        critical_escapes = {
            '<script': '&lt;script',
            'javascript:': 'javascript&#x3A;',
            '; DROP ': '&#x3B; DROP ',
            '; DELETE ': '&#x3B; DELETE '
        }
        
        for pattern, replacement in critical_escapes.items():
            sanitized = advanced_regex.sub(pattern, replacement, sanitized, flags=advanced_regex.IGNORECASE)
        
        return sanitized
    
    @staticmethod
    @performance_monitor
    async def _sanitize_homograph_attack(value: Any, config: Dict[str, Any] = None) -> Any:
        """Detect and sanitize homograph attacks"""
        if not isinstance(value, str):
            return value
        
        config = config or {}
        
        # Extended homograph detection
        homograph_sets = {
            # Latin vs Cyrillic
            'a': ['а', 'ɑ', 'α'],  # Latin a vs Cyrillic а, etc.
            'e': ['е', 'ε', 'ℯ'],
            'o': ['о', 'ο', '᧐', '߀'],
            'p': ['р', 'ρ', '𝗉'],
            'c': ['с', 'ϲ', '𝖼'],
            'y': ['у', 'γ', '𝗒'],
            'x': ['х', 'χ', '𝗑'],
            'i': ['і', 'ι', '𝗂', '1', '|', 'l'],
            'B': ['В', 'Β', '𝐁'],
            'H': ['Н', 'Η', '𝐇'],
            'K': ['К', 'Κ', '𝐊'],
            'M': ['М', 'Μ', '𝐌'],
            'P': ['Р', 'Ρ', '𝐏'],
            'T': ['Т', 'Τ', '𝐓'],
            'X': ['Х', 'Χ', '𝐗'],
            
            # Digits and similar
            '0': ['О', 'Ο', '𝟎', '𝟘', '𝟢', '𝟬', '𝟶'],
            '1': ['l', 'I', '|', '1', '𝟏', '𝟙', '𝟣', '𝟭', '𝟷'],
            '2': ['Ƨ', '𝟐', '𝟚', '𝟤', '𝟮', '𝟸'],
            '3': ['Ʒ', 'З', '𝟑', '𝟛', '𝟥', '𝟯', '𝟹'],
            '5': ['Ƽ', 'Ϛ', '𝟓', '𝟝', '𝟧', '𝟱', '𝟻'],
            '6': ['б', 'Ϭ', '𝟔', '𝟞', '𝟨', '𝟲', '𝟼'],
            '8': ['Ȣ', '𝟖', '𝟠', '𝟪', '𝟴', '𝟾'],
            '9': ['Ϲ', '𝟗', '𝟡', '𝟫', '𝟵', '𝟿']
        }
        
        # Detect homograph usage
        homograph_score = 0.0
        suspicious_chars = []
        
        for char in value:
            for normal_char, homographs in homograph_sets.items():
                if char in homographs:
                    homograph_score += 1.0
                    suspicious_chars.append((char, normal_char))
        
        # Calculate suspicion ratio
        if len(value) > 0:
            suspicion_ratio = homograph_score / len(value)
            
            # High homograph usage is suspicious
            if suspicion_ratio > config.get('homograph_threshold', 0.3):
                # Apply normalization
                normalized = SecuritySanitizers._normalize_homographs(value, homograph_sets)
                
                # Log the detection
                logging.getLogger("SCAFAD.Layer1.SecuritySanitizers").warning(
                    f"Homograph attack detected: {len(suspicious_chars)} suspicious characters, "
                    f"ratio: {suspicion_ratio:.2%}"
                )
                
                return normalized
        
        return value
    
    @staticmethod
    def _normalize_homographs(text: str, homograph_sets: Dict[str, List[str]]) -> str:
        """Normalize homographs to standard characters"""
        normalized = text
        
        # Create reverse mapping
        homograph_to_normal = {}
        for normal_char, homographs in homograph_sets.items():
            for homograph in homographs:
                homograph_to_normal[homograph] = normal_char
        
        # Replace homographs
        for char in text:
            if char in homograph_to_normal:
                normalized = normalized.replace(char, homograph_to_normal[char])
        
        return normalized
    
    @staticmethod
    @performance_monitor
    async def _sanitize_steganography(value: Any, config: Dict[str, Any] = None) -> Any:
        """Detect and sanitize potential steganography"""
        if not isinstance(value, str):
            return value
        
        config = config or {}
        stego_indicators = []
        
        # Check for hidden Unicode characters
        hidden_chars = [
            '\u200b',  # Zero-width space
            '\u200c',  # Zero-width non-joiner
            '\u200d',  # Zero-width joiner
            '\ufeff',  # Zero-width no-break space
            '\u2060',  # Word joiner
            '\u2061',  # Function application
            '\u2062',  # Invisible times
            '\u2063',  # Invisible separator
            '\u2064',  # Invisible plus
        ]
        
        hidden_char_count = sum(value.count(char) for char in hidden_chars)
        if hidden_char_count > 0:
            stego_indicators.append(f"Hidden Unicode chars: {hidden_char_count}")
        
        # Check for excessive whitespace patterns
        whitespace_patterns = [
            (' ' * 10, 'Long spaces'),  # 10+ consecutive spaces
            ('\t' * 5, 'Long tabs'),    # 5+ consecutive tabs
            ('\n' * 5, 'Long newlines') # 5+ consecutive newlines
        ]
        
        for pattern, description in whitespace_patterns:
            if pattern in value:
                stego_indicators.append(description)
        
        # Check for unusual character frequencies
        char_frequency = {}
        for char in value:
            char_frequency[char] = char_frequency.get(char, 0) + 1
        
        # Look for characters that appear exactly specific counts (potential bit encoding)
        suspicious_frequencies = [8, 16, 32, 64, 128]  # Powers of 2
        for char, freq in char_frequency.items():
            if freq in suspicious_frequencies and not char.isalnum():
                stego_indicators.append(f"Suspicious frequency pattern: '{char}' appears {freq} times")
        
        # Check for base64-like patterns in unexpected places
        if len(value) > 20:
            # Look for base64 patterns
            base64_pattern = advanced_regex.compile(r'[A-Za-z0-9+/]{20,}={0,2}')
            base64_matches = base64_pattern.findall(value)
            if base64_matches:
                for match in base64_matches:
                    if len(match) % 4 == 0:  # Valid base64 length
                        stego_indicators.append(f"Potential base64 steganography: {match[:20]}...")
        
        # Apply sanitization if indicators found
        if stego_indicators:
            sanitized = value
            
            # Remove hidden Unicode characters
            for char in hidden_chars:
                sanitized = sanitized.replace(char, '')
            
            # Normalize excessive whitespace
            sanitized = advanced_regex.sub(r' {10,}', ' ', sanitized)  # Max 1 space
            sanitized = advanced_regex.sub(r'\t{5,}', '\t', sanitized)  # Max 1 tab
            sanitized = advanced_regex.sub(r'\n{5,}', '\n\n', sanitized)  # Max 2 newlines
            
            logging.getLogger("SCAFAD.Layer1.SecuritySanitizers").info(
                f"Steganography indicators detected: {stego_indicators}"
            )
            
            return sanitized
        
        return value
    
    @staticmethod
    @performance_monitor
    async def _sanitize_covert_channel(value: Any, config: Dict[str, Any] = None) -> Any:
        """Detect and sanitize covert channel communications"""
        if not isinstance(value, str):
            return value
        
        config = config or {}
        covert_indicators = []
        
        # Check for timing-based patterns in text
        timing_patterns = [
            r'\b\d{10,13}\b',  # Unix timestamps (potential timing channel)
            r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3,6}Z?\b',  # High-precision timestamps
            r'\bsleep\s*\(\s*\d+\s*\)',  # Sleep commands
            r'\bwait\s*\(\s*\d+\s*\)',   # Wait commands
            r'\bdelay\s*\(\s*\d+\s*\)',  # Delay commands
        ]
        
        for pattern in timing_patterns:
            matches = advanced_regex.findall(pattern, value, advanced_regex.IGNORECASE)
            if matches:
                covert_indicators.append(f"Timing pattern: {pattern}")
        
        # Check for size-based covert channels
        if len(value) in [256, 512, 1024, 2048, 4096]:  # Suspicious exact sizes
            covert_indicators.append(f"Suspicious exact size: {len(value)} bytes")
        
        # Check for frequency-based patterns
        char_counts = {}
        for char in value:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Look for characters with suspicious frequencies
        for char, count in char_counts.items():
            if count > 0 and (count & (count - 1)) == 0:  # Power of 2
                if count >= 8 and not char.isspace():
                    covert_indicators.append(f"Power-of-2 frequency: '{char}' appears {count} times")
        
        # Check for modulation patterns (e.g., alternating case)
        case_changes = 0
        for i in range(len(value) - 1):
            if value[i].isalpha() and value[i+1].isalpha():
                if (value[i].islower() and value[i+1].isupper()) or \
                   (value[i].isupper() and value[i+1].islower()):
                    case_changes += 1
        
        if case_changes > len(value) * 0.3:  # More than 30% case changes
            covert_indicators.append(f"Excessive case changes: {case_changes}")
        
        # Check for DNS-style covert channels
        dns_pattern = r'\b[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+'
        dns_matches = advanced_regex.findall(dns_pattern, value)
        if len(dns_matches) > 5:  # Many domain-like strings
            covert_indicators.append("Multiple DNS-like strings detected")
        
        # Apply sanitization if covert channel indicators found
        if covert_indicators:
            sanitized = value
            
            # Remove or normalize timing-related content
            for pattern in timing_patterns:
                sanitized = advanced_regex.sub(pattern, '[TIMING_BLOCKED]', sanitized, flags=advanced_regex.IGNORECASE)
            
            # Normalize case to prevent case-based covert channels
            if case_changes > len(value) * 0.3:
                sanitized = sanitized.lower()
            
            logging.getLogger("SCAFAD.Layer1.SecuritySanitizers").warning(
                f"Covert channel indicators detected: {covert_indicators}"
            )
            
            return sanitized
        
        return value


# =============================================================================
# Advanced Performance Optimizations
# =============================================================================

class PerformanceOptimizer:
    """Advanced performance optimization for sanitization operations"""
    
    def __init__(self, engine: EnhancedSanitizationEngine):
        self.engine = engine
        self.performance_history = deque(maxlen=1000)
        self.optimization_rules = self._initialize_optimization_rules()
        self.adaptive_thresholds = {
            'latency_threshold_ms': 1.0,
            'memory_threshold_mb': 100,
            'cpu_threshold_percent': 50,
            'error_rate_threshold': 0.05
        }
    
    def _initialize_optimization_rules(self) -> List[Dict[str, Any]]:
        """Initialize performance optimization rules"""
        return [
            {
                'name': 'cache_size_optimization',
                'condition': lambda m: m.cache_hit_rate < 0.4,
                'action': self._optimize_cache_size,
                'priority': 90
            },
            {
                'name': 'processing_mode_optimization',
                'condition': lambda m: m.throughput_records_per_second < 500,
                'action': self._optimize_processing_mode,
                'priority': 80
            },
            {
                'name': 'sanitization_level_optimization',
                'condition': lambda m: m.average_latency_ms > 2.0,
                'action': self._optimize_sanitization_level,
                'priority': 70
            },
            {
                'name': 'memory_optimization',
                'condition': lambda m: m.memory_usage_mb > 200,
                'action': self._optimize_memory_usage,
                'priority': 95
            },
            {
                'name': 'batch_size_optimization',
                'condition': lambda m: m.cpu_utilization_percent > 80,
                'action': self._optimize_batch_processing,
                'priority': 60
            }
        ]
    
    def optimize(self, metrics: SanitizationMetrics) -> Dict[str, Any]:
        """Apply performance optimizations based on current metrics"""
        optimizations_applied = []
        
        # Record performance history
        self.performance_history.append({
            'timestamp': time.time(),
            'metrics': asdict(metrics)
        })
        
        # Apply optimization rules
        for rule in sorted(self.optimization_rules, key=lambda r: r['priority'], reverse=True):
            try:
                if rule['condition'](metrics):
                    result = rule['action'](metrics)
                    if result:
                        optimizations_applied.append({
                            'rule': rule['name'],
                            'result': result,
                            'timestamp': time.time()
                        })
            except Exception as e:
                logging.getLogger("SCAFAD.Layer1.PerformanceOptimizer").error(
                    f"Error applying optimization rule {rule['name']}: {e}"
                )
        
        # Update adaptive thresholds
        self._update_adaptive_thresholds()
        
        return {
            'optimizations_applied': optimizations_applied,
            'adaptive_thresholds': self.adaptive_thresholds,
            'performance_trend': self._analyze_performance_trend()
        }
    
    def _optimize_cache_size(self, metrics: SanitizationMetrics) -> Dict[str, Any]:
        """Optimize cache size based on hit rate"""
        current_size = len(self.engine.result_cache)
        target_hit_rate = 0.7
        
        if metrics.cache_hit_rate < target_hit_rate:
            # Increase cache size
            new_max_size = min(current_size * 2, 50000)
            self.engine.result_cache = TTLCache(maxsize=new_max_size, ttl=300)
            
            return {
                'action': 'increased_cache_size',
                'old_size': current_size,
                'new_max_size': new_max_size,
                'expected_improvement': 'higher_hit_rate'
            }
        
        return None
    
    def _optimize_processing_mode(self, metrics: SanitizationMetrics) -> Dict[str, Any]:
        """Optimize processing mode based on throughput"""
        current_mode = self.engine.processing_mode
        
        if metrics.throughput_records_per_second < 500:
            if current_mode == ProcessingMode.SYNCHRONOUS:
                self.engine.processing_mode = ProcessingMode.ASYNCHRONOUS
                return {
                    'action': 'switched_to_async',
                    'old_mode': current_mode.value,
                    'new_mode': ProcessingMode.ASYNCHRONOUS.value
                }
            elif current_mode == ProcessingMode.ASYNCHRONOUS:
                self.engine.processing_mode = ProcessingMode.PARALLEL
                return {
                    'action': 'switched_to_parallel',
                    'old_mode': current_mode.value,
                    'new_mode': ProcessingMode.PARALLEL.value
                }
        
        return None
    
    def _optimize_sanitization_level(self, metrics: SanitizationMetrics) -> Dict[str, Any]:
        """Optimize sanitization level based on latency"""
        current_level = self.engine.sanitization_level
        
        if metrics.average_latency_ms > 2.0 and metrics.error_rate < 0.01:
            # Reduce sanitization level if latency is high and error rate is low
            if current_level == SanitizationLevel.PARANOID:
                self.engine.sanitization_level = SanitizationLevel.AGGRESSIVE
                return {
                    'action': 'reduced_sanitization_level',
                    'old_level': current_level.value,
                    'new_level': SanitizationLevel.AGGRESSIVE.value,
                    'reason': 'high_latency_low_errors'
                }
            elif current_level == SanitizationLevel.AGGRESSIVE:
                self.engine.sanitization_level = SanitizationLevel.STANDARD
                return {
                    'action': 'reduced_sanitization_level',
                    'old_level': current_level.value,
                    'new_level': SanitizationLevel.STANDARD.value,
                    'reason': 'high_latency_low_errors'
                }