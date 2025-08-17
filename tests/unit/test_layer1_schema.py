"""
Comprehensive unit tests for layer1_schema.py

Tests schema evolution engine, migration strategies, and backward compatibility
with extensive coverage of schema changes and version management.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
import json
from copy import deepcopy

# Add the core directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'core'))

from layer1_schema import (
    SchemaEvolutionEngine, SchemaMetadata, MigrationResult,
    SchemaVersion, SchemaChange, MigrationStrategy, CompatibilityLevel
)


class TestSchemaVersion:
    """Test the SchemaVersion class."""
    
    def test_version_creation(self):
        """Test creating schema versions."""
        version = SchemaVersion(
            major=1,
            minor=2,
            patch=3,
            timestamp=datetime.now(),
            description="Test schema version"
        )
        
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.description == "Test schema version"
    
    def test_version_comparison(self):
        """Test version comparison operations."""
        v1 = SchemaVersion(1, 0, 0)
        v2 = SchemaVersion(1, 1, 0)
        v3 = SchemaVersion(2, 0, 0)
        
        assert v1 < v2
        assert v2 < v3
        assert v3 > v1
        assert v1 <= v2
        assert v2 >= v1
    
    def test_version_string_representation(self):
        """Test version string formatting."""
        version = SchemaVersion(2, 1, 5)
        version_str = str(version)
        assert version_str == "2.1.5"
        
        # Test with patch 0
        version_no_patch = SchemaVersion(1, 0, 0)
        version_str = str(version_no_patch)
        assert version_str == "1.0"
    
    def test_version_serialization(self):
        """Test version serialization."""
        version = SchemaVersion(
            major=3,
            minor=2,
            patch=1,
            timestamp=datetime(2024, 1, 1, 0, 0, 0),
            description="Production schema"
        )
        
        serialized = version.to_dict()
        assert serialized["major"] == 3
        assert serialized["minor"] == 2
        assert serialized["patch"] == 1
        assert serialized["description"] == "Production schema"
        assert "timestamp" in serialized


class TestCompatibilityLevel:
    """Test the CompatibilityLevel enum."""
    
    def test_compatibility_levels(self):
        """Test all compatibility levels are defined."""
        levels = list(CompatibilityLevel)
        assert CompatibilityLevel.BACKWARD in levels
        assert CompatibilityLevel.FORWARD in levels
        assert CompatibilityLevel.FULL in levels
        assert CompatibilityLevel.NONE in levels
    
    def test_compatibility_level_values(self):
        """Test compatibility level string values."""
        assert CompatibilityLevel.BACKWARD.value == "backward"
        assert CompatibilityLevel.FORWARD.value == "forward"
        assert CompatibilityLevel.FULL.value == "full"
        assert CompatibilityLevel.NONE.value == "none"


class TestSchemaChange:
    """Test the SchemaChange class."""
    
    def test_schema_change_creation(self):
        """Test creating schema changes."""
        change = SchemaChange(
            change_type="field_addition",
            field_name="new_field",
            old_value=None,
            new_value="string",
            compatibility=CompatibilityLevel.BACKWARD,
            description="Added new optional field"
        )
        
        assert change.change_type == "field_addition"
        assert change.field_name == "new_field"
        assert change.old_value is None
        assert change.new_value == "string"
        assert change.compatibility == CompatibilityLevel.BACKWARD
        assert change.description == "Added new optional field"
    
    def test_schema_change_serialization(self):
        """Test schema change serialization."""
        change = SchemaChange(
            change_type="field_removal",
            field_name="deprecated_field",
            old_value="string",
            new_value=None,
            compatibility=CompatibilityLevel.NONE,
            description="Removed deprecated field"
        )
        
        serialized = change.to_dict()
        assert serialized["change_type"] == "field_removal"
        assert serialized["field_name"] == "deprecated_field"
        assert serialized["old_value"] == "string"
        assert serialized["new_value"] is None
        assert serialized["compatibility"] == "none"
    
    def test_schema_change_impact_assessment(self):
        """Test schema change impact assessment."""
        # Backward compatible change
        backward_change = SchemaChange(
            change_type="field_addition",
            field_name="optional_field",
            old_value=None,
            new_value="string",
            compatibility=CompatibilityLevel.BACKWARD,
            description="Added optional field"
        )
        assert backward_change.is_backward_compatible() is True
        
        # Breaking change
        breaking_change = SchemaChange(
            change_type="field_removal",
            field_name="required_field",
            old_value="string",
            new_value=None,
            compatibility=CompatibilityLevel.NONE,
            description="Removed required field"
        )
        assert breaking_change.is_backward_compatible() is False
        
        # Forward compatible change
        forward_change = SchemaChange(
            change_type="field_type_change",
            field_name="existing_field",
            old_value="string",
            new_value="number",
            compatibility=CompatibilityLevel.FORWARD,
            description="Changed field type"
        )
        assert forward_change.is_forward_compatible() is True


class TestSchemaMetadata:
    """Test the SchemaMetadata class."""
    
    def test_metadata_creation(self):
        """Test creating schema metadata."""
        metadata = SchemaMetadata(
            schema_id="user_profile_v1",
            version=SchemaVersion(1, 0, 0),
            fields={
                "id": {"type": "string", "required": True},
                "name": {"type": "string", "required": True},
                "email": {"type": "string", "required": False}
            },
            required_fields=["id", "name"],
            optional_fields=["email"],
            constraints={
                "id": {"pattern": "^[a-zA-Z0-9_]+$"},
                "email": {"format": "email"}
            }
        )
        
        assert metadata.schema_id == "user_profile_v1"
        assert metadata.version.major == 1
        assert "id" in metadata.fields
        assert "id" in metadata.required_fields
        assert "email" in metadata.optional_fields
        assert "pattern" in metadata.constraints["id"]
    
    def test_metadata_validation(self):
        """Test metadata validation."""
        # Valid metadata
        valid_metadata = SchemaMetadata(
            schema_id="test_schema",
            version=SchemaVersion(1, 0, 0),
            fields={"id": {"type": "string", "required": True}},
            required_fields=["id"],
            optional_fields=[],
            constraints={}
        )
        assert valid_metadata.is_valid() is True
        
        # Invalid metadata - required field not in fields
        invalid_metadata = SchemaMetadata(
            schema_id="test_schema",
            version=SchemaVersion(1, 0, 0),
            fields={"id": {"type": "string", "required": True}},
            required_fields=["id", "missing_field"],
            optional_fields=[],
            constraints={}
        )
        assert invalid_metadata.is_valid() is False
    
    def test_metadata_serialization(self):
        """Test metadata serialization."""
        metadata = SchemaMetadata(
            schema_id="serialization_test",
            version=SchemaVersion(2, 1, 0),
            fields={"test": {"type": "string", "required": True}},
            required_fields=["test"],
            optional_fields=[],
            constraints={}
        )
        
        serialized = metadata.to_dict()
        assert serialized["schema_id"] == "serialization_test"
        assert serialized["version"]["major"] == 2
        assert "test" in serialized["fields"]
        assert "test" in serialized["required_fields"]
    
    def test_field_access(self):
        """Test field access methods."""
        metadata = SchemaMetadata(
            schema_id="field_test",
            version=SchemaVersion(1, 0, 0),
            fields={
                "required_field": {"type": "string", "required": True},
                "optional_field": {"type": "number", "required": False}
            },
            required_fields=["required_field"],
            optional_fields=["optional_field"],
            constraints={}
        )
        
        # Test field existence
        assert metadata.has_field("required_field") is True
        assert metadata.has_field("optional_field") is True
        assert metadata.has_field("non_existent") is False
        
        # Test field requirements
        assert metadata.is_required("required_field") is True
        assert metadata.is_required("optional_field") is False
        
        # Test field types
        assert metadata.get_field_type("required_field") == "string"
        assert metadata.get_field_type("optional_field") == "number"
    
    def test_constraint_validation(self):
        """Test constraint validation."""
        metadata = SchemaMetadata(
            schema_id="constraint_test",
            version=SchemaVersion(1, 0, 0),
            fields={
                "id": {"type": "string", "required": True},
                "age": {"type": "number", "required": True}
            },
            required_fields=["id", "age"],
            optional_fields=[],
            constraints={
                "id": {"pattern": "^[A-Z]{2}\\d{4}$"},
                "age": {"min": 0, "max": 150}
            }
        )
        
        # Test constraint access
        id_constraints = metadata.get_field_constraints("id")
        assert "pattern" in id_constraints
        assert id_constraints["pattern"] == "^[A-Z]{2}\\d{4}$"
        
        age_constraints = metadata.get_field_constraints("age")
        assert "min" in age_constraints
        assert "max" in age_constraints
        assert age_constraints["min"] == 0
        assert age_constraints["max"] == 150


class TestMigrationStrategy:
    """Test the MigrationStrategy class."""
    
    def test_strategy_creation(self):
        """Test creating migration strategies."""
        strategy = MigrationStrategy(
            name="add_optional_field",
            source_version=SchemaVersion(1, 0, 0),
            target_version=SchemaVersion(1, 1, 0),
            changes=[
                SchemaChange(
                    change_type="field_addition",
                    field_name="new_field",
                    old_value=None,
                    new_value="string",
                    compatibility=CompatibilityLevel.BACKWARD,
                    description="Added optional field"
                )
            ],
            migration_function=lambda data: data
        )
        
        assert strategy.name == "add_optional_field"
        assert strategy.source_version.major == 1
        assert strategy.target_version.major == 1
        assert strategy.target_version.minor == 1
        assert len(strategy.changes) == 1
        assert strategy.migration_function is not None
    
    def test_strategy_validation(self):
        """Test migration strategy validation."""
        # Valid strategy
        valid_strategy = MigrationStrategy(
            name="valid",
            source_version=SchemaVersion(1, 0, 0),
            target_version=SchemaVersion(1, 1, 0),
            changes=[],
            migration_function=lambda data: data
        )
        assert valid_strategy.is_valid() is True
        
        # Invalid strategy - source version >= target version
        invalid_strategy = MigrationStrategy(
            name="invalid",
            source_version=SchemaVersion(2, 0, 0),
            target_version=SchemaVersion(1, 0, 0),
            changes=[],
            migration_function=lambda data: data
        )
        assert invalid_strategy.is_valid() is False
    
    def test_strategy_execution(self):
        """Test migration strategy execution."""
        def add_field_migration(data):
            data["new_field"] = "default_value"
            return data
        
        strategy = MigrationStrategy(
            name="add_field",
            source_version=SchemaVersion(1, 0, 0),
            target_version=SchemaVersion(1, 1, 0),
            changes=[],
            migration_function=add_field_migration
        )
        
        test_data = {"existing_field": "value"}
        migrated_data = strategy.execute(test_data)
        
        assert "new_field" in migrated_data
        assert migrated_data["new_field"] == "default_value"
        assert migrated_data["existing_field"] == "value"
    
    def test_strategy_serialization(self):
        """Test migration strategy serialization."""
        strategy = MigrationStrategy(
            name="serialization_test",
            source_version=SchemaVersion(1, 0, 0),
            target_version=SchemaVersion(1, 1, 0),
            changes=[],
            migration_function=lambda data: data
        )
        
        serialized = strategy.to_dict()
        assert serialized["name"] == "serialization_test"
        assert serialized["source_version"]["major"] == 1
        assert serialized["target_version"]["minor"] == 1


class TestMigrationResult:
    """Test the MigrationResult class."""
    
    def test_result_creation(self):
        """Test creating migration results."""
        result = MigrationResult(
            success=True,
            source_version=SchemaVersion(1, 0, 0),
            target_version=SchemaVersion(1, 1, 0),
            migrated_data={"id": "123", "new_field": "value"},
            changes_applied=[
                SchemaChange(
                    change_type="field_addition",
                    field_name="new_field",
                    old_value=None,
                    new_value="string",
                    compatibility=CompatibilityLevel.BACKWARD,
                    description="Added field"
                )
            ],
            warnings=["Field 'new_field' has no default value"],
            errors=[],
            processing_time=0.5
        )
        
        assert result.success is True
        assert result.source_version.major == 1
        assert result.target_version.minor == 1
        assert "new_field" in result.migrated_data
        assert len(result.changes_applied) == 1
        assert len(result.warnings) == 1
        assert len(result.errors) == 0
        assert result.processing_time == 0.5
    
    def test_result_serialization(self):
        """Test migration result serialization."""
        result = MigrationResult(
            success=True,
            source_version=SchemaVersion(1, 0, 0),
            target_version=SchemaVersion(1, 1, 0),
            migrated_data={"test": "data"},
            changes_applied=[],
            warnings=[],
            errors=[],
            processing_time=0.3
        )
        
        serialized = result.to_dict()
        assert serialized["success"] is True
        assert serialized["source_version"]["major"] == 1
        assert serialized["target_version"]["minor"] == 1
        assert serialized["migrated_data"]["test"] == "data"
        assert serialized["processing_time"] == 0.3
    
    def test_result_summary(self):
        """Test migration result summary generation."""
        result = MigrationResult(
            success=True,
            source_version=SchemaVersion(1, 0, 0),
            target_version=SchemaVersion(1, 1, 0),
            migrated_data={"id": "123"},
            changes_applied=[],
            warnings=["Minor warning"],
            errors=[],
            processing_time=0.2
        )
        
        summary = result.generate_summary()
        assert "1.0" in summary  # Source version
        assert "1.1" in summary  # Target version
        assert "successful" in summary.lower()
        assert "0.2s" in summary  # Processing time


class TestSchemaEvolutionEngine:
    """Test the SchemaEvolutionEngine class."""
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = SchemaEvolutionEngine()
        assert engine is not None
        assert hasattr(engine, 'schemas')
        assert hasattr(engine, 'migration_strategies')
        assert hasattr(engine, 'current_version')
    
    def test_register_schema(self):
        """Test schema registration."""
        engine = SchemaEvolutionEngine()
        
        schema = SchemaMetadata(
            schema_id="test_schema",
            version=SchemaVersion(1, 0, 0),
            fields={"id": {"type": "string", "required": True}},
            required_fields=["id"],
            optional_fields=[],
            constraints={}
        )
        
        engine.register_schema(schema)
        assert "test_schema" in engine.schemas
        assert engine.schemas["test_schema"] == schema
    
    def test_register_migration_strategy(self):
        """Test migration strategy registration."""
        engine = SchemaEvolutionEngine()
        
        strategy = MigrationStrategy(
            name="test_migration",
            source_version=SchemaVersion(1, 0, 0),
            target_version=SchemaVersion(1, 1, 0),
            changes=[],
            migration_function=lambda data: data
        )
        
        engine.register_migration_strategy(strategy)
        assert len(engine.migration_strategies) == 1
        assert engine.migration_strategies[0] == strategy
    
    def test_schema_validation(self, sample_telemetry_data):
        """Test schema validation against registered schemas."""
        engine = SchemaEvolutionEngine()
        
        # Register a schema
        schema = SchemaMetadata(
            schema_id="telemetry_schema",
            version=SchemaVersion(1, 0, 0),
            fields={
                "id": {"type": "string", "required": True},
                "timestamp": {"type": "string", "required": True},
                "data": {"type": "object", "required": True}
            },
            required_fields=["id", "timestamp", "data"],
            optional_fields=[],
            constraints={}
        )
        
        engine.register_schema(schema)
        
        # Test validation
        validation_result = engine.validate_data(sample_telemetry_data, "telemetry_schema")
        assert validation_result.is_valid is True
        
        # Test invalid data
        invalid_data = {"id": "test"}  # Missing required fields
        validation_result = engine.validate_data(invalid_data, "telemetry_schema")
        assert validation_result.is_valid is False
    
    def test_schema_migration(self):
        """Test schema migration functionality."""
        engine = SchemaEvolutionEngine()
        
        # Register source schema
        source_schema = SchemaMetadata(
            schema_id="user_v1",
            version=SchemaVersion(1, 0, 0),
            fields={
                "id": {"type": "string", "required": True},
                "name": {"type": "string", "required": True}
            },
            required_fields=["id", "name"],
            optional_fields=[],
            constraints={}
        )
        
        # Register target schema
        target_schema = SchemaMetadata(
            schema_id="user_v2",
            version=SchemaVersion(1, 1, 0),
            fields={
                "id": {"type": "string", "required": True},
                "name": {"type": "string", "required": True},
                "email": {"type": "string", "required": False}
            },
            required_fields=["id", "name"],
            optional_fields=["email"],
            constraints={}
        )
        
        engine.register_schema(source_schema)
        engine.register_schema(target_schema)
        
        # Register migration strategy
        def add_email_migration(data):
            data["email"] = "default@example.com"
            return data
        
        strategy = MigrationStrategy(
            name="add_email_field",
            source_version=SchemaVersion(1, 0, 0),
            target_version=SchemaVersion(1, 1, 0),
            changes=[
                SchemaChange(
                    change_type="field_addition",
                    field_name="email",
                    old_value=None,
                    new_value="string",
                    compatibility=CompatibilityLevel.BACKWARD,
                    description="Added email field"
                )
            ],
            migration_function=add_email_migration
        )
        
        engine.register_migration_strategy(strategy)
        
        # Test migration
        source_data = {"id": "user123", "name": "John Doe"}
        migration_result = engine.migrate_data(
            source_data,
            "user_v1",
            "user_v2"
        )
        
        assert migration_result.success is True
        assert "email" in migration_result.migrated_data
        assert migration_result.migrated_data["email"] == "default@example.com"
        assert migration_result.migrated_data["id"] == "user123"
        assert migration_result.migrated_data["name"] == "John Doe"
    
    def test_migration_chain(self):
        """Test migration through multiple versions."""
        engine = SchemaEvolutionEngine()
        
        # Register schemas
        schemas = []
        for i in range(3):
            schema = SchemaMetadata(
                schema_id=f"schema_v{i+1}",
                version=SchemaVersion(1, i, 0),
                fields={"id": {"type": "string", "required": True}},
                required_fields=["id"],
                optional_fields=[],
                constraints={}
            )
            schemas.append(schema)
            engine.register_schema(schema)
        
        # Register migration strategies
        for i in range(2):
            def create_migration(version):
                def migration_func(data):
                    data[f"field_v{version+1}"] = f"value_v{version+1}"
                    return data
                return migration_func
            
            strategy = MigrationStrategy(
                name=f"migration_{i+1}",
                source_version=SchemaVersion(1, i, 0),
                target_version=SchemaVersion(1, i+1, 0),
                changes=[],
                migration_function=create_migration(i+1)
            )
            engine.register_migration_strategy(strategy)
        
        # Test migration chain
        source_data = {"id": "test123"}
        migration_result = engine.migrate_data(
            source_data,
            "schema_v1",
            "schema_v3"
        )
        
        assert migration_result.success is True
        assert "field_v2" in migration_result.migrated_data
        assert "field_v3" in migration_result.migrated_data
        assert migration_result.migrated_data["field_v2"] == "value_v2"
        assert migration_result.migrated_data["field_v3"] == "value_v3"
    
    def test_migration_error_handling(self):
        """Test migration error handling."""
        engine = SchemaEvolutionEngine()
        
        # Register schemas
        source_schema = SchemaMetadata(
            schema_id="error_test_source",
            version=SchemaVersion(1, 0, 0),
            fields={"id": {"type": "string", "required": True}},
            required_fields=["id"],
            optional_fields=[],
            constraints={}
        )
        
        target_schema = SchemaMetadata(
            schema_id="error_test_target",
            version=SchemaVersion(1, 1, 0),
            fields={"id": {"type": "string", "required": True}},
            required_fields=["id"],
            optional_fields=[],
            constraints={}
        )
        
        engine.register_schema(source_schema)
        engine.register_schema(target_schema)
        
        # Test migration with non-existent schemas
        with pytest.raises(ValueError):
            engine.migrate_data({"id": "test"}, "non_existent", "error_test_target")
        
        with pytest.raises(ValueError):
            engine.migrate_data({"id": "test"}, "error_test_source", "non_existent")
        
        # Test migration without strategy
        with pytest.raises(ValueError):
            engine.migrate_data({"id": "test"}, "error_test_source", "error_test_target")
    
    def test_migration_performance(self, large_dataset):
        """Test migration performance with large datasets."""
        engine = SchemaEvolutionEngine()
        
        # Register schemas
        source_schema = SchemaMetadata(
            schema_id="performance_source",
            version=SchemaVersion(1, 0, 0),
            fields={"id": {"type": "string", "required": True}},
            required_fields=["id"],
            optional_fields=[],
            constraints={}
        )
        
        target_schema = SchemaMetadata(
            schema_id="performance_target",
            version=SchemaVersion(1, 1, 0),
            fields={
                "id": {"type": "string", "required": True},
                "processed": {"type": "boolean", "required": False}
            },
            required_fields=["id"],
            optional_fields=["processed"],
            constraints={}
        )
        
        engine.register_schema(source_schema)
        engine.register_schema(target_schema)
        
        # Register migration strategy
        def performance_migration(data):
            data["processed"] = True
            return data
        
        strategy = MigrationStrategy(
            name="performance_test",
            source_version=SchemaVersion(1, 0, 0),
            target_version=SchemaVersion(1, 1, 0),
            changes=[],
            migration_function=performance_migration
        )
        
        engine.register_migration_strategy(strategy)
        
        # Test with subset of large dataset
        test_data = large_dataset[:1000]
        
        start_time = datetime.now()
        
        for record in test_data:
            result = engine.migrate_data(record, "performance_source", "performance_target")
            assert result.success is True
            assert "processed" in result.migrated_data
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time
        assert duration < 30.0  # 30 seconds for 1000 records
    
    def test_schema_compatibility_checking(self):
        """Test schema compatibility checking."""
        engine = SchemaEvolutionEngine()
        
        # Register compatible schemas
        schema_v1 = SchemaMetadata(
            schema_id="compatible_v1",
            version=SchemaVersion(1, 0, 0),
            fields={"id": {"type": "string", "required": True}},
            required_fields=["id"],
            optional_fields=[],
            constraints={}
        )
        
        schema_v2 = SchemaMetadata(
            schema_id="compatible_v2",
            version=SchemaVersion(1, 1, 0),
            fields={
                "id": {"type": "string", "required": True},
                "name": {"type": "string", "required": False}
            },
            required_fields=["id"],
            optional_fields=["name"],
            constraints={}
        )
        
        engine.register_schema(schema_v1)
        engine.register_schema(schema_v2)
        
        # Check compatibility
        compatibility = engine.check_compatibility("compatible_v1", "compatible_v2")
        assert compatibility.level == CompatibilityLevel.BACKWARD
        assert compatibility.is_compatible is True
        
        # Check reverse compatibility
        reverse_compatibility = engine.check_compatibility("compatible_v2", "compatible_v1")
        assert reverse_compatibility.level == CompatibilityLevel.FORWARD
        assert reverse_compatibility.is_compatible is True


class TestSchemaIntegration:
    """Test integration between schema components."""
    
    def test_full_schema_evolution_workflow(self):
        """Test complete schema evolution workflow."""
        engine = SchemaEvolutionEngine()
        
        # Register initial schema
        initial_schema = SchemaMetadata(
            schema_id="workflow_v1",
            version=SchemaVersion(1, 0, 0),
            fields={
                "id": {"type": "string", "required": True},
                "name": {"type": "string", "required": True}
            },
            required_fields=["id", "name"],
            optional_fields=[],
            constraints={}
        )
        
        engine.register_schema(initial_schema)
        
        # Register evolved schema
        evolved_schema = SchemaMetadata(
            schema_id="workflow_v2",
            version=SchemaVersion(1, 1, 0),
            fields={
                "id": {"type": "string", "required": True},
                "name": {"type": "string", "required": True},
                "email": {"type": "string", "required": False},
                "age": {"type": "number", "required": False}
            },
            required_fields=["id", "name"],
            optional_fields=["email", "age"],
            constraints={
                "age": {"min": 0, "max": 150}
            }
        )
        
        engine.register_schema(evolved_schema)
        
        # Register migration strategy
        def workflow_migration(data):
            data["email"] = "default@example.com"
            data["age"] = 25
            return data
        
        strategy = MigrationStrategy(
            name="workflow_migration",
            source_version=SchemaVersion(1, 0, 0),
            target_version=SchemaVersion(1, 1, 0),
            changes=[
                SchemaChange(
                    change_type="field_addition",
                    field_name="email",
                    old_value=None,
                    new_value="string",
                    compatibility=CompatibilityLevel.BACKWARD,
                    description="Added email field"
                ),
                SchemaChange(
                    change_type="field_addition",
                    field_name="age",
                    old_value=None,
                    new_value="number",
                    compatibility=CompatibilityLevel.BACKWARD,
                    description="Added age field"
                )
            ],
            migration_function=workflow_migration
        )
        
        engine.register_migration_strategy(strategy)
        
        # Test complete workflow
        source_data = {"id": "user123", "name": "John Doe"}
        
        # Validate against source schema
        validation_result = engine.validate_data(source_data, "workflow_v1")
        assert validation_result.is_valid is True
        
        # Migrate to target schema
        migration_result = engine.migrate_data(source_data, "workflow_v1", "workflow_v2")
        assert migration_result.success is True
        
        # Validate migrated data against target schema
        validation_result = engine.validate_data(migration_result.migrated_data, "workflow_v2")
        assert validation_result.is_valid is True
        
        # Check that all fields are present
        migrated_data = migration_result.migrated_data
        assert "id" in migrated_data
        assert "name" in migrated_data
        assert "email" in migrated_data
        assert "age" in migrated_data
        
        # Check that migration changes were applied
        assert len(migration_result.changes_applied) == 2
        change_types = {change.change_type for change in migration_result.changes_applied}
        assert "field_addition" in change_types


if __name__ == '__main__':
    pytest.main([__file__])
