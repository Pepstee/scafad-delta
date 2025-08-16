# Schema Evolution Guide

## Overview

This document describes the schema evolution capabilities of the Scafad Layer1 system, including versioning, migration strategies, backward compatibility, and automated schema management.

## Schema Evolution Concepts

### 1. Schema Versioning

The system maintains multiple schema versions to support data evolution while preserving backward compatibility:

```python
# Schema version structure
schema_version = {
    "version": "2.1.0",
    "created_at": "2024-01-01T00:00:00Z",
    "changes": [
        "Added new_field for enhanced analytics",
        "Deprecated old_metric in favor of new_metric",
        "Extended metadata structure with additional properties"
    ],
    "compatibility": {
        "backward_compatible": True,
        "forward_compatible": True,
        "breaking_changes": False
    }
}
```

#### Version Naming Convention:
- **Major Version (X.0.0)**: Breaking changes, requires migration
- **Minor Version (X.Y.0)**: New features, backward compatible
- **Patch Version (X.Y.Z)**: Bug fixes, fully compatible

### 2. Schema Lifecycle Management

#### Schema States:
1. **Draft**: Under development, not yet active
2. **Active**: Currently in use for data processing
3. **Deprecated**: Still supported but not recommended
4. **Retired**: No longer supported, data migration required

#### Schema Transitions:
```python
# Schema state transition workflow
schema_workflow = {
    "draft": ["active", "cancelled"],
    "active": ["deprecated", "retired"],
    "deprecated": ["active", "retired"],
    "retired": []  # Terminal state
}
```

## Schema Evolution Strategies

### 1. Backward Compatibility

#### Additive Changes:
- **New Fields**: Adding optional fields with default values
- **Extended Enums**: Adding new values to existing enumerations
- **Nested Structures**: Extending nested object properties

```python
# Example: Adding new field with default value
schema_v1 = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "content": {"type": "string"}
    }
}

schema_v2 = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "content": {"type": "string"},
        "priority": {"type": "string", "default": "normal"}  # New field
    }
}
```

#### Non-Breaking Modifications:
- **Field Renaming**: Using aliases for field name changes
- **Type Widening**: Expanding type constraints (e.g., string to string|number)
- **Constraint Relaxation**: Making required fields optional

### 2. Forward Compatibility

#### Future-Proofing:
- **Unknown Field Handling**: Graceful processing of unexpected fields
- **Version Detection**: Automatic schema version identification
- **Fallback Mechanisms**: Default behavior for missing schema elements

```python
# Forward compatibility configuration
forward_compatibility_config = {
    "allow_unknown_fields": True,
    "unknown_field_strategy": "preserve",  # preserve, ignore, validate
    "version_fallback": True,
    "strict_mode": False
}
```

### 3. Breaking Changes Management

#### Migration Strategies:
1. **Gradual Migration**: Phased rollout with dual-write support
2. **Data Transformation**: Automatic data conversion between versions
3. **Version Coexistence**: Multiple schema versions running simultaneously
4. **Rollback Capability**: Quick reversion to previous schema versions

## Schema Registry

### 1. Centralized Schema Management

The Schema Registry provides a centralized repository for all schema definitions:

```python
from subsystems.schema_registry import SchemaRegistry

# Initialize schema registry
registry = SchemaRegistry()

# Register new schema version
schema_id = registry.register_schema(
    name="user_profile",
    version="2.1.0",
    schema_definition=user_profile_schema,
    metadata={
        "description": "Enhanced user profile with analytics fields",
        "owner": "analytics_team",
        "tags": ["user_data", "analytics"]
    }
)
```

#### Registry Features:
- **Schema Discovery**: Search and browse available schemas
- **Version History**: Complete change history and documentation
- **Dependency Tracking**: Schema relationships and dependencies
- **Validation Rules**: Schema validation and quality checks

### 2. Schema Validation

#### Validation Levels:
1. **Syntax Validation**: JSON Schema compliance
2. **Semantic Validation**: Business rule enforcement
3. **Compatibility Validation**: Version compatibility checks
4. **Quality Validation**: Schema design best practices

```python
# Schema validation example
validation_result = registry.validate_schema(
    schema_id=schema_id,
    validation_level="comprehensive"
)

if validation_result.is_valid:
    print("Schema validation passed")
    print(f"Quality score: {validation_result.quality_score}")
else:
    print("Schema validation failed:")
    for error in validation_result.errors:
        print(f"- {error.message}")
```

## Migration Strategies

### 1. Automatic Migration

#### Data Transformation:
```python
from core.layer1_schema import SchemaMigrator

# Initialize schema migrator
migrator = SchemaMigrator()

# Define migration rules
migration_rules = [
    {
        "from_version": "1.0.0",
        "to_version": "2.0.0",
        "transformations": [
            {
                "field": "old_metric",
                "action": "rename",
                "new_name": "new_metric"
            },
            {
                "field": "status",
                "action": "transform",
                "function": "convert_status_format"
            }
        ]
    }
]

# Apply migration
migrated_data = migrator.migrate_data(
    data=original_data,
    from_schema="user_profile_v1",
    to_schema="user_profile_v2",
    rules=migration_rules
)
```

#### Migration Types:
- **Field Mapping**: Direct field-to-field mapping
- **Data Transformation**: Value conversion and formatting
- **Structure Restructuring**: Nested object reorganization
- **Default Value Assignment**: Missing field population

### 2. Manual Migration

#### Custom Migration Scripts:
```python
# Custom migration function
def custom_user_migration(data, source_version, target_version):
    """Custom migration logic for user data."""
    
    if source_version == "1.0.0" and target_version == "2.0.0":
        # Transform user preferences
        if "preferences" in data:
            old_prefs = data["preferences"]
            new_prefs = {
                "theme": old_prefs.get("ui_theme", "default"),
                "notifications": {
                    "email": old_prefs.get("email_notifications", True),
                    "push": old_prefs.get("push_notifications", False)
                },
                "privacy": {
                    "profile_visibility": old_prefs.get("profile_public", False)
                }
            }
            data["preferences"] = new_prefs
        
        # Add new required fields
        data["created_at"] = data.get("created_at", "2024-01-01T00:00:00Z")
        data["updated_at"] = "2024-01-01T00:00:00Z"
    
    return data

# Register custom migration
migrator.register_custom_migration(
    schema_name="user_profile",
    migration_function=custom_user_migration
)
```

### 3. Migration Orchestration

#### Migration Workflow:
1. **Planning**: Schema change analysis and impact assessment
2. **Testing**: Migration validation in staging environment
3. **Execution**: Production migration with rollback capability
4. **Verification**: Data integrity and performance validation
5. **Cleanup**: Removal of deprecated schemas and data

## Schema Evolution Patterns

### 1. Field Addition Pattern

#### Safe Addition:
```python
# Adding new optional field
def add_optional_field(data, field_name, default_value):
    """Safely add new optional field to existing data."""
    if field_name not in data:
        data[field_name] = default_value
    return data

# Apply to existing data
existing_data = {"id": "user_123", "name": "John Doe"}
updated_data = add_optional_field(existing_data, "email", "unknown@example.com")
```

#### Required Field Addition:
```python
# Adding new required field with migration
def add_required_field(data, field_name, generator_function):
    """Add required field using generator function for existing data."""
    if field_name not in data:
        data[field_name] = generator_function(data)
    return data

# Generate unique identifier for existing users
def generate_user_code(user_data):
    return f"USER_{user_data['id']}_{hash(user_data['name'])}"

# Apply migration
user_data = {"id": "user_123", "name": "John Doe"}
migrated_data = add_required_field(user_data, "user_code", generate_user_code)
```

### 2. Field Removal Pattern

#### Safe Removal:
```python
# Safely remove deprecated field
def remove_deprecated_field(data, field_name, preserve_history=True):
    """Remove deprecated field while preserving history if needed."""
    if preserve_history and field_name in data:
        # Store in deprecated_fields for audit purposes
        if "deprecated_fields" not in data:
            data["deprecated_fields"] = {}
        data["deprecated_fields"][field_name] = data.pop(field_name)
    else:
        data.pop(field_name, None)
    return data

# Apply removal
user_data = {"id": "user_123", "old_metric": "deprecated_value"}
cleaned_data = remove_deprecated_field(user_data, "old_metric")
```

### 3. Type Evolution Pattern

#### Type Widening:
```python
# Widen field type for enhanced flexibility
def widen_field_type(data, field_name, new_type, converter_function):
    """Convert field to wider type while preserving existing values."""
    if field_name in data:
        current_value = data[field_name]
        if not isinstance(current_value, new_type):
            data[field_name] = converter_function(current_value)
    return data

# Convert string numbers to actual numbers
def string_to_number(value):
    try:
        return int(value) if value.isdigit() else float(value)
    except (ValueError, TypeError):
        return value

# Apply type widening
user_data = {"id": "user_123", "score": "95"}
converted_data = widen_field_type(user_data, "score", (int, float), string_to_number)
```

## Monitoring and Observability

### 1. Schema Usage Metrics

#### Tracking Metrics:
- **Schema Version Distribution**: Usage patterns across versions
- **Migration Frequency**: How often schemas change
- **Compatibility Issues**: Version compatibility problems
- **Performance Impact**: Schema evolution performance effects

```python
# Schema usage monitoring
schema_metrics = {
    "active_versions": {
        "user_profile": {"1.0.0": 1500, "2.0.0": 3200},
        "analytics_event": {"1.0.0": 500, "2.1.0": 1200}
    },
    "migration_stats": {
        "total_migrations": 45,
        "successful_migrations": 43,
        "failed_migrations": 2,
        "average_migration_time": "2.3s"
    }
}
```

### 2. Schema Health Monitoring

#### Health Indicators:
- **Validation Success Rate**: Percentage of data validation success
- **Migration Error Rate**: Frequency of migration failures
- **Schema Drift**: Deviation from expected schema structure
- **Performance Degradation**: Impact on processing performance

## Best Practices

### 1. Schema Design Principles

#### Evolution-Friendly Design:
- **Extensible Structures**: Design for future additions
- **Default Values**: Provide sensible defaults for new fields
- **Optional Fields**: Make non-critical fields optional
- **Version Metadata**: Include version information in data

#### Naming Conventions:
- **Consistent Patterns**: Use consistent naming across schemas
- **Clear Semantics**: Field names should be self-explanatory
- **Avoid Abbreviations**: Use full descriptive names
- **Namespace Separation**: Use prefixes for related fields

### 2. Migration Best Practices

#### Safe Migration:
- **Incremental Changes**: Make small, manageable changes
- **Backward Compatibility**: Maintain compatibility when possible
- **Testing**: Thorough testing in staging environments
- **Rollback Plan**: Always have a rollback strategy

#### Performance Considerations:
- **Batch Processing**: Process migrations in batches for large datasets
- **Indexing**: Update database indexes after schema changes
- **Monitoring**: Monitor performance during and after migration
- **Cleanup**: Remove deprecated data and schemas

### 3. Documentation Standards

#### Schema Documentation:
- **Change Log**: Document all schema changes
- **Migration Guides**: Provide step-by-step migration instructions
- **Examples**: Include usage examples and data samples
- **Deprecation Notices**: Clear deprecation timelines

## Troubleshooting

### Common Schema Evolution Issues

#### 1. Migration Failures
- **Data Type Mismatches**: Incompatible data types between versions
- **Missing Required Fields**: Required fields not present in source data
- **Validation Errors**: Data failing new schema validation rules
- **Performance Issues**: Slow migration due to large datasets

#### 2. Compatibility Problems
- **Breaking Changes**: Incompatible schema modifications
- **Version Conflicts**: Multiple schema versions causing confusion
- **Dependency Issues**: Schema dependencies not properly managed
- **Rollback Problems**: Inability to revert to previous versions

#### 3. Performance Degradation
- **Schema Complexity**: Overly complex schemas affecting performance
- **Validation Overhead**: Excessive validation rules slowing processing
- **Migration Bottlenecks**: Large-scale migrations blocking operations
- **Memory Issues**: Schema caching consuming excessive memory

## Support and Resources

### Documentation
- **API Reference**: Complete schema evolution API documentation
- **Migration Examples**: Common migration patterns and solutions
- **Best Practices**: Schema design and evolution guidelines
- **Troubleshooting Guide**: Common issues and solutions

### Tools and Utilities
- **Schema Validator**: Schema validation and quality assessment
- **Migration Generator**: Automated migration script generation
- **Compatibility Checker**: Version compatibility analysis
- **Performance Analyzer**: Schema performance impact assessment

### Support Channels
- **Schema Team**: Dedicated schema evolution support
- **Technical Support**: Implementation and configuration help
- **Migration Assistance**: Complex migration planning and execution
- **Training Resources**: Schema evolution best practices training

---

*This document is maintained by the Scafad Schema Team and updated regularly to reflect current schema evolution capabilities and best practices.*
