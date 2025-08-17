"""
Comprehensive unit tests for layer1_hashing.py

Tests deferred hashing management, hash registry, and cryptographic
hashing functionality with extensive coverage of hash operations.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
import hashlib
import json
import base64

# Add the core directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'core'))

from layer1_hashing import (
    DeferredHashingManager, HashedRecord, HashRegistry,
    HashAlgorithm, HashStrategy, HashResult
)


class TestHashAlgorithm:
    """Test the HashAlgorithm enum."""
    
    def test_hash_algorithms(self):
        """Test all hash algorithms are defined."""
        algorithms = list(HashAlgorithm)
        assert HashAlgorithm.MD5 in algorithms
        assert HashAlgorithm.SHA1 in algorithms
        assert HashAlgorithm.SHA256 in algorithms
        assert HashAlgorithm.SHA512 in algorithms
        assert HashAlgorithm.BLAKE2B in algorithms
    
    def test_algorithm_values(self):
        """Test hash algorithm string values."""
        assert HashAlgorithm.MD5.value == "md5"
        assert HashAlgorithm.SHA1.value == "sha1"
        assert HashAlgorithm.SHA256.value == "sha256"
        assert HashAlgorithm.SHA512.value == "sha512"
        assert HashAlgorithm.BLAKE2B.value == "blake2b"
    
    def test_algorithm_security_levels(self):
        """Test hash algorithm security levels."""
        # MD5 and SHA1 are considered weak
        assert HashAlgorithm.MD5.security_level == "weak"
        assert HashAlgorithm.SHA1.security_level == "weak"
        
        # SHA256 and SHA512 are considered strong
        assert HashAlgorithm.SHA256.security_level == "strong"
        assert HashAlgorithm.SHA512.security_level == "strong"
        
        # BLAKE2B is considered strong
        assert HashAlgorithm.BLAKE2B.security_level == "strong"


class TestHashStrategy:
    """Test the HashStrategy class."""
    
    def test_strategy_creation(self):
        """Test creating hash strategies."""
        strategy = HashStrategy(
            name="secure_hashing",
            algorithm=HashAlgorithm.SHA256,
            salt_length=32,
            iterations=10000,
            description="Secure hashing with salt and iterations"
        )
        
        assert strategy.name == "secure_hashing"
        assert strategy.algorithm == HashAlgorithm.SHA256
        assert strategy.salt_length == 32
        assert strategy.iterations == 10000
        assert strategy.description == "Secure hashing with salt and iterations"
    
    def test_strategy_validation(self):
        """Test hash strategy validation."""
        # Valid strategy
        valid_strategy = HashStrategy(
            name="valid",
            algorithm=HashAlgorithm.SHA256,
            salt_length=16,
            iterations=1000
        )
        assert valid_strategy.is_valid() is True
        
        # Invalid strategy - no algorithm
        invalid_strategy = HashStrategy(
            name="invalid",
            algorithm=None,
            salt_length=16,
            iterations=1000
        )
        assert invalid_strategy.is_valid() is False
        
        # Invalid strategy - negative salt length
        invalid_strategy = HashStrategy(
            name="invalid",
            algorithm=HashAlgorithm.SHA256,
            salt_length=-1,
            iterations=1000
        )
        assert invalid_strategy.is_valid() is False
    
    def test_strategy_serialization(self):
        """Test hash strategy serialization."""
        strategy = HashStrategy(
            name="serialization_test",
            algorithm=HashAlgorithm.SHA512,
            salt_length=64,
            iterations=20000,
            description="Test serialization"
        )
        
        serialized = strategy.to_dict()
        assert serialized["name"] == "serialization_test"
        assert serialized["algorithm"] == "sha512"
        assert serialized["salt_length"] == 64
        assert serialized["iterations"] == 20000
        assert serialized["description"] == "Test serialization"


class TestHashedRecord:
    """Test the HashedRecord class."""
    
    def test_record_creation(self):
        """Test creating hashed records."""
        record = HashedRecord(
            original_value="sensitive_data",
            hash_value="abc123hash",
            algorithm=HashAlgorithm.SHA256,
            salt="random_salt_123",
            timestamp=datetime.now(),
            metadata={"source": "test", "field": "email"}
        )
        
        assert record.original_value == "sensitive_data"
        assert record.hash_value == "abc123hash"
        assert record.algorithm == HashAlgorithm.SHA256
        assert record.salt == "random_salt_123"
        assert record.timestamp is not None
        assert record.metadata["source"] == "test"
    
    def test_record_serialization(self):
        """Test hashed record serialization."""
        record = HashedRecord(
            original_value="test_value",
            hash_value="test_hash",
            algorithm=HashAlgorithm.SHA256,
            salt="test_salt",
            timestamp=datetime(2024, 1, 1, 0, 0, 0),
            metadata={"test": "data"}
        )
        
        serialized = record.to_dict()
        assert serialized["original_value"] == "test_value"
        assert serialized["hash_value"] == "test_hash"
        assert serialized["algorithm"] == "sha256"
        assert serialized["salt"] == "test_salt"
        assert "timestamp" in serialized
        assert serialized["metadata"]["test"] == "data"
    
    def test_record_verification(self):
        """Test hash verification."""
        # Create a record with known values
        original_value = "password123"
        salt = "random_salt"
        
        # Generate hash manually for verification
        import hashlib
        hash_input = original_value + salt
        expected_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        
        record = HashedRecord(
            original_value=original_value,
            hash_value=expected_hash,
            algorithm=HashAlgorithm.SHA256,
            salt=salt,
            timestamp=datetime.now(),
            metadata={}
        )
        
        # Verify hash
        assert record.verify_hash(original_value) is True
        assert record.verify_hash("wrong_password") is False
    
    def test_record_metadata_access(self):
        """Test metadata access methods."""
        metadata = {
            "source": "user_input",
            "field": "password",
            "privacy_level": "high",
            "retention_days": 30
        }
        
        record = HashedRecord(
            original_value="test",
            hash_value="hash",
            algorithm=HashAlgorithm.SHA256,
            salt="salt",
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        # Test metadata access
        assert record.get_metadata("source") == "user_input"
        assert record.get_metadata("field") == "password"
        assert record.get_metadata("privacy_level") == "high"
        assert record.get_metadata("retention_days") == 30
        assert record.get_metadata("non_existent") is None
        
        # Test metadata update
        record.update_metadata("retention_days", 60)
        assert record.get_metadata("retention_days") == 60


class TestHashRegistry:
    """Test the HashRegistry class."""
    
    def test_registry_initialization(self):
        """Test hash registry initialization."""
        registry = HashRegistry()
        assert registry is not None
        assert hasattr(registry, 'records')
        assert len(registry.records) == 0
    
    def test_add_hash_record(self):
        """Test adding hash records to registry."""
        registry = HashRegistry()
        
        record = HashedRecord(
            original_value="test_value",
            hash_value="test_hash",
            algorithm=HashAlgorithm.SHA256,
            salt="test_salt",
            timestamp=datetime.now(),
            metadata={}
        )
        
        registry.add_record(record)
        assert len(registry.records) == 1
        assert registry.records[0] == record
    
    def test_find_record_by_hash(self):
        """Test finding records by hash value."""
        registry = HashRegistry()
        
        # Add multiple records
        record1 = HashedRecord(
            original_value="value1",
            hash_value="hash1",
            algorithm=HashAlgorithm.SHA256,
            salt="salt1",
            timestamp=datetime.now(),
            metadata={}
        )
        
        record2 = HashedRecord(
            original_value="value2",
            hash_value="hash2",
            algorithm=HashAlgorithm.SHA256,
            salt="salt2",
            timestamp=datetime.now(),
            metadata={}
        )
        
        registry.add_record(record1)
        registry.add_record(record2)
        
        # Test finding by hash
        found_record = registry.find_by_hash("hash1")
        assert found_record is not None
        assert found_record.original_value == "value1"
        
        found_record = registry.find_by_hash("hash2")
        assert found_record is not None
        assert found_record.original_value == "value2"
        
        # Test finding non-existent hash
        not_found = registry.find_by_hash("non_existent")
        assert not_found is None
    
    def test_find_record_by_original_value(self):
        """Test finding records by original value."""
        registry = HashRegistry()
        
        record = HashedRecord(
            original_value="sensitive_email@example.com",
            hash_value="email_hash",
            algorithm=HashAlgorithm.SHA256,
            salt="email_salt",
            timestamp=datetime.now(),
            metadata={"field": "email"}
        )
        
        registry.add_record(record)
        
        # Test finding by original value
        found_record = registry.find_by_original_value("sensitive_email@example.com")
        assert found_record is not None
        assert found_record.hash_value == "email_hash"
        
        # Test finding non-existent value
        not_found = registry.find_by_original_value("other@example.com")
        assert not_found is None
    
    def test_registry_serialization(self):
        """Test registry serialization."""
        registry = HashRegistry()
        
        # Add a record
        record = HashedRecord(
            original_value="test",
            hash_value="hash",
            algorithm=HashAlgorithm.SHA256,
            salt="salt",
            timestamp=datetime(2024, 1, 1, 0, 0, 0),
            metadata={"test": "data"}
        )
        
        registry.add_record(record)
        
        # Serialize registry
        serialized = registry.to_dict()
        assert "records" in serialized
        assert len(serialized["records"]) == 1
        assert serialized["records"][0]["hash_value"] == "hash"
    
    def test_registry_cleanup(self):
        """Test registry cleanup functionality."""
        registry = HashRegistry()
        
        # Add records with different timestamps
        old_record = HashedRecord(
            original_value="old_value",
            hash_value="old_hash",
            algorithm=HashAlgorithm.SHA256,
            salt="old_salt",
            timestamp=datetime(2020, 1, 1, 0, 0, 0),  # Old timestamp
            metadata={}
        )
        
        new_record = HashedRecord(
            original_value="new_value",
            hash_value="new_hash",
            algorithm=HashAlgorithm.SHA256,
            salt="new_salt",
            timestamp=datetime.now(),  # Current timestamp
            metadata={}
        )
        
        registry.add_record(old_record)
        registry.add_record(new_record)
        
        # Clean up old records (older than 1 year)
        cutoff_date = datetime(2023, 1, 1, 0, 0, 0)
        removed_count = registry.cleanup_old_records(cutoff_date)
        
        assert removed_count == 1  # Should remove old record
        assert len(registry.records) == 1  # Should keep new record
        assert registry.records[0].original_value == "new_value"


class TestDeferredHashingManager:
    """Test the DeferredHashingManager class."""
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = DeferredHashingManager()
        assert manager is not None
        assert hasattr(manager, 'registry')
        assert hasattr(manager, 'strategies')
        assert hasattr(manager, 'deferred_operations')
    
    def test_add_hashing_strategy(self):
        """Test adding hashing strategies."""
        manager = DeferredHashingManager()
        
        strategy = HashStrategy(
            name="secure",
            algorithm=HashAlgorithm.SHA256,
            salt_length=32,
            iterations=10000
        )
        
        manager.add_strategy(strategy)
        assert "secure" in manager.strategies
        assert manager.strategies["secure"] == strategy
    
    def test_basic_hashing(self):
        """Test basic hashing functionality."""
        manager = DeferredHashingManager()
        
        # Add default strategy
        strategy = HashStrategy(
            name="default",
            algorithm=HashAlgorithm.SHA256,
            salt_length=16,
            iterations=1000
        )
        
        manager.add_strategy(strategy)
        
        # Test hashing
        result = manager.hash_value("sensitive_data", "default")
        
        assert result.is_successful is True
        assert result.hash_value is not None
        assert result.algorithm == HashAlgorithm.SHA256
        assert result.salt is not None
        assert len(result.salt) == 16
        
        # Verify hash
        assert result.verify_hash("sensitive_data") is True
        assert result.verify_hash("wrong_data") is False
    
    def test_deferred_hashing(self):
        """Test deferred hashing functionality."""
        manager = DeferredHashingManager()
        
        # Add strategy
        strategy = HashStrategy(
            name="deferred",
            algorithm=HashAlgorithm.SHA256,
            salt_length=16,
            iterations=1000
        )
        
        manager.add_strategy(strategy)
        
        # Queue deferred operation
        operation_id = manager.queue_hash_operation(
            "sensitive_data",
            "deferred",
            priority="high"
        )
        
        assert operation_id is not None
        assert len(manager.deferred_operations) == 1
        
        # Process deferred operations
        results = manager.process_deferred_operations()
        
        assert len(results) == 1
        assert results[0].is_successful is True
        assert results[0].original_value == "sensitive_data"
    
    def test_batch_hashing(self):
        """Test batch hashing functionality."""
        manager = DeferredHashingManager()
        
        # Add strategy
        strategy = HashStrategy(
            name="batch",
            algorithm=HashAlgorithm.SHA256,
            salt_length=16,
            iterations=1000
        )
        
        manager.add_strategy(strategy)
        
        # Test data
        test_values = [
            "value1",
            "value2",
            "value3",
            "value4",
            "value5"
        ]
        
        # Hash in batch
        results = manager.hash_batch(test_values, "batch")
        
        assert len(results) == 5
        
        # Verify all hashes
        for i, result in enumerate(results):
            assert result.is_successful is True
            assert result.original_value == f"value{i+1}"
            assert result.hash_value is not None
            assert result.verify_hash(f"value{i+1}") is True
    
    def test_hash_verification(self):
        """Test hash verification functionality."""
        manager = DeferredHashingManager()
        
        # Add strategy
        strategy = HashStrategy(
            name="verify",
            algorithm=HashAlgorithm.SHA256,
            salt_length=16,
            iterations=1000
        )
        
        manager.add_strategy(strategy)
        
        # Hash a value
        result = manager.hash_value("password123", "verify")
        
        # Test verification
        assert manager.verify_hash("password123", result.hash_value, result.salt, "verify") is True
        assert manager.verify_hash("wrong_password", result.hash_value, result.salt, "verify") is False
        
        # Test verification with wrong salt
        assert manager.verify_hash("password123", result.hash_value, "wrong_salt", "verify") is False
    
    def test_hash_registry_integration(self):
        """Test integration with hash registry."""
        manager = DeferredHashingManager()
        
        # Add strategy
        strategy = HashStrategy(
            name="registry_test",
            algorithm=HashAlgorithm.SHA256,
            salt_length=16,
            iterations=1000
        )
        
        manager.add_strategy(strategy)
        
        # Hash a value
        result = manager.hash_value("test_value", "registry_test")
        
        # Check that it was added to registry
        registry_record = manager.registry.find_by_hash(result.hash_value)
        assert registry_record is not None
        assert registry_record.original_value == "test_value"
        
        # Check that it can be found by original value
        registry_record = manager.registry.find_by_original_value("test_value")
        assert registry_record is not None
        assert registry_record.hash_value == result.hash_value
    
    def test_error_handling(self):
        """Test error handling during hashing."""
        manager = DeferredHashingManager()
        
        # Test hashing with non-existent strategy
        with pytest.raises(ValueError):
            manager.hash_value("test", "non_existent")
        
        # Test hashing with invalid data
        with pytest.raises(ValueError):
            manager.hash_value(None, "default")
        
        # Test hashing with empty data
        with pytest.raises(ValueError):
            manager.hash_value("", "default")
    
    def test_performance_with_large_data(self, large_dataset):
        """Test hashing performance with large datasets."""
        manager = DeferredHashingManager()
        
        # Add strategy
        strategy = HashStrategy(
            name="performance",
            algorithm=HashAlgorithm.SHA256,
            salt_length=16,
            iterations=1000
        )
        
        manager.add_strategy(strategy)
        
        # Test with subset of large dataset
        test_data = large_dataset[:1000]
        
        start_time = datetime.now()
        
        # Hash each record ID
        for record in test_data:
            result = manager.hash_value(record["id"], "performance")
            assert result.is_successful is True
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time
        assert duration < 30.0  # 30 seconds for 1000 hashes
        
        # Check registry size
        assert len(manager.registry.records) == 1000
    
    def test_hash_strategy_selection(self):
        """Test automatic hash strategy selection."""
        manager = DeferredHashingManager()
        
        # Add multiple strategies
        weak_strategy = HashStrategy(
            name="weak",
            algorithm=HashAlgorithm.MD5,
            salt_length=8,
            iterations=100
        )
        
        strong_strategy = HashStrategy(
            name="strong",
            algorithm=HashAlgorithm.SHA512,
            salt_length=64,
            iterations=50000
        )
        
        manager.add_strategy(weak_strategy)
        manager.add_strategy(strong_strategy)
        
        # Test automatic selection based on data sensitivity
        sensitive_data = "highly_sensitive_password"
        result = manager.hash_value(sensitive_data, auto_select=True)
        
        # Should select strong strategy for sensitive data
        assert result.algorithm == HashAlgorithm.SHA512
        assert result.salt_length == 64
        assert result.iterations == 50000
        
        # Test with less sensitive data
        less_sensitive = "public_identifier"
        result = manager.hash_value(less_sensitive, auto_select=True)
        
        # Should select appropriate strategy
        assert result.is_successful is True


class TestHashResult:
    """Test the HashResult class."""
    
    def test_result_creation(self):
        """Test creating hash results."""
        result = HashResult(
            is_successful=True,
            hash_value="abc123hash",
            algorithm=HashAlgorithm.SHA256,
            salt="random_salt",
            original_value="sensitive_data",
            processing_time=0.5,
            metadata={"strategy": "secure", "iterations": 10000}
        )
        
        assert result.is_successful is True
        assert result.hash_value == "abc123hash"
        assert result.algorithm == HashAlgorithm.SHA256
        assert result.salt == "random_salt"
        assert result.original_value == "sensitive_data"
        assert result.processing_time == 0.5
        assert result.metadata["strategy"] == "secure"
    
    def test_result_serialization(self):
        """Test hash result serialization."""
        result = HashResult(
            is_successful=True,
            hash_value="test_hash",
            algorithm=HashAlgorithm.SHA256,
            salt="test_salt",
            original_value="test_value",
            processing_time=0.3,
            metadata={"test": "data"}
        )
        
        serialized = result.to_dict()
        assert serialized["is_successful"] is True
        assert serialized["hash_value"] == "test_hash"
        assert serialized["algorithm"] == "sha256"
        assert serialized["salt"] == "test_salt"
        assert serialized["processing_time"] == 0.3
        assert serialized["metadata"]["test"] == "data"
    
    def test_result_verification(self):
        """Test hash result verification."""
        # Create result with known values
        original_value = "password123"
        salt = "random_salt"
        
        # Generate hash manually
        import hashlib
        hash_input = original_value + salt
        hash_value = hashlib.sha256(hash_input.encode()).hexdigest()
        
        result = HashResult(
            is_successful=True,
            hash_value=hash_value,
            algorithm=HashAlgorithm.SHA256,
            salt=salt,
            original_value=original_value,
            processing_time=0.1,
            metadata={}
        )
        
        # Test verification
        assert result.verify_hash(original_value) is True
        assert result.verify_hash("wrong_password") is False
    
    def test_result_comparison(self):
        """Test hash result comparison."""
        result1 = HashResult(
            is_successful=True,
            hash_value="hash1",
            algorithm=HashAlgorithm.SHA256,
            salt="salt1",
            original_value="value1",
            processing_time=0.1,
            metadata={}
        )
        
        result2 = HashResult(
            is_successful=True,
            hash_value="hash2",
            algorithm=HashAlgorithm.SHA256,
            salt="salt2",
            original_value="value2",
            processing_time=0.2,
            metadata={}
        )
        
        # Test equality
        assert result1 != result2  # Different hash values
        
        # Test processing time comparison
        assert result1.processing_time < result2.processing_time


class TestHashingIntegration:
    """Test integration between hashing components."""
    
    def test_full_hashing_pipeline(self):
        """Test complete hashing pipeline."""
        manager = DeferredHashingManager()
        
        # Add comprehensive strategy
        strategy = HashStrategy(
            name="comprehensive",
            algorithm=HashAlgorithm.SHA512,
            salt_length=64,
            iterations=25000,
            description="Comprehensive hashing strategy"
        )
        
        manager.add_strategy(strategy)
        
        # Test data
        test_data = [
            "password123",
            "sensitive_email@example.com",
            "credit_card_number",
            "social_security_number"
        ]
        
        # Hash all data
        results = []
        for data in test_data:
            result = manager.hash_value(data, "comprehensive")
            results.append(result)
            
            # Verify each result
            assert result.is_successful is True
            assert result.algorithm == HashAlgorithm.SHA512
            assert result.salt_length == 64
            assert result.iterations == 25000
            assert result.verify_hash(data) is True
        
        # Check registry integration
        assert len(manager.registry.records) == 4
        
        # Verify all records can be found
        for i, data in enumerate(test_data):
            registry_record = manager.registry.find_by_original_value(data)
            assert registry_record is not None
            assert registry_record.hash_value == results[i].hash_value
    
    def test_deferred_hashing_workflow(self):
        """Test deferred hashing workflow."""
        manager = DeferredHashingManager()
        
        # Add strategy
        strategy = HashStrategy(
            name="deferred_workflow",
            algorithm=HashAlgorithm.SHA256,
            salt_length=32,
            iterations=10000
        )
        
        manager.add_strategy(strategy)
        
        # Queue multiple operations
        operation_ids = []
        test_values = ["value1", "value2", "value3", "value4", "value5"]
        
        for value in test_values:
            op_id = manager.queue_hash_operation(value, "deferred_workflow", priority="normal")
            operation_ids.append(op_id)
        
        # Verify operations are queued
        assert len(manager.deferred_operations) == 5
        
        # Process all operations
        results = manager.process_deferred_operations()
        
        # Verify results
        assert len(results) == 5
        
        for i, result in enumerate(results):
            assert result.is_successful is True
            assert result.original_value == f"value{i+1}"
            assert result.verify_hash(f"value{i+1}") is True
        
        # Check registry
        assert len(manager.registry.records) == 5
        
        # Verify no pending operations
        assert len(manager.deferred_operations) == 0
    
    def test_hash_strategy_evolution(self):
        """Test hash strategy evolution and migration."""
        manager = DeferredHashingManager()
        
        # Add old strategy
        old_strategy = HashStrategy(
            name="legacy",
            algorithm=HashAlgorithm.SHA1,
            salt_length=8,
            iterations=1000
        )
        
        # Add new strategy
        new_strategy = HashStrategy(
            name="modern",
            algorithm=HashAlgorithm.SHA512,
            salt_length=64,
            iterations=50000
        )
        
        manager.add_strategy(old_strategy)
        manager.add_strategy(new_strategy)
        
        # Hash with old strategy
        old_result = manager.hash_value("password", "legacy")
        assert old_result.algorithm == HashAlgorithm.SHA1
        
        # Hash with new strategy
        new_result = manager.hash_value("password", "modern")
        assert new_result.algorithm == HashAlgorithm.SHA512
        
        # Verify both hashes work
        assert old_result.verify_hash("password") is True
        assert new_result.verify_hash("password") is True
        
        # Check registry has both
        assert len(manager.registry.records) == 2
        
        # Test strategy migration
        migrated_result = manager.migrate_hash(
            old_result.hash_value,
            old_result.salt,
            "legacy",
            "modern"
        )
        
        assert migrated_result.is_successful is True
        assert migrated_result.algorithm == HashAlgorithm.SHA512
        assert migrated_result.verify_hash("password") is True


if __name__ == '__main__':
    pytest.main([__file__])
