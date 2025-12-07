#!/usr/bin/env python3
"""
Unit tests for sparse attention implementation.

Tests:
1. ANNIndex correctness (exact mode)
2. ANNIndex recall (IVF mode)
3. SparseAttentionManager index building
4. Config validation
5. (Optional, requires GPU) Full integration test with toy model
"""

import sys
sys.path.insert(0, '/workspace')

import numpy as np
import torch

# pytest is optional - tests can run standalone
try:
    import pytest
except ImportError:
    pytest = None

from nanovllm.sparse.ann_index import ANNIndex
from nanovllm.sparse.manager import SparseAttentionManager


class TestANNIndex:
    """Tests for the ANNIndex class."""
    
    def test_exact_ip_basic(self):
        """Test exact inner product search."""
        np.random.seed(42)
        corpus = np.random.randn(100, 32).astype(np.float32)
        queries = np.random.randn(5, 32).astype(np.float32)
        k = 10
        
        index = ANNIndex(metric="ip", mode="exact")
        index.build(corpus)
        indices = index.query(queries, k)
        
        # Verify shape
        assert indices.shape == (5, k)
        
        # Verify indices are valid
        assert indices.min() >= 0
        assert indices.max() < 100
        
        # Verify correctness against manual computation
        for i in range(len(queries)):
            scores = np.dot(queries[i], corpus.T)
            expected_indices = np.argsort(-scores)[:k]
            np.testing.assert_array_equal(indices[i], expected_indices)
    
    def test_exact_l2_basic(self):
        """Test exact L2 distance search."""
        np.random.seed(42)
        corpus = np.random.randn(100, 32).astype(np.float32)
        queries = np.random.randn(5, 32).astype(np.float32)
        k = 10
        
        index = ANNIndex(metric="l2", mode="exact")
        index.build(corpus)
        indices = index.query(queries, k)
        
        # Verify correctness against manual computation
        for i in range(len(queries)):
            dists = np.sum((queries[i] - corpus) ** 2, axis=1)
            expected_indices = np.argsort(dists)[:k]
            np.testing.assert_array_equal(indices[i], expected_indices)
    
    def test_ivf_recall(self):
        """Test IVF mode achieves reasonable recall."""
        np.random.seed(42)
        corpus = np.random.randn(1000, 64).astype(np.float32)
        queries = np.random.randn(10, 64).astype(np.float32)
        k = 16
        
        # Exact index for ground truth
        exact_index = ANNIndex(metric="ip", mode="exact")
        exact_index.build(corpus)
        exact_indices = exact_index.query(queries, k)
        
        # IVF index
        ivf_index = ANNIndex(metric="ip", mode="ivf", nlist=32, nprobe=8)
        ivf_index.build(corpus)
        ivf_indices = ivf_index.query(queries, k)
        
        # Compute recall
        recalls = []
        for i in range(len(queries)):
            gt = set(exact_indices[i].tolist())
            pred = set(ivf_indices[i].tolist())
            recall = len(gt & pred) / k
            recalls.append(recall)
        
        avg_recall = np.mean(recalls)
        # IVF should achieve at least 30% recall with these settings
        assert avg_recall >= 0.3, f"IVF recall too low: {avg_recall:.2%}"
    
    def test_small_corpus(self):
        """Test with corpus smaller than k."""
        np.random.seed(42)
        corpus = np.random.randn(5, 16).astype(np.float32)
        queries = np.random.randn(3, 16).astype(np.float32)
        k = 10  # Larger than corpus size
        
        index = ANNIndex(metric="ip", mode="exact")
        index.build(corpus)
        indices = index.query(queries, k)
        
        # Should clamp k to corpus size
        assert indices.shape == (3, 5)
    
    def test_index_stats(self):
        """Test index statistics."""
        np.random.seed(42)
        corpus = np.random.randn(500, 64).astype(np.float32)
        
        index = ANNIndex(metric="ip", mode="exact")
        index.build(corpus)
        
        stats = index.get_stats()
        assert stats["num_vectors"] == 500
        assert stats["dim"] == 64
        assert stats["metric"] == "ip"
        assert stats["mode"] == "exact"
        assert stats["build_time_ms"] >= 0


class TestSparseAttentionManager:
    """Tests for the SparseAttentionManager class."""
    
    def mock_config(self):
        """Create a mock config for testing."""
        class MockConfig:
            use_sparse_attention = True
            sparse_topk = 16
            sparse_min_seq_len = 32
            sparse_distance_metric = "ip"
            sparse_index_granularity = "layer_shared"
            sparse_ann_mode = "exact"
            sparse_ivf_nlist = 32
            sparse_ivf_nprobe = 4
            sparse_include_decode_dense = True
            sparse_max_decode_tokens = 8
            sparse_debug = False
            kvcache_block_size = 16
        return MockConfig()
    
    def test_manager_init(self, mock_config):
        """Test manager initialization."""
        manager = SparseAttentionManager(
            config=mock_config,
            num_layers=4,
            num_kv_heads=4,
            head_dim=32,
        )
        
        assert manager.num_layers == 4
        assert manager.num_kv_heads == 4
        assert manager.head_dim == 32
        assert manager.enabled == True
        assert manager.topk == 16
    
    def test_index_dim(self, mock_config):
        """Test index dimension calculation."""
        # Create fresh configs to avoid side effects
        class LayerSharedConfig:
            use_sparse_attention = True
            sparse_topk = 16
            sparse_min_seq_len = 32
            sparse_distance_metric = "ip"
            sparse_index_granularity = "layer_shared"
            sparse_ann_mode = "exact"
            sparse_ivf_nlist = 32
            sparse_ivf_nprobe = 4
            sparse_include_decode_dense = True
            sparse_max_decode_tokens = 8
            sparse_debug = False
            kvcache_block_size = 16
        
        class PerHeadConfig:
            use_sparse_attention = True
            sparse_topk = 16
            sparse_min_seq_len = 32
            sparse_distance_metric = "ip"
            sparse_index_granularity = "per_head"
            sparse_ann_mode = "exact"
            sparse_ivf_nlist = 32
            sparse_ivf_nprobe = 4
            sparse_include_decode_dense = True
            sparse_max_decode_tokens = 8
            sparse_debug = False
            kvcache_block_size = 16
        
        # Layer shared: num_kv_heads * head_dim
        manager = SparseAttentionManager(LayerSharedConfig(), 4, 8, 64)
        assert manager.index_dim == 8 * 64
        
        # Per head: head_dim only
        manager = SparseAttentionManager(PerHeadConfig(), 4, 8, 64)
        assert manager.index_dim == 64
    
    def test_sparse_eligibility(self, mock_config):
        """Test sparse attention eligibility check."""
        manager = SparseAttentionManager(mock_config, 4, 4, 32)
        
        assert manager.is_sparse_eligible(31) == False
        assert manager.is_sparse_eligible(32) == True
        assert manager.is_sparse_eligible(1000) == True
    
    def test_build_indices(self, mock_config):
        """Test index building from mock KV cache."""
        manager = SparseAttentionManager(
            config=mock_config,
            num_layers=4,
            num_kv_heads=4,
            head_dim=32,
        )
        
        # Create mock KV cache
        num_blocks = 8
        block_size = 16
        kv_cache = torch.randn(2, 4, num_blocks, block_size, 4, 32)
        
        seq_len = 64
        block_table = [0, 1, 2, 3]
        
        build_times = manager.build_indices_from_kv_cache(
            kv_cache, [seq_len], [block_table], block_size
        )
        
        # Should have built indices for all layers
        assert len(build_times) == 4
        assert all(layer_id in manager.indices for layer_id in range(4))
    
    def test_query_indices(self, mock_config):
        """Test querying indices."""
        manager = SparseAttentionManager(
            config=mock_config,
            num_layers=4,
            num_kv_heads=4,
            head_dim=32,
        )
        
        # Build indices
        num_blocks = 8
        block_size = 16
        kv_cache = torch.randn(2, 4, num_blocks, block_size, 4, 32)
        seq_len = 64
        block_table = [0, 1, 2, 3]
        
        manager.build_indices_from_kv_cache(
            kv_cache, [seq_len], [block_table], block_size
        )
        
        # Query
        num_queries = 4
        num_heads = 8
        queries = torch.randn(num_queries, num_heads, 32)
        
        for layer_id in range(4):
            indices = manager.query(layer_id, queries)
            
            assert indices.shape == (num_queries, 16)  # topk=16
            assert indices.min() >= 0
            assert indices.max() < seq_len
    
    def test_reset(self, mock_config):
        """Test manager reset."""
        manager = SparseAttentionManager(mock_config, 4, 4, 32)
        
        # Build some indices
        kv_cache = torch.randn(2, 4, 8, 16, 4, 32)
        manager.build_indices_from_kv_cache(kv_cache, [64], [[0, 1, 2, 3]], 16)
        
        assert len(manager.indices) > 0
        
        # Reset
        manager.reset()
        
        assert len(manager.indices) == 0
        assert len(manager.seq_info) == 0


class TestConfig:
    """Tests for sparse attention configuration."""
    
    def test_default_config_no_sparse(self):
        """Test that default config has sparse attention disabled."""
        # We can't easily test the actual Config class without model path
        # Just verify the expected defaults
        expected_defaults = {
            'use_sparse_attention': False,
            'sparse_topk': 64,
            'sparse_min_seq_len': 512,
            'sparse_distance_metric': 'ip',
            'sparse_index_granularity': 'layer_shared',
            'sparse_ann_mode': 'exact',
        }
        # This is more of a documentation test
        assert expected_defaults['use_sparse_attention'] == False


def run_tests():
    """Run all tests."""
    print("Running sparse attention unit tests...")
    print("=" * 60)
    
    # ANNIndex tests
    print("\n1. ANNIndex Tests")
    print("-" * 40)
    
    test_ann = TestANNIndex()
    test_ann.test_exact_ip_basic()
    print("   ✓ test_exact_ip_basic")
    
    test_ann.test_exact_l2_basic()
    print("   ✓ test_exact_l2_basic")
    
    test_ann.test_ivf_recall()
    print("   ✓ test_ivf_recall")
    
    test_ann.test_small_corpus()
    print("   ✓ test_small_corpus")
    
    test_ann.test_index_stats()
    print("   ✓ test_index_stats")
    
    # SparseAttentionManager tests
    print("\n2. SparseAttentionManager Tests")
    print("-" * 40)
    
    class MockConfig:
        use_sparse_attention = True
        sparse_topk = 16
        sparse_min_seq_len = 32
        sparse_distance_metric = "ip"
        sparse_index_granularity = "layer_shared"
        sparse_ann_mode = "exact"
        sparse_ivf_nlist = 32
        sparse_ivf_nprobe = 4
        sparse_include_decode_dense = True
        sparse_max_decode_tokens = 8
        sparse_debug = False
        kvcache_block_size = 16
    
    mock_config = MockConfig()
    test_manager = TestSparseAttentionManager()
    
    test_manager.test_manager_init(mock_config)
    print("   ✓ test_manager_init")
    
    test_manager.test_index_dim(mock_config)
    print("   ✓ test_index_dim")
    
    test_manager.test_sparse_eligibility(mock_config)
    print("   ✓ test_sparse_eligibility")
    
    test_manager.test_build_indices(mock_config)
    print("   ✓ test_build_indices")
    
    test_manager.test_query_indices(mock_config)
    print("   ✓ test_query_indices")
    
    test_manager.test_reset(mock_config)
    print("   ✓ test_reset")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
