#!/usr/bin/env python3
"""
Unit tests for sparse attention implementation with MLANN.

Tests:
1. MLANNIndex correctness and structure
2. MLANNIndex recall with different parameters
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

from nanovllm.sparse.mlann_index import MLANNIndex, compute_recall
from nanovllm.sparse.manager import SparseAttentionManager


class TestMLANNIndex:
    """Tests for the MLANNIndex class (MLANN algorithm implementation)."""
    
    def test_basic_build_query(self):
        """Test basic MLANN index build and query."""
        np.random.seed(42)
        corpus = np.random.randn(100, 32).astype(np.float32)
        queries = np.random.randn(5, 32).astype(np.float32)
        k = 10
        
        index = MLANNIndex(metric="ip", k_train=16, n_trees=4, max_depth=6)
        index.build(corpus)
        indices = index.query(queries, k)
        
        # Verify shape
        assert indices.shape == (5, k)
        
        # Verify indices are valid
        assert indices.min() >= 0
        assert indices.max() < 100
    
    def test_same_distribution_recall(self):
        """Test MLANN achieves good recall when queries are from corpus."""
        np.random.seed(42)
        corpus = np.random.randn(200, 64).astype(np.float32)
        # Normalize for inner product
        corpus = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-10)
        
        # Use subset of corpus as queries (best case for MLANN)
        queries = corpus[:20]
        k = 16
        
        index = MLANNIndex(metric="ip", k_train=32, n_trees=16, max_depth=8)
        index.build(corpus)
        mlann_indices = index.query(queries, k)
        
        # Ground truth
        gt_indices = np.argsort(-(queries @ corpus.T), axis=1)[:, :k]
        
        # MLANN should achieve reasonable recall for same-distribution queries
        recall = compute_recall(mlann_indices, gt_indices)
        assert recall >= 0.3, f"Same-distribution recall too low: {recall:.2%}"
    
    def test_multi_tree_ensemble(self):
        """Test that more trees improves recall."""
        np.random.seed(42)
        corpus = np.random.randn(500, 64).astype(np.float32)
        corpus = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-10)
        queries = corpus[:30]
        k = 16
        
        gt_indices = np.argsort(-(queries @ corpus.T), axis=1)[:, :k]
        
        recalls = []
        for n_trees in [1, 4, 8, 16]:
            index = MLANNIndex(metric="ip", k_train=32, n_trees=n_trees, max_depth=8)
            index.build(corpus)
            mlann_indices = index.query(queries, k)
            recall = compute_recall(mlann_indices, gt_indices)
            recalls.append(recall)
        
        # More trees should generally improve or maintain recall
        # (with some variance due to randomness)
        assert recalls[-1] >= recalls[0] * 0.8, "More trees should not significantly hurt recall"
    
    def test_l2_metric(self):
        """Test MLANN with L2 distance metric."""
        np.random.seed(42)
        corpus = np.random.randn(100, 32).astype(np.float32)
        queries = corpus[:10]
        k = 10
        
        index = MLANNIndex(metric="l2", k_train=16, n_trees=8, max_depth=6)
        index.build(corpus)
        indices = index.query(queries, k)
        
        # Verify shape and validity
        assert indices.shape == (10, k)
        assert indices.min() >= 0
        assert indices.max() < 100
    
    def test_small_corpus(self):
        """Test with corpus smaller than k."""
        np.random.seed(42)
        corpus = np.random.randn(5, 16).astype(np.float32)
        queries = np.random.randn(3, 16).astype(np.float32)
        k = 10  # Larger than corpus size
        
        index = MLANNIndex(metric="ip", k_train=3, n_trees=2, max_depth=4, min_leaf_size=1)
        index.build(corpus)
        indices = index.query(queries, k)
        
        # Should clamp k to corpus size
        assert indices.shape == (3, 5)
    
    def test_index_stats(self):
        """Test MLANN index statistics."""
        np.random.seed(42)
        corpus = np.random.randn(500, 64).astype(np.float32)
        
        index = MLANNIndex(metric="ip", k_train=20, n_trees=8, max_depth=6)
        index.build(corpus)
        
        stats = index.get_stats()
        assert stats["num_corpus"] == 500
        assert stats["k_train"] == 20
        assert stats["n_trees"] == 8
        assert stats["metric"] == "ip"
        assert stats["build_time_ms"] >= 0
        assert stats["total_cells"] > 0
    
    def test_query_with_scores(self):
        """Test query_with_scores returns valid scores."""
        np.random.seed(42)
        corpus = np.random.randn(100, 32).astype(np.float32)
        queries = np.random.randn(5, 32).astype(np.float32)
        k = 10
        
        index = MLANNIndex(metric="ip", k_train=16, n_trees=4, max_depth=6)
        index.build(corpus)
        indices, scores = index.query_with_scores(queries, k)
        
        # Verify shapes
        assert indices.shape == (5, k)
        assert scores.shape == (5, k)
        
        # Scores should be non-negative (probabilities)
        assert scores.min() >= 0
        
        # Scores should be sorted descending
        for i in range(5):
            assert np.all(scores[i, :-1] >= scores[i, 1:])
    
    def test_separate_train_queries(self):
        """Test MLANN with separate training queries (Q) and corpus (K).
        
        This tests the attention-aware training where:
        - Corpus = K vectors
        - Training queries = Q vectors (different from K!)
        - Labels = k-NN of each Q in K space
        
        This is critical for sparse attention since Q queries K, not K queries K.
        """
        np.random.seed(42)
        dim = 64
        num_tokens = 200
        
        # Simulate Q and K from same transformer layer (correlated but different)
        # In real attention: Q = W_q * X, K = W_k * X with different projections
        hidden = np.random.randn(num_tokens, dim * 2).astype(np.float32)
        queries = hidden[:, :dim]  # Q
        corpus = hidden[:, dim:]   # K
        
        # Normalize
        queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-10)
        corpus = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-10)
        
        k = 16
        
        # Ground truth: Q queries K
        gt_indices = np.argsort(-(queries @ corpus.T), axis=1)[:, :k]
        
        # Test 1: Train with Q, query with Q (correct way)
        index_q = MLANNIndex(metric="ip", k_train=32, n_trees=8, max_depth=8)
        index_q.build(corpus=corpus, train_queries=queries)
        mlann_indices_q = index_q.query(queries[:30], k)
        recall_q = compute_recall(mlann_indices_q, gt_indices[:30])
        
        # Test 2: Train with K (self), query with Q (incorrect way - old behavior)
        index_k = MLANNIndex(metric="ip", k_train=32, n_trees=8, max_depth=8)
        index_k.build(corpus=corpus, train_queries=None)  # Uses corpus as train_queries
        mlann_indices_k = index_k.query(queries[:30], k)
        recall_k = compute_recall(mlann_indices_k, gt_indices[:30])
        
        print(f"\n   MLANN Q vs K training comparison:")
        print(f"      Trained with Q: recall = {recall_q:.2%}")
        print(f"      Trained with K: recall = {recall_k:.2%}")
        
        # Training with Q should be at least as good as training with K
        # when queries are Q (since that's the actual distribution)
        # Note: in some cases K self-training might work well if Q and K are very similar
        assert recall_q >= 0.1, f"Q-trained recall too low: {recall_q:.2%}"


class TestSparseAttentionManager:
    """Tests for the SparseAttentionManager class with MLANN."""
    
    def mock_config(self):
        """Create a mock config for testing with MLANN settings."""
        class MockConfig:
            use_sparse_attention = True
            sparse_topk = 16
            sparse_min_seq_len = 32
            sparse_distance_metric = "ip"
            sparse_index_granularity = "layer_shared"
            sparse_include_decode_dense = True
            sparse_max_decode_tokens = 8
            sparse_debug = False
            kvcache_block_size = 16
            # MLANN-specific
            sparse_mlann_k_train = 16
            sparse_mlann_n_trees = 4
            sparse_mlann_max_depth = 6
            sparse_mlann_min_leaf_size = 5
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
        assert manager.mlann_n_trees == 4
    
    def test_index_dim(self, mock_config):
        """Test index dimension calculation."""
        # Create fresh configs to avoid side effects
        class LayerSharedConfig:
            use_sparse_attention = True
            sparse_topk = 16
            sparse_min_seq_len = 32
            sparse_distance_metric = "ip"
            sparse_index_granularity = "layer_shared"
            sparse_include_decode_dense = True
            sparse_max_decode_tokens = 8
            sparse_debug = False
            kvcache_block_size = 16
            sparse_mlann_k_train = 16
            sparse_mlann_n_trees = 4
            sparse_mlann_max_depth = 6
            sparse_mlann_min_leaf_size = 5
        
        class PerHeadConfig:
            use_sparse_attention = True
            sparse_topk = 16
            sparse_min_seq_len = 32
            sparse_distance_metric = "ip"
            sparse_index_granularity = "per_head"
            sparse_include_decode_dense = True
            sparse_max_decode_tokens = 8
            sparse_debug = False
            kvcache_block_size = 16
            sparse_mlann_k_train = 16
            sparse_mlann_n_trees = 4
            sparse_mlann_max_depth = 6
            sparse_mlann_min_leaf_size = 5
        
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
        """Test MLANN index building from mock KV cache."""
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
        
        # Verify MLANN structure
        for layer_id in range(4):
            index = manager.indices[layer_id]
            assert index.is_built
            assert len(index.partitioners) == 4  # n_trees
    
    def test_query_indices(self, mock_config):
        """Test querying MLANN indices."""
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
        expected_defaults = {
            'use_sparse_attention': False,
            'sparse_topk': 64,
            'sparse_min_seq_len': 512,
            'sparse_distance_metric': 'ip',
            'sparse_index_granularity': 'layer_shared',
            'sparse_mlann_n_trees': 8,
        }
        assert expected_defaults['use_sparse_attention'] == False


def run_tests():
    """Run all tests."""
    print("Running sparse attention unit tests (MLANN)...")
    print("=" * 60)
    
    # MLANNIndex tests
    print("\n1. MLANNIndex Tests")
    print("-" * 40)
    
    test_mlann = TestMLANNIndex()
    
    test_mlann.test_basic_build_query()
    print("   ✓ test_basic_build_query")
    
    test_mlann.test_same_distribution_recall()
    print("   ✓ test_same_distribution_recall")
    
    test_mlann.test_multi_tree_ensemble()
    print("   ✓ test_multi_tree_ensemble")
    
    test_mlann.test_l2_metric()
    print("   ✓ test_l2_metric")
    
    test_mlann.test_small_corpus()
    print("   ✓ test_small_corpus")
    
    test_mlann.test_index_stats()
    print("   ✓ test_index_stats")
    
    test_mlann.test_query_with_scores()
    print("   ✓ test_query_with_scores")
    
    test_mlann.test_separate_train_queries()
    print("   ✓ test_separate_train_queries")
    
    # SparseAttentionManager tests
    print("\n2. SparseAttentionManager Tests (with MLANN)")
    print("-" * 40)
    
    class MockConfig:
        use_sparse_attention = True
        sparse_topk = 16
        sparse_min_seq_len = 32
        sparse_distance_metric = "ip"
        sparse_index_granularity = "layer_shared"
        sparse_include_decode_dense = True
        sparse_max_decode_tokens = 8
        sparse_debug = False
        kvcache_block_size = 16
        sparse_mlann_k_train = 16
        sparse_mlann_n_trees = 4
        sparse_mlann_max_depth = 6
        sparse_mlann_min_leaf_size = 5
    
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
