"""
Sparse Attention Manager for nano-vllm.

This module manages per-layer MLANN indices for sparse attention:
- Builds MLANN indices from prefill keys using the natural classifier formulation
- Queries indices during decode to find top-k relevant keys
- Handles device transfers (GPU <-> CPU) and shape transformations

The manager supports two granularity modes:
- "layer_shared": One index per layer, all heads share (flatten num_kv_heads * head_dim)
- "per_head": One index per head (TODO: partially implemented)

MLANN (Multilabel ANN) treats candidate selection as a multilabel classification:
- Training: For each corpus vector, compute its k-NN and use as labels
- Build a partitioner (RP tree ensemble) and estimate per-cell label probabilities
- Query: Route query to cells, aggregate probabilities, return top-k candidates

Usage:
    # Initialize (typically in ModelRunner)
    manager = SparseAttentionManager(config, num_layers, num_kv_heads, head_dim)
    
    # After prefill, build indices from KV cache
    manager.build_indices_from_kv_cache(kv_cache, seq_info)
    
    # During decode, query for top-k keys
    topk_indices = manager.query(layer_id, queries, k)
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Optional, List, Dict, Tuple
import time
import logging

from nanovllm.sparse.mlann_index import MLANNIndex


logger = logging.getLogger(__name__)


class SparseAttentionManager:
    """
    Manages MLANN indices for sparse attention across all layers.
    
    Uses the MLANN algorithm (Multilabel Classification for ANN) which:
    - Treats candidate selection as multilabel classification
    - Builds random projection tree ensemble for space partitioning
    - Estimates per-cell label probabilities during training
    - Aggregates scores across trees during query
    
    Attributes:
        config: The nano-vllm Config object
        num_layers: Number of transformer layers
        num_kv_heads: Number of key-value heads (per GPU in tensor parallel)
        head_dim: Dimension of each head
        granularity: "layer_shared" or "per_head"
        indices: Dict mapping layer_id -> MLANNIndex (or list of MLANNIndex for per_head)
    """
    
    def __init__(
        self,
        config,  # Config object
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        """
        Initialize the sparse attention manager.
        
        Args:
            config: nano-vllm Config with sparse attention settings
            num_layers: Number of transformer layers
            num_kv_heads: Number of KV heads (after tensor parallel division)
            head_dim: Dimension per head
        """
        self.config = config
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        # Sparse attention config
        self.enabled = config.use_sparse_attention
        self.topk = config.sparse_topk
        self.min_seq_len = config.sparse_min_seq_len
        self.metric = config.sparse_distance_metric
        self.granularity = config.sparse_index_granularity
        self.include_decode_dense = config.sparse_include_decode_dense
        self.max_decode_tokens = config.sparse_max_decode_tokens
        self.debug = config.sparse_debug
        
        # MLANN-specific config
        self.mlann_k_train = getattr(config, 'sparse_mlann_k_train', 32)
        self.mlann_n_trees = getattr(config, 'sparse_mlann_n_trees', 8)
        self.mlann_max_depth = getattr(config, 'sparse_mlann_max_depth', 8)
        self.mlann_min_leaf_size = getattr(config, 'sparse_mlann_min_leaf_size', 10)
        
        # Per-layer indices
        # For "layer_shared": indices[layer_id] = MLANNIndex
        # For "per_head": indices[layer_id] = [MLANNIndex] * num_kv_heads
        self.indices: Dict[int, MLANNIndex | List[MLANNIndex]] = {}
        
        # Sequence info for each active sequence
        # Maps seq_id -> {prefill_len, block_table, ...}
        self.seq_info: Dict[int, Dict] = {}
        
        # Stats
        self.stats = {
            "total_index_build_time_ms": 0.0,
            "total_query_time_ms": 0.0,
            "num_index_builds": 0,
            "num_queries": 0,
        }
    
    @property
    def index_dim(self) -> int:
        """Dimension of vectors in the index."""
        if self.granularity == "layer_shared":
            return self.num_kv_heads * self.head_dim
        else:  # per_head
            return self.head_dim
    
    def is_sparse_eligible(self, seq_len: int) -> bool:
        """Check if a sequence is eligible for sparse attention."""
        return self.enabled and seq_len >= self.min_seq_len
    
    def reset(self):
        """Reset all indices and sequence info (e.g., for new batch)."""
        self.indices.clear()
        self.seq_info.clear()
    
    def _create_index(self) -> MLANNIndex:
        """Create a new MLANNIndex with current config."""
        return MLANNIndex(
            metric=self.metric,
            k_train=self.mlann_k_train,
            n_trees=self.mlann_n_trees,
            max_depth=self.mlann_max_depth,
            min_leaf_size=self.mlann_min_leaf_size,
        )
    
    def build_index_for_layer(
        self,
        layer_id: int,
        keys: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim] on GPU
    ) -> float:
        """
        Build MLANN index for a single layer from prefill keys.
        
        The MLANN algorithm:
        1. Use keys as both corpus and training queries
        2. Compute exact k-NN for each training query (ground truth labels)
        3. Build RP tree ensemble partitioner
        4. Estimate per-cell label probabilities p(r, j)
        
        Args:
            layer_id: The layer index
            keys: Key tensor [num_tokens, num_kv_heads, head_dim] on GPU
            
        Returns:
            Build time in milliseconds
        """
        start_time = time.perf_counter()
        
        num_tokens = keys.shape[0]
        
        if self.granularity == "layer_shared":
            # Flatten heads: [num_tokens, num_kv_heads * head_dim]
            keys_flat = keys.reshape(num_tokens, -1)
            
            # Transfer to CPU and convert to numpy
            keys_np = keys_flat.float().cpu().numpy()
            
            # Create and build MLANN index
            # Uses keys as both corpus and training queries
            # MLANN will:
            # 1. Compute exact k-NN for each key (training labels)
            # 2. Build RP tree ensemble
            # 3. Estimate per-cell label probabilities
            index = self._create_index()
            index.build(corpus=keys_np, train_queries=None, train_knn_indices=None)
            self.indices[layer_id] = index
            
        else:  # per_head
            # Build one index per head
            head_indices = []
            for h in range(self.num_kv_heads):
                keys_h = keys[:, h, :]  # [num_tokens, head_dim]
                keys_np = keys_h.float().cpu().numpy()
                
                index = self._create_index()
                index.build(corpus=keys_np)
                head_indices.append(index)
            
            self.indices[layer_id] = head_indices
        
        build_time = (time.perf_counter() - start_time) * 1000
        self.stats["total_index_build_time_ms"] += build_time
        self.stats["num_index_builds"] += 1
        
        if self.debug:
            if self.granularity == "layer_shared":
                stats = self.indices[layer_id].get_stats()
                logger.info(f"Layer {layer_id} MLANN index built: {num_tokens} tokens, "
                           f"{stats['n_trees']} trees, {stats['total_cells']} cells, "
                           f"{build_time:.2f} ms")
            else:
                logger.info(f"Layer {layer_id} MLANN index built: {num_tokens} tokens, "
                           f"{build_time:.2f} ms")
        
        return build_time
    
    def build_indices_from_kv_cache(
        self,
        kv_cache: torch.Tensor,  # [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        seq_lens: List[int],  # Prefill lengths for each sequence
        block_tables: List[List[int]],  # Block tables for each sequence
        block_size: int,
    ) -> Dict[int, float]:
        """
        Build indices for all layers from the KV cache after prefill.
        
        This is the main entry point called after prefill completes.
        For v1, this is synchronous. Comments indicate where async/overlap could be added.
        
        Args:
            kv_cache: The full KV cache tensor [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
            seq_lens: List of prefill sequence lengths
            block_tables: List of block tables (one per sequence)
            block_size: Size of each KV block
            
        Returns:
            Dict mapping layer_id -> build_time_ms
        """
        build_times = {}
        
        # Currently only support single-sequence sparse attention
        # TODO: Extend to batched sparse attention
        if len(seq_lens) > 1:
            if self.debug:
                logger.warning("Sparse attention currently only supports single sequence, using first sequence")
        
        seq_len = seq_lens[0]
        block_table = block_tables[0]
        
        if not self.is_sparse_eligible(seq_len):
            if self.debug:
                logger.info(f"Sequence length {seq_len} below threshold {self.min_seq_len}, skipping index build")
            return build_times
        
        # Extract keys from KV cache
        # kv_cache shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        # kv_cache[0] = K cache, kv_cache[1] = V cache
        k_cache = kv_cache[0]  # [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        
        # TODO: For overlap, we could spawn background threads here.
        # Each layer's index build could run asynchronously once that layer's prefill is done.
        # For v1, we do synchronous sequential build.
        
        for layer_id in range(self.num_layers):
            # Gather keys from scattered blocks into contiguous tensor
            keys = self._gather_keys_from_cache(
                k_cache[layer_id],  # [num_blocks, block_size, num_kv_heads, head_dim]
                block_table,
                seq_len,
                block_size,
            )
            
            build_times[layer_id] = self.build_index_for_layer(layer_id, keys)
        
        # Store sequence info for later use in decode
        # Use a simple seq_id (0 for now, extend for batching later)
        self.seq_info[0] = {
            "prefill_len": seq_len,
            "block_table": block_table,
            "block_size": block_size,
        }
        
        return build_times
    
    def _gather_keys_from_cache(
        self,
        k_cache_layer: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
        block_table: List[int],
        seq_len: int,
        block_size: int,
    ) -> torch.Tensor:
        """
        Gather keys for a sequence from the block-organized KV cache.
        
        Args:
            k_cache_layer: K cache for one layer [num_blocks, block_size, num_kv_heads, head_dim]
            block_table: List of block IDs for the sequence
            seq_len: Total sequence length
            block_size: Tokens per block
            
        Returns:
            keys: [seq_len, num_kv_heads, head_dim]
        """
        # Calculate number of full blocks and remaining tokens
        num_full_blocks = seq_len // block_size
        remaining = seq_len % block_size
        
        gathered_keys = []
        
        for i, block_id in enumerate(block_table):
            if i < num_full_blocks:
                # Full block
                gathered_keys.append(k_cache_layer[block_id])  # [block_size, num_kv_heads, head_dim]
            elif i == num_full_blocks and remaining > 0:
                # Partial last block
                gathered_keys.append(k_cache_layer[block_id, :remaining])  # [remaining, num_kv_heads, head_dim]
        
        if gathered_keys:
            keys = torch.cat(gathered_keys, dim=0)  # [seq_len, num_kv_heads, head_dim]
        else:
            # Empty sequence (shouldn't happen in practice)
            keys = torch.empty(0, self.num_kv_heads, self.head_dim, 
                             dtype=k_cache_layer.dtype, device=k_cache_layer.device)
        
        return keys
    
    def query(
        self,
        layer_id: int,
        queries: torch.Tensor,  # [num_queries, num_heads, head_dim] on GPU
        k: int | None = None,
    ) -> torch.Tensor:
        """
        Query the ANN index for top-k keys.
        
        Args:
            layer_id: The layer index
            queries: Query tensor [num_queries, num_heads, head_dim] on GPU
            k: Number of neighbors (default: self.topk)
            
        Returns:
            indices: [num_queries, k] tensor of token indices (on GPU)
                    For per_head mode: [num_queries, num_heads, k]
        """
        if k is None:
            k = self.topk
        
        if layer_id not in self.indices:
            raise RuntimeError(f"No index built for layer {layer_id}")
        
        start_time = time.perf_counter()
        
        num_queries, num_heads, head_dim = queries.shape
        device = queries.device
        
        if self.granularity == "layer_shared":
            index = self.indices[layer_id]
            
            # For layer_shared, we need to handle the head dimension
            # Option 1: Flatten query heads and search once (may not align with KV heads)
            # Option 2: Use one representative query per token (e.g., average across heads)
            # For v1, we flatten and search, then each query gets same top-k for all its heads
            
            # Since queries have num_heads but index was built with num_kv_heads,
            # we need to handle GQA (grouped query attention)
            # For simplicity, we'll use the first query head scaled or average
            
            # Actually, for proper sparse attention, each query token should get
            # the same set of key indices. So we can just use an average or first head.
            # Let's use average across heads as a representative query.
            
            # For GQA: queries [N, num_heads, head_dim], keys [num_tokens, num_kv_heads, head_dim]
            # The index is built with flattened keys [num_tokens, num_kv_heads * head_dim]
            # We need to transform queries similarly...
            
            # Actually, the proper way for GQA is:
            # - Each query head h attends to KV head h // groups
            # - For sparse attention, we want to find which KEY TOKENS are most relevant
            # - So we should query with a representative of the query
            
            # Simple approach: average query across heads, then tile to match KV heads
            # This works for layer_shared granularity
            
            # More sophisticated: for each query head, find its corresponding KV head
            # But for v1, let's keep it simple: use mean of query heads
            
            # Compute mean query across heads: [num_queries, head_dim]
            query_mean = queries.mean(dim=1)  # [num_queries, head_dim]
            
            # Tile to match KV head structure for index query
            # Index dimension is num_kv_heads * head_dim
            # We can tile the mean query num_kv_heads times
            query_for_index = query_mean.unsqueeze(1).expand(-1, self.num_kv_heads, -1)
            query_for_index = query_for_index.reshape(num_queries, -1)  # [num_queries, num_kv_heads * head_dim]
            
            # Transfer to CPU and query
            query_np = query_for_index.float().cpu().numpy()
            indices_np = index.query(query_np, k)  # [num_queries, k]
            
            # Transfer back to GPU
            indices = torch.from_numpy(indices_np).to(device=device, dtype=torch.long)
            
        else:  # per_head
            # Query each head's index separately
            head_indices = self.indices[layer_id]
            all_indices = []
            
            for h in range(self.num_kv_heads):
                # Map query heads to KV heads (for GQA)
                # Assuming simple mapping: query head h maps to KV head h % num_kv_heads
                kv_head = h % self.num_kv_heads
                
                # For each query token, get the query for this head
                # But we have num_heads query heads and num_kv_heads KV heads
                # For GQA, multiple query heads share one KV head
                # Let's use the first query head that maps to this KV head
                
                # Simple approach: just iterate over KV heads and use queries that map to them
                # For now, assume num_heads == num_kv_heads or GQA ratio
                queries_h = queries[:, h, :]  # [num_queries, head_dim]
                query_np = queries_h.float().cpu().numpy()
                
                indices_np = head_indices[kv_head].query(query_np, k)  # [num_queries, k]
                all_indices.append(torch.from_numpy(indices_np))
            
            # Stack: [num_queries, num_kv_heads, k]
            indices = torch.stack(all_indices, dim=1).to(device=device, dtype=torch.long)
        
        query_time = (time.perf_counter() - start_time) * 1000
        self.stats["total_query_time_ms"] += query_time
        self.stats["num_queries"] += 1
        
        if self.debug:
            logger.info(f"Layer {layer_id} query: {num_queries} queries, k={k}, {query_time:.2f} ms")
        
        return indices
    
    def get_sparse_attention_indices(
        self,
        layer_id: int,
        queries: torch.Tensor,  # [num_queries, num_heads, head_dim]
        current_seq_len: int,  # Current total sequence length (prefill + decode so far)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get indices for sparse attention computation.
        
        This returns indices into the KV cache for sparse attention, handling:
        1. Sparse top-k from prefill region (via ANN index)
        2. Optionally dense attention over recent decode tokens
        
        Args:
            layer_id: Layer index
            queries: Query tensor [num_queries, num_heads, head_dim]
            current_seq_len: Current sequence length
            
        Returns:
            sparse_indices: Token indices for sparse attention [num_queries, k]
            decode_range: (start, end) tuple for decode tokens to include densely, or None
        """
        # Get sparse indices from ANN
        sparse_indices = self.query(layer_id, queries)  # [num_queries, k] or [num_queries, num_heads, k]
        
        # Determine decode token range
        decode_range = None
        if self.include_decode_dense and 0 in self.seq_info:
            prefill_len = self.seq_info[0]["prefill_len"]
            num_decode_tokens = current_seq_len - prefill_len
            
            if num_decode_tokens > 0:
                # Include up to max_decode_tokens recent decode tokens
                decode_start = max(prefill_len, current_seq_len - self.max_decode_tokens)
                decode_range = (decode_start, current_seq_len)
        
        return sparse_indices, decode_range
    
    def get_stats(self) -> Dict:
        """Get manager statistics."""
        stats = dict(self.stats)
        stats["num_layers_indexed"] = len(self.indices)
        if self.stats["num_index_builds"] > 0:
            stats["avg_index_build_time_ms"] = (
                self.stats["total_index_build_time_ms"] / self.stats["num_index_builds"]
            )
        if self.stats["num_queries"] > 0:
            stats["avg_query_time_ms"] = (
                self.stats["total_query_time_ms"] / self.stats["num_queries"]
            )
        return stats
    
    def has_index(self, layer_id: int) -> bool:
        """Check if an index exists for the given layer."""
        return layer_id in self.indices


def test_sparse_attention_manager():
    """Test the SparseAttentionManager with MLANN."""
    import torch
    
    # Mock config with MLANN settings
    class MockConfig:
        use_sparse_attention = True
        sparse_topk = 16
        sparse_min_seq_len = 32
        sparse_distance_metric = "ip"
        sparse_index_granularity = "layer_shared"
        sparse_include_decode_dense = True
        sparse_max_decode_tokens = 8
        sparse_debug = True
        kvcache_block_size = 16
        # MLANN-specific
        sparse_mlann_k_train = 16
        sparse_mlann_n_trees = 4
        sparse_mlann_max_depth = 6
        sparse_mlann_min_leaf_size = 5
    
    config = MockConfig()
    num_layers = 4
    num_kv_heads = 4
    head_dim = 32
    
    # Create manager
    manager = SparseAttentionManager(config, num_layers, num_kv_heads, head_dim)
    
    # Create mock KV cache
    num_blocks = 8
    block_size = 16
    kv_cache = torch.randn(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
    
    # Mock sequence info
    seq_len = 64  # Use 4 blocks
    block_table = [0, 1, 2, 3]
    
    print("Testing SparseAttentionManager with MLANN...")
    
    # Build indices
    print("\n1. Building MLANN indices from KV cache...")
    build_times = manager.build_indices_from_kv_cache(
        kv_cache, [seq_len], [block_table], block_size
    )
    print(f"   Build times per layer: {build_times}")
    
    # Verify MLANN structure
    print("\n2. Verifying MLANN index structure...")
    for layer_id in range(num_layers):
        index = manager.indices[layer_id]
        stats = index.get_stats()
        print(f"   Layer {layer_id}: {stats['n_trees']} trees, {stats['total_cells']} cells")
    
    # Query indices
    print("\n3. Querying indices...")
    num_queries = 4
    num_heads = 8  # GQA: 8 query heads, 4 KV heads
    queries = torch.randn(num_queries, num_heads, head_dim)
    
    for layer_id in range(num_layers):
        sparse_indices, decode_range = manager.get_sparse_attention_indices(
            layer_id, queries, current_seq_len=seq_len + 10  # 10 decode tokens
        )
        print(f"   Layer {layer_id}: sparse_indices shape = {sparse_indices.shape}, decode_range = {decode_range}")
    
    # Get stats
    print("\n4. Manager stats:")
    stats = manager.get_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    # Test with queries from same distribution as keys (best case for MLANN)
    print("\n5. Testing with same-distribution queries...")
    # Extract some keys to use as queries
    layer_0_keys = manager._gather_keys_from_cache(
        kv_cache[0, 0], block_table, seq_len, block_size
    )
    same_dist_queries = layer_0_keys[:4]  # Use first 4 keys as queries
    
    indices = manager.query(0, same_dist_queries)
    print(f"   Same-distribution query result shape: {indices.shape}")
    print(f"   Top-3 indices for query 0: {indices[0, :3].tolist()}")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_sparse_attention_manager()
