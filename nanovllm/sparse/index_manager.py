"""
Index manager for sparse attention.

This module manages ANN indices for each layer, building them after prefill
and providing query interfaces for decode.
"""

import torch
import numpy as np
from typing import Dict, Optional, List
from nanovllm.sparse.mlann_index import MLANNIndex
from nanovllm.config import Config


class SparseIndexManager:
    """
    Manages ANN indices for sparse attention across all layers.
    
    In v1, indices are built synchronously after prefill completes.
    Future versions can support asynchronous/overlapped index building.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the index manager.
        
        Args:
            config: Configuration object with sparse attention settings.
        """
        self.config = config
        self.use_sparse = config.use_sparse_attention
        self.granularity = config.sparse_index_granularity
        self.topk = config.sparse_topk
        self.min_seq_len = config.sparse_min_seq_len
        
        # Per-layer indices: layer_id -> {seq_id: MLANNIndex}
        # For v1, we build one index per sequence
        self.indices: Dict[int, Dict[int, MLANNIndex | List[MLANNIndex]]] = {}
        
        # Track prefill token ranges per sequence for each layer
        # Format: {layer_id: {seq_id: (start_idx, end_idx)}}
        self.prefill_ranges: Dict[int, Dict[int, tuple]] = {}
        
        # Track which sequences have indices built
        self.indexed_sequences: set = set()
        
        # Store block_tables for each sequence (needed for gather)
        self.seq_block_tables: Dict[int, List[int]] = {}
    
    def should_use_sparse(self, seq_len: int) -> bool:
        """Check if sparse attention should be used for a given sequence length."""
        return self.use_sparse and seq_len >= self.min_seq_len
    
    def build_indices_for_layer(
        self,
        layer_id: int,
        kv_cache: torch.Tensor,
        seqs: List,
        block_size: int,
        block_tables: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Build ANN index for a specific layer after prefill.
        
        This extracts K vectors from KV cache for all prefill tokens and builds
        the index. In v1, this runs synchronously after prefill.
        
        Args:
            layer_id: Layer index (0-based).
            kv_cache: KV cache tensor of shape [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim].
            seqs: List of Sequence objects that just completed prefill.
            block_size: Size of each KV cache block.
        """
        if not self.use_sparse:
            return
        
        # Extract K cache for this layer: [num_blocks, block_size, num_kv_heads, head_dim]
        k_cache = kv_cache[0, layer_id]  # [num_blocks, block_size, num_kv_heads, head_dim]
        num_blocks, block_size_dim, num_kv_heads, head_dim = k_cache.shape
        
        # Build index for each sequence separately (v1: one index per sequence)
        if layer_id not in self.indices:
            self.indices[layer_id] = {}
        if layer_id not in self.prefill_ranges:
            self.prefill_ranges[layer_id] = {}
        
        for seq in seqs:
            if seq.seq_id in self.indexed_sequences:
                continue  # Skip sequences already indexed
            
            seq_len = len(seq)
            if not self.should_use_sparse(seq_len):
                continue
            
            # Extract keys for this sequence's prefill tokens
            seq_keys = []
            for logical_block_idx, physical_block_idx in enumerate(seq.block_table):
                if physical_block_idx < 0:
                    break
                # Get the actual number of tokens in this block
                if logical_block_idx == len(seq.block_table) - 1:
                    num_tokens_in_block = seq.last_block_num_tokens
                else:
                    num_tokens_in_block = block_size
                
                # Extract keys from physical block: [num_tokens_in_block, num_kv_heads, head_dim]
                if physical_block_idx < num_blocks:
                    block_keys = k_cache[physical_block_idx, :num_tokens_in_block]
                    seq_keys.append(block_keys)
            
            if not seq_keys:
                continue
            
            # Concatenate all blocks for this sequence
            seq_keys_tensor = torch.cat(seq_keys, dim=0)  # [seq_len, num_kv_heads, head_dim]
            
            if self.granularity == "layer_shared":
                # Flatten to [seq_len, num_kv_heads * head_dim]
                seq_keys_flat = seq_keys_tensor.flatten(1)  # [seq_len, num_kv_heads * head_dim]
            elif self.granularity == "per_head":
                # For per_head, we'll handle this separately
                # For now, fall back to layer_shared
                seq_keys_flat = seq_keys_tensor.flatten(1)
            else:
                raise ValueError(f"Unknown granularity: {self.granularity}")
            
            # Convert to CPU numpy for MLANN
            corpus_np = seq_keys_flat.detach().cpu().numpy().astype(np.float32)
            
            # Build index for this sequence
            index = MLANNIndex(
                metric=self.config.sparse_distance_metric,
                index_type="PCA"
            )
            index.build(
                corpus_np,
                n_trees=self.config.sparse_mlann_n_trees,
                depth=self.config.sparse_mlann_depth,
                votes_required=self.config.sparse_mlann_votes_required,
            )
            
            # Store index and prefill range
            self.indices[layer_id][seq.seq_id] = index
            self.prefill_ranges[layer_id][seq.seq_id] = (0, seq_len)
            self.seq_block_tables[seq.seq_id] = seq.block_table.copy()
            
            # Mark sequence as indexed
            self.indexed_sequences.add(seq.seq_id)
    
    def query_layer(
        self,
        layer_id: int,
        queries: torch.Tensor,
        seq_id: int,
    ) -> Optional[torch.Tensor]:
        """
        Query the ANN index for a layer.
        
        Args:
            layer_id: Layer index.
            queries: Query vectors of shape [batch_size, num_heads, head_dim] or
                    [batch_size, num_heads * head_dim] depending on granularity.
            seq_id: Sequence ID to query for.
            
        Returns:
            Indices tensor of shape [batch_size, topk] or None if sparse not applicable.
            Indices are relative to the sequence start (0-based within the sequence).
        """
        if not self.use_sparse:
            return None
        
        if layer_id not in self.indices:
            return None
        
        if seq_id not in self.indices[layer_id]:
            return None
        
        index = self.indices[layer_id][seq_id]
        
        # Reshape queries based on granularity
        if self.granularity == "layer_shared":
            # queries should be [batch_size, num_heads * head_dim]
            if queries.ndim == 3:  # [batch_size, num_heads, head_dim]
                batch_size, num_heads, head_dim = queries.shape
                queries = queries.flatten(1)  # [batch_size, num_heads * head_dim]
        else:
            raise NotImplementedError("per_head granularity not yet implemented")
        
        # Query the index
        indices = index.query_torch(
            queries,
            k=self.topk,
            votes_required=self.config.sparse_mlann_votes_required
        )
        
        return indices
    
    def get_block_table(self, seq_id: int) -> Optional[List[int]]:
        """Get block table for a sequence."""
        return self.seq_block_tables.get(seq_id)
    
    def get_prefill_range(self, layer_id: int, seq_id: int) -> Optional[tuple]:
        """Get the prefill token range for a sequence in a layer."""
        if layer_id not in self.prefill_ranges:
            return None
        return self.prefill_ranges[layer_id].get(seq_id)
    
    def clear_indices(self):
        """Clear all indices (e.g., for new batch)."""
        self.indices.clear()
        self.prefill_ranges.clear()
        self.indexed_sequences.clear()
        self.seq_block_tables.clear()
