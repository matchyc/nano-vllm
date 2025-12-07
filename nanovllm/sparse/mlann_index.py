"""
MLANN index wrapper for sparse attention.

This module provides a thin wrapper around the MLANN library to build and query
ANN indices over key vectors for sparse attention computation.
"""

import numpy as np
import torch
import mlann
from typing import Optional


class MLANNIndex:
    """
    A wrapper around MLANN index for building and querying ANN indices.
    
    This class handles device round-trips (GPU <-> CPU) and provides a simple
    interface for building indices from prefill keys and querying them during decode.
    """
    
    def __init__(self, metric: str = "ip", index_type: str = "PCA"):
        """
        Initialize an MLANN index.
        
        Args:
            metric: Distance metric, either "ip" (inner product) or "l2" (L2 distance).
            index_type: MLANN index type, one of "PCA", "RP", or "RF".
        """
        self.metric = metric
        self.index_type = index_type
        self.mlann_dist = mlann.IP if metric == "ip" else mlann.L2
        self.index: Optional[mlann.MLANNIndex] = None
        self.built = False
        self.corpus_size = 0
        self.dim = 0
        
    def build(
        self,
        corpus: np.ndarray,
        training_data: Optional[np.ndarray] = None,
        training_k: int = 50,
        n_trees: int = 10,
        depth: int = 6,
        votes_required: int = 5,
    ) -> None:
        """
        Build the ANN index from a corpus of key vectors.
        
        Args:
            corpus: Key vectors as numpy array of shape [num_tokens, dim], dtype float32.
            training_data: Optional training data for MLANN. If None, uses a subset of corpus.
            training_k: Number of nearest neighbors to use for training.
            n_trees: Number of trees in the MLANN index.
            depth: Depth of trees in the MLANN index.
            votes_required: Voting threshold for candidates in linear search phase.
        """
        if self.built:
            raise RuntimeError("Index has already been built")
        
        if not isinstance(corpus, np.ndarray):
            raise ValueError("corpus must be a numpy array")
        if corpus.dtype != np.float32:
            corpus = corpus.astype(np.float32)
        if not corpus.flags["C_CONTIGUOUS"]:
            corpus = np.ascontiguousarray(corpus)
        
        self.corpus_size, self.dim = corpus.shape
        
        # Use a subset of corpus as training data if not provided
        if training_data is None:
            # Use first 30% of corpus as training data, but at least 100 samples
            n_train = max(100, min(int(0.3 * self.corpus_size), 10000))
            training_data = corpus[:n_train].copy()
        else:
            if training_data.dtype != np.float32:
                training_data = training_data.astype(np.float32)
            if not training_data.flags["C_CONTIGUOUS"]:
                training_data = np.ascontiguousarray(training_data)
        
        # Initialize MLANN index
        self.index = mlann.MLANNIndex(corpus, self.index_type)
        
        # Compute training KNN using exact search
        knn = self.index.exact_search(training_data, training_k, dist=self.mlann_dist)
        
        # Build the index
        self.index.build(training_data, knn, n_trees, depth, density="auto", b=votes_required)
        self.built = True
        self.votes_required = votes_required
    
    def query(self, queries: np.ndarray, k: int, votes_required: Optional[int] = None) -> np.ndarray:
        """
        Query the index for top-k nearest neighbors.
        
        Args:
            queries: Query vectors as numpy array of shape [num_queries, dim], dtype float32.
            k: Number of nearest neighbors to return.
            
        Returns:
            Indices array of shape [num_queries, k] with dtype int64.
        """
        if not self.built:
            raise RuntimeError("Cannot query before building index")
        
        if not isinstance(queries, np.ndarray):
            raise ValueError("queries must be a numpy array")
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)
        if not queries.flags["C_CONTIGUOUS"]:
            queries = np.ascontiguousarray(queries)
        
        # Handle single query vector
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        
        # Query the index
        if votes_required is None:
            votes_required = getattr(self, 'votes_required', 5)
        indices = self.index.ann(
            queries,
            k,
            votes_required=votes_required,
            dist=self.mlann_dist,
            return_distances=False
        )
        
        # Ensure output is int64
        if indices.dtype != np.int64:
            indices = indices.astype(np.int64)
        
        return indices
    
    def query_torch(self, queries: torch.Tensor, k: int, votes_required: Optional[int] = None) -> torch.Tensor:
        """
        Query the index using PyTorch tensors (handles GPU->CPU conversion).
        
        Args:
            queries: Query vectors as torch.Tensor of shape [num_queries, dim].
                    Can be on GPU or CPU.
            k: Number of nearest neighbors to return.
            
        Returns:
            Indices tensor of shape [num_queries, k] on the same device as queries.
        """
        device = queries.device
        # Convert to CPU numpy
        queries_cpu = queries.detach().cpu().numpy()
        if queries_cpu.ndim == 1:
            queries_cpu = queries_cpu.reshape(1, -1)
        
        # Query on CPU
        indices_cpu = self.query(queries_cpu, k, votes_required=votes_required)
        
        # Convert back to torch tensor on original device
        return torch.from_numpy(indices_cpu).to(device)
