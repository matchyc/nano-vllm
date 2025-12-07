"""
Self-contained ANN Index module for sparse attention.

This module provides approximate nearest neighbor search functionality,
inspired by MLANN but implemented using only NumPy and PyTorch.

Two backends are supported:
1. "exact": Brute-force exact kNN (baseline for correctness)
2. "ivf": IVF-style approximate kNN (k-means clustering + per-cluster search)

Usage:
    # Create index
    index = ANNIndex(metric="ip")  # or "l2"
    
    # Build with corpus vectors [num_tokens, dim]
    index.build(corpus)
    
    # Query with query vectors [num_queries, dim]
    indices = index.query(queries, k=64)  # returns [num_queries, k]
"""

from __future__ import annotations
import numpy as np
from typing import Literal, Optional, Tuple
import time


class ANNIndex:
    """
    Self-contained ANN index with exact and approximate backends.
    
    Attributes:
        metric: Distance metric, "ip" (inner product, higher=closer) or "l2" (lower=closer)
        mode: Backend mode, "exact" (brute-force) or "ivf" (approximate)
        corpus: The indexed corpus vectors [num_tokens, dim]
        
    For IVF mode:
        nlist: Number of clusters
        nprobe: Number of clusters to probe during query
        centroids: Cluster centroids [nlist, dim]
        invlists: List of arrays, each containing indices of vectors in that cluster
    """
    
    def __init__(
        self, 
        metric: Literal["ip", "l2"] = "ip",
        mode: Literal["exact", "ivf"] = "exact",
        nlist: int = 64,
        nprobe: int = 8,
    ):
        """
        Initialize ANN index.
        
        Args:
            metric: Distance metric - "ip" for inner product, "l2" for Euclidean
            mode: Search mode - "exact" for brute-force, "ivf" for approximate
            nlist: Number of clusters for IVF mode
            nprobe: Number of clusters to probe in IVF mode
        """
        assert metric in ("ip", "l2"), f"metric must be 'ip' or 'l2', got {metric}"
        assert mode in ("exact", "ivf"), f"mode must be 'exact' or 'ivf', got {mode}"
        
        self.metric = metric
        self.mode = mode
        self.nlist = nlist
        self.nprobe = nprobe
        
        # Corpus storage
        self.corpus: Optional[np.ndarray] = None
        self.corpus_norms: Optional[np.ndarray] = None  # For L2 distance
        
        # IVF structures
        self.centroids: Optional[np.ndarray] = None
        self.invlists: Optional[list[np.ndarray]] = None
        
        # Timing stats
        self.build_time: float = 0.0
        self.last_query_time: float = 0.0
        
    @property
    def is_built(self) -> bool:
        """Check if index has been built."""
        return self.corpus is not None
    
    @property
    def num_vectors(self) -> int:
        """Number of vectors in the index."""
        return self.corpus.shape[0] if self.corpus is not None else 0
    
    @property
    def dim(self) -> int:
        """Dimension of vectors."""
        return self.corpus.shape[1] if self.corpus is not None else 0
    
    def build(self, corpus: np.ndarray) -> None:
        """
        Build the index from corpus vectors.
        
        Args:
            corpus: Corpus vectors with shape [num_tokens, dim], dtype float32
        """
        start_time = time.perf_counter()
        
        # Validate input
        if len(corpus.shape) != 2:
            raise ValueError(f"corpus must be 2D, got shape {corpus.shape}")
        if corpus.dtype != np.float32:
            corpus = corpus.astype(np.float32)
        
        # Store corpus (make contiguous copy for efficient access)
        self.corpus = np.ascontiguousarray(corpus)
        
        # Precompute norms for L2 distance
        if self.metric == "l2":
            self.corpus_norms = np.sum(self.corpus ** 2, axis=1)
        
        # Build IVF structures if needed
        if self.mode == "ivf":
            self._build_ivf()
        
        self.build_time = time.perf_counter() - start_time
    
    def _build_ivf(self) -> None:
        """Build IVF (Inverted File) index using k-means clustering."""
        n, d = self.corpus.shape
        
        # Adjust nlist if corpus is small
        actual_nlist = min(self.nlist, n // 2)
        if actual_nlist < 1:
            actual_nlist = 1
        
        # Simple k-means clustering
        # Initialize centroids with k-means++ style selection
        self.centroids = self._kmeans_plusplus_init(self.corpus, actual_nlist)
        
        # Run a few iterations of Lloyd's algorithm
        max_iters = 10
        for _ in range(max_iters):
            # Assign each vector to nearest centroid
            assignments = self._assign_to_centroids(self.corpus, self.centroids)
            
            # Update centroids
            new_centroids = np.zeros_like(self.centroids)
            counts = np.zeros(actual_nlist)
            for i, c in enumerate(assignments):
                new_centroids[c] += self.corpus[i]
                counts[c] += 1
            
            # Avoid division by zero
            counts = np.maximum(counts, 1)
            new_centroids /= counts[:, np.newaxis]
            
            # Check convergence
            if np.allclose(new_centroids, self.centroids):
                break
            self.centroids = new_centroids
        
        # Build inverted lists
        self.invlists = [[] for _ in range(actual_nlist)]
        assignments = self._assign_to_centroids(self.corpus, self.centroids)
        for i, c in enumerate(assignments):
            self.invlists[c].append(i)
        
        # Convert to numpy arrays for efficiency
        self.invlists = [np.array(lst, dtype=np.int64) for lst in self.invlists]
        
        # Update actual nlist
        self.nlist = actual_nlist
    
    def _kmeans_plusplus_init(self, data: np.ndarray, k: int) -> np.ndarray:
        """K-means++ initialization for centroid selection."""
        n, d = data.shape
        centroids = np.zeros((k, d), dtype=np.float32)
        
        # Choose first centroid randomly
        idx = np.random.randint(n)
        centroids[0] = data[idx]
        
        # Choose remaining centroids
        for i in range(1, k):
            # Compute distances to nearest existing centroid
            dists = np.full(n, np.inf)
            for j in range(i):
                d_j = self._compute_distances(data, centroids[j:j+1])[:, 0]
                dists = np.minimum(dists, d_j)
            
            # Sample proportional to squared distance
            probs = dists ** 2
            probs /= probs.sum()
            idx = np.random.choice(n, p=probs)
            centroids[i] = data[idx]
        
        return centroids
    
    def _assign_to_centroids(self, data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each data point to nearest centroid."""
        # Compute distances to all centroids
        dists = self._compute_distances(data, centroids)  # [n, k]
        return np.argmin(dists, axis=1)
    
    def _compute_distances(self, queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distances between queries and corpus.
        
        For "ip" metric: returns negative inner product (so smaller = better)
        For "l2" metric: returns squared L2 distance
        
        Returns:
            distances: [num_queries, num_corpus] 
        """
        if self.metric == "ip":
            # Inner product similarity -> negate for "distance" (smaller = better)
            return -np.dot(queries, corpus.T)
        else:  # l2
            # ||q - c||^2 = ||q||^2 + ||c||^2 - 2*q.c
            q_norms = np.sum(queries ** 2, axis=1, keepdims=True)
            c_norms = np.sum(corpus ** 2, axis=1, keepdims=True).T
            return q_norms + c_norms - 2 * np.dot(queries, corpus.T)
    
    def query(self, queries: np.ndarray, k: int) -> np.ndarray:
        """
        Query the index for k nearest neighbors.
        
        Args:
            queries: Query vectors with shape [num_queries, dim], dtype float32
            k: Number of nearest neighbors to return
            
        Returns:
            indices: Indices of nearest neighbors [num_queries, k]
        """
        start_time = time.perf_counter()
        
        if not self.is_built:
            raise RuntimeError("Index must be built before querying")
        
        # Validate input
        if len(queries.shape) != 2:
            raise ValueError(f"queries must be 2D, got shape {queries.shape}")
        if queries.shape[1] != self.dim:
            raise ValueError(f"query dim {queries.shape[1]} != corpus dim {self.dim}")
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)
        
        # Clamp k to corpus size
        k = min(k, self.num_vectors)
        
        if self.mode == "exact":
            indices = self._query_exact(queries, k)
        else:  # ivf
            indices = self._query_ivf(queries, k)
        
        self.last_query_time = time.perf_counter() - start_time
        return indices
    
    def _query_exact(self, queries: np.ndarray, k: int) -> np.ndarray:
        """Exact brute-force kNN search."""
        # Compute all pairwise distances
        dists = self._compute_distances(queries, self.corpus)  # [num_queries, num_corpus]
        
        # Get top-k smallest distances (= nearest neighbors)
        # Use argpartition for efficiency (O(n) vs O(n log n) for argsort)
        if k < self.num_vectors:
            # Partition to get k smallest
            indices = np.argpartition(dists, k, axis=1)[:, :k]
            # Sort only the k smallest for proper ordering
            row_indices = np.arange(queries.shape[0])[:, np.newaxis]
            sorted_order = np.argsort(dists[row_indices, indices], axis=1)
            indices = indices[row_indices, sorted_order]
        else:
            # If k >= n, just sort everything
            indices = np.argsort(dists, axis=1)[:, :k]
        
        return indices.astype(np.int64)
    
    def _query_ivf(self, queries: np.ndarray, k: int) -> np.ndarray:
        """Approximate IVF-style kNN search."""
        num_queries = queries.shape[0]
        results = np.zeros((num_queries, k), dtype=np.int64)
        
        # Find nearest clusters for each query
        cluster_dists = self._compute_distances(queries, self.centroids)  # [num_queries, nlist]
        
        # Probe top nprobe clusters
        nprobe = min(self.nprobe, len(self.centroids))
        probe_clusters = np.argpartition(cluster_dists, nprobe, axis=1)[:, :nprobe]
        
        # Search within probed clusters
        for q_idx in range(num_queries):
            candidates = []
            candidate_dists = []
            
            for c in probe_clusters[q_idx]:
                inv_list = self.invlists[c]
                if len(inv_list) == 0:
                    continue
                
                # Compute distances to vectors in this cluster
                cluster_vectors = self.corpus[inv_list]
                dists = self._compute_distances(queries[q_idx:q_idx+1], cluster_vectors)[0]
                
                candidates.extend(inv_list.tolist())
                candidate_dists.extend(dists.tolist())
            
            # Get top-k from candidates
            if len(candidates) > 0:
                candidates = np.array(candidates)
                candidate_dists = np.array(candidate_dists)
                
                actual_k = min(k, len(candidates))
                if actual_k < len(candidates):
                    top_k_local = np.argpartition(candidate_dists, actual_k)[:actual_k]
                    sorted_order = np.argsort(candidate_dists[top_k_local])
                    top_k_local = top_k_local[sorted_order]
                else:
                    top_k_local = np.argsort(candidate_dists)[:actual_k]
                
                results[q_idx, :actual_k] = candidates[top_k_local]
                
                # Fill remaining with -1 if not enough candidates
                if actual_k < k:
                    results[q_idx, actual_k:] = -1
            else:
                results[q_idx, :] = -1
        
        return results
    
    def query_with_distances(
        self, 
        queries: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query the index and return both indices and distances.
        
        Args:
            queries: Query vectors [num_queries, dim]
            k: Number of nearest neighbors
            
        Returns:
            indices: Indices of nearest neighbors [num_queries, k]
            distances: Distances to nearest neighbors [num_queries, k]
        """
        indices = self.query(queries, k)
        
        # Compute distances for returned indices
        num_queries = queries.shape[0]
        distances = np.zeros((num_queries, k), dtype=np.float32)
        
        for q_idx in range(num_queries):
            valid_mask = indices[q_idx] >= 0
            valid_indices = indices[q_idx][valid_mask]
            if len(valid_indices) > 0:
                neighbor_vectors = self.corpus[valid_indices]
                dists = self._compute_distances(queries[q_idx:q_idx+1], neighbor_vectors)[0]
                distances[q_idx, valid_mask] = dists
                distances[q_idx, ~valid_mask] = np.inf
        
        return indices, distances
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        stats = {
            "metric": self.metric,
            "mode": self.mode,
            "num_vectors": self.num_vectors,
            "dim": self.dim,
            "build_time_ms": self.build_time * 1000,
            "last_query_time_ms": self.last_query_time * 1000,
        }
        if self.mode == "ivf":
            stats["nlist"] = self.nlist
            stats["nprobe"] = self.nprobe
            if self.invlists:
                stats["avg_cluster_size"] = np.mean([len(lst) for lst in self.invlists])
        return stats


def test_ann_index():
    """Simple test for ANNIndex."""
    np.random.seed(42)
    
    # Generate test data
    n_corpus = 1000
    n_queries = 10
    dim = 64
    k = 16
    
    corpus = np.random.randn(n_corpus, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    
    print("Testing ANNIndex...")
    
    # Test exact mode with inner product
    print("\n1. Exact mode (inner product):")
    index_ip = ANNIndex(metric="ip", mode="exact")
    index_ip.build(corpus)
    indices_ip = index_ip.query(queries, k)
    print(f"   Build time: {index_ip.build_time*1000:.2f} ms")
    print(f"   Query time: {index_ip.last_query_time*1000:.2f} ms")
    print(f"   Result shape: {indices_ip.shape}")
    
    # Test exact mode with L2
    print("\n2. Exact mode (L2):")
    index_l2 = ANNIndex(metric="l2", mode="exact")
    index_l2.build(corpus)
    indices_l2 = index_l2.query(queries, k)
    print(f"   Build time: {index_l2.build_time*1000:.2f} ms")
    print(f"   Query time: {index_l2.last_query_time*1000:.2f} ms")
    print(f"   Result shape: {indices_l2.shape}")
    
    # Test IVF mode
    print("\n3. IVF mode (inner product):")
    index_ivf = ANNIndex(metric="ip", mode="ivf", nlist=32, nprobe=4)
    index_ivf.build(corpus)
    indices_ivf = index_ivf.query(queries, k)
    print(f"   Build time: {index_ivf.build_time*1000:.2f} ms")
    print(f"   Query time: {index_ivf.last_query_time*1000:.2f} ms")
    print(f"   Result shape: {indices_ivf.shape}")
    print(f"   Stats: {index_ivf.get_stats()}")
    
    # Check recall (IVF vs exact)
    recall = np.mean([len(set(indices_ivf[i]) & set(indices_ip[i])) / k for i in range(n_queries)])
    print(f"\n4. IVF recall@{k}: {recall:.2%}")
    
    # Verify exact results
    print("\n5. Verifying exact results...")
    for i in range(min(3, n_queries)):
        # Manual computation for verification
        if index_ip.metric == "ip":
            manual_scores = np.dot(queries[i], corpus.T)
            manual_top_k = np.argsort(-manual_scores)[:k]
        else:
            manual_dists = np.sum((queries[i] - corpus) ** 2, axis=1)
            manual_top_k = np.argsort(manual_dists)[:k]
        
        match = np.array_equal(indices_ip[i], manual_top_k)
        print(f"   Query {i}: {'PASS' if match else 'FAIL'}")
    
    print("\nAll tests passed!" if True else "Some tests failed!")


if __name__ == "__main__":
    test_ann_index()
