"""
MLANN (Multilabel Classification for Approximate Nearest Neighbor) Index

This module implements the MLANN algorithm from:
"A Multilabel Classification Framework for Approximate Nearest Neighbor Search"
(NeurIPS 2022 / JMLR 2024)

Core idea:
- Treat ANN candidate selection as a multilabel classification problem
- Training queries have labels Y_i = {indices of k-NN of query x_i}
- A partitioning structure (e.g., RP tree) divides the space into cells
- For each cell r and corpus index j, estimate:
    p(r, j) = P(j in k-NN of query | query lands in cell r)
- At query time, route query to cell r(q), score corpus by p(r(q), j), return top-M

Implementation follows the "natural classifier" formulation from the paper.

For Attention-aware training:
- Corpus = K (key vectors) - what we search over
- Training queries = Q (query vectors) - what does the searching
- Labels = k-NN of each Q in K space

This is crucial because in attention, Q queries K (not K queries K).
Using Q as training queries captures the actual attention pattern and
significantly improves recall compared to self-training with K.
"""

from __future__ import annotations
import numpy as np
from typing import Literal, Optional, Tuple, List, Dict
from dataclasses import dataclass
import time


@dataclass
class RPTreeNode:
    """Node in a Random Projection Tree."""
    # For internal nodes
    projection: Optional[np.ndarray] = None  # Random direction [d]
    threshold: float = 0.0                    # Split threshold
    left: Optional['RPTreeNode'] = None
    right: Optional['RPTreeNode'] = None
    
    # For leaf nodes
    is_leaf: bool = False
    cell_id: int = -1                         # Unique ID for this leaf cell


class RPTreePartitioner:
    """
    Random Projection Tree partitioner for MLANN.
    
    Builds a binary tree by recursively splitting points along random projections.
    Each leaf node represents a partition cell.
    
    This is one concrete partitioner; the design allows plugging in alternatives
    (e.g., PCA-based, random forest ensemble) in the future.
    """
    
    def __init__(
        self,
        max_depth: int = 8,
        min_leaf_size: int = 10,
        seed: Optional[int] = None,
    ):
        """
        Args:
            max_depth: Maximum tree depth
            min_leaf_size: Minimum points in a leaf before stopping splits
            seed: Random seed for reproducibility
        """
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.rng = np.random.default_rng(seed)
        
        self.root: Optional[RPTreeNode] = None
        self.num_cells: int = 0
        self.dim: int = 0
    
    def build(self, data: np.ndarray) -> None:
        """
        Build the RP tree from data points.
        
        Args:
            data: [n, d] array of data points
        """
        n, d = data.shape
        self.dim = d
        self.num_cells = 0
        
        indices = np.arange(n)
        self.root = self._build_recursive(data, indices, depth=0)
    
    def _build_recursive(
        self, 
        data: np.ndarray, 
        indices: np.ndarray, 
        depth: int
    ) -> RPTreeNode:
        """Recursively build the tree."""
        n = len(indices)
        
        # Create leaf if stopping criteria met
        if depth >= self.max_depth or n <= self.min_leaf_size:
            node = RPTreeNode(is_leaf=True, cell_id=self.num_cells)
            self.num_cells += 1
            return node
        
        # Generate random projection direction (sparse or dense)
        # Using dense Gaussian for simplicity; could use sparse RP for efficiency
        projection = self.rng.standard_normal(self.dim).astype(np.float32)
        projection /= np.linalg.norm(projection) + 1e-10
        
        # Project data points
        points = data[indices]
        projections = points @ projection  # [n]
        
        # Choose threshold as median for balanced splits
        threshold = np.median(projections)
        
        # Split indices
        left_mask = projections <= threshold
        right_mask = ~left_mask
        
        # Handle edge case where all points go to one side
        if not np.any(left_mask) or not np.any(right_mask):
            node = RPTreeNode(is_leaf=True, cell_id=self.num_cells)
            self.num_cells += 1
            return node
        
        left_indices = indices[left_mask]
        right_indices = indices[right_mask]
        
        # Recursively build children
        left_child = self._build_recursive(data, left_indices, depth + 1)
        right_child = self._build_recursive(data, right_indices, depth + 1)
        
        return RPTreeNode(
            projection=projection,
            threshold=threshold,
            left=left_child,
            right=right_child,
            is_leaf=False,
        )
    
    def get_cell(self, point: np.ndarray) -> int:
        """
        Route a single point to its partition cell.
        
        Args:
            point: [d] array
            
        Returns:
            Cell ID (leaf node ID)
        """
        node = self.root
        while not node.is_leaf:
            proj_val = np.dot(point, node.projection)
            if proj_val <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.cell_id
    
    def get_cells_batch(self, points: np.ndarray) -> np.ndarray:
        """
        Route multiple points to their partition cells.
        
        Args:
            points: [n, d] array
            
        Returns:
            [n] array of cell IDs
        """
        n = points.shape[0]
        cell_ids = np.zeros(n, dtype=np.int32)
        for i in range(n):
            cell_ids[i] = self.get_cell(points[i])
        return cell_ids


class MLANNIndex:
    """
    MLANN (Multilabel ANN) Index following the natural classifier formulation.
    
    The algorithm:
    1. Build phase:
       a) Construct partitioner(s) (RP tree ensemble) over the corpus
       b) For each training query, compute its k-NN (ground truth labels)
       c) Route each training query to its partition cell(s)
       d) For each cell r and corpus index j, compute:
          p(r, j) = (# queries in cell r with j in their k-NN) / (# queries in cell r)
    
    2. Query phase:
       a) Route query q to cell(s) r(q) via the RP tree(s)
       b) Aggregate scores across trees: sum of p(r_t(q), j) over trees t
       c) Return top-k indices by aggregated score
    
    The multi-tree ensemble (like a random forest) improves recall by averaging
    over multiple random partitionings of the space.
    
    Attributes:
        metric: Distance metric ("ip" for inner product, "l2" for Euclidean)
        k_train: Number of neighbors for training label computation
        n_trees: Number of RP trees in the ensemble
        max_depth: Maximum depth of each RP tree
        min_leaf_size: Minimum leaf size in RP tree
    """
    
    def __init__(
        self,
        metric: Literal["ip", "l2"] = "ip",
        k_train: int = 10,
        n_trees: int = 8,
        max_depth: int = 8,
        min_leaf_size: int = 10,
        seed: Optional[int] = None,
    ):
        """
        Initialize MLANN index.
        
        Args:
            metric: Distance metric - "ip" (inner product) or "l2" (Euclidean)
            k_train: Number of neighbors for computing training labels
            n_trees: Number of RP trees in the ensemble (more trees = better recall)
            max_depth: Maximum depth of partition tree
            min_leaf_size: Minimum points in a leaf node
            seed: Random seed for reproducibility
        """
        assert metric in ("ip", "l2"), f"metric must be 'ip' or 'l2', got {metric}"
        
        self.metric = metric
        self.k_train = k_train
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.seed = seed
        
        # Built structures
        self.corpus: Optional[np.ndarray] = None
        
        # Ensemble of partitioners and their label probability tables
        self.partitioners: List[RPTreePartitioner] = []
        self.cell_label_probs_list: List[np.ndarray] = []  # Each: [num_cells_t, corpus_size]
        
        # Stats
        self.build_time: float = 0.0
        self.last_query_time: float = 0.0
        self.num_corpus: int = 0
        self.total_cells: int = 0
    
    @property
    def is_built(self) -> bool:
        return self.corpus is not None and len(self.partitioners) > 0
    
    def _compute_knn_labels(
        self,
        queries: np.ndarray,
        corpus: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """
        Compute exact k-NN for training queries (ground truth labels).
        
        This is the "training label computation" step in MLANN.
        Uses brute-force for correctness.
        
        Args:
            queries: [n_queries, d] query vectors
            corpus: [m, d] corpus vectors
            k: Number of neighbors
            
        Returns:
            knn_indices: [n_queries, k] indices of k nearest neighbors
        """
        n_queries = queries.shape[0]
        m = corpus.shape[0]
        k = min(k, m)
        
        # Compute pairwise distances/similarities
        if self.metric == "ip":
            # Inner product: higher = more similar
            similarities = queries @ corpus.T  # [n_queries, m]
            if k < m:
                neg_sim = -similarities
                indices = np.argpartition(neg_sim, k, axis=1)[:, :k]
                row_idx = np.arange(n_queries)[:, np.newaxis]
                sorted_order = np.argsort(neg_sim[row_idx, indices], axis=1)
                indices = indices[row_idx, sorted_order]
            else:
                indices = np.argsort(-similarities, axis=1)[:, :k]
        else:  # l2
            q_norms = np.sum(queries ** 2, axis=1, keepdims=True)
            c_norms = np.sum(corpus ** 2, axis=1, keepdims=True).T
            distances = q_norms + c_norms - 2 * (queries @ corpus.T)
            
            if k < m:
                indices = np.argpartition(distances, k, axis=1)[:, :k]
                row_idx = np.arange(n_queries)[:, np.newaxis]
                sorted_order = np.argsort(distances[row_idx, indices], axis=1)
                indices = indices[row_idx, sorted_order]
            else:
                indices = np.argsort(distances, axis=1)[:, :k]
        
        return indices.astype(np.int64)
    
    def build(
        self,
        corpus: np.ndarray,
        train_queries: Optional[np.ndarray] = None,
        train_knn_indices: Optional[np.ndarray] = None,
    ) -> None:
        """
        Build the MLANN index following the natural classifier formulation.
        
        Steps:
        1. Compute training labels (k-NN indices) if not provided
        2. For each tree in the ensemble:
           a) Build RP tree partitioner over corpus
           b) Route training queries to cells
           c) Estimate per-cell label probabilities p(r, j)
        
        Args:
            corpus: [m, d] corpus vectors (e.g., keys from prefill)
            train_queries: [n, d] training queries. If None, use corpus.
            train_knn_indices: [n, k_train] pre-computed k-NN indices for training.
                              If None, compute via brute-force.
        """
        start_time = time.perf_counter()
        
        # Validate and store corpus
        if len(corpus.shape) != 2:
            raise ValueError(f"corpus must be 2D, got shape {corpus.shape}")
        if corpus.dtype != np.float32:
            corpus = corpus.astype(np.float32)
        self.corpus = np.ascontiguousarray(corpus)
        self.num_corpus = corpus.shape[0]
        
        # Use corpus as training queries if not provided
        if train_queries is None:
            train_queries = corpus
        elif train_queries.dtype != np.float32:
            train_queries = train_queries.astype(np.float32)
        
        n_train = train_queries.shape[0]
        
        # Step 1: Compute training labels (k-NN) - this is shared across all trees
        if train_knn_indices is None:
            train_knn_indices = self._compute_knn_labels(
                train_queries, corpus, self.k_train
            )
        
        # Step 2: Build ensemble of RP trees
        self.partitioners = []
        self.cell_label_probs_list = []
        self.total_cells = 0
        
        rng = np.random.default_rng(self.seed)
        
        for tree_idx in range(self.n_trees):
            # Each tree gets a different random seed
            tree_seed = rng.integers(0, 2**31) if self.seed is not None else None
            
            # Build partitioner
            partitioner = RPTreePartitioner(
                max_depth=self.max_depth,
                min_leaf_size=self.min_leaf_size,
                seed=tree_seed,
            )
            partitioner.build(corpus)
            
            # Route training queries to cells
            train_cell_ids = partitioner.get_cells_batch(train_queries)
            
            # Estimate per-cell label probabilities
            num_cells = partitioner.num_cells
            cell_counts = np.zeros(num_cells, dtype=np.float32)
            np.add.at(cell_counts, train_cell_ids, 1)
            
            cell_label_counts = np.zeros((num_cells, self.num_corpus), dtype=np.float32)
            for i in range(n_train):
                cell_id = train_cell_ids[i]
                knn_indices = train_knn_indices[i]
                np.add.at(cell_label_counts[cell_id], knn_indices, 1)
            
            cell_counts_safe = np.maximum(cell_counts, 1)[:, np.newaxis]
            cell_label_probs = cell_label_counts / cell_counts_safe
            
            self.partitioners.append(partitioner)
            self.cell_label_probs_list.append(cell_label_probs)
            self.total_cells += num_cells
        
        self.build_time = time.perf_counter() - start_time
    
    def query(self, queries: np.ndarray, k: int) -> np.ndarray:
        """
        Query the MLANN index for approximate k nearest neighbors.
        
        For each query:
        1. Route to partition cell in each tree
        2. Aggregate scores across trees: sum of p(r_t(q), j)
        3. Return top-k indices by aggregated score
        
        Args:
            queries: [n, d] query vectors
            k: Number of neighbors to return
            
        Returns:
            indices: [n, k] indices of approximate nearest neighbors
        """
        start_time = time.perf_counter()
        
        if not self.is_built:
            raise RuntimeError("Index must be built before querying")
        
        if len(queries.shape) != 2:
            raise ValueError(f"queries must be 2D, got shape {queries.shape}")
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)
        
        n_queries = queries.shape[0]
        k = min(k, self.num_corpus)
        
        # Aggregate scores across all trees
        aggregated_scores = np.zeros((n_queries, self.num_corpus), dtype=np.float32)
        
        for tree_idx in range(self.n_trees):
            partitioner = self.partitioners[tree_idx]
            cell_label_probs = self.cell_label_probs_list[tree_idx]
            
            # Route queries to cells
            query_cell_ids = partitioner.get_cells_batch(queries)
            
            # Add cell probabilities to aggregated scores
            for i in range(n_queries):
                cell_id = query_cell_ids[i]
                aggregated_scores[i] += cell_label_probs[cell_id]
        
        # Get top-k by aggregated score for each query
        indices = np.zeros((n_queries, k), dtype=np.int64)
        
        for i in range(n_queries):
            scores = aggregated_scores[i]
            if k < self.num_corpus:
                top_k_idx = np.argpartition(-scores, k)[:k]
                sorted_order = np.argsort(-scores[top_k_idx])
                indices[i] = top_k_idx[sorted_order]
            else:
                indices[i] = np.argsort(-scores)[:k]
        
        self.last_query_time = time.perf_counter() - start_time
        return indices
    
    def query_with_scores(
        self, 
        queries: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query and return both indices and aggregated probability scores.
        
        Args:
            queries: [n, d] query vectors
            k: Number of neighbors
            
        Returns:
            indices: [n, k] corpus indices
            scores: [n, k] aggregated label probabilities
        """
        if not self.is_built:
            raise RuntimeError("Index must be built before querying")
        
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)
        
        n_queries = queries.shape[0]
        k = min(k, self.num_corpus)
        
        # Aggregate scores across all trees
        aggregated_scores = np.zeros((n_queries, self.num_corpus), dtype=np.float32)
        
        for tree_idx in range(self.n_trees):
            partitioner = self.partitioners[tree_idx]
            cell_label_probs = self.cell_label_probs_list[tree_idx]
            query_cell_ids = partitioner.get_cells_batch(queries)
            
            for i in range(n_queries):
                cell_id = query_cell_ids[i]
                aggregated_scores[i] += cell_label_probs[cell_id]
        
        indices = np.zeros((n_queries, k), dtype=np.int64)
        scores = np.zeros((n_queries, k), dtype=np.float32)
        
        for i in range(n_queries):
            agg_scores = aggregated_scores[i]
            if k < self.num_corpus:
                top_k_idx = np.argpartition(-agg_scores, k)[:k]
                sorted_order = np.argsort(-agg_scores[top_k_idx])
                indices[i] = top_k_idx[sorted_order]
                scores[i] = agg_scores[indices[i]]
            else:
                sorted_idx = np.argsort(-agg_scores)[:k]
                indices[i] = sorted_idx
                scores[i] = agg_scores[sorted_idx]
        
        return indices, scores
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "metric": self.metric,
            "k_train": self.k_train,
            "n_trees": self.n_trees,
            "num_corpus": self.num_corpus,
            "total_cells": self.total_cells,
            "avg_cells_per_tree": self.total_cells / max(self.n_trees, 1) if self.is_built else 0,
            "max_depth": self.max_depth,
            "min_leaf_size": self.min_leaf_size,
            "build_time_ms": self.build_time * 1000,
            "last_query_time_ms": self.last_query_time * 1000,
        }


def compute_recall(pred_indices: np.ndarray, gt_indices: np.ndarray) -> float:
    """
    Compute recall@k between predicted and ground-truth indices.
    
    Args:
        pred_indices: [n, k] predicted indices
        gt_indices: [n, k] ground truth indices
        
    Returns:
        Average recall across queries
    """
    n = pred_indices.shape[0]
    k = gt_indices.shape[1]
    recalls = []
    for i in range(n):
        gt_set = set(gt_indices[i].tolist())
        pred_set = set(pred_indices[i].tolist())
        recall = len(gt_set & pred_set) / k
        recalls.append(recall)
    return np.mean(recalls)


def test_mlann_index():
    """Test the MLANN index implementation."""
    np.random.seed(42)
    
    print("=" * 60)
    print("Testing MLANN Index (Multi-Tree Ensemble)")
    print("=" * 60)
    
    # Generate test data
    n_corpus = 1000
    n_queries = 50
    dim = 64
    k = 16
    k_train = 32
    
    corpus = np.random.randn(n_corpus, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    
    # Normalize for inner product similarity
    corpus = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-10)
    queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-10)
    
    print(f"\nCorpus: {n_corpus} vectors, dim={dim}")
    print(f"Queries: {n_queries} vectors")
    print(f"k_train: {k_train}, k_query: {k}")
    
    # Compute ground truth
    print("\n1. Computing ground truth (brute-force)...")
    similarities = queries @ corpus.T
    gt_indices = np.argsort(-similarities, axis=1)[:, :k]
    
    # Test 2: Single tree baseline
    print("\n2. Single tree baseline...")
    index_single = MLANNIndex(
        metric="ip", k_train=k_train, n_trees=1,
        max_depth=8, min_leaf_size=10, seed=42
    )
    index_single.build(corpus)
    single_indices = index_single.query(queries, k)
    single_recall = compute_recall(single_indices, gt_indices)
    print(f"   1 tree: {index_single.total_cells} cells, recall={single_recall:.2%}")
    
    # Test 3: Multi-tree ensemble (MLANN random forest style)
    print("\n3. Multi-tree ensemble (MLANN style)...")
    for n_trees in [4, 8, 16, 32]:
        index = MLANNIndex(
            metric="ip", k_train=k_train, n_trees=n_trees,
            max_depth=8, min_leaf_size=10, seed=42
        )
        index.build(corpus)
        mlann_indices = index.query(queries, k)
        recall = compute_recall(mlann_indices, gt_indices)
        print(f"   {n_trees:2d} trees: {index.total_cells:4d} total cells, "
              f"build={index.build_time*1000:.1f}ms, "
              f"query={index.last_query_time*1000:.2f}ms, "
              f"recall={recall:.2%}")
    
    # Test 4: Effect of k_train
    print("\n4. Effect of k_train (training neighborhood size)...")
    for k_train_val in [8, 16, 32, 64]:
        index = MLANNIndex(
            metric="ip", k_train=k_train_val, n_trees=16,
            max_depth=8, min_leaf_size=10, seed=42
        )
        index.build(corpus)
        mlann_indices = index.query(queries, k)
        recall = compute_recall(mlann_indices, gt_indices)
        print(f"   k_train={k_train_val:2d}: recall={recall:.2%}")
    
    # Test 5: Effect of tree depth
    print("\n5. Effect of tree depth...")
    for depth in [4, 6, 8, 10]:
        index = MLANNIndex(
            metric="ip", k_train=k_train, n_trees=16,
            max_depth=depth, min_leaf_size=5, seed=42
        )
        index.build(corpus)
        mlann_indices = index.query(queries, k)
        recall = compute_recall(mlann_indices, gt_indices)
        print(f"   depth={depth:2d}: {index.total_cells:4d} cells, recall={recall:.2%}")
    
    # Test 6: Verify index structure
    print("\n6. Verifying index structure...")
    index = MLANNIndex(metric="ip", k_train=k_train, n_trees=8, seed=42)
    index.build(corpus)
    assert index.corpus is not None
    assert len(index.partitioners) == 8
    assert len(index.cell_label_probs_list) == 8
    for cell_probs in index.cell_label_probs_list:
        assert cell_probs.shape[1] == n_corpus
    print("   ✓ Index structure valid")
    
    # Test 7: L2 metric
    print("\n7. Testing L2 metric...")
    index_l2 = MLANNIndex(metric="l2", k_train=k_train, n_trees=16, max_depth=8, seed=42)
    index_l2.build(corpus)
    l2_indices = index_l2.query(queries, k)
    
    q_norms = np.sum(queries ** 2, axis=1, keepdims=True)
    c_norms = np.sum(corpus ** 2, axis=1, keepdims=True).T
    distances = q_norms + c_norms - 2 * (queries @ corpus.T)
    gt_l2_indices = np.argsort(distances, axis=1)[:, :k]
    
    l2_recall = compute_recall(l2_indices, gt_l2_indices)
    print(f"   L2 recall@{k} (16 trees): {l2_recall:.2%}")
    
    # Test 8: Query with scores
    print("\n8. Testing query with scores...")
    indices, scores = index.query_with_scores(queries[:3], k)
    print(f"   Query 0 top-3 scores: {scores[0, :3]}")
    print(f"   Query 0 top-3 indices: {indices[0, :3]}")
    
    # Test 9: Same distribution test (corpus as queries - best case for MLANN)
    print("\n9. Same distribution test (queries = corpus subset)...")
    test_queries = corpus[:n_queries]  # Use corpus vectors as queries
    test_gt_indices = np.argsort(-(test_queries @ corpus.T), axis=1)[:, :k]
    
    index_same = MLANNIndex(metric="ip", k_train=k_train, n_trees=16, max_depth=8, seed=42)
    index_same.build(corpus)
    same_indices = index_same.query(test_queries, k)
    same_recall = compute_recall(same_indices, test_gt_indices)
    print(f"   Same distribution recall@{k}: {same_recall:.2%}")
    
    print("\n" + "=" * 60)
    print("All MLANN tests passed!")
    print("=" * 60)
    
    # Return best recall achieved
    return same_recall


if __name__ == "__main__":
    test_mlann_index()
