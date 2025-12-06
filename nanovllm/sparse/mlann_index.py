import logging
import os
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger("nanovllm.sparse.mlann")

try:
    from mlann.mlann import MLANNIndex as _CoreMLANNIndex
    from mlann import IP as _MLANN_IP
    from mlann import L2 as _MLANN_L2
except Exception:  # pragma: no cover - exercised when mlann is not installed
    _CoreMLANNIndex = None
    _MLANN_IP = None
    _MLANN_L2 = None


Metric = Literal["ip", "l2"]


def _normalize_metric(metric: str) -> Metric:
    metric = metric.lower()
    if metric not in {"ip", "l2"}:
        raise ValueError(f"Unsupported metric: {metric}")
    return metric  # type: ignore[return-value]


@dataclass
class MLANNIndex:
    metric: Metric = "ip"
    index_type: str = "PCA"
    n_trees: int = 4
    depth: int = 6
    training_k: int = 64

    def __post_init__(self):
        self.metric = _normalize_metric(self.metric)
        self.backend = "mlann" if _CoreMLANNIndex is not None else "naive"
        force_naive = os.getenv("NANOVLLM_FORCE_NAIVE_MLANN", "").lower() in {"1", "true", "yes"}
        if force_naive:
            self.backend = "naive"
        if self.backend == "mlann":
            logger.info("Using MLANN backend (%s)", self.index_type)
        else:
            logger.warning("Falling back to naive numpy indexer for sparse attention prototype.")
        self._index = None
        self._corpus = None

    @staticmethod
    def _to_numpy(matrix) -> np.ndarray:
        arr = np.asarray(matrix, dtype=np.float32)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
        return arr

    def build(self, corpus: np.ndarray):
        corpus = self._to_numpy(corpus)
        if corpus.ndim != 2:
            raise ValueError("Corpus must be 2D [num_tokens, dim]")
        if corpus.shape[0] == 0:
            raise ValueError("Cannot build index for zero tokens")
        if self.backend == "mlann":
            self._build_mlann(corpus)
        else:
            self._corpus = corpus

    def _build_mlann(self, corpus: np.ndarray):
        assert _CoreMLANNIndex is not None
        index = _CoreMLANNIndex(corpus, self.index_type)
        dist = _MLANN_IP if self.metric == "ip" else _MLANN_L2
        training_k = min(max(self.training_k, 8), corpus.shape[0])
        knn = index.exact_search(corpus, training_k, dist=dist)
        index.build(corpus, knn, self.n_trees, self.depth)
        self._index = index

    def query(self, queries: np.ndarray, k: int) -> np.ndarray:
        if k <= 0:
            raise ValueError("k must be positive")
        queries = self._to_numpy(queries)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        if self.backend == "mlann":
            return self._query_mlann(queries, k)
        return self._query_naive(queries, k)

    def _query_mlann(self, queries: np.ndarray, k: int) -> np.ndarray:
        assert self._index is not None
        dist = _MLANN_IP if self.metric == "ip" else _MLANN_L2
        votes = max(1, min(k, int(np.ceil(k / 4))))
        return self._index.ann(queries, k, votes, dist=dist)

    def _query_naive(self, queries: np.ndarray, k: int) -> np.ndarray:
        assert self._corpus is not None
        if self.metric == "ip":
            scores = queries @ self._corpus.T
            idx = np.argpartition(scores, kth=-k, axis=1)[:, -k:]
            scores = np.take_along_axis(scores, idx, axis=1)
            order = np.argsort(-scores, axis=1)
            return np.take_along_axis(idx, order, axis=1)
        # L2 distance
        dists = np.sum((queries[:, None, :] - self._corpus[None, :, :]) ** 2, axis=2)
        idx = np.argpartition(dists, kth=k-1, axis=1)[:, :k]
        dists = np.take_along_axis(dists, idx, axis=1)
        order = np.argsort(dists, axis=1)
        return np.take_along_axis(idx, order, axis=1)
