from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("nanovllm.sparse.ann")

Metric = str
Mode = str


def _to_numpy(arr) -> np.ndarray:
    array = np.asarray(arr, dtype=np.float32)
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)
    return array


def _normalize_metric(metric: str) -> Metric:
    metric = metric.lower()
    if metric not in {"ip", "l2"}:
        raise ValueError(f"Unsupported metric {metric}")
    return metric


def _normalize_mode(mode: str) -> Mode:
    mode = mode.lower()
    if mode not in {"exact", "ivf"}:
        raise ValueError(f"Unsupported ANN mode {mode}")
    return mode


def _pairwise_scores(metric: Metric, queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    if metric == "ip":
        return queries @ corpus.T
    # metric == "l2": return negative distance so that higher is better
    q_norm = np.sum(queries * queries, axis=1, keepdims=True)
    c_norm = np.sum(corpus * corpus, axis=1, keepdims=True).T
    dists = q_norm + c_norm - 2.0 * (queries @ corpus.T)
    return -dists


@dataclass
class ANNIndex:
    metric: Metric = "ip"
    mode: Mode = "exact"
    ivf_num_lists: int = 64
    ivf_num_probe: int = 4
    ivf_max_iters: int = 6
    random_state: int = 0

    def __post_init__(self):
        self.metric = _normalize_metric(self.metric)
        self.mode = _normalize_mode(self.mode)
        self._corpus: np.ndarray | None = None
        self._centroids: np.ndarray | None = None
        self._lists: list[np.ndarray] | None = None
        self._effective_mode: Mode = "exact"

    def build(self, corpus: np.ndarray):
        corpus = _to_numpy(corpus)
        if corpus.ndim != 2 or corpus.shape[0] == 0:
            raise ValueError("Corpus must be a non-empty 2D array")
        self._corpus = corpus
        if self.mode == "ivf" and corpus.shape[0] > self.ivf_num_lists:
            self._build_ivf()
        else:
            self._effective_mode = "exact"
            self._centroids = None
            self._lists = None
            if self.mode == "ivf":
                logger.debug("Falling back to exact ANN: corpus smaller than ivf_num_lists.")

    def query(self, queries: np.ndarray, k: int) -> np.ndarray:
        if self._corpus is None:
            raise RuntimeError("ANNIndex must be built before querying")
        queries = _to_numpy(queries)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        k = min(max(k, 1), self._corpus.shape[0])
        if self._effective_mode == "exact":
            return self._query_exact(queries, k)
        return self._query_ivf(queries, k)

    # ------------------------------------------------------------------ #
    def _query_exact(self, queries: np.ndarray, k: int, subset: np.ndarray | None = None) -> np.ndarray:
        corpus = self._corpus if subset is None else self._corpus[subset]
        k = min(max(k, 1), corpus.shape[0])
        scores = _pairwise_scores(self.metric, queries, corpus)
        kth = max(scores.shape[1] - k, 0)
        idx = np.argpartition(scores, kth=kth, axis=1)[:, -k:]
        top_scores = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(top_scores, axis=1)
        top = np.take_along_axis(idx, order, axis=1)[:, ::-1]
        if subset is not None:
            top = subset[top]
        return top

    def _build_ivf(self):
        corpus = self._corpus
        assert corpus is not None
        rng = np.random.default_rng(self.random_state)
        n_lists = min(self.ivf_num_lists, corpus.shape[0])
        centroids = corpus[rng.choice(corpus.shape[0], n_lists, replace=False)].copy()

        assignment = np.zeros(corpus.shape[0], dtype=np.int32)
        for _ in range(self.ivf_max_iters):
            scores = _pairwise_scores("l2", corpus, centroids)
            assignment = np.argmin(-scores, axis=1)
            for i in range(n_lists):
                members = corpus[assignment == i]
                if members.size == 0:
                    centroids[i] = corpus[rng.integers(0, corpus.shape[0])]
                else:
                    centroids[i] = members.mean(axis=0)

        lists = [np.nonzero(assignment == i)[0] for i in range(n_lists)]
        self._centroids = centroids
        self._lists = lists
        self._effective_mode = "ivf"
        logger.debug("Built IVF index: n_lists=%d avg_list_size=%.1f", n_lists, corpus.shape[0] / n_lists)

    def _query_ivf(self, queries: np.ndarray, k: int) -> np.ndarray:
        assert self._lists is not None and self._centroids is not None
        probe = min(self.ivf_num_probe, len(self._lists))
        outputs = []
        for q in queries:
            centroid_scores = _pairwise_scores(self.metric, q.reshape(1, -1), self._centroids)[0]
            kth = max(centroid_scores.shape[0] - probe, 0)
            candidate_lists = np.argpartition(centroid_scores, kth=kth)[-probe:]
            candidate_lists = candidate_lists[np.argsort(centroid_scores[candidate_lists])[::-1]]
            candidates = [self._lists[i] for i in candidate_lists if self._lists[i].size > 0]
            if not candidates:
                subset = None
            else:
                subset = np.unique(np.concatenate(candidates))
            top = self._query_exact(q.reshape(1, -1), k, subset=subset)[0]
            outputs.append(top)
        return np.stack(outputs, axis=0)
