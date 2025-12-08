import unittest

import numpy as np
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from nanovllm.sparse.ann_index import ANNIndex
if torch is not None:
    from nanovllm.sparse.ops import dense_attention_reference, sparse_subset_attention


class ANNIndexTests(unittest.TestCase):

    def test_ann_index_exact_shapes(self):
        corpus = np.random.randn(32, 16).astype(np.float32)
        index = ANNIndex(metric="ip", mode="exact")
        index.build(corpus)
        queries = np.random.randn(3, 16).astype(np.float32)
        out = index.query(queries, k=10)
        self.assertEqual(out.shape, (3, 10))

    def test_ann_index_ivf_mode(self):
        corpus = np.random.randn(128, 8).astype(np.float32)
        index = ANNIndex(metric="l2", mode="ivf", ivf_num_lists=16, ivf_num_probe=3)
        index.build(corpus)
        queries = np.random.randn(2, 8).astype(np.float32)
        out = index.query(queries, k=7)
        self.assertEqual(out.shape, (2, 7))


@unittest.skipUnless(torch is not None, "PyTorch is required for sparse attention tests")
class SparseAttentionTests(unittest.TestCase):

    def test_sparse_matches_dense_when_topk_covers_context(self):
        torch.manual_seed(0)
        num_heads = 4
        num_kv_heads = 2
        head_dim = 8
        seq_len = 16
        scale = head_dim ** -0.5

        q = torch.randn(num_heads, head_dim, dtype=torch.float32)
        k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.float32)
        v = torch.randn_like(k)

        corpus = k.reshape(seq_len, -1).numpy().astype(np.float32)
        index = ANNIndex(metric="ip", mode="exact")
        index.build(corpus)
        neighbors = index.query(q.reshape(1, -1).numpy().astype(np.float32), k=seq_len)[0]
        slots = torch.arange(seq_len, dtype=torch.int64)
        gathered = slots.index_select(0, torch.from_numpy(neighbors).to(torch.int64))
        k_subset = k.index_select(0, gathered)
        v_subset = v.index_select(0, gathered)

        dense_out = dense_attention_reference(q, k, v, num_heads, num_kv_heads, scale)
        sparse_out = sparse_subset_attention(q, k_subset, v_subset, num_heads, num_kv_heads, scale)
        self.assertTrue(torch.allclose(dense_out, sparse_out, atol=1e-5, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
