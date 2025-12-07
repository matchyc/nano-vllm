# Sparse Attention with ANN Index

本文档描述了 nano-vllm 中稀疏注意力功能的实现和使用方法。

## 概述

稀疏注意力功能通过在 prefill 阶段为每个层的 key 向量构建 ANN（近似最近邻）索引，在 decode 阶段使用该索引找到 top-k 最相关的 keys，从而减少注意力计算量。

## 功能特性

- **MLANN 索引**: 使用 MLANN 库构建高效的 ANN 索引
- **每层索引**: 为每个 Transformer 层单独构建索引
- **可配置粒度**: 支持 layer_shared（默认）和 per_head（待实现）两种索引粒度
- **自动回退**: 当序列长度小于阈值时自动使用密集注意力
- **保持兼容**: 默认情况下行为与原始 nano-vllm 完全相同

## 配置参数

在创建 `LLM` 实例时，可以通过以下参数配置稀疏注意力：

```python
from nanovllm import LLM

llm = LLM(
    model="/path/to/model",
    # 稀疏注意力配置
    use_sparse_attention=True,        # 启用稀疏注意力
    sparse_topk=64,                    # Top-k 最相关的 keys
    sparse_min_seq_len=512,            # 最小序列长度（低于此值使用密集注意力）
    sparse_distance_metric="ip",       # 距离度量："ip" (内积) 或 "l2" (L2距离)
    sparse_index_granularity="layer_shared",  # 索引粒度："layer_shared" 或 "per_head"
    # MLANN 构建参数
    sparse_mlann_n_trees=10,           # MLANN 树的数量
    sparse_mlann_depth=6,              # MLANN 树的深度
    sparse_mlann_votes_required=5,     # MLANN 投票阈值
)
```

## 使用示例

### 基本使用

```python
from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams

# 创建启用稀疏注意力的 LLM
llm = LLM(
    model="/path/to/model",
    use_sparse_attention=True,
    sparse_topk=64,
    sparse_min_seq_len=512,
)

# 生成文本
sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
prompt = "Your long prompt here... " * 100  # 确保提示足够长
outputs = llm.generate([prompt], sampling_params)
print(outputs[0]["text"])

llm.exit()
```

### 性能对比

使用 `benchmark_sparse.py` 脚本可以对比密集和稀疏注意力的性能：

```bash
python benchmark_sparse.py /path/to/model "Your prompt here"
```

## 实现细节

### 架构设计

1. **Config**: 扩展了 `Config` 类，添加稀疏注意力相关配置标志
2. **MLANNIndex**: 提供了 MLANN 库的包装器，处理 GPU/CPU 数据转换
3. **SparseIndexManager**: 管理每层的 ANN 索引，负责构建和查询
4. **Attention 层**: 修改了 `Attention` 层，添加稀疏注意力路径
5. **ModelRunner**: 在 prefill 后构建索引，在 decode 时使用索引

### 工作流程

1. **Prefill 阶段**:
   - 正常计算所有 token 的 K/V 并写入 KV cache
   - Prefill 完成后，为每个层提取 K 向量
   - 为每个序列构建 MLANN 索引（v1 版本同步构建）

2. **Decode 阶段**:
   - 对于每个查询 Q，使用 ANN 索引找到 top-k 最相关的 keys
   - 从 KV cache 中 gather 对应的 K/V
   - 计算稀疏注意力（仅对 top-k keys）
   - 如果序列长度小于阈值或索引查询失败，回退到密集注意力

### 限制和注意事项

**v1 版本的限制**:

1. **仅支持单序列**: 当前实现假设每个 batch 只有一个序列
2. **仅 prefill 稀疏**: 索引只覆盖 prefill tokens，decode 阶段的 tokens 不在索引中
3. **同步索引构建**: 索引构建是同步的，未实现与后续层计算的重叠
4. **单 GPU**: 仅支持单 GPU 运行
5. **固定长度 prefill**: 不支持动态长度的 prefill context

**性能考虑**:

- MLANN 索引构建和查询在 CPU 上进行，会有一定的开销
- 对于短序列（< sparse_min_seq_len），稀疏注意力可能不如密集注意力快
- 当前实现使用简单的 PyTorch matmul 进行稀疏注意力计算，未使用优化的 CUDA kernel

## 测试

运行单元测试：

```bash
python test_sparse_attention.py
```

测试包括：
- MLANN 索引构建和查询
- 稀疏注意力计算
- 正确性验证（小规模情况下稀疏注意力应与密集注意力一致）

## 未来改进

- [ ] 支持多序列批处理
- [ ] 实现异步/重叠的索引构建
- [ ] 支持 per_head 索引粒度
- [ ] 优化稀疏注意力的 CUDA kernel
- [ ] 支持在线索引更新（decode 阶段的 tokens）
- [ ] 添加更多性能指标和日志

## 故障排除

### 索引构建失败

如果遇到索引构建错误，检查：
- 序列长度是否 >= sparse_min_seq_len
- MLANN 库是否正确安装
- 内存是否充足

### 性能未提升

如果稀疏注意力没有带来性能提升：
- 检查序列长度是否足够长
- 尝试调整 sparse_topk 参数
- 确认是否真的使用了稀疏路径（添加日志）

### 回退到密集注意力

稀疏注意力会在以下情况自动回退：
- 序列长度 < sparse_min_seq_len
- 索引查询失败
- Batch size > 1（v1 限制）

## 参考

- MLANN 库: https://github.com/your-repo/mlann
- nano-vllm: 原始实现
