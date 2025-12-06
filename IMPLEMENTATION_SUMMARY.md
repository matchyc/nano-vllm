# 稀疏注意力实现总结

## 实现概述

本次实现为 nano-vllm 添加了基于 MLANN ANN 索引的稀疏注意力原型功能。实现遵循了最小化、增量式和可逆的设计原则。

## 文件结构

### 新增文件

1. **nanovllm/sparse/__init__.py**: 稀疏注意力模块初始化文件
2. **nanovllm/sparse/mlann_index.py**: MLANN 索引包装器
3. **nanovllm/sparse/index_manager.py**: 索引管理器，负责构建和管理每层的索引
4. **nanovllm/sparse/sparse_attention.py**: 稀疏注意力计算函数
5. **test_sparse_attention.py**: 单元测试脚本
6. **benchmark_sparse.py**: 性能基准测试脚本
7. **SPARSE_ATTENTION.md**: 使用文档

### 修改文件

1. **nanovllm/config.py**: 添加稀疏注意力配置参数
2. **nanovllm/layers/attention.py**: 修改 Attention 层支持稀疏注意力路径
3. **nanovllm/engine/model_runner.py**: 添加索引构建逻辑和索引管理器初始化

## 核心组件

### 1. Config 扩展

添加了以下配置参数：
- `use_sparse_attention`: 是否启用稀疏注意力
- `sparse_topk`: Top-k 最相关的 keys
- `sparse_min_seq_len`: 最小序列长度阈值
- `sparse_distance_metric`: 距离度量（"ip" 或 "l2"）
- `sparse_index_granularity`: 索引粒度（"layer_shared" 或 "per_head"）
- MLANN 构建参数（n_trees, depth, votes_required）

### 2. MLANNIndex 包装器

- 封装 MLANN 库的接口
- 处理 GPU/CPU 数据转换
- 提供 `build()` 和 `query()` 方法
- 支持 torch tensor 接口 (`query_torch()`)

### 3. SparseIndexManager

- 管理每层、每序列的 ANN 索引
- 在 prefill 后构建索引
- 在 decode 时查询索引
- 跟踪 prefill token 范围和 block_table

### 4. 稀疏注意力计算

`sparse_attention_decode()` 函数：
- 使用 ANN 索引找到 top-k keys
- 通过 block_table 映射 gather K/V
- 计算稀疏注意力（naive matmul）
- 正确处理 GQA（Grouped Query Attention）

### 5. Attention 层修改

- 添加 `layer_id` 和 `index_manager` 属性
- 在 decode 阶段检查是否使用稀疏路径
- 查询 ANN 索引并调用稀疏注意力计算
- 失败时自动回退到密集注意力

### 6. ModelRunner 集成

- 初始化 `SparseIndexManager`
- 为所有 Attention 层设置 `layer_id` 和 `index_manager`
- 在 prefill 后构建索引（`_build_indices_after_prefill()`）
- 在新 batch 前清除索引

## 工作流程

### Prefill 阶段

1. 正常执行 prefill 计算，K/V 写入 KV cache
2. Prefill 完成后，`_build_indices_after_prefill()` 被调用
3. 对每个层和每个序列：
   - 从 KV cache 提取 K 向量
   - 根据 `sparse_index_granularity` 重塑（layer_shared: flatten）
   - 构建 MLANN 索引
   - 存储索引和 block_table

### Decode 阶段

1. 对于每个查询 Q：
   - 检查是否启用稀疏注意力且序列长度足够
   - 查询 ANN 索引获取 top-k indices
   - 使用 block_table 映射 indices 到 KV cache 位置
   - Gather K/V 并计算稀疏注意力
   - 如果失败，回退到密集注意力

## 设计决策

### v1 限制

1. **单序列支持**: 当前实现假设每个 batch 只有一个序列，简化了索引查询逻辑
2. **仅 prefill 稀疏**: 索引只覆盖 prefill tokens，符合 v1 设计目标
3. **同步索引构建**: 索引构建是同步的，为未来异步实现留下了清晰的接口
4. **Layer-shared 索引**: 默认实现 layer_shared，per_head 留作 TODO

### 正确性保证

- 当 `use_sparse_attention=False` 时，行为与原始实现完全相同
- 当序列长度 < `sparse_min_seq_len` 时，自动使用密集注意力
- 稀疏注意力失败时自动回退
- 单元测试验证了小规模情况下的正确性

### 性能考虑

- MLANN 索引构建和查询在 CPU 上进行（符合研究原型定位）
- 使用简单的 PyTorch matmul 进行稀疏注意力计算
- 为未来的 CUDA/Triton kernel 优化留下了 TODO 标记

## 测试

### 单元测试 (`test_sparse_attention.py`)

1. **MLANN 索引测试**: 验证索引构建和查询功能
2. **稀疏注意力计算测试**: 验证 gather 和 attention 计算
3. **正确性测试**: 验证小规模情况下稀疏注意力与密集注意力的一致性

### 基准测试 (`benchmark_sparse.py`)

- 对比密集和稀疏注意力的性能
- 支持自定义 prompt 和参数

## 未来改进方向

1. **多序列批处理**: 支持 batch_size > 1 的情况
2. **异步索引构建**: 重叠索引构建与后续层计算
3. **Per-head 索引**: 实现 per_head 粒度
4. **CUDA kernel**: 优化稀疏注意力计算
5. **在线索引更新**: 支持 decode 阶段的索引更新
6. **性能监控**: 添加详细的性能指标和日志

## 使用示例

```python
from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams

# 启用稀疏注意力
llm = LLM(
    model="/path/to/model",
    use_sparse_attention=True,
    sparse_topk=64,
    sparse_min_seq_len=512,
)

# 生成文本
sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
outputs = llm.generate(["Your long prompt..."], sampling_params)
```

## 注意事项

1. **序列长度**: 确保 prompt 足够长（>= sparse_min_seq_len）才能使用稀疏注意力
2. **内存**: MLANN 索引构建需要额外的 CPU 内存
3. **性能**: 对于短序列，稀疏注意力可能不如密集注意力快
4. **单序列**: v1 版本仅支持单序列批处理

## 代码质量

- ✅ 所有代码通过 linter 检查
- ✅ 遵循项目代码风格
- ✅ 添加了详细的注释和文档
- ✅ 实现了单元测试
- ✅ 保持了向后兼容性

## 总结

本次实现成功地为 nano-vllm 添加了稀疏注意力原型功能，实现了所有核心需求：

1. ✅ 扩展 Config 添加配置标志
2. ✅ 实现 MLANN 包装器
3. ✅ 在 prefill 后构建索引
4. ✅ 在 decode 时使用稀疏注意力
5. ✅ 保持密集路径完整并可回退
6. ✅ 添加测试和文档

实现遵循了最小化、增量式和可逆的设计原则，为未来的优化和扩展打下了良好的基础。
