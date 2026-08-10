# 支持 Qwen3.5 Hybrid 模型

## 背景

当前 nano-vllm 仅支持 Qwen3 系列的纯 Dense Transformer 模型。Qwen3.5 采用了全新的 **Hybrid 混合架构**，核心创新在于：

1. **混合注意力**：75% 的层使用 **Gated DeltaNet**（线性注意力），25% 的层使用 **Gated Full Attention**（标准自注意力），以 3:1 的节奏交替排列。
2. **Gated DeltaNet** 是一种线性注意力变体，维护固定大小的隐状态矩阵 `S`，通过 delta rule 进行增量更新，而非构建 N×N 的注意力矩阵。它的推理内存占用为 O(1)，计算复杂度为 O(L)。
3. **全注意力层带有 output gate**：`attn_output = attn_output * sigmoid(gate)`。
4. **部分旋转嵌入（Partial RoPE）**：RoPE 仅应用于 head_dim 的 25%（`partial_rotary_factor=0.25`），而非全部维度。
5. **线性注意力层拥有独立的 head 配置**：使用 `linear_num_key_heads`、`linear_num_value_heads`、`linear_key_head_dim`、`linear_value_head_dim` 等独立参数，与全注意力层的 head 配置不同。

> [!IMPORTANT]
> 这是一次架构级别的重大改动。Qwen3.5 的推理方式与 Qwen3 有本质不同：
> - DeltaNet 层不使用 KV Cache，而是维护一个固定大小的 recurrent state `(num_heads, key_dim, value_dim)`。
> - Full Attention 层仍然使用 KV Cache（类似 Qwen3）。
> - 这意味着 KV Cache 分配、block management、CUDA graph 捕获等引擎层面的逻辑都需要适配。

## User Review Required

> [!WARNING]
> **依赖项变化**：Qwen3.5 的 Gated DeltaNet 在生产中通常依赖 `fla`（flash-linear-attention）库提供高性能 kernel。在本实现中，我们将先提供**纯 PyTorch 的参考实现**（recurrent 和 chunked 两种路径），不依赖外部包。这意味着 DeltaNet 的速度不会达到最优，但可以正确运行并保持代码的轻量级特性。如果后续需要高性能版本，可以加入 `fla` 作为可选依赖。

> [!IMPORTANT]
> **视觉部分**：Qwen3.5 原版是一个多模态模型（Vision-Language），但本项目 nano-vllm 专注于文本推理。本计划只实现 **Qwen3.5 的文本部分**（`qwen3_5_text` 或 `qwen3_next` text config），不涉及视觉编码器。用户需要使用纯文本版本的 Qwen3.5 权重（如社区提取的纯文本版本），或者使用 `Qwen3-Next` 系列。

## Open Questions

> [!IMPORTANT]
> 1. **目标模型**：你计划使用哪个具体的 Qwen3.5 模型？（如 Qwen3.5-4B、Qwen3.5-9B、Qwen3.5-27B？）这影响 GPU 内存需求。
> 2. **`fla` 库支持**：是否需要集成 `fla`（flash-linear-attention）库来加速 DeltaNet？还是先用纯 PyTorch 参考实现？
> 3. **Conv1d 依赖**：DeltaNet 层使用了 `causal_conv1d`，是否可以接受纯 PyTorch 实现的 conv1d（稍慢但无额外依赖）？

---

## Proposed Changes

### 组件一：配置层

#### [MODIFY] [config.py](file:///d:/vibecoding/nano-vllm/nanovllm/config.py)

- 从 `hf_config` 中自动检测 `model_type`（`qwen3_5_text` / `qwen3_next`），在 config 中记录 `model_type` 字段。
- 提取并缓存 `layer_types` 列表（`["linear_attention", "full_attention", ...]`），方便后续按层类型分别处理。
- 适配 `max_position_embeddings` 的读取（Qwen3.5 的配置结构可能嵌套在 `text_config` 中）。

---

### 组件二：模型层 (layers)

#### [MODIFY] [rotary_embedding.py](file:///d:/vibecoding/nano-vllm/nanovllm/layers/rotary_embedding.py)

- 支持 **Partial RoPE**：Qwen3.5 的 `partial_rotary_factor=0.25`，即 RoPE 仅应用于 `head_dim` 的前 25%。修改 `RotaryEmbedding` 使 `rotary_dim` 可以不等于 `head_dim`（当前代码有 `assert rotary_dim == head_dim` 的断言）。
- 修改 `apply_rotary_emb` 以支持 partial rotation：只对前 `rotary_dim` 维度应用旋转，剩余维度直接拼接。
- 修改 `get_rope` 的缓存 key 以区分不同的 rotary_dim 配置。

#### [NEW] [gated_deltanet.py](file:///d:/vibecoding/nano-vllm/nanovllm/layers/gated_deltanet.py)

新建 Gated DeltaNet 层实现，核心组件包括：

- **`RMSNormGated`**：带 gate 的 RMSNorm（norm → weight → silu_gate）。
- **`GatedDeltaNet`** 类：
  - `in_proj_qkv`：将 hidden_states 投影为 key、value、query（注意这里 linear attention 有自己的 head 配置）。
  - `in_proj_z`：output gate 投影。
  - `in_proj_b`：beta（delta rule 中的学习率）投影。
  - `in_proj_a`：decay gate 投影。
  - `conv1d`：causal 1D 卷积，用于捕获局部上下文。
  - `dt_bias` / `A_log`：时间步离散化参数。
  - `norm`：Gated RMSNorm。
  - `out_proj`：输出投影。
  - **前向传播逻辑**：
    - **Prefill 路径（chunked）**：使用 chunk-wise delta rule 算法，将序列分成 chunk 来高效处理。
    - **Decode 路径（recurrent）**：逐 token 更新 recurrent state，O(1) 内存。
  - **State 管理**：每个 DeltaNet 层维护一个 `recurrent_state`，形状为 `(batch, num_heads, key_dim, value_dim)`，以及一个 `conv_state`。

---

### 组件三：Qwen3.5 模型文件

#### [NEW] [qwen3_5.py](file:///d:/vibecoding/nano-vllm/nanovllm/models/qwen3_5.py)

新建 Qwen3.5 模型文件，主要类包括：

- **`Qwen3_5Attention`**（继承/改造自 Qwen3Attention）：
  - Full attention 层，增加 output gate：`q_proj` 投影维度翻倍（一半做 query，一半做 gate），输出经过 `sigmoid(gate)` 缩放。
  - 使用 partial RoPE。
  - 使用 QK norm（不带 bias）。

- **`Qwen3_5DecoderLayer`**：
  - 根据 `layer_types[layer_idx]` 选择使用 `Qwen3_5Attention`（full_attention）还是 `GatedDeltaNet`（linear_attention）。
  - MLP 部分复用 Qwen3MLP。
  - LayerNorm 部分使用 RMSNorm。

- **`Qwen3_5Model`**：
  - 类似 Qwen3Model，但 layers 列表根据 `layer_types` 混合创建不同类型的层。

- **`Qwen3_5ForCausalLM`**：
  - 顶层模型类，包含 `packed_modules_mapping` 以支持权重加载。

---

### 组件四：引擎层适配

#### [MODIFY] [model_runner.py](file:///d:/vibecoding/nano-vllm/nanovllm/engine/model_runner.py)

- **模型选择**：根据 `hf_config.model_type` 自动选择实例化 `Qwen3ForCausalLM` 或 `Qwen3_5ForCausalLM`。
- **KV Cache 分配**：
  - 仅为 `full_attention` 层分配 KV Cache block，`linear_attention` 层不需要 KV Cache。
  - 修改 `allocate_kv_cache` 中的 `num_hidden_layers` 计算，使用实际的 full_attention 层数量。
  - KV cache 的 `num_kv_heads` 和 `head_dim` 取自 full attention 层的配置。
- **Recurrent State 分配**：
  - 为每个 `linear_attention` 层分配 recurrent state 和 conv state。
  - Recurrent state 形状：`(max_batch_size, num_linear_heads, key_dim, value_dim)`。
  - Conv state 形状：`(max_batch_size, conv_dim, conv_kernel_size - 1)`。
- **CUDA Graph 适配**：
  - Decode 阶段的 CUDA graph 需要能处理 DeltaNet 层的 recurrent state 更新。
  - 初步方案：对于 Qwen3.5 模型，可能需要暂时禁用 CUDA graph（`enforce_eager=True`），后续再优化。

#### [MODIFY] [block_manager.py](file:///d:/vibecoding/nano-vllm/nanovllm/engine/block_manager.py)

- KV cache block 的计算需要考虑 Qwen3.5 中只有 full_attention 层使用 KV cache，`block_bytes` 的计算应使用 `num_full_attention_layers` 而非 `num_hidden_layers`。

---

### 组件五：权重加载

#### [MODIFY] [loader.py](file:///d:/vibecoding/nano-vllm/nanovllm/utils/loader.py)

- Qwen3.5 的权重名称中使用 `linear_attn` 前缀来标识 DeltaNet 层，与 `self_attn` 区分。
- 需要在 `packed_modules_mapping` 中添加 DeltaNet 相关的映射规则。
- 处理 Qwen3.5 的 `text_model.` 前缀（如果使用完整的多模态权重）。

---

### 组件六：Context 适配

#### [MODIFY] [context.py](file:///d:/vibecoding/nano-vllm/nanovllm/utils/context.py)

- 可能需要在 Context 中添加 `recurrent_states` 相关的字段，以便 DeltaNet 层在 forward 时能够访问和更新 state。
- 或者，DeltaNet 层的 state 直接作为模块属性存储（类似当前 Attention 层的 `k_cache`、`v_cache`），这样不需要修改 Context。

---

## 文件修改总结

| 文件 | 操作 | 说明 |
|------|------|------|
| [config.py](file:///d:/vibecoding/nano-vllm/nanovllm/config.py) | MODIFY | 支持 Qwen3.5 配置读取 |
| [rotary_embedding.py](file:///d:/vibecoding/nano-vllm/nanovllm/layers/rotary_embedding.py) | MODIFY | 支持 partial RoPE |
| [gated_deltanet.py](file:///d:/vibecoding/nano-vllm/nanovllm/layers/gated_deltanet.py) | NEW | Gated DeltaNet 层实现 |
| [qwen3_5.py](file:///d:/vibecoding/nano-vllm/nanovllm/models/qwen3_5.py) | NEW | Qwen3.5 Hybrid 模型 |
| [model_runner.py](file:///d:/vibecoding/nano-vllm/nanovllm/engine/model_runner.py) | MODIFY | 模型选择 + KV/State 分配 |
| [block_manager.py](file:///d:/vibecoding/nano-vllm/nanovllm/engine/block_manager.py) | MODIFY | block 计算适配 |
| [loader.py](file:///d:/vibecoding/nano-vllm/nanovllm/utils/loader.py) | MODIFY | 权重加载映射 |

---

## Verification Plan

### Automated Tests

```bash
# 1. 确保代码可以正确加载 Qwen3.5 模型权重（无报错）
python -c "from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM; print('Import OK')"

# 2. 使用 Qwen3.5 模型运行端到端推理
python example_qwen3_5.py
```

### Manual Verification

1. 下载 Qwen3.5 的文本模型权重。
2. 使用 `enforce_eager=True` 进行推理，验证输出文本的语义正确性。
3. 对比 HuggingFace Transformers 库的推理输出，确保结果一致。
4. 检查内存占用是否符合预期（DeltaNet 层不应分配 KV cache block）。
