# Jina Embedding & Reranker Server

OpenAI-compatible API server for Jina embeddings and reranking models.
## Note：
可将`custom_st.py` 放入 `\jinaai\jina-embeddings-v5-text-small` 下，连续进行相同任务时在本机获得1.27x的加速

## Performance (Ryzen 9 9900X, BF16, 12 threads, 实测 HTTP 基准，多次运行区间)

默认配置（零额外内存）：

| 指标 | 优化前 | 默认配置 | 提升 |
|---|---|---|---|
| 串行单条 embedding（中位） | ~135 ms | 65–73 ms | **~1.9x** |
| 就绪后首请求延迟 | 135 ms | 74 ms | 1.8x |
| 批量 8 条（吞吐） | 34.8 texts/s | 43–47 texts/s | ~1.3x |
| 批量 32 条（吞吐） | 69.1 texts/s | 68–70 texts/s | 持平（前向 GEMM 主导，固定开销占比小） |
| 8 并发客户端（RPS） | 44.6 | 45–46 | ~持平（服务端批量时间 -25%，被客户端开销压缩） |
| Rerank 8/32 docs | 254 / 906 ms | 279–297 / 896–948 ms | 持平（前向级 -2~4%，运行噪声内） |
| 就绪时间 | 12.4 s | 13.2 s | +0.8 s（全任务预热） |

默认配置的优化手段（全部 PyTorch + 运行时任务切换，无内存代价）：

1. **短路 SentenceTransformers 每请求的 `.to()/.eval()` 全树扫描**：`encode()` 每次调用遍历 ~4.6k 个子模块（单条 -17ms、10 条批量 -21ms，实测）。默认对 PEFT 模型做实例级 no-op 绑定。
2. **reranker 直连主干**：运行时补丁绕过 lm_head 与全部 29 层 hidden states 的物化（仅用最后一层），~3% 前向提速，分数逐位一致；不修改模型目录文件，重新下载模型后依然生效。
3. **动态批处理窗口 5 ms → 2 ms**：串行 -3 ms；2 ms 相对 0 ms 在 8 并发下保留突发合并成批（RPS +42%）。
4. **启动序列**：retrieval 两种 prompt + 32 条批量 + 两种 rerank 块形状预热，首请求免冷路径。

可选加速（`JINA_LORA_MERGE=1`，**+4.8 GB 内存**，96 GB 机器无压力）：

| 指标 | 默认配置 | LORA_MERGE=1 |
|---|---|---|
| 串行单条 | 65–73 ms | 48–76 ms |
| 批量 8 条 | 43–47 texts/s | 49–63 texts/s |
| 批量 32 条 | 68–70 texts/s | 83–90 texts/s |
| 8 并发 RPS | 45–46 | 52–80 |
| 就绪时间 | 13.2 s | 14.0 s |

原理：**按任务预合并 LoRA（TaskModelPool）**——启动时把 4 个任务的 LoRA 适配器分别合并成独立的纯 Qwen3 主干副本，推理时 `set_adapter` 退化为指针切换。未合并 PEFT 每层 7 个模块有 wrapper 分发与额外 matmul（裸前向串行 3.0x、批量 1.7x），但合并副本越多、上述 `.to()/.eval()` 扫描越贵，因此 pool 内置了同样的扫描短路。合并质量：嵌入 cosine ≥ 0.9998，检索排序 Spearman 0.9998。retrieval 同步合并后即就绪，其余任务后台合并（各 ~1.5 s，期间走 PEFT 回退，输出等价）。

torch.compile 结论（已充分探索，默认关闭）：本机需 `PYTHONUTF8=0` + `CXX=<MSVC cl.exe>` + `TORCHINDUCTOR_VEC_ISA_OK=1` + `INCLUDE/LIB` 才能编译（全局 `CXX` 指向的 g++ 已损坏、GBK 输出解码崩溃、omp.h 缺失）。打通后生成的内核在 MSVC 上有语法错误，且 profile 显示可融合算子仅占 ~10%（GEMM 占 56-69%），收益上限 5-15%，不值得引入。详见 `bench/compile_ceiling.py` 与 `bench/merged_variants.py`。

## CPU Settings

默认 CPU 参数：BF16、12 个计算线程、2 个 interop 线程、embedding 动态批处理 `2 ms / 32`、rerank block `16`、LoRA 按任务预合并**关闭**（内存与基线相同）。

可按需覆盖：

```powershell
$env:JINA_TORCH_THREADS = "12"
$env:JINA_TORCH_INTEROP_THREADS = "2"
$env:JINA_MODEL_DTYPE = "bfloat16"
$env:JINA_EMBEDDING_BATCH_WINDOW_MS = "2"
$env:JINA_EMBEDDING_BATCH_MAX_SIZE = "32"
$env:JINA_RERANKER_BLOCK_SIZE = "16"
$env:JINA_LORA_MERGE = "0"          # 1 = 按任务预合并 LoRA：串行再 -30%，代价 +4.8 GB 常驻内存
```

## Features

- **Embedding**: jina-embeddings-v5-text-small (with LoRA task adapters)
- **Reranking**: jina-reranker-v3
- **OpenAI-compatible API**: Works with existing OpenAI client libraries
- **Task-adaptive embeddings**: Switch LoRA adapter per request (`retrieval`, `text-matching`, `clustering`, `classification`)

## Installation

```bash
uv sync

Fetch_Models.bat  #This is for windows, it contains two 'hf download' commands, adapt for linux accordingly
```

## Usage

Start the server:

```bash
uv run jina_server.py
```

Or activate the virtual environment:

```bash
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python jina_server.py
```

## API Endpoints

### `POST /v1/embeddings`

Create embeddings for input text. Supports task-adaptive LoRA adapters.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | `string \| string[]` | *(required)* | Text(s) to embed |
| `task` | `string` | `"retrieval"` | Task adapter: `retrieval`, `text-matching`, `clustering`, `classification` |
| `prompt_name` | `string` | `null` | Required when `task="retrieval"`: `"query"` or `"document"` |
| `model` | `string` | `"jina-embeddings-v5-text-small"` | Model name |
| `batch_size` | `int` | `32` | Processing batch size (1–128) |

#### Examples

Retrieval (query side):

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "What is machine learning?", "task": "retrieval", "prompt_name": "query"}'
```

Retrieval (document side):

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Machine learning is a field of AI...", "task": "retrieval", "prompt_name": "document"}'
```

Text matching:

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["Hello", "Hi"], "task": "text-matching"}'
```

Classification:

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "This product is great!", "task": "classification"}'
```

Clustering:

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["neural networks", "monetary policy"], "task": "clustering"}'
```

### `POST /v1/rerank`

Rerank documents based on query relevance.

```bash
curl http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "documents": ["Doc 1", "Doc 2"],
    "top_n": 3
  }'
```

### `GET /v1/models`

List available models.

### `GET /`

Health check endpoint.

## Testing

Run the test suite:

```bash
uv run test_server.py
```
## WIP:
I'm telling my opencode to implement avx512 for my 9900x rig, brb

## License

MIT
