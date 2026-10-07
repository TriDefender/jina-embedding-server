# Jina Embedding & Reranker Server

OpenAI-compatible API server for Jina embeddings and reranking models.
## Note：
可将`custom_st.py` 放入 `\jinaai\jina-embeddings-v5-text-small` 下，连续进行相同任务时在本机获得1.27x的加速

## Performance (Ryzen 9 9900X, BF16, 12 threads, 实测 HTTP 基准，多次运行区间)

| 指标 | 优化前 | 优化后 | 提升 |
|---|---|---|---|
| 串行单条 embedding（中位） | ~135 ms | 48–76 ms | **~2.3x** |
| 批量 8 条（吞吐） | 34.8 texts/s | 49–63 texts/s | ~1.6x |
| 批量 32 条（吞吐） | 69.1 texts/s | 83–90 texts/s | ~1.3x |
| 8 并发客户端（RPS） | 44.6 | 52–80 | ~1.4x |
| 就绪后首请求延迟 | 135 ms | 57–93 ms | ~2x |
| Rerank 8/32 docs | 254 / 906 ms | 239–270 / 886–953 ms | ~持平 |
| 就绪时间 | 12.4 s | 14.0 s（+1.6s 合并+预热） | — |

注：本机连续重负载下存在 ±30% 的运行间波动（温控/睿频）；上表为多次独立运行区间，任何单次运行均显著优于基线。常驻内存增加约 4.8 GB（4 份合并模型副本）。

优化手段（全部保持 PyTorch + 运行时任务切换，`JINA_LORA_MERGE=0` 可整体关闭）：

1. **按任务预合并 LoRA（TaskModelPool）**：启动时把 4 个任务的 LoRA 适配器分别合并成独立的纯 Qwen3 主干副本（每份 ~1.2 GB），推理时 `set_adapter` 退化为指针切换。相比未合并的 PEFT 路径，单条前向 ~3x、批量 ~1.7x（省掉每层 7 个模块的 wrapper 分发与额外 matmul）。合并后的嵌入与 PEFT 输出 cosine ≥ 0.9998，检索排序 Spearman 0.9998。
2. **短路 SentenceTransformers 每请求的 `.to()/.eval()` 全树遍历**：`encode()` 每次调用都会扫过全部子模块（pool 持有 5 份模型时约 68 ms/次的纯 Python 开销），pool 对 `to/_apply/train` 做 no-op 覆盖。
3. **reranker 直连主干**：绕过 lm_head 与全部 29 层 hidden states 的物化（仅用最后一层），~3% 前向提速。
4. **动态批处理窗口 5 ms → 2 ms**：串行延迟 -3 ms，且 8 并发下突发合并成批仍保留（RPS 比窗口 0 ms 高 ~40%）。
5. **启动序列**：retrieval（主任务）同步合并+预热后立即就绪；其余任务后台合并（各 ~1.5 s，期间请求自动走 PEFT 回退路径，输出等价）。

torch.compile 结论（已充分探索，默认关闭）：本机需 `PYTHONUTF8=0` + `CXX=<MSVC cl.exe>` + `TORCHINDUCTOR_VEC_ISA_OK=1` + `INCLUDE/LIB` 才能编译（全局 `CXX` 指向的 g++ 已损坏、GBK 输出解码崩溃、omp.h 缺失）。打通后生成的内核在 MSVC 上有语法错误，且 profile 显示可融合算子仅占 ~10%（GEMM 占 56-69%），收益上限 5-15%，不值得引入。详见 `bench/compile_ceiling.py` 与 `bench/merged_variants.py`。

## CPU Settings

默认 CPU 参数：BF16、12 个计算线程、2 个 interop 线程、embedding 动态批处理 `2 ms / 32`、rerank block `16`、LoRA 按任务预合并开启（常驻内存约 6 GB：PEFT 原模型 + 4 份合并副本）。

可按需覆盖：

```powershell
$env:JINA_TORCH_THREADS = "12"
$env:JINA_TORCH_INTEROP_THREADS = "2"
$env:JINA_MODEL_DTYPE = "bfloat16"
$env:JINA_EMBEDDING_BATCH_WINDOW_MS = "2"
$env:JINA_EMBEDDING_BATCH_MAX_SIZE = "32"
$env:JINA_RERANKER_BLOCK_SIZE = "16"
$env:JINA_LORA_MERGE = "1"          # 0 = 回退到原生 PEFT 路径（省内存，慢 ~2-3x）
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
