# Qwen3.5 vLLM Serverless Endpoint Worker

[![Deploy on RunPod](https://img.shields.io/badge/RunPod-Deploy-blue?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PC9zdmc+)](https://www.runpod.io/console/hub)

Deploy OpenAI-compatible LLM endpoints powered by [vLLM](https://github.com/vllm-project/vllm) on RunPod Serverless.

Forked from [runpod-workers/worker-vllm](https://github.com/runpod-workers/worker-vllm) with the following changes:

- **vLLM 0.18.0** (includes full Qwen3.5 Gated Delta Networks architecture support)
- **CUDA 12.9.1** base image
- **FlashInfer** bundled via `vllm[flashinfer]`
- PyTorch cu129 wheels
- Dependency install order fixed (vLLM first, then requirements.txt) to prevent PyTorch version conflicts

## Quick Start

### Option 1: Deploy from RunPod Hub

1. Go to the [RunPod Hub](https://console.runpod.io/hub)
2. Find this repo and click **Deploy**
3. Set your environment variables and GPU
4. Done

### Option 2: Deploy with Docker Image

Use a pre-built image or build your own:

```bash
# Deploy on RunPod Serverless with these environment variables:
MODEL_NAME=Qwen/Qwen3.5-35B-A3B-FP8
MAX_MODEL_LEN=32768
GPU_MEMORY_UTILIZATION=0.90
REASONING_PARSER=qwen3
```

GPU: 48GB (L40S, A40) or 80GB (A100, H100)

### Option 3: Build Docker Image with Model Baked In

```bash
export DOCKER_BUILDKIT=1
export HF_TOKEN="hf_your_token_here"

docker build \
  --build-arg MODEL_NAME="Qwen/Qwen3.5-35B-A3B-FP8" \
  --build-arg BASE_PATH="/models" \
  --secret id=HF_TOKEN,env=HF_TOKEN \
  -t your-registry/worker-vllm-qwen35-fp8:latest \
  .

docker push your-registry/worker-vllm-qwen35-fp8:latest
```

When deploying a baked image, do **not** attach a Network Volume.

## Configuration

### Core Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `facebook/opt-125m` | Hugging Face model ID or local path |
| `HF_TOKEN` | | Hugging Face token for gated models |
| `MAX_MODEL_LEN` | | Maximum context length in tokens |
| `GPU_MEMORY_UTILIZATION` | `0.95` | Fraction of GPU VRAM to use (0.0-1.0) |
| `REASONING_PARSER` | | Reasoning parser (`qwen3` for Qwen3/3.5) |
| `TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs for tensor parallelism |
| `QUANTIZATION` | | Quantization method (awq, gptq, squeezellm, bitsandbytes) |
| `ENFORCE_EAGER` | `false` | Disable CUDA graphs (set `true` if startup crashes) |

### Tool Calling

| Variable | Default | Description |
|---|---|---|
| `ENABLE_AUTO_TOOL_CHOICE` | `false` | Enable automatic tool selection |
| `TOOL_CALL_PARSER` | | Parser for tool calls (e.g. `qwen3_coder`) |

### Advanced

| Variable | Default | Description |
|---|---|---|
| `MAX_NUM_SEQS` | `256` | Max concurrent sequences per iteration |
| `MAX_CONCURRENCY` | `30` | Max concurrent requests per worker |
| `ENABLE_PREFIX_CACHING` | `false` | Enable automatic prefix caching |
| `ENABLE_EXPERT_PARALLEL` | `false` | Enable Expert Parallel for MoE models |
| `VLLM_NIGHTLY` | `false` | Build arg: replace pinned vLLM with nightly |

Any vLLM `AsyncEngineArgs` field can be set as an **UPPERCASED** environment variable. The worker auto-discovers all fields. See [full configuration reference](docs/configuration.md).

### Build Arguments (for baking models)

| Argument | Default | Description |
|---|---|---|
| `MODEL_NAME` | | **(Required)** Hugging Face model ID |
| `BASE_PATH` | `/runpod-volume` | Storage path (set to `/models` for baked images) |
| `MODEL_REVISION` | `main` | Model revision |
| `TOKENIZER_NAME` | | Custom tokenizer (defaults to model tokenizer) |
| `VLLM_NIGHTLY` | `false` | Use latest nightly vLLM build |

## Usage: OpenAI API

The worker is fully compatible with OpenAI's API. Change 3 lines in your existing code:

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_RUNPOD_API_KEY",
    base_url="https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-35B-A3B-FP8",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
    max_tokens=100,
)
print(response.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="Qwen/Qwen3.5-35B-A3B-FP8",
    messages=[{"role": "user", "content": "Explain quantum computing."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### curl

```bash
curl https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -d '{
    "model": "Qwen/Qwen3.5-35B-A3B-FP8",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

## Usage: Standard (Non-OpenAI)

You can also use RunPod's native API format:

```json
{
  "input": {
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 2+2?"}
    ],
    "sampling_params": {
      "temperature": 0.7,
      "max_tokens": 100
    }
  }
}
```

## Supported Models

Any model supported by vLLM 0.18.0 works, including:

- **Qwen3.5** (35B-A3B, 27B, 122B-A10B, 397B-A17B, and small variants)
- **Qwen3** / **Qwen3-Next**
- **Llama 3 / 4**, **Gemma 3**, **Mistral**, **DeepSeek V3**
- And [many more](https://docs.vllm.ai/en/latest/models/supported_models.html)

## License

MIT
