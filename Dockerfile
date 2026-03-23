FROM vllm/vllm-openai:v0.18.0-cu130

# Install additional Python dependencies for RunPod worker
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade -r /requirements.txt

# Setup for building the image with the model included
ARG MODEL_NAME="Qwen/Qwen3.5-35B-A3B-FP8"
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""
ARG VLLM_NIGHTLY="false"

# --- Local model support ---
# Set to "true" and place model files in ./local_model/ next to this Dockerfile
# to skip downloading from HuggingFace during build.
ARG USE_LOCAL_MODEL="false"

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    RAY_METRICS_EXPORT_ENABLED=0 \
    RAY_DISABLE_USAGE_STATS=1 \
    TOKENIZERS_PARALLELISM=false \
    RAYON_NUM_THREADS=4

ENV PYTHONPATH="/:/vllm-workspace"

RUN if [ "${VLLM_NIGHTLY}" = "true" ]; then \
      pip install -U vllm --pre --index-url https://pypi.org/simple --extra-index-url https://wheels.vllm.ai/nightly && \
      apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/* && \
      pip install git+https://github.com/huggingface/transformers.git; \
    fi

COPY src /src

# Option A: Copy model from local directory (fast, no download)
# Requires: ./local_model/ folder next to Dockerfile with model files
# The wildcard + true trick makes COPY optional (won't fail if folder is empty)
COPY local_mode[l]/ ${BASE_PATH}/huggingface-cache/hub/

# Option B: Download model from HuggingFace (only if local_model/ was empty)
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
      export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
      # Check if model already exists from local copy
      SNAPSHOT_DIR=$(find ${BASE_PATH}/huggingface-cache/hub/ -type d -name "snapshots" 2>/dev/null | head -1); \
      if [ -n "$SNAPSHOT_DIR" ] && [ "$(ls -A "$SNAPSHOT_DIR" 2>/dev/null)" ]; then \
        echo "Model already present from local copy, skipping download."; \
      else \
        echo "Downloading model from HuggingFace..."; \
        python3 /src/download_model.py; \
      fi; \
    fi

# Override vllm-openai default entrypoint with RunPod handler
ENTRYPOINT []
CMD ["python3", "/src/handler.py"]