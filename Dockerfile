# Build command (run from repo root):
#   docker build -t alpasim-base .
#
# For private dependencies (requires ~/.netrc with credentials):
#   docker build --secret id=netrc,src=$HOME/.netrc -t alpasim-base .
#
# Automatically detects architecture:
#   x86_64  -> nvidia/cuda base
#   aarch64 -> NGC PyTorch base (only CUDA-enabled PyTorch source on ARM)

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base-amd64
FROM nvcr.io/nvidia/pytorch:25.08-py3 AS base-arm64
ARG TARGETARCH
FROM base-${TARGETARCH}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    curl \
    build-essential \
    gcc-11 \
    g++-11 \
    ninja-build \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# gsplat builds CUDA extensions at runtime. Pin the host compiler to a CUDA-compatible
# toolchain and use ninja when torch compiles the extension.
ENV CC=/usr/bin/gcc-11
ENV CXX=/usr/bin/g++-11
ENV MAX_JOBS=4

# Install Rust toolchain (required for utils_rs maturin build)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

COPY . /repo

# Configure uv
ENV UV_LINK_MODE=copy

# Compile protos
WORKDIR /repo/src/grpc
RUN --mount=type=secret,id=netrc,target=/root/.netrc \
    --mount=type=cache,target=/root/.cache/uv \
    sh -c 'if [ -f /root/.netrc ]; then export NETRC=/root/.netrc; fi && uv sync'
RUN uv run compile-protos --no-sync

WORKDIR /repo

RUN --mount=type=secret,id=netrc,target=/root/.netrc \
    --mount=type=cache,target=/root/.cache/uv \
    sh -c 'if [ -f /root/.netrc ]; then export NETRC=/root/.netrc; fi && uv sync --extra all --extra mtgs --extra pdms'

# --- NAVSIM/nuPlan PDM-scoring stack (alpasim-pdms plugin) ------------------
# The ``--extra pdms`` above installs the alpasim-pdms plugin (registers the
# ``pdms`` eval scorer) but intentionally NOT nuplan-devkit/navsim: their pinned
# dependency sets are ancient (numpy<2, old pandas/scipy) and fight the modern
# mtgs+torch stack.  Install them with ``--no-deps`` on top of the already
# uv-synced env and add only the leaf runtime deps they need that aren't already
# present.  ``numpy>=2`` is passed as a floor so a stray transitive pin cannot
# silently downgrade numpy below what torch/mtgs require.
# See plugins/pdms_eval/README.md for the scoring pipeline details.
ARG NUPLAN_REF=e9241677997dd86bfc0bcd44817ab04fe631405b
ARG NAVSIM_REF=0a380a9063d7162ec93d0f51e9990ebac585f720
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /repo/.venv/bin/python --no-deps \
      "nuplan-devkit @ git+https://github.com/motional/nuplan-devkit.git@${NUPLAN_REF}" \
      "navsim @ git+https://github.com/autonomousvision/navsim.git@${NAVSIM_REF}" \
    && uv pip install --python /repo/.venv/bin/python \
      "numpy>=2" rasterio retry aioboto3 \
    && /repo/.venv/bin/python -c "import numpy,nuplan,navsim,alpasim_pdms; assert numpy.__version__.startswith('2.'), numpy.__version__; print('PDMS stack OK; numpy', numpy.__version__)"

ENV UV_CACHE_DIR=/tmp/uv-cache
ENV UV_NO_SYNC=1
