FROM nvcr.io/nvidia/pytorch:25.06-py3

# создаем пользователя с ID 1000 и группой с ID 1000
# cоздаем пустую директорию с правами user

RUN userdel ubuntu && useradd -u 1000 -m user --shell /bin/bash \
    && mkdir -p /home/jovyan && chown user:user /home/jovyan

# Ставим OpenSSH для MPI что бы соединятся между контейнирами
# Генерим ключи
RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-server\
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd \
    && ssh-keygen -A

# Allow OpenSSH to talk to containers without asking for confirmation
# Disable OpenSSH PAM
# Set port 2222
# chmod 0600 /etc/ssh/ssh_host_*_key
RUN echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config.d/mlspace.conf \
    && echo "UsePAM no" > /etc/ssh/sshd_config.d/mlspace.conf \
    && echo "PubkeyAcceptedAlgorithms +ssh-rsa" >> /etc/ssh/sshd_config.d/mlspace.conf \
    && echo "Port 2222" >> /etc/ssh/sshd_config.d/mlspace.conf \
    && echo "PidFile /run/sshd/sshd.pid" >> /etc/ssh/sshd_config.d/mlspace.conf \
    && chmod 600 /etc/ssh/ssh_host_* \
    && chown user:user /etc/ssh/ssh_host_* \
    && chown user:user /run/sshd

# Clearing the apt caches
RUN apt-get clean

# Building and installing nccl tests for current container
RUN cd /workspace \
    && n=0; until git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git; do \
        n=$((n+1)); [ "$n" -ge 6 ] && { echo "nccl-tests clone failed after 6 tries"; exit 1; }; \
        echo "retry nccl-tests clone ($n)"; rm -rf nccl-tests; sleep 30; \
    done \
    && cd nccl-tests \
    && make MPI=1 MPI_HOME=/opt/hpcx/ompi/ CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/lib/x86_64-linux-gnu/ \
    && mv build /opt/nccl-tests \
    && cd /workspace \
    && rm -rf nccl-tests

####
# Все необходимые дополнительные действия от рута делаются в секции ниже
# таким образом данные действия будут глобальными для всего образа
# Не устанавливайте переменные через действие ENV, только через добавление в файлы
# /etc/profile.d/<profile>.sh|csh
####

ARG NVSHMEM_DIR=/home/user/nvshmem

RUN echo $'\n\
export LD_LIBRARY_PATH=$NVSHMEM_DIR/lib:/opt/hpcx/nccl_rdma_sharp_plugin/lib:/opt/hpcx/ucc/lib/ucc:/opt/hpcx/ucc/lib:/opt/hpcx/ucx/lib/ucx:/opt/hpcx/ucx/lib:/opt/hpcx/sharp/lib:/opt/hpcx/hcoll/lib:/opt/hpcx/ompi/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64\n\
export PATH=$NVSHMEM_DIR/bin:/opt/hpcx/sharp/bin:/opt/hpcx/clusterkit/bin:/opt/hpcx/hcoll/bin:/opt/hpcx/ucc/bin:/opt/hpcx/ucx/bin:/opt/hpcx/ompi/bin:/home/user/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n\
export NVSHMEM_DIR=$NVSHMEM_DIR\n\
export OPAL_PREFIX=/opt/hpcx/ompi' > /etc/profile.d/wm.sh

####
# Все необходимые действия от пользователя делаются в секции ниже под USER
# Тут важно помнить, что если делается экспорт переменных в секции выше, то пользователь
# в данной секции их не увидит, следовательно все должно располагаться тут
####

WORKDIR /home/user

# check if pytorch version has been changed
RUN PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)") && \
    echo $PYTORCH_VERSION > /tmp/pytorch_version.txt

#### start audio
RUN apt-get update && apt-get install -y \
    libavcodec-dev \
    libavformat-dev \
    libavfilter-dev \
    libavdevice-dev \
    libavutil-dev \
    libswresample-dev \
    libswscale-dev \
    libopus-dev \
    libfftw3-dev \
    libsndfile1-dev \
    libmp3lame-dev \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --chmod=777 ./audioaug_src /home/user/audioaug_src

RUN git config --global http.version HTTP/1.1 && \
    git config --global http.postBuffer 524288000 && \
    cd /home/user/audioaug_src/contrib/ && \
    rm -rf pybind11 && \
    n=0; until git clone --depth 1 https://github.com/pybind/pybind11.git; do \
        n=$((n+1)); [ "$n" -ge 6 ] && { echo "pybind11 clone failed after 6 tries"; exit 1; }; \
        echo "retry pybind11 clone ($n)"; rm -rf pybind11; sleep 30; \
    done

RUN mkdir /home/user/audioaug_src/build && \
    cd /home/user/audioaug_src/ && \
    cmake -B./build -DCMAKE_BUILD_TYPE:STRING=Release -Wno-dev && \
    cmake --build ./build --config Release --target all -j && cd /home/user/audioaug_src && pip install -e .

WORKDIR /home/user
#### end audio


###############
# Flash attention
###############
USER root

ENV FLASH_ATTENTION_VERSION=2.8.3
RUN pip uninstall -y flash-attn
RUN cd /home/user && git clone -b v${FLASH_ATTENTION_VERSION} --single-branch --depth 1 https://github.com/Dao-AILab/flash-attention.git
RUN pip install packaging==24.1 setuptools==69.5.1
# Build output goes to a file: the FA compile streams megabytes of ptxas stats,
# which overflows GitLab's 4 MB job log limit and hides every later error.
RUN cd /home/user/flash-attention && \
    MAX_JOBS=8 python setup.py install > /tmp/fa2_build.log 2>&1 \
    || (tail -n 300 /tmp/fa2_build.log; exit 1)
RUN cd /home/user/flash-attention/hopper && \
    mkdir flash_attn_3 && \
    touch flash_attn_3/__init__.py && \
    cp flash_attn_interface.py flash_attn_3/ && \
    MAX_JOBS=8 python setup.py install > /tmp/fa3_build.log 2>&1 \
    || (tail -n 300 /tmp/fa3_build.log; exit 1)

WORKDIR /home/user/flash-attention
RUN cd csrc/layer_norm && pip install .
RUN cd csrc/fused_dense_lib && \
    pip install .

WORKDIR /home/user


#######################
# DeepEP
#######################
# can be "sm80" or "sm90"
# for sm80 FP8, launch methods, TMA (Tensor Memory Access), all internode and low-latency features from NVSHMEM are disabled
ARG CUDA_ARCH="sm_90"
ARG DEEPEP_COMMIT=e6012370b01a86e821ace99193f5d912b3f66d06

# Cloning DeepEP from GitHub breaks off (curl 56 / early EOF), so fetch exactly the one
# pinned commit with --depth 1 and retry, after 6c836c1b on main.
RUN git config --global http.version HTTP/1.1 && \
    git config --global http.postBuffer 524288000 && \
    mkdir -p DeepEP && cd DeepEP && git init -q && \
    git remote add origin https://github.com/deepseek-ai/DeepEP.git && \
    n=0; until git fetch --depth 1 --no-tags origin ${DEEPEP_COMMIT}; do \
        n=$((n+1)); [ "$n" -ge 6 ] && { echo "DeepEP fetch failed after 6 tries"; exit 1; }; \
        echo "retry DeepEP fetch ($n)"; sleep 30; \
    done && \
    git checkout -q FETCH_HEAD

# NVSHMEM
RUN wget https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz && \
    tar xvf nvshmem_src_3.2.5-1.txz && \
    cd nvshmem_src && git apply /home/user/DeepEP/third-party/nvshmem.patch && \
    # sed -i for commenting warnings from ibgda about cpu fallbacks
    sed -i '968,971 s/^/\/\/ /' src/modules/transport/ibgda/ibgda.cpp && \
    sed -i '1815,1816 s/^/\/\/ /' src/modules/transport/ibgda/ibgda.cpp && \
    sed -i '3468,3470 s/^/\/\/ /' src/modules/transport/ibgda/ibgda.cpp && \
    NVSHMEM_SHMEM_SUPPORT=0 \
    NVSHMEM_UCX_SUPPORT=0 \
    NVSHMEM_USE_NCCL=0 \
    NVSHMEM_MPI_SUPPORT=0 \
    NVSHMEM_IBGDA_SUPPORT=1 \
    NVSHMEM_PMIX_SUPPORT=0 \
    NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
    NVSHMEM_USE_GDRCOPY=1 \
    cmake -S . -B build/ \
        -DCMAKE_INSTALL_PREFIX=${NVSHMEM_DIR} \
        -DNVSHMEM_BUILD_EXAMPLES=OFF \
        -DNVSHMEM_BUILD_NVSHMEM4PY=OFF \
        -DMLX5_lib=/lib/x86_64-linux-gnu/libmlx5.so.1 \
        -DCMAKE_CUDA_ARCHITECTURES=90 && \
    (cmake --build build --target install --parallel $(nproc) > /tmp/nvshmem_build.log 2>&1 \
        || (tail -n 300 /tmp/nvshmem_build.log; exit 1)) && \
    cd .. && \
    rm -rf nvshmem_src*

# install DeepEP
RUN cd DeepEP && \
    export CUDA_HOME=/usr/local/cuda && \
    if [ "${CUDA_ARCH}" = "sm_90" ]; then \
        export TORCH_CUDA_ARCH_LIST="9.0" && \
        NVSHMEM_DIR=${NVSHMEM_DIR} \
        pip install -v .; \
    else \
        export TORCH_CUDA_ARCH_LIST="8.0" && \
        DISABLE_SM90_FEATURES=1 \
        pip install -v .; \
    fi && \
    cd .. && \
    rm -rf DeepEP

###############
# LLM F
###############
COPY --chmod=777 . ./llm-foundry
COPY ./deepgemm /opt/deepgemm
COPY ./cudacherry /opt/cudacherry

ARG CUDACHERRY_COMMIT=8a439c2bd0069f1a716f37054e3de7caf9f12512

#### start copy deepgemm cache
COPY --chmod=777 ./cache ./dg_cache
RUN for i in $(seq 0 7); do \
    mkdir -p /temp/.cache/rank_${i}/deepgemm/cache/ && \
    cp -r ./dg_cache/cache/* /temp/.cache/rank_${i}/deepgemm/cache/; \
    done && \
    chmod -R 777 /temp/.cache
#### end copy deepgemm cache
RUN chmod +x /opt/deepgemm/install.sh

RUN pip install setuptools -U && \
    pip install packaging==24.2 && \
    export CUDA_VERSION_PT=$(python3 -c "import torch; print(torch.version.cuda)") && \
    n=0; until pip install  \
        --no-build-isolation \
        --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VERSION_PT//./} \
            "./llm-foundry/contrib/composer[libcloud,wandb,mlflow,oci,gcs,nlp,sentencepiece]" \
            "./llm-foundry/contrib/streaming" \
            "./llm-foundry/contrib/metric-sampler" \
            "./llm-foundry"; do \
        n=$((n+1)); [ "$n" -ge 6 ] && { echo "llm-foundry deps install failed after 6 tries"; exit 1; }; \
        echo "retry llm-foundry deps install ($n)"; sleep 30; \
    done && \
    pip uninstall -y llm-foundry && \
    export PYTORCH_VERSION=$(cat /tmp/pytorch_version.txt) && \
    export PYTORCH_VERSION_AFTER=$(python3 -c "import torch; print(torch.__version__)") && \
    if [ "$PYTORCH_VERSION" != "$PYTORCH_VERSION_AFTER" ]; then \
        echo "ERROR: PyTorch version was reinstalled during dependency installation." && \
        echo "Before: $PYTORCH_VERSION, After: $PYTORCH_VERSION_AFTER" && \
        exit 1; \
    fi && \
    if [ "${CUDA_ARCH}" = "sm_90" ]; then ARCH_LIST="9.0"; else ARCH_LIST="8.0"; fi && \
    (cd /opt/cudacherry && git reset --hard ${CUDACHERRY_COMMIT} && \
        TORCH_CUDA_ARCH_LIST="$ARCH_LIST" MAX_JOBS=16 \
        pip install --no-build-isolation --no-deps --no-cache-dir /opt/cudacherry \
        > /tmp/cudacherry_build.log 2>&1 \
        || (echo "===== cudacherry build FAILED; errors: ====="; \
            grep -inE "error|fatal" /tmp/cudacherry_build.log | head -50; \
            echo "===== last 300 lines: ====="; \
            tail -n 300 /tmp/cudacherry_build.log; exit 1)) && \
    rm -rf /opt/cudacherry && \
    rm -rf ./llm-foundry && \
    rm -rf /usr/local/lib/python3.12/dist-packages/composer

RUN python3 -c "\
import importlib; \
from cudacherry._build_spec import FAMILIES; \
[importlib.import_module(f.module_path) for f in FAMILIES]; \
print(f'cudacherry: {len(FAMILIES)} fp8 cuda extensions OK')"

# Never JIT-compile in a training image.
ENV CUDACHERRY_DISABLE_JIT=1

WORKDIR /opt
RUN pip uninstall -y deep_gemm && \
    n=0; until bash ./deepgemm/install.sh; do \
        n=$((n+1)); [ "$n" -ge 6 ] && { echo "deepgemm install failed after 6 tries"; exit 1; }; \
        echo "retry deepgemm install ($n)"; sleep 30; \
    done && \
    rm -rf ./deepgemm
RUN echo $(python3 -c "import deep_gemm; print('DeepGEMM version: ', deep_gemm.__version__)")
RUN echo $(python3 -c "import deep_gemm; print('DeepGEMM commit: ', deep_gemm.__commit__)")

###############
# DeepSeek Sparse Attention (DSA): FlashMLA (sparse fwd) + vendored CuTeDSL kernels
# (llmfoundry/models/layers/cudnn_dsa in the repo) + the cutlass-dsl cu12 mix
###############
USER root

# DSA is Hopper-only (sm90 CuTeDSL kernels, FlashMLA sm90, tilelang SM90). On any other
# CUDA_ARCH each step below is a no-op, so the image still builds - just without DSA, and
# the attention registry then resolves LlamaLatentSparseAttention to its ImportError stub.

# Runtime deps of the vendored kernels (previously pulled in by nvidia-cudnn-frontend[cutedsl],
# which is no longer installed - the kernels live in the repo with our fixes applied).
RUN if [ "${CUDA_ARCH}" != "sm_90" ]; then echo "CUDA_ARCH=${CUDA_ARCH}: building without the DSA kernels"; exit 0; fi; \
    n=0; until env -u PIP_CONSTRAINT pip install --no-cache-dir --no-deps cuda-python==12.9.0 cuda-bindings==12.9.0; do \
        n=$((n+1)); [ "$n" -ge 6 ] && { echo "cuda-python wheels failed after 6 tries"; exit 1; }; \
        echo "retry cuda-python wheels ($n)"; sleep 30; \
    done && \
    n=0; until env -u PIP_CONSTRAINT pip install --no-cache-dir apache-tvm-ffi==0.1.11; do \
        n=$((n+1)); [ "$n" -ge 6 ] && { echo "tvm-ffi failed after 6 tries"; exit 1; }; \
        echo "retry tvm-ffi ($n)"; sleep 30; \
    done

# cutlass-dsl mix: the vendored kernels need the 4.5.x python API (make_fake_stream; ships cu13-only libs),
# but the cluster's CUDA-12.8 driver needs cu12 libs (only in 4.6.x) -> 4.5.3 python + 4.6.1 cu12 libs.
RUN if [ "${CUDA_ARCH}" != "sm_90" ]; then echo "CUDA_ARCH=${CUDA_ARCH}: building without the DSA kernels"; exit 0; fi; \
    pip uninstall -y nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-cu13 nvidia-cutlass-dsl-libs-cu12 || true && \
    env -u PIP_CONSTRAINT pip install --no-cache-dir --no-deps nvidia-cutlass-dsl==4.5.3 nvidia-cutlass-dsl-libs-base==4.5.3 && \
    env -u PIP_CONSTRAINT pip install --no-cache-dir --no-deps nvidia-cutlass-dsl-libs-cu12==4.6.1

# tilelang (stage-2 top-k selector + 32-head sparse-MLA fwd/bwd). --no-deps avoids pulling a conflicting
RUN if [ "${CUDA_ARCH}" != "sm_90" ]; then echo "CUDA_ARCH=${CUDA_ARCH}: building without the DSA kernels"; exit 0; fi; \
    env -u PIP_CONSTRAINT pip install --no-cache-dir --no-deps tilelang==0.1.12 z3-solver==4.15.4.0

# FlashMLA (sparse MLA fwd), from source. csrc/cutlass is shallow-fetched at the pin so setup.py skips
# the recursive-submodule clone (which stalls on GitHub egress).
ARG FLASHMLA_COMMIT=9241ae3ef9bac614dd25e45e507e089f888280e0
ARG FLASHMLA_CUTLASS_COMMIT=147f5673d0c1c3dcf66f78d677fd647e4a020219
RUN if [ "${CUDA_ARCH}" != "sm_90" ]; then echo "CUDA_ARCH=${CUDA_ARCH}: building without the DSA kernels"; exit 0; fi; \
    cd /home/user && git config --global http.version HTTP/1.1 && \
    mkdir -p FlashMLA && cd FlashMLA && git init -q && \
    git remote add origin https://github.com/deepseek-ai/FlashMLA.git && \
    n=0; until git fetch --depth 1 --no-tags origin ${FLASHMLA_COMMIT}; do \
        n=$((n+1)); [ "$n" -ge 6 ] && { echo "FlashMLA fetch failed after 6 tries"; exit 1; }; \
        echo "retry FlashMLA fetch ($n)"; sleep 30; \
    done && \
    git checkout -q FETCH_HEAD && \
    rm -rf csrc/cutlass && mkdir -p csrc/cutlass && cd csrc/cutlass && \
    git init -q && git remote add origin https://github.com/NVIDIA/cutlass.git && \
    n=0; until git fetch --depth 1 origin ${FLASHMLA_CUTLASS_COMMIT}; do \
        n=$((n+1)); [ "$n" -ge 6 ] && { echo "cutlass fetch failed after 6 tries"; exit 1; }; \
        echo "retry cutlass fetch ($n)"; sleep 30; \
    done && git checkout -q FETCH_HEAD && \
    cd /home/user/FlashMLA && \
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=8 pip install --no-build-isolation --no-cache-dir . && \
    cd /home/user && rm -rf FlashMLA

# Sanity: DSA kernel deps import (the vendored kernels themselves live in the repo, not the image;
# imports fine on the GPU-less builder)
RUN if [ "${CUDA_ARCH}" != "sm_90" ]; then echo "CUDA_ARCH=${CUDA_ARCH}: building without the DSA kernels"; exit 0; fi; \
    python3 -c "import flash_mla, cutlass, cuda.bindings.driver, tilelang; print('DSA kernel deps OK; cutlass-dsl', cutlass.__version__, 'tilelang', tilelang.__version__)"
USER user
WORKDIR /home/user
###############
## MegaBlocks
###############
ENV MEGABLOCKS_VERSION=v0.5.1
RUN cd /home/user/ && \
    n=0; until git clone -b ${MEGABLOCKS_VERSION} --single-branch --depth 1 https://github.com/stanford-futuredata/megablocks.git; do \
        n=$((n+1)); [ "$n" -ge 6 ] && { echo "megablocks clone failed after 6 tries"; exit 1; }; \
        echo "retry megablocks clone ($n)"; rm -rf megablocks; sleep 30; \
    done && \
    cd megablocks && \
    n=0; until git submodule update --init --recursive --depth 1; do \
        n=$((n+1)); [ "$n" -ge 6 ] && { echo "megablocks submodules failed after 6 tries"; exit 1; }; \
        echo "retry megablocks submodules ($n)"; sleep 30; \
    done

WORKDIR /home/user/megablocks

RUN pip install --no-build-isolation "."

RUN rm -rf /home/user/megablocks
###############

WORKDIR /home/user
USER root

RUN pip uninstall transformer_engine -y

USER user

# final check
RUN echo "checking torch version in the end" && \
    export PYTORCH_VERSION=$(cat /tmp/pytorch_version.txt) && \
    export PYTORCH_VERSION_AFTER=$(python3 -c "import torch; print(torch.__version__)") && \
    if [ "$PYTORCH_VERSION" != "$PYTORCH_VERSION_AFTER" ]; then \
        echo "ERROR: PyTorch version was reinstalled during dependency installation." && \
        echo "Before: $PYTORCH_VERSION, After: $PYTORCH_VERSION_AFTER" && \
        exit 1; \
    fi

USER user
WORKDIR /home/user

RUN pip freeze

USER user

ENV NVIDIA_GDRCOPY=enabled
