ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.03-py3
ARG UV_VERSION=0.12.10
ARG UV_TARGET=x86_64-unknown-linux-gnu
ARG USER_NAME=user
ARG USER_UID=1000
ARG USER_HOME=/home/user
ARG TORCHTITAN_HOME=/home/user/torchtitan
ARG CUDA_HOME=/usr/local/cuda
ARG NVSHMEM_DIR=/home/user/nvshmem
ARG NCCL_TESTS_REF=a0b82b2260cf5152b9f8c061bbf7eaf0ba096432
ARG DEEPEP_BRANCH=hybrid-ep
ARG DEEPEP_REF=1b8f467
ARG CUDACHERRY_COMMIT=585bd50a641857ccae8b5ec691bb15791c9529e7
ARG USE_CPP_20_STANDARD=1

FROM ${BASE_IMAGE} AS base
ARG CUDA_HOME
ARG DEEPEP_BRANCH
ARG DEEPEP_REF
ARG NVSHMEM_DIR
ARG TORCHTITAN_HOME
ARG UV_TARGET
ARG UV_VERSION
ARG USER_HOME
ARG USER_NAME
ARG USER_UID
ARG CUDACHERRY_COMMIT
ARG USE_CPP_20_STANDARD

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV CUDA_HOME=${CUDA_HOME} \
    CUDACHERRY_DISABLE_JIT=1 \
    DEEPEP_BRANCH=${DEEPEP_BRANCH} \
    DEEPEP_REF=${DEEPEP_REF} \
    CUDACHERRY_COMMIT=${CUDACHERRY_COMMIT} \
    NVSHMEM_DIR=${NVSHMEM_DIR} \
    OPAL_PREFIX=/opt/hpcx/ompi \
    PIP_CONSTRAINT="" \
    TORCHTITAN_HOME=${TORCHTITAN_HOME} \
    VIRTUAL_ENV=/opt/torchtitan-venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/torchtitan-venv \
    UV_PYTHON_DOWNLOADS=never \
    USER_HOME=${USER_HOME} \
    USER_NAME=${USER_NAME} \
    USER_UID=${USER_UID} \
    NVIDIA_GDRCOPY=enabled \
    NVIDIA_IMEX_CHANNELS=0 \
    USE_CPP_20_STANDARD=${USE_CPP_20_STANDARD}

# Needed for non-login Docker RUN commands and runtime processes
ENV LD_LIBRARY_PATH=${NVSHMEM_DIR}/lib:/usr/local/cuda-13.2/compat/lib:/opt/hpcx/nccl_rdma_sharp_plugin/lib:/opt/hpcx/ucc/lib/ucc:/opt/hpcx/ucc/lib:/opt/hpcx/ucx/lib/ucx:/opt/hpcx/ucx/lib:/opt/hpcx/sharp/lib:/opt/hpcx/hcoll/lib:/opt/hpcx/ompi/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PATH=${VIRTUAL_ENV}/bin:${NVSHMEM_DIR}/bin:/opt/hpcx/sharp/bin:/opt/hpcx/clusterkit/bin:/opt/hpcx/hcoll/bin:/opt/hpcx/ucc/bin:/opt/hpcx/ucx/bin:/opt/hpcx/ompi/bin:${USER_HOME}/conda/bin:/usr/local/nvidia/bin:${CUDA_HOME}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

ADD https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-${UV_TARGET}.tar.gz /tmp/uv.tar.gz

RUN userdel ubuntu || true \
    && useradd -u "${USER_UID}" -m "${USER_NAME}" --shell /bin/bash \
    && mkdir -p /home/jovyan \
    && chown "${USER_NAME}:${USER_NAME}" /home/jovyan \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-compat-13-2 \
        openssh-server \
    && rm -rf /var/lib/apt/lists/* \
    && tar -xzf /tmp/uv.tar.gz -C /tmp \
    && install "/tmp/uv-${UV_TARGET}/uv" "/tmp/uv-${UV_TARGET}/uvx" /usr/local/bin/ \
    && rm -rf /tmp/uv.tar.gz "/tmp/uv-${UV_TARGET}" \
    && mkdir -p /var/run/sshd \
    && ssh-keygen -A \
    && echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config.d/mlspace.conf \
    && { \
        echo "UsePAM no"; \
        echo "PubkeyAcceptedAlgorithms +ssh-rsa"; \
        echo "Port 2222"; \
        echo "PidFile /run/sshd/sshd.pid"; \
    } > /etc/ssh/sshd_config.d/mlspace.conf \
    && chmod 600 /etc/ssh/ssh_host_* \
    && chown "${USER_NAME}:${USER_NAME}" /etc/ssh/ssh_host_* /run/sshd \
    && { \
        echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"; \
        echo "export PATH=${PATH}"; \
        echo "export NVSHMEM_DIR=${NVSHMEM_DIR}"; \
        echo "export CUDA_HOME=${CUDA_HOME}"; \
        echo "export OPAL_PREFIX=${OPAL_PREFIX}"; \
        echo "export USE_CPP_20_STANDARD=${USE_CPP_20_STANDARD}"; \
    } > /etc/profile.d/wm.sh

FROM base AS nccl-tests-builder
ARG NCCL_TESTS_REF

RUN cd /workspace \
    && mkdir nccl-tests \
    && cd nccl-tests \
    && git init \
    && git remote add origin https://github.com/NVIDIA/nccl-tests.git \
    && git fetch --depth 1 origin "${NCCL_TESTS_REF}" \
    && git checkout FETCH_HEAD \
    && make MPI=1 \
        MPI_HOME=/opt/hpcx/ompi/ \
        CUDA_HOME="${CUDA_HOME}" \
        NCCL_HOME=/usr/lib/x86_64-linux-gnu/ \
    && mv build /opt/nccl-tests

FROM base AS runtime

COPY --from=nccl-tests-builder /opt/nccl-tests /opt/nccl-tests

RUN mkdir -p "${TORCHTITAN_HOME}/gift_docker"
COPY gift_docker/pyproject.toml gift_docker/uv.lock ${TORCHTITAN_HOME}/gift_docker/
RUN --mount=type=secret,id=ci_job_token,required=true,env=UV_INDEX_PYTORCH_NIGHTLY_CU130_PASSWORD \
    cd "${TORCHTITAN_HOME}/gift_docker" \
    && UV_INDEX_PYTORCH_NIGHTLY_CU130_USERNAME=gitlab-ci-token uv sync \
        --python python3.12 \
        --frozen \
        --inexact \
        --all-extras \
        --no-install-project \
        --prerelease allow

COPY deepgemm /opt/deepgemm

### create fp8 deepgemm cache folder
RUN for i in $(seq 0 7); do \
        mkdir -p /temp/.cache/rank_${i}/deepgemm/cache/; \
    done \
    && chmod -R 777 /temp/.cache

RUN (uv pip install wheel \
        && (uv pip uninstall deep_gemm || true) \
        && cd /opt/deepgemm \
        && rm -rf build dist *.egg-info \
        && python setup.py bdist_wheel \
        && uv pip install dist/*.whl) > /tmp/deepgemm_build.log 2>&1 \
    || (echo "===== deep_gemm build FAILED; errors: ====="; \
        grep -inE "error|fatal" /tmp/deepgemm_build.log | head -50; \
        echo "===== last 300 lines: ====="; \
        tail -n 300 /tmp/deepgemm_build.log; exit 1)
RUN python3 -c "import deep_gemm; print('DeepGEMM', deep_gemm.__version__, getattr(deep_gemm, '__commit__', ''))" \
    && rm -rf /opt/deepgemm

COPY --chown=${USER_NAME}:${USER_NAME} . ${TORCHTITAN_HOME}
RUN uv pip install \
        --no-build-isolation \
        --no-deps \
        --editable "${TORCHTITAN_HOME}" \
    && rm -rf "${TORCHTITAN_HOME}/cudacherry" "${TORCHTITAN_HOME}/deepgemm"

RUN uv pip install \
        --no-build-isolation \
        --no-deps \
        "${TORCHTITAN_HOME}/contrib/streaming"

RUN NVSHMEM_PKG_DIR="$(python3 -c "import nvidia.nvshmem as n; print(n.__path__[0])")" \
    && test -d "${NVSHMEM_PKG_DIR}/lib" \
    && test -d "${NVSHMEM_PKG_DIR}/include" \
    && HOST_LIB="$(ls "${NVSHMEM_PKG_DIR}"/lib/libnvshmem_host.so.* | head -n1)" \
    && test -n "${HOST_LIB}" \
    && ln -sfn "$(basename "${HOST_LIB}")" "${NVSHMEM_PKG_DIR}/lib/libnvshmem_host.so" \
    && ln -sfn "${NVSHMEM_PKG_DIR}" "${NVSHMEM_DIR}" \
    && git clone --branch "${DEEPEP_BRANCH}" https://github.com/deepseek-ai/DeepEP.git /tmp/DeepEP \
    && cd /tmp/DeepEP \
    && git checkout "${DEEPEP_REF}" \
    && sed -i "s|f'{nvshmem_dir}/include']|f'{nvshmem_dir}/include', '${CUDA_HOME}/include/cccl']|" setup.py \
    && sed -i '/"-std=c++[0-9]*",/d' setup.py \
    && TORCH_CUDA_ARCH_LIST="9.0" uv pip install \
        -v \
        --no-build-isolation \
        . \
    && rm -rf /tmp/DeepEP \
    && chown -R "${USER_NAME}:${USER_NAME}" "${TORCHTITAN_HOME}" "${NVSHMEM_DIR}"

# flash-linear-attention (fla): GDN/KDA kernels for gift_moe gift_3_5.
# --no-deps so it does not perturb the pinned torch nightly.
RUN uv pip install \
        --no-build-isolation \
        --no-deps \
        "${TORCHTITAN_HOME}/contrib/flash-linear-attention"

COPY cudacherry /opt/cudacherry
RUN git -C /opt/cudacherry checkout --detach "${CUDACHERRY_COMMIT}"
RUN TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=16 uv pip install \
        --no-build-isolation \
        --no-deps \
        /opt/cudacherry \
        > /tmp/cudacherry_build.log 2>&1 \
    || (echo "===== cudacherry build FAILED; errors: ====="; \
        grep -inE "error|fatal" /tmp/cudacherry_build.log | head -50; \
        echo "===== last 300 lines: ====="; \
        tail -n 300 /tmp/cudacherry_build.log; exit 1)
RUN python3 -c "import importlib; from cudacherry._build_spec import FAMILIES; [importlib.import_module(f.module_path) for f in FAMILIES]; print(f'cudacherry: {len(FAMILIES)} fp8 cuda extensions OK')" \
    && rm -rf /opt/cudacherry

USER ${USER_NAME}
WORKDIR ${TORCHTITAN_HOME}

FROM runtime AS import-check

RUN python3 "${TORCHTITAN_HOME}/scripts/ci/validate_docker_imports.py"

FROM import-check AS final
