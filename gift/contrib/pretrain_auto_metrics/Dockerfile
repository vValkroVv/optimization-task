FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04 AS os_base

# создаем пользователя с ID 1000 и группой с ID 1000
# cоздаем пустую директорию с правами user
RUN userdel ubuntu && useradd -u 1000 -m user --shell /bin/bash \
    && mkdir -p /home/jovyan && chown user:user /home/jovyan

# Ставим OpenSSH для MPI что бы соединятся между контейнирами
# Генерим ключи
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server tmux python3 python3-venv python3-pip ca-certificates \
    libnuma1 numactl git build-essential openmpi-bin libopenmpi-dev librdmacm-dev python3-dev python3.12-dev zstd bash ninja-build \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libncursesw5-dev xz-utils tk-dev \
    libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev curl cmake file libavcodec-dev libavformat-dev libavfilter-dev \
    libavdevice-dev libavutil-dev libswresample-dev libswscale-dev libopus-dev libopusenc-dev libfftw3-dev libsndfile1-dev libmp3lame-dev \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd \
    && ssh-keygen -A

RUN apt-get update && apt-get install -y --no-install-recommends cuda-gdb-12-9 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --break-system-packages py-spy && \
    py-spy --version

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

RUN mkdir -p /workspace && \
    cd /workspace && \
    git clone https://github.com/NVIDIA/nccl-tests.git && \
    cd nccl-tests && \
    make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/lib/x86_64-linux-gnu/ && \
    mv build /opt/nccl-tests && \
    cd /workspace && \
    rm -rf nccl-tests && \
    rm -rf /var/lib/apt/lists/*

RUN echo "export LD_LIBRARY_PATH=/opt/hpcx/nccl_rdma_sharp_plugin/lib:/opt/hpcx/ucc/lib/ucc:/opt/hpcx/ucc/lib:/opt/hpcx/ucx/lib/ucx:/opt/hpcx/ucx/lib:/opt/hpcx/sharp/lib:/opt/hpcx/hcoll/lib:/opt/hpcx/ompi/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64" > /etc/profile.d/wm.sh && \
    echo "export PATH=/usr/local/gdrcopy/bin:/home/user/nvshmem/bin:/opt/hpcx/sharp/bin:/opt/hpcx/clusterkit/bin:/opt/hpcx/hcoll/bin:/opt/hpcx/ucc/bin:/opt/hpcx/ucx/bin:/opt/hpcx/ompi/bin:/home/user/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" >> /etc/profile.d/wm.sh && \
    echo "export OPAL_PREFIX=/opt/hpcx/ompi" >> /etc/profile.d/wm.sh
RUN chown user:user /etc/profile.d/wm.sh \
    && chmod 644 /etc/profile.d/wm.sh

USER user
WORKDIR /home/user

ENV PIP_NO_CACHE_DIR=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1


FROM os_base AS lm_eval_bundle

# setup lm_eval
RUN python3 -m venv lm_eval --system-site-packages
COPY --chown=user:user ./contrib/lm_evaluation_harness /home/user/lm-evaluation-harness
COPY --chown=user:user ./build_utils/lm_eval_prefetch.py /home/user/build_utils/

RUN /home/user/lm_eval/bin/pip install python-Levenshtein
RUN cd /home/user/lm-evaluation-harness && \
    /home/user/lm_eval/bin/pip install ".[api, math, longbench, ruler, libra, slava, spellcheck_punctuation_ru]"
RUN /home/user/lm_eval/bin/pip install "jsonschema[format]"
RUN /home/user/lm_eval/bin/pip install emoji syllapy nltk
# datasets 3.6.0 uses legacy dataset IDs rejected by newer Hub URI parsing.
RUN /home/user/lm_eval/bin/pip install -U "transformers==5.8.1" "peft==0.18.1" "huggingface_hub==1.5.0"
RUN /home/user/lm_eval/bin/pip install -U datasets==3.6.0
RUN /home/user/lm_eval/bin/pip install hf_transfer
RUN /home/user/lm_eval/bin/pip install sentencepiece
# filelock >3.29.7 adds _audit_fork_safety, which aborts the os.fork() that
# code_eval does in multiprocessing.Manager() -> breaks mbpp/humaneval scoring
RUN /home/user/lm_eval/bin/pip install "filelock==3.29.7"
COPY --chown=user:user ./contrib/lm_evaluation_harness/lm_eval_version.txt /home/user/lm_eval_version.txt
RUN cat /home/user/lm_eval_version.txt

ENV NLTK_DATA=/home/user/nltk_data
COPY --chown=user:user ./build_utils/nltk_prefetch.py /home/user/build_utils/
RUN env -u HTTP_PROXY -u HTTPS_PROXY -u NO_PROXY \
        -u http_proxy -u https_proxy -u no_proxy \
        /home/user/lm_eval/bin/python /home/user/build_utils/nltk_prefetch.py \
        --resources punkt_tab

ENV HF_ALLOW_CODE_EVAL=1
RUN --mount=type=secret,id=gpqa_hf_token,uid=1000,gid=1000,mode=0400 \
    bash -lc ' \
        export HF_TOKEN="$(cat /run/secrets/gpqa_hf_token)" && \
        /home/user/lm_eval/bin/python /home/user/build_utils/lm_eval_prefetch.py \
            --python-bin /home/user/lm_eval/bin/python \
            --tasks truthfulqa,agieval,piqa,triviaqa,mmlu,mmlu_pro,mmlu_redux_generative,hellaswag,gpqa,winogrande,supergpqa,arc_challenge,bbh,mmmlu'

RUN --mount=type=secret,id=gpqa_hf_token,uid=1000,gid=1000,mode=0400 \
    bash -lc ' \
        export HF_TOKEN="$(cat /run/secrets/gpqa_hf_token)" && \
        /home/user/lm_eval/bin/python /home/user/build_utils/lm_eval_prefetch.py \
            --python-bin /home/user/lm_eval/bin/python \
            --tasks gsm8k,minerva_math,mgsm_direct,mgsm_cot_native,hendrycks_math,aime24,aime25,aime26,mmmlu_zh_cn,gpqa_diamond_n_shot,arc_easy,nq_open,gpqa_diamond_zeroshot,minerva_math500,hendrycks_math500,hmmt,asdiv_cot_llama'

RUN --mount=type=secret,id=gpqa_hf_token,uid=1000,gid=1000,mode=0400 \
    bash -lc ' \
        export HF_TOKEN="$(cat /run/secrets/gpqa_hf_token)" && \
        /home/user/lm_eval/bin/python /home/user/build_utils/lm_eval_prefetch.py \
            --python-bin /home/user/lm_eval/bin/python \
            --tasks lcb_code_gen_lite,humaneval_plus,mbpp_plus,mbpp,humaneval,lcb_code_gen_lite_instruct,cruxeval'

RUN --mount=type=secret,id=gpqa_hf_token,uid=1000,gid=1000,mode=0400 \
    bash -lc ' \
        export HF_TOKEN="$(cat /run/secrets/gpqa_hf_token)" && \
        export MULTIPL_E_CACHE_ONLY=1 && \
        /home/user/lm_eval/bin/python /home/user/build_utils/lm_eval_prefetch.py \
            --python-bin /home/user/lm_eval/bin/python \
            --tasks multipl_e_humaneval'

RUN --mount=type=secret,id=gpqa_hf_token,uid=1000,gid=1000,mode=0400 \
    bash -lc ' \
        export HF_TOKEN="$(cat /run/secrets/gpqa_hf_token)" && \
        export BCB_CACHE_ONLY=1 && \
        /home/user/lm_eval/bin/python /home/user/build_utils/lm_eval_prefetch.py \
            --python-bin /home/user/lm_eval/bin/python \
            --tasks bigcodebench'

RUN --mount=type=secret,id=gpqa_hf_token,uid=1000,gid=1000,mode=0400 \
    bash -lc ' \
        export HF_TOKEN="$(cat /run/secrets/gpqa_hf_token)" && \
        /home/user/lm_eval/bin/python /home/user/build_utils/lm_eval_prefetch.py \
            --python-bin /home/user/lm_eval/bin/python \
            --tasks ruler,longbench2,longbench \
            --metadata "{\"max_seq_lengths\":[4096],\"tokenizer\":\"gpt2\"}"'

RUN --mount=type=secret,id=gpqa_hf_token,uid=1000,gid=1000,mode=0400 \
    bash -lc ' \
        export HF_TOKEN="$(cat /run/secrets/gpqa_hf_token)" && \
        /home/user/lm_eval/bin/python /home/user/build_utils/lm_eval_prefetch.py \
            --python-bin /home/user/lm_eval/bin/python \
            --tasks mrcr \
            --download-only \
            --metadata "{\"max_seq_lengths\":[4096],\"tokenizer\":\"gpt2\"}"'

RUN --mount=type=secret,id=gpqa_hf_token,uid=1000,gid=1000,mode=0400 \
    bash -lc ' \
        export HF_TOKEN="$(cat /run/secrets/gpqa_hf_token)" && \
        /home/user/lm_eval/bin/python /home/user/build_utils/lm_eval_prefetch.py \
            --python-bin /home/user/lm_eval/bin/python \
            --tasks mmlu_ru,pollux,babymmlu2'

RUN --mount=type=secret,id=sage_hf_token,uid=1000,gid=1000,mode=0400 \
    bash -lc ' \
        export HF_TOKEN="$(cat /run/secrets/sage_hf_token)" && \
        /home/user/lm_eval/bin/python /home/user/build_utils/lm_eval_prefetch.py \
            --python-bin /home/user/lm_eval/bin/python \
            --tasks doom,gsm8k_ru,minerva_math_ru,fakemmlu,rummlu'

RUN --mount=type=secret,id=sage_hf_token,uid=1000,gid=1000,mode=0400 \
    bash -lc ' \
        export HF_TOKEN="$(cat /run/secrets/sage_hf_token)" && \
        /home/user/lm_eval/bin/python /home/user/build_utils/lm_eval_prefetch.py \
            --python-bin /home/user/lm_eval/bin/python \
            --tasks libra_tasks,repa,ruscontext,russian_gec,rublimp,bouquet,coat,rucola'

RUN --mount=type=secret,id=sage_hf_token,uid=1000,gid=1000,mode=0400 \
    bash -lc ' \
        export HF_TOKEN="$(cat /run/secrets/sage_hf_token)" && \
        /home/user/lm_eval/bin/python /home/user/build_utils/lm_eval_prefetch.py \
            --python-bin /home/user/lm_eval/bin/python \
            --tasks slava,spellcheck_punctuation_ru,jsonschema_bench,drop'

RUN --mount=type=secret,id=gpqa_hf_token,uid=1000,gid=1000,mode=0400 \
    bash -lc ' \
        export HF_TOKEN="$(cat /run/secrets/gpqa_hf_token)" && \
        /home/user/lm_eval/bin/python /home/user/build_utils/lm_eval_prefetch.py \
            --python-bin /home/user/lm_eval/bin/python \
            --tasks ifbench'

RUN --mount=type=secret,id=gpqa_hf_token,uid=1000,gid=1000,mode=0400 \
    bash -lc ' \
        export HF_TOKEN="$(cat /run/secrets/gpqa_hf_token)" && \
        /home/user/lm_eval/bin/python /home/user/build_utils/lm_eval_prefetch.py \
            --python-bin /home/user/lm_eval/bin/python \
            --tasks agieval_deepseek_card,evalplus_official,gpqa_diamond_cot_official,gpqa_main_cot_official,gpqa_extended_cot_official,supergpqa_cot,supergpqa_cot_official'

RUN rm -rf /home/user/lm-evaluation-harness

RUN bash -lc '\
    set -euo pipefail; \
    src=/home/user/.cache/huggingface/hub; \
    mkdir -p /home/user/hf_hub_chunk_{0,1,2,3,4,5}; \
    i=0; \
    while IFS= read -r -d "" d; do \
        b=$((i % 6)); \
        mv "$d" "/home/user/hf_hub_chunk_${b}/"; \
        i=$((i+1)); \
    done < <(find "$src" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)'


FROM os_base AS mera_bundle

COPY --chown=user:user ./mera_united /home/user/mera-united
COPY --chown=user:user ./build_utils/lm_eval_prefetch.py /home/user/build_utils/
RUN cp -r /home/user/mera-united/mera_tasks /home/user/mera_tasks && \
    cp /home/user/mera-united/mera_version.txt /home/user/mera_version.txt && \
    cp -r /home/user/mera-united/scripts /home/user/mera_scripts

RUN python3 -m venv mera_eval --system-site-packages
RUN cd /home/user/mera-united/lm-evaluation-harness && \
    /home/user/mera_eval/bin/pip install . && \
    /home/user/mera_eval/bin/pip install -U "transformers==4.56.2" "peft==0.18.0" && \
    /home/user/mera_eval/bin/pip install -U datasets==3.6.0 && \
    /home/user/mera_eval/bin/pip install sentencepiece && \
    /home/user/mera_eval/bin/pip install hf_transfer && \
    /home/user/mera_eval/bin/pip install flake8 flake8-json

COPY --chown=user:user ./node_runner/resources/mera_system_instruction.txt /home/user/mera_system_instruction.txt
RUN cat /home/user/mera_version.txt

ENV HF_ALLOW_CODE_EVAL=1
ENV LINTER_PATH=/home/user/mera_eval/bin/flake8
RUN bash -lc ' \
        /home/user/mera_eval/bin/python /home/user/build_utils/lm_eval_prefetch.py \
            --python-bin /home/user/mera_eval/bin/python \
            --include-path /home/user/mera_tasks \
            --tasks simplear,bps_gen,lcs_gen,chegeka,mathlogicqa_gen,parus_gen,rcb_gen,rudetox,ruhatespeech_gen,rummlu_gen,ruworldtree_gen,ruopenbookqa_gen'

RUN bash -lc ' \
        /home/user/mera_eval/bin/python /home/user/build_utils/lm_eval_prefetch.py \
            --python-bin /home/user/mera_eval/bin/python \
            --include-path /home/user/mera_tasks \
            --tasks rumultiar,use,rwsd_gen,mamuramu_gen,multiq,rumodar,ruhhh_gen,ruethics_gen,rutie_gen,rucodeeval,ruhumaneval'

RUN bash -lc ' \
        /home/user/mera_eval/bin/python /home/user/build_utils/lm_eval_prefetch.py \
            --python-bin /home/user/mera_eval/bin/python \
            --include-path /home/user/mera_tasks \
            --tasks agro_bench,aqua_bench,med_bench,codecorrectness,strucom,codelintereval'


FROM os_base AS obm_bundle

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV UV_PYTHON_INSTALL_DIR=/home/user/obm_python

RUN /home/user/.local/bin/uv python install 3.10.19

RUN /home/user/.local/bin/uv venv --python 3.10.19 /home/user/obm_env

COPY --chown=user:user ./one-button-metrics /home/user/one-button-metrics
COPY --chown=user:user ./one-button-metrics/obm_version.txt /home/user/obm_version.txt
RUN cat /home/user/obm_version.txt

RUN cd /home/user/one-button-metrics && \
    /home/user/.local/bin/uv pip install --python /home/user/obm_env/bin/python ".[full]" -c requirements/constraints.txt && \
    /home/user/.local/bin/uv pip uninstall --python /home/user/obm_env/bin/python obm && \
    /home/user/obm_env/bin/python -m obm.versions_cli check

# new_obm: тот же OBM после рефакторинга, свой venv (constraints разошлись),
# данные не пуллим второй раз — хардлинки на данные one-button-metrics ниже
RUN /home/user/.local/bin/uv venv --python 3.10.19 /home/user/new_obm_env

COPY --chown=user:user ./new_obm /home/user/new_obm
COPY --chown=user:user ./new_obm/new_obm_version.txt /home/user/new_obm_version.txt
RUN cat /home/user/new_obm_version.txt

RUN cd /home/user/new_obm && \
    /home/user/.local/bin/uv pip install --python /home/user/new_obm_env/bin/python ".[full]" -c requirements/constraints.txt && \
    /home/user/.local/bin/uv pip uninstall --python /home/user/new_obm_env/bin/python obm

ENV NLTK_DATA=/home/user/nltk_data
COPY --chown=user:user ./build_utils/nltk_prefetch.py /home/user/build_utils/
RUN /home/user/obm_env/bin/python /home/user/build_utils/nltk_prefetch.py \
        --resources punkt,wordnet,punkt_tab,stopwords,averaged_perceptron_tagger_eng

RUN mkdir -p /home/user/one-button-metrics/.dvc/tmp /home/user/new_obm/.dvc/tmp && \
    chown -R 1000:1000 /home/user/one-button-metrics/.dvc /home/user/new_obm/.dvc && \
    chmod -R 777 /home/user/one-button-metrics/.dvc /home/user/new_obm/.dvc

WORKDIR /home/user/one-button-metrics
RUN --mount=type=secret,id=obm_s3_access_key,uid=1000,gid=1000,mode=0400 \
    --mount=type=secret,id=obm_s3_secret_key,uid=1000,gid=1000,mode=0400 \
    --mount=type=secret,id=obm_s3_endpoint_url,uid=1000,gid=1000,mode=0400 \
    bash -lc ' \
        export DVC_S3_ACCESS_KEY_ID="$(cat /run/secrets/obm_s3_access_key)" && \
        export DVC_S3_SECRET_ACCESS_KEY="$(cat /run/secrets/obm_s3_secret_key)" && \
        export DVC_S3_ENDPOINT_URL="$(cat /run/secrets/obm_s3_endpoint_url)" && \
        cd /home/user/one-button-metrics && \
        /home/user/obm_env/bin/dvc config --local core.no_scm true && \
        /home/user/obm_env/bin/dvc remote modify --local remote-obm url s3://d-gccore-onebuttonmetrics/obm && \
        /home/user/obm_env/bin/dvc remote modify --local remote-obm endpointurl "$DVC_S3_ENDPOINT_URL" && \
        /home/user/obm_env/bin/dvc remote modify --local remote-obm access_key_id "$DVC_S3_ACCESS_KEY_ID" && \
        /home/user/obm_env/bin/dvc remote modify --local remote-obm secret_access_key "$DVC_S3_SECRET_ACCESS_KEY" && \
        /home/user/obm_env/bin/dvc remote modify --local remote-obm jobs 8 && \
        /home/user/obm_env/bin/dvc pull && \
        rm -rf /home/user/one-button-metrics/.dvc/cache /home/user/one-button-metrics/.dvc/config.local && \
        cd /home/user && \
        find one-button-metrics -name "*.dvc" -type f | while read -r ptr; do \
            rel="${ptr#one-button-metrics/}"; rel="${rel%.dvc}"; \
            mkdir -p "new_obm/$(dirname "$rel")" && \
            cp -al "one-button-metrics/$rel" "new_obm/$(dirname "$rel")/" || exit 1; \
        done && \
        tar --zstd -cf obm.tar.zst one-button-metrics new_obm && \
        rm -rf one-button-metrics new_obm'
WORKDIR /home/user

FROM os_base AS dredd_bundle

COPY --chown=user:user ./contrib/dredd/requirements.txt /tmp/dredd_requirements.txt
RUN python3 -m venv /home/user/dredd_env --system-site-packages \
    && /home/user/dredd_env/bin/pip install --upgrade pip setuptools wheel \
    && /home/user/dredd_env/bin/pip install -r /tmp/dredd_requirements.txt

COPY --chown=user:user ./contrib/dredd/data.tar.zst /tmp/data.tar.zst
RUN mkdir -p /home/user/data && tar --zstd -xf /tmp/data.tar.zst -C /home/user/data && rm /tmp/data.tar.zst

COPY --chown=user:user ./contrib/dredd/dredd_version.txt /home/user/dredd_version.txt
RUN cat /home/user/dredd_version.txt


FROM os_base AS safety_unified_bundle

COPY --chown=user:user ./safety-unified/data.tar.zst /tmp/safety_unified_data.tar.zst
COPY --chown=user:user ./safety-unified/runner /home/user/safety_unified/runner
RUN mkdir -p /home/user/safety_unified/data \
    && tar --zstd -xf /tmp/safety_unified_data.tar.zst -C /home/user/safety_unified/data \
    && rm /tmp/safety_unified_data.tar.zst
COPY --chown=user:user ./safety-unified/data/ /home/user/safety_unified/data/
COPY --chown=user:user ./safety-unified/safety_unified_version.txt /home/user/safety_unified/safety_unified_version.txt


# setup aa_lcr
FROM os_base AS aa_lcr_bundle

COPY --chown=user:user ./aa-lcr-bench/data /home/user/aa_lcr/data
COPY --chown=user:user ./aa-lcr-bench/runner /home/user/aa_lcr/runner
COPY --chown=user:user ./aa-lcr-bench/aa_lcr_version.txt /home/user/aa_lcr/aa_lcr_version.txt
RUN cat /home/user/aa_lcr/aa_lcr_version.txt


FROM os_base AS lmms_eval_bundle

ARG GIGAVISION_RELEASE_TAG=v0.7.1+gv.0.3.8

RUN python3 -m venv /home/user/lmms_eval \
    && /home/user/lmms_eval/bin/pip install --upgrade pip setuptools wheel
RUN set -eux; \
    release_tag="${GIGAVISION_RELEASE_TAG}"; \
    printf '%s\n' "${release_tag}" | grep -Eq '^v[0-9]+\.[0-9]+\.[0-9]+\+gv\.[0-9]+\.[0-9]+\.[0-9]+$' \
        || { echo "Invalid GIGAVISION_RELEASE_TAG: ${release_tag}" >&2; exit 1; }; \
    lmms_eval_version="${release_tag#v}"; \
    gigavision_eval_version="${release_tag##*+gv.}"; \
    /home/user/lmms_eval/bin/python -m pip install \
        --extra-index-url https://gitlab.ai.cloud.ru/api/v4/projects/1859/packages/pypi/simple \
        "lmms_eval[transformers]==${lmms_eval_version}" \
        "gigavision-eval[integrations]==${gigavision_eval_version}" \
    && /home/user/lmms_eval/bin/python -c "import lmms_eval; import transformers; import gigavision_eval.cli.main; import shooters.models.obm_proxy"
RUN /home/user/lmms_eval/bin/python -c "import lmms_eval; print(lmms_eval.__version__)" \
    > /home/user/lmms_eval_version.txt
RUN cat /home/user/lmms_eval_version.txt


FROM os_base AS tau2_bundle

USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
USER user

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/user/.local/bin:$PATH"

COPY --chown=user:user ./tau2-bench /home/user/tau2-bench
RUN cd /home/user/tau2-bench && \
    uv venv /home/user/tau2_eval --python 3.12 && \
    VIRTUAL_ENV=/home/user/tau2_eval uv pip install -e ".[knowledge]" && \
    rm -rf data/tau2/results outputs results.json


FROM os_base AS forge_bundle

USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
USER user

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/user/.local/bin:$PATH"

COPY --chown=user:user ./contrib/forge_lite /home/user/forge_lite
RUN cd /home/user/forge_lite && \
    uv venv /home/user/forge_lite_eval --python 3.12 && \
    VIRTUAL_ENV=/home/user/forge_lite_eval uv pip install ".[vision,translation]" && \
    cd /home/user && \
    tar --zstd --exclude='.git' -cf forge_lite.tar.zst forge_lite


FROM os_base AS bfcl_bundle

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/user/.local/bin:$PATH"


RUN python3 -m venv bfcl_eval --system-site-packages
COPY --chown=user:user ./contrib/gorilla /home/user/gorilla
RUN cd /home/user/gorilla/berkeley-function-call-leaderboard && \
    /home/user/bfcl_eval/bin/pip install -e .


FROM os_base AS papayametric_bundle

COPY --chown=user:user ./contrib/papayametric /home/user/papayametric
COPY --chown=user:user ./contrib/gigarli/src /home/user/rli_sdk
RUN python3 -m venv /home/user/papayametric_uv \
    && /home/user/papayametric_uv/bin/pip install --no-cache-dir uv==0.5.16 \
    && /home/user/papayametric_uv/bin/uv venv --seed /home/user/papayametric_venv \
    && /home/user/papayametric_uv/bin/uv pip install \
        --python /home/user/papayametric_venv/bin/python \
        --no-cache /home/user/papayametric \
    && mkdir -p /home/user/papayametric_uv_envs \
    && ln -s /home/user/papayametric_uv_envs /home/user/papayametric/.uv_envs
RUN --mount=type=secret,id=gigametric_s3_access_key,uid=1000,gid=1000,mode=0400,required=true \
    --mount=type=secret,id=gigametric_s3_secret_key,uid=1000,gid=1000,mode=0400,required=true \
    --mount=type=secret,id=gigametric_s3_bucket_name,uid=1000,gid=1000,mode=0400,required=true \
    --mount=type=secret,id=gigametric_s3_endpoint_url,uid=1000,gid=1000,mode=0400,required=true \
    --mount=type=secret,id=gigametric_s3_region_name,uid=1000,gid=1000,mode=0400,required=true \
    bash -lc ' \
        export S3_ACCESS_KEY="$(cat /run/secrets/gigametric_s3_access_key)" && \
        export S3_SECRET_KEY="$(cat /run/secrets/gigametric_s3_secret_key)" && \
        export S3_BUCKET_NAME="$(cat /run/secrets/gigametric_s3_bucket_name)" && \
        export S3_ENDPOINT_URL="$(cat /run/secrets/gigametric_s3_endpoint_url)" && \
        export S3_REGION_NAME="$(cat /run/secrets/gigametric_s3_region_name)" && \
        cd /home/user/papayametric && \
        PATH="/home/user/papayametric_uv/bin:${PATH}" \
        UV_HTTP_TIMEOUT=300 \
        UV_PYTHON_INSTALL_DIR=/home/user/papayametric_uv_envs/.python \
        UV_LINK_MODE=copy \
        DEPS_FAST_INSTALL=1 \
            /home/user/papayametric_venv/bin/python prepare_offline.py'
RUN cd /home/user/papayametric && \
    PYTHONPATH=/home/user/rli_sdk \
    /home/user/papayametric_uv_envs/toolathlon/bin/python \
    -c "from gigarli.client import RLIClient; from src.benchmarks.toolathlon.rli_env.rli_runner import run_rli_attempt"
RUN mkdir -p /home/user/papayametric_uv_envs_heavy/papayametric_uv_envs \
    && for v in sber-esg-agent gpqa weather_reports bfcl-v4; do \
        if [ -e "/home/user/papayametric_uv_envs/$v" ]; then \
            mv "/home/user/papayametric_uv_envs/$v" \
               "/home/user/papayametric_uv_envs_heavy/papayametric_uv_envs/"; \
        fi; \
    done
RUN cd /home/user \
    && tar --zstd -cf papayametric.tar.zst papayametric rli_sdk .cache/gigametric \
    && rm -rf papayametric rli_sdk .cache/gigametric


FROM os_base AS planbench_bundle

RUN python3 -m venv planbench_eval --system-site-packages
COPY --chown=user:user ./plan-bench-pam /home/user/planbench
RUN /home/user/planbench_eval/bin/pip install -r /home/user/planbench/requirements.txt


FROM os_base AS harness_bench_task_index

RUN python3 -m venv /home/user/harness_bench \
    && /home/user/harness_bench/bin/pip install PyYAML==6.0.3
COPY --chown=user:user ./harness-bench-016 /home/user/harness_bench_fast_src
RUN PYTHONPATH=/home/user/harness_bench_fast_src /home/user/harness_bench/bin/python -c \
    'import json; from harness_bench.tasks import ALL_TASKS; from harness_bench.runner import DEFAULT_TASK_TIMEOUT_SECONDS; print(json.dumps([{"id": task.id, "prompt": task.prompt, "timeout_sec": DEFAULT_TASK_TIMEOUT_SECONDS} for task in ALL_TASKS], ensure_ascii=False))' \
    > /home/user/harness-bench-task-index.json

COPY --chown=user:user ./harness-bench-017 /home/user/harness-bench-017-src
COPY --chown=user:user ./harness-bench-018 /home/user/harness-bench-018-src
RUN set -e; for version in 017 018; do \
        PYTHONPATH="/home/user/harness-bench-${version}-src" \
            /home/user/harness_bench/bin/python -c \
            'import json; from harness_bench.tasks import ALL_TASKS; from harness_bench.runner import DEFAULT_TASK_TIMEOUT_SECONDS; from harness_bench.versioning import validate_task_set_metadata; assert not validate_task_set_metadata(ALL_TASKS); print(json.dumps([{ "id": task.id, "prompt": task.prompt, "timeout_sec": max(DEFAULT_TASK_TIMEOUT_SECONDS, task.min_timeout_seconds or 0) } for task in ALL_TASKS], ensure_ascii=False))' \
            > "/home/user/harness-bench-${version}-task-index.json"; \
    done


FROM os_base AS audio_bundle

COPY --chown=user:user --chmod=777 ./audioaug_src /home/user/audioaug_src

RUN cd /home/user/audioaug_src && \
    cmake -B./build -DCMAKE_BUILD_TYPE:STRING=Release -Wno-dev && \
    cmake --build ./build --config Release --target all -j && \
    pip wheel . --no-deps -w /home/user/audioaug_wheel


FROM os_base AS node_runner_bundle

RUN python3 -m venv convertation --system-site-packages
COPY convertation_requirements.txt /tmp/convertation_requirements.txt
RUN /home/user/convertation/bin/pip install -r /tmp/convertation_requirements.txt

COPY --from=audio_bundle /home/user/audioaug_wheel /tmp/audioaug_wheel
RUN /home/user/convertation/bin/pip install /tmp/audioaug_wheel/*.whl

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt


FROM os_base AS terminal_bench_bundle

ENV TERMINAL_BENCH_VENV=/home/user/terminal_bench_env

RUN python3 -m venv /home/user/terminal_bench_env --system-site-packages

RUN /home/user/terminal_bench_env/bin/pip install --upgrade \
    pip \
    setuptools \
    wheel

RUN /home/user/terminal_bench_env/bin/pip install \
    harbor==0.8.0 \
    mini-swe-agent==2.3.0 \
    litellm==1.83.14 \
    orjson==3.11.6 \
    requests \
    httpx \
    pydantic

COPY --chown=user:user ./terminal-env-gigarli/src/gigarli \
    ${TERMINAL_BENCH_VENV}/gigarli

COPY --chown=user:user ./harbor-rli-terminal-data /home/user/terminal-bench-2


FROM os_base AS pier_bundle

ENV PIER_RLI_VENV=/home/user/pier_rli_env

RUN python3 -m venv "$PIER_RLI_VENV"

RUN ${PIER_RLI_VENV}/bin/pip install --upgrade \
    pip \
    setuptools \
    wheel

RUN ${PIER_RLI_VENV}/bin/pip install \
    datacurve-pier==0.3.1 \
    litellm==1.100.1 \
    orjson==3.11.6 \
    httpx==0.28.1 \
    pydantic==2.13.5

COPY --chown=user:user ./pier-gigarli/src/gigarli \
    ${PIER_RLI_VENV}/gigarli
COPY --chown=user:user ./deep-swe/tasks \
    ${PIER_RLI_VENV}/datasets/deepswe
COPY --chown=user:user ./pier-rli ${PIER_RLI_VENV}/pier_rli
COPY --chown=user:user ./pier-rli/pier_rli_version.txt \
    /home/user/pier_rli/pier_rli_version.txt


# section appworld
FROM os_base AS appworld_bundle

ARG APPWORLD_DATA_URL=https://s3.us-west-2.amazonaws.com/appworld.dev/data-0.2.0.bundle
ARG APPWORLD_DATA_SHA256=c9299e6cafe92bce4592a3c117c047c973d1554a667c21dd81537e78ab2f532e

RUN python3 -m venv /home/user/appworld_env --system-site-packages && \
    /home/user/appworld_env/bin/pip install --upgrade pip setuptools wheel

COPY --chown=user:user ./appworld-bench /home/user/appworld

RUN /home/user/appworld_env/bin/pip install -e /home/user/appworld && \
    /home/user/appworld_env/bin/pip install -e "/home/user/appworld/experiments[simplified]"

RUN cd /home/user/appworld && \
    /home/user/appworld_env/bin/appworld install --repo

RUN curl -fsSL "$APPWORLD_DATA_URL" -o /tmp/appworld_data.bundle && \
    echo "$APPWORLD_DATA_SHA256  /tmp/appworld_data.bundle" | sha256sum -c - && \
    cd /home/user/appworld && \
    /home/user/appworld_env/bin/python -c "from appworld.common.constants import PASSWORD, SALT; from appworld.common.crypto import unpack_bundle; unpack_bundle(bundle_file_path='/tmp/appworld_data.bundle', base_directory='/home/user/appworld', password=PASSWORD, salt=SALT)" && \
    rm /tmp/appworld_data.bundle

RUN printf '%s+data.%s\n' \
        "$(cat /home/user/appworld/pam_shooter_version.txt)" \
        "$APPWORLD_DATA_SHA256" \
        > /home/user/appworld_version.txt


FROM os_base AS swe_bench_dataset_bundle

COPY ./harbor-rli-swe-data /home/user/swe-bench-verified


FROM os_base AS claweval_bundle

# setup claweval
ARG CLAWEVAL_FIXTURES_URL=https://huggingface.co/datasets/claw-eval/Claw-Eval/resolve/main/data/fixtures.tar.gz
ARG CLAWEVAL_FIXTURES_SHA256=01c349b8eb3ae8bcd741578d6a7f94ba3f76e0aad74bd98233e02e629e65ff5b

COPY --chown=user:user ./claweval-bench /home/user/claweval/upstream
COPY --chown=user:user ./claweval-bench/pam_shooter_version.txt /home/user/claweval/upstream_commit.txt

RUN curl -fsSL "$CLAWEVAL_FIXTURES_URL" -o /tmp/claweval_fixtures.tar.gz && \
    echo "$CLAWEVAL_FIXTURES_SHA256  /tmp/claweval_fixtures.tar.gz" | sha256sum -c - && \
    tar -xzf /tmp/claweval_fixtures.tar.gz -C /home/user/claweval/upstream/tasks && \
    rm /tmp/claweval_fixtures.tar.gz

RUN python3 -m venv /home/user/claweval_env && \
    /home/user/claweval_env/bin/pip install --upgrade pip setuptools wheel && \
    /home/user/claweval_env/bin/pip install -e "/home/user/claweval/upstream[mock]" && \
    /home/user/claweval_env/bin/python -c "import claw_eval"

COPY --chown=user:user ./claweval-bench/runner/run_claweval.py /home/user/claweval/runner/run_claweval.py


FROM os_base AS harbor_rli_bundle

ENV HARBOR_RLI_VENV=/home/user/harbor_rli_env

RUN python3 -m venv "$HARBOR_RLI_VENV"

RUN ${HARBOR_RLI_VENV}/bin/pip install --upgrade \
    pip \
    setuptools \
    wheel

RUN ${HARBOR_RLI_VENV}/bin/pip install \
    harbor==0.8.0 \
    mini-swe-agent==2.3.0 \
    litellm==1.83.14 \
    orjson==3.11.6 \
    requests==2.34.2 \
    httpx==0.28.1

COPY --chown=user:user ./harbor-gigarli/src/gigarli \
    ${HARBOR_RLI_VENV}/gigarli
COPY --chown=user:user ./harbor-rli-swe-data/datasets/swebench-verified \
    ${HARBOR_RLI_VENV}/datasets/swe-bench-verified
COPY --chown=user:user ./harbor-rli-swe-data/datasets/swebench_multilingual \
    ${HARBOR_RLI_VENV}/datasets/swe-bench-multilingual
COPY --chown=user:user ./harbor-rli-terminal-data \
    ${HARBOR_RLI_VENV}/datasets/terminal-bench-2.0
COPY --chown=user:user ./harbor-rli-terminal-21-data/tasks \
    ${HARBOR_RLI_VENV}/datasets/terminal-bench-2.1
COPY --chown=user:user ./harbor-rli-cowork-data/tasks \
    ${HARBOR_RLI_VENV}/datasets/cowork-bench
COPY --from=harness_bench_task_index --chown=user:user \
    /home/user/harness-bench-task-index.json \
    ${HARBOR_RLI_VENV}/datasets/harness-bench/task-index.json
COPY --from=harness_bench_task_index --chown=user:user \
    /home/user/harness-bench-017-task-index.json \
    ${HARBOR_RLI_VENV}/datasets/harness-bench-0.17.0/task-index.json
COPY --from=harness_bench_task_index --chown=user:user \
    /home/user/harness-bench-018-task-index.json \
    ${HARBOR_RLI_VENV}/datasets/harness-bench-0.18.0/task-index.json
COPY --chown=user:user ./harbor-rli ${HARBOR_RLI_VENV}/harbor_rli
COPY --chown=user:user ./harbor-rli/harbor_rli_version.txt \
    /home/user/harbor_rli/harbor_rli_version.txt


FROM os_base AS skillsbench_bundle

COPY --chown=user:user ./skillsbench-bench/skillsbench \
    /home/user/skillsbench/skillsbench
COPY --chown=user:user ./skillsbench-bench/tasks \
    /home/user/skillsbench/tasks
COPY --chown=user:user ./skillsbench-bench/UPSTREAM_COMMIT \
    /home/user/skillsbench/UPSTREAM_COMMIT
COPY --chown=user:user ./skillsbench-bench/skillsbench_version.txt \
    /home/user/skillsbench/skillsbench_version.txt


FROM os_base AS nl2repo_bundle

COPY --chown=user:user ./nl2repo-bench/nl2repo \
    /home/user/nl2repo/nl2repo
COPY --chown=user:user ./nl2repo-bench/data \
    /home/user/nl2repo/data
COPY --chown=user:user ./nl2repo-bench/UPSTREAM_COMMIT \
    /home/user/nl2repo/UPSTREAM_COMMIT
COPY --chown=user:user ./nl2repo-bench/nl2repo_version.txt \
    /home/user/nl2repo/nl2repo_version.txt


FROM os_base AS nccl_bundle

ARG NCCL_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100"

COPY --chown=user:user ./nccl-pretrain /home/user/nccl-src

RUN cd /home/user/nccl-src && \
    make -j"$(nproc)" src.build NVCC_GENCODE="${NCCL_GENCODE}" && \
    mkdir -p /home/user/nccl/lib && \
    cp -a build/lib/libnccl.so* /home/user/nccl/lib/ && \
    cp -a nccl_version.txt /home/user/nccl/nccl_version.txt && \
    rm -rf /home/user/nccl-src


FROM os_base AS sglang_opensource_bundle

# setup sglang
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/user/.local/bin:${PATH}" \
    VIRTUAL_ENV="/home/user/sglang"

COPY --chown=user:user ./build_utils/cuda-cu13-exclude-list.txt /tmp/excludes-cu13.txt

RUN uv venv --seed "$VIRTUAL_ENV" && \
    uv pip install --excludes /tmp/excludes-cu13.txt "sglang[all]==0.5.14"

RUN uv pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu129 \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

COPY --from=nccl_bundle --chown=user:user /home/user/nccl /home/user/nccl
RUN NCCL_LIB_DIR="$("$VIRTUAL_ENV/bin/python" -c 'import nvidia.nccl; print(nvidia.nccl.__path__[0] + "/lib")')" && \
    test -f "$NCCL_LIB_DIR/libnccl.so.2" && \
    cp -a /home/user/nccl/lib/libnccl.so* "$NCCL_LIB_DIR/" && \
    echo "NCCL from $(cat /home/user/nccl/nccl_version.txt)" && \
    "$VIRTUAL_ENV/bin/python" -c "import ctypes, sys; v = ctypes.c_int(); ctypes.CDLL(sys.argv[1]).ncclGetVersion(ctypes.byref(v)); print('NCCL', v.value)" \
        "$NCCL_LIB_DIR/libnccl.so.2"

RUN uv pip install --force-reinstall --no-deps \
        "https://github.com/sgl-project/whl/releases/download/v0.4.4/sglang_kernel-0.4.4+cu129-cp310-abi3-manylinux2014_$(uname -m).whl" && \
    uv pip install --force-reinstall \
        "https://github.com/sgl-project/whl/releases/download/v0.1.3/sgl_deep_gemm-0.1.3+cu129-py3-none-manylinux2014_$(uname -m).whl"

RUN uv pip install --force-reinstall cuda-python==12.9.0 cuda-bindings==12.9.4

RUN uv pip install --force-reinstall --no-deps \
        flash-attn-4==4.0.0b15 torch-c-dlpack-ext==0.1.5 \
        tilelang==0.1.8 z3-solver==4.15.4.0 ml-dtypes==0.5.4 \
        quack-kernels==0.5.0 cuda-tile==1.3.0 \
        nvidia-cutlass-dsl==4.5.2 nvidia-cutlass-dsl-libs-base==4.5.2

ARG DEEPEP_COMMIT=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
ARG DEEPEP_BUILD_JOBS=3
RUN curl --retry 3 --retry-delay 2 -fsSL -o /tmp/deepep.tar.gz \
        "https://github.com/deepseek-ai/DeepEP/archive/${DEEPEP_COMMIT}.tar.gz" && \
    mkdir -p /tmp/DeepEP /tmp/deepep_wheels && \
    tar -xzf /tmp/deepep.tar.gz --strip-components=1 -C /tmp/DeepEP && \
    cd /tmp/DeepEP && \
    sed -i 's/#define NUM_CPU_TIMEOUT_SECS 100/#define NUM_CPU_TIMEOUT_SECS 1000/' csrc/kernels/configs.cuh && \
    sed -i 's/#define NUM_TIMEOUT_CYCLES 200000000000ull/#define NUM_TIMEOUT_CYCLES 2000000000000ull/' csrc/kernels/configs.cuh && \
    uv pip install wheel && \
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=${DEEPEP_BUILD_JOBS} \
        "$VIRTUAL_ENV/bin/python" setup.py bdist_wheel -d /tmp/deepep_wheels && \
    uv pip install /tmp/deepep_wheels/*.whl && \
    rm -rf /tmp/DeepEP /tmp/deepep.tar.gz /tmp/deepep_wheels

RUN uv pip install sglang-router==0.3.2 mooncake-transfer-engine==0.3.13.post1 nixl==1.4.1

RUN "$VIRTUAL_ENV/bin/python" -c "import sglang" && \
    "$VIRTUAL_ENV/bin/python" -c "import deep_ep" && \
    uv pip list && \
    "$VIRTUAL_ENV/bin/python" -c "import torch; print('TORCH', torch.__version__, torch.version.cuda)"


FROM os_base AS sglang_router_bundle

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git libssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*
USER user

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN curl --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal --default-toolchain 1.90
ENV PATH="/home/user/.cargo/bin:/home/user/.local/bin:${PATH}"

ARG PROTOC_VERSION=25.3
RUN curl --retry 3 --retry-delay 2 -fsSL -o /tmp/protoc.zip \
        "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-$(uname -m).zip" && \
    python3 -m zipfile -e /tmp/protoc.zip /home/user/protoc && \
    chmod +x /home/user/protoc/bin/protoc && \
    rm /tmp/protoc.zip && \
    /home/user/protoc/bin/protoc --version
ENV PROTOC="/home/user/protoc/bin/protoc" \
    PROTOC_INCLUDE="/home/user/protoc/include"

COPY --chown=user:user ./contrib/sglang_pretrain/sgl-model-gateway /home/user/sgl-model-gateway

ARG SGLANG_ROUTER_BUILD_JOBS=4
RUN uv venv --seed /home/user/router_build && \
    VIRTUAL_ENV=/home/user/router_build uv pip install "maturin>=1,<2" && \
    cd /home/user/sgl-model-gateway/bindings/python && \
    CARGO_BUILD_JOBS=${SGLANG_ROUTER_BUILD_JOBS} \
        /home/user/router_build/bin/maturin build --release --compatibility linux \
            --interpreter /home/user/router_build/bin/python \
            --out /home/user/router_wheels && \
    ls -la /home/user/router_wheels && \
    rm -rf /home/user/sgl-model-gateway/target /home/user/.cargo/registry


FROM os_base AS sglang_custom_bundle

# setup custom sglang
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/user/.local/bin:${PATH}" \
    VIRTUAL_ENV="/home/user/sglang"

COPY --chown=user:user ./build_utils/cuda-cu13-exclude-list.txt /tmp/excludes-cu13.txt
COPY --chown=user:user ./contrib/sglang_pretrain /home/user/tmp_sglang

RUN uv venv --seed "$VIRTUAL_ENV" && \
    uv pip install --excludes /tmp/excludes-cu13.txt -r /home/user/tmp_sglang/python/pyproject.toml

RUN uv pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu129 \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

COPY --from=nccl_bundle --chown=user:user /home/user/nccl /home/user/nccl
RUN NCCL_LIB_DIR="$("$VIRTUAL_ENV/bin/python" -c 'import nvidia.nccl; print(nvidia.nccl.__path__[0] + "/lib")')" && \
    test -f "$NCCL_LIB_DIR/libnccl.so.2" && \
    cp -a /home/user/nccl/lib/libnccl.so* "$NCCL_LIB_DIR/" && \
    echo "NCCL from $(cat /home/user/nccl/nccl_version.txt)" && \
    "$VIRTUAL_ENV/bin/python" -c "import ctypes, sys; v = ctypes.c_int(); ctypes.CDLL(sys.argv[1]).ncclGetVersion(ctypes.byref(v)); print('NCCL', v.value)" \
        "$NCCL_LIB_DIR/libnccl.so.2"

RUN uv pip install --force-reinstall --no-deps \
        "https://github.com/sgl-project/whl/releases/download/v0.4.4/sglang_kernel-0.4.4+cu129-cp310-abi3-manylinux2014_$(uname -m).whl" && \
    uv pip install --force-reinstall \
        "https://github.com/sgl-project/whl/releases/download/v0.1.3/sgl_deep_gemm-0.1.3+cu129-py3-none-manylinux2014_$(uname -m).whl"

RUN uv pip install --force-reinstall cuda-python==12.9.0 cuda-bindings==12.9.4

RUN uv pip install --force-reinstall --no-deps \
        flash-attn-4==4.0.0b15 torch-c-dlpack-ext==0.1.5 \
        tilelang==0.1.8 z3-solver==4.15.4.0 ml-dtypes==0.5.4 \
        quack-kernels==0.5.0 cuda-tile==1.3.0 \
        nvidia-cutlass-dsl==4.5.2 nvidia-cutlass-dsl-libs-base==4.5.2

ARG DEEPEP_COMMIT=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
ARG DEEPEP_BUILD_JOBS=3
RUN curl --retry 3 --retry-delay 2 -fsSL -o /tmp/deepep.tar.gz \
        "https://github.com/deepseek-ai/DeepEP/archive/${DEEPEP_COMMIT}.tar.gz" && \
    mkdir -p /tmp/DeepEP /tmp/deepep_wheels && \
    tar -xzf /tmp/deepep.tar.gz --strip-components=1 -C /tmp/DeepEP && \
    cd /tmp/DeepEP && \
    sed -i 's/#define NUM_CPU_TIMEOUT_SECS 100/#define NUM_CPU_TIMEOUT_SECS 1000/' csrc/kernels/configs.cuh && \
    sed -i 's/#define NUM_TIMEOUT_CYCLES 200000000000ull/#define NUM_TIMEOUT_CYCLES 2000000000000ull/' csrc/kernels/configs.cuh && \
    uv pip install wheel && \
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=${DEEPEP_BUILD_JOBS} \
        "$VIRTUAL_ENV/bin/python" setup.py bdist_wheel -d /tmp/deepep_wheels && \
    uv pip install /tmp/deepep_wheels/*.whl && \
    rm -rf /tmp/DeepEP /tmp/deepep.tar.gz /tmp/deepep_wheels

COPY --from=sglang_router_bundle --chown=user:user /home/user/router_wheels /tmp/router_wheels
RUN uv pip install /tmp/router_wheels/sglang_router-*.whl \
        mooncake-transfer-engine==0.3.13.post1 nixl==1.4.1 && \
    rm -rf /tmp/router_wheels && \
    "$VIRTUAL_ENV/bin/python" -c "import sglang_router.sglang_router_rs as rs; print('ROUTER', rs.__file__)"

RUN uv pip install av==17.1.0 decord==0.6.0

COPY --from=audio_bundle /home/user/audioaug_wheel /tmp/audioaug_wheel
RUN uv pip install /tmp/audioaug_wheel/*.whl

RUN PYTHONPATH=/home/user/tmp_sglang/python "$VIRTUAL_ENV/bin/python" -P -c "import sglang" && \
    "$VIRTUAL_ENV/bin/python" -c "import deep_ep" && \
    uv pip list && \
    "$VIRTUAL_ENV/bin/python" -c "import torch; print('TORCH', torch.__version__, torch.version.cuda)" && \
    rm -rf /home/user/tmp_sglang && \
    ! "$VIRTUAL_ENV/bin/python" -P -c "import sglang"


FROM os_base AS final_common

ENV HF_ALLOW_CODE_EVAL=1
ENV NLTK_DATA=/home/user/nltk_data

# lm_eval artifacts
COPY --from=lm_eval_bundle --chown=user:user /home/user/lm_eval /home/user/lm_eval
COPY --from=lm_eval_bundle --chown=user:user /home/user/lm_eval_version.txt /home/user/lm_eval_version.txt

# mera artifacts
COPY --from=mera_bundle --chown=user:user /home/user/mera_eval /home/user/mera_eval
COPY --from=mera_bundle --chown=user:user /home/user/mera_tasks /home/user/mera_tasks
COPY --from=mera_bundle --chown=user:user /home/user/mera_scripts /home/user/mera_scripts
COPY --from=mera_bundle --chown=user:user /home/user/mera_version.txt /home/user/mera_version.txt
COPY --from=mera_bundle --chown=user:user /home/user/mera_system_instruction.txt /home/user/mera_system_instruction.txt

# lmms_eval artifacts
COPY --from=lmms_eval_bundle --chown=user:user /home/user/lmms_eval /home/user/lmms_eval
COPY --from=lmms_eval_bundle --chown=user:user /home/user/lmms_eval_version.txt /home/user/lmms_eval_version.txt

# warmup datasets cache (huggingface) for lm_eval + mera
COPY --from=lm_eval_bundle --chown=user:user /home/user/hf_hub_chunk_0/ /home/user/.cache/huggingface/hub/
COPY --from=lm_eval_bundle --chown=user:user /home/user/hf_hub_chunk_1/ /home/user/.cache/huggingface/hub/
COPY --from=lm_eval_bundle --chown=user:user /home/user/hf_hub_chunk_2/ /home/user/.cache/huggingface/hub/
COPY --from=lm_eval_bundle --chown=user:user /home/user/hf_hub_chunk_3/ /home/user/.cache/huggingface/hub/
COPY --from=lm_eval_bundle --chown=user:user /home/user/hf_hub_chunk_4/ /home/user/.cache/huggingface/hub/
COPY --from=lm_eval_bundle --chown=user:user /home/user/hf_hub_chunk_5/ /home/user/.cache/huggingface/hub/
COPY --from=lm_eval_bundle --chown=user:user /home/user/.cache/huggingface /home/user/.cache/huggingface
COPY --from=mera_bundle --chown=user:user /home/user/.cache/huggingface /home/user/.cache/huggingface


# obm artifacts (obm.tar.zst содержит one-button-metrics и new_obm, данные общие через хардлинки)
COPY --from=obm_bundle --chown=user:user /home/user/obm_python /home/user/obm_python
COPY --from=obm_bundle --chown=user:user /home/user/obm_env /home/user/obm_env
COPY --from=obm_bundle --chown=user:user /home/user/new_obm_env /home/user/new_obm_env
COPY --from=obm_bundle --chown=user:user /home/user/obm_version.txt /home/user/obm_version.txt
COPY --from=obm_bundle --chown=user:user /home/user/new_obm_version.txt /home/user/new_obm_version.txt
COPY --from=obm_bundle --chown=user:user /home/user/obm.tar.zst /home/user/obm.tar.zst
COPY --from=obm_bundle --chown=user:user /home/user/nltk_data /home/user/nltk_data

# dredd artifacts (venv + unpacked data for DREDD_DATA_ROOT)
COPY --from=dredd_bundle --chown=user:user /home/user/dredd_env /home/user/dredd_env
COPY --from=dredd_bundle --chown=user:user /home/user/data /home/user/data
COPY --from=dredd_bundle --chown=user:user /home/user/dredd_version.txt /home/user/dredd_version.txt
ENV DREDD_DATA_ROOT="/home/user/data"

COPY --from=safety_unified_bundle --chown=user:user /home/user/safety_unified /home/user/safety_unified

# tau2 artifacts
COPY --from=tau2_bundle --chown=user:user /home/user/tau2_eval /home/user/tau2_eval
COPY --from=tau2_bundle --chown=user:user /home/user/tau2-bench /home/user/tau2-bench

# forge_lite artifacts
COPY --chown=user:user ./forge_lite_version.txt /home/user/forge_lite_version.txt
COPY --from=forge_bundle --chown=user:user /home/user/forge_lite_eval /home/user/forge_lite_eval
COPY --from=forge_bundle --chown=user:user /home/user/forge_lite.tar.zst /home/user/forge_lite.tar.zst

# bfcl artifacts
COPY --from=bfcl_bundle --chown=user:user /home/user/bfcl_eval /home/user/bfcl_eval

# planbench artifacts
COPY --from=planbench_bundle --chown=user:user /home/user/planbench_eval /home/user/planbench_eval
COPY --from=planbench_bundle --chown=user:user /home/user/planbench /home/user/planbench

# node_runner dependencies and convertation artifacts
COPY --from=node_runner_bundle --chown=user:user /home/user/convertation /home/user/convertation
COPY --from=node_runner_bundle --chown=user:user /home/user/.local /home/user/.local

# papayametric artifacts
COPY --from=papayametric_bundle --chown=user:user /home/user/papayametric.tar.zst /home/user/papayametric.tar.zst
COPY --from=papayametric_bundle --chown=user:user /home/user/papayametric_venv /home/user/papayametric_venv
COPY --from=papayametric_bundle --chown=user:user /home/user/papayametric_uv /home/user/papayametric_uv
COPY --from=papayametric_bundle --chown=user:user /home/user/papayametric_uv_envs /home/user/papayametric_uv_envs
COPY --from=papayametric_bundle --chown=user:user /home/user/papayametric_uv_envs_heavy/papayametric_uv_envs /home/user/papayametric_uv_envs
COPY --from=papayametric_bundle --chown=user:user /home/user/.cache/huggingface /home/user/.cache/huggingface

# terminal_bench artifacts
COPY --from=terminal_bench_bundle --chown=user:user /home/user/terminal_bench_env /home/user/terminal_bench_env
COPY --from=terminal_bench_bundle --chown=user:user /home/user/terminal-bench-2 /home/user/terminal-bench-2
COPY --chown=user:user ./terminal_bench_version.txt /home/user/terminal_bench_version.txt

# harbor_rli artifacts
COPY --from=harbor_rli_bundle --chown=user:user /home/user/harbor_rli_env /home/user/harbor_rli_env
COPY --from=harbor_rli_bundle --chown=user:user /home/user/harbor_rli /home/user/harbor_rli

# pier_rli artifacts
COPY --from=pier_bundle --chown=user:user /home/user/pier_rli_env /home/user/pier_rli_env
COPY --from=pier_bundle --chown=user:user /home/user/pier_rli /home/user/pier_rli

# appworld artifacts
COPY --from=appworld_bundle --chown=user:user /home/user/appworld /home/user/appworld
COPY --from=appworld_bundle --chown=user:user /home/user/appworld_env /home/user/appworld_env
COPY --from=appworld_bundle --chown=user:user /home/user/appworld_version.txt /home/user/appworld_version.txt

# swe_bench dataset artifacts
COPY --from=swe_bench_dataset_bundle --chown=user:user /home/user/swe-bench-verified /home/user/swe-bench-verified
COPY --chown=user:user ./bfcl_version.txt /home/user/bfcl_version.txt
COPY --chown=user:user ./swe_bench_version.txt /home/user/swe_bench_version.txt

# claweval artifacts
COPY --from=claweval_bundle --chown=user:user /home/user/claweval /home/user/claweval
COPY --from=claweval_bundle --chown=user:user /home/user/claweval_env /home/user/claweval_env

# skillsbench artifacts
COPY --from=skillsbench_bundle --chown=user:user /home/user/skillsbench /home/user/skillsbench

# nl2repo artifacts
COPY --from=nl2repo_bundle --chown=user:user /home/user/nl2repo /home/user/nl2repo

# aa_lcr artifacts
COPY --from=aa_lcr_bundle --chown=user:user /home/user/aa_lcr /home/user/aa_lcr

USER root
RUN apt-get update && apt-get install -y --no-install-recommends rclone \
    && rm -rf /var/lib/apt/lists/*
USER user


FROM final_common AS opensource

COPY --from=sglang_opensource_bundle --chown=user:user /home/user/sglang /home/user/sglang
WORKDIR /home/user/node_runner


FROM final_common AS custom

COPY --from=sglang_custom_bundle --chown=user:user /home/user/sglang /home/user/sglang
WORKDIR /home/user/node_runner


FROM opensource AS default
