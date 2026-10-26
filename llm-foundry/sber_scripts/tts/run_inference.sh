# make dataset
SPEAKER="Ermilova"
FILE_NAME=logs_dev_qa
FILE_PATH=inference_tests/tts/${FILE_NAME}
cat ${FILE_PATH}.json | sed 's|\[speaker\]|'$SPEAKER'|' > ${FILE_PATH}_${SPEAKER}.json
DATA_JSON_PATH=${FILE_PATH}_${SPEAKER}.json
echo $DATA_JSON_PATH


# ENVIRONMENT
export MODEL_CFG=logs/checkpoints/tts-9b-qa-lora-dialogs-use-speech/artifacts/llm_foundry_config.yaml  # path to model config (see training artifacts)
export MODEL_HF_CKPT_PATH=logs/checkpoints/tts-9b-qa-lora-dialogs-use-speech/hf_ckpt_30k # path to hf checkpoint
export INFERENCE_CFG='sber_configs/tts/inference/config.yaml' # maybe custom cfg
export SAVE_PATH=./dump/${FILE_NAME}_${SPEAKER}


# additional
export CHATMLV2_CONFIGS_PATH='/home/jovyan/shares/SR008.fs2/emlagutin/data/giga_tts_datasets/tts_norm_val/converted_chatmlv2_9b/configs' # chatml v2 config
export AUDIO_CODE_BASE_PATH='.' # options for metrics


# script start
CODE_PATH='.'
export NGPUS_PER_NODE=8
export PPORT=6002
export NNODES=1
export PYTHONPATH=$CODE_PATH:$CODE_PATH/contrib/composer:$CODE_PATH/contrib/streaming:$CODE_PATH/contrib/gigafsdp:$CODE_PATH/contrib/sft-data-loaders\

python -m torch.distributed.launch \
    --nproc_per_node gpu \
    --nnodes 1 \
    --master_port $PPORT \
    --use_env \
    $CODE_PATH/sber_scripts/tts/inference.py \
        --model-cfg-path $MODEL_CFG \
	--inference-cfg-path $INFERENCE_CFG \
        --model-hf-ckpt-path $MODEL_HF_CKPT_PATH \
        --data-json-path $DATA_JSON_PATH \
        --audio-code-base-path $AUDIO_CODE_BASE_PATH \
        --chatmlv2-configs-path $CHATMLV2_CONFIGS_PATH \
        --save-path $SAVE_PATH
