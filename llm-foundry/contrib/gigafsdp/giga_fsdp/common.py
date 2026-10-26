from enum import Enum

EMA_SUFFIX = "_ema"

CHECKPOINT_VERSION = 2

class ParamBufferPrefix(Enum):
    TRAINABLE = "main"
    FREEZED = "freezed"
    SEPARATE = "layernorm"
    SEPARATE_FREEZED = "layernormfreezed"

    def __str__(self):
        return str(self.value)

class MetaInfoKeys:
    OPTIMIZER_STATE_IDX_TO_LAYER_NAME = "optimizer_state_idx_to_layer_name"
    SEPARATE_MODULE_NAMES = "separate_module_names"
    STATE_DICT_KEYS = "state_dict_keys"
    LAYER_NAME_TO_META_INFO_IDX = "layer_name_to_meta_info_idx"
    FREEZED_LAYER_NAME_TO_META_INFO_IDX = "freezed_layer_name_to_meta_info_idx"
    PARAM_NAME_TO_LAYER_NAME = "param_name_to_layer_name"
    PARAM_NAME_TO_FSDP_GROUP = "param_name_to_fsdp_group"
    PARAMS_MAPPING = "params_mapping" # for legacy optimizer state dict
    CHECKPOINT_VERSION_FIELD = "checkpoint_version"

COLUMN_SPLIT_CLASS_NAMES = [ "ColumnParallelLinear", "VocabParallelEmbedding"]

ROW_SPLIT_CLASS_NAMES = [ "RowParallelLinear", "EmbeddingParallelEmbedding"]