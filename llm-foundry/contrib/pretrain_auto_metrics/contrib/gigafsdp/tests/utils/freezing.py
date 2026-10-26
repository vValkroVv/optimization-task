import enum


class FreezeStrategy(enum.Enum):
    EMBS_MNORM_HEAD = "only embs, lm_head and model.norm are trainable"
    EMBS_HEAD = "only embs and lm_head are trainable"
    LNS = "only layernorms are trainable"
    WITHOUT_LNS = "everyone trainable without layernorms"
    MIXED = "Some layers are frozen, some layers are not frozen, some layers are frozen without layernorms"
    SOME_LAYERS_ALL_LNS = "Some layers are frozen, all layernorms are frozen"
    PF_MLPS = "only mlps are trainable"
    PF_WITHOUT_MLPS = "everyone trainable without mlps"
    PF_ALL_LNS_OTHER_MIXED = "there are freezed / partially freezed / train layers, all lns trainable"
    PF_ALL_MIXED = "there are freezed / partially freezed / train layers and ln buffers"
    MM_PROJ = "only projectors are trainable"
    MM_TOWERS = "only towers are trainable"
    MM_ALL_WITHOUT_LLM_WITH_LNS = "everyone trainable without llm, but layernorms are trainable"
    INFERENCE = "inference mode"
    FULL_TRAINABLE = "train mode"

def give_flags_to_gradients(model, flag):
    for param in model.parameters():
        param.requires_grad = flag

def get_freeze_dict(model):
    freeze_dict = {}
    for name, param in model.named_parameters():
        freeze_dict[name] = param.requires_grad
    return freeze_dict

def freezing_layer_module_by_name(model, config, module_name):
    if module_name in config:
        layers = config[module_name]
        if layers:
            for i, layer in enumerate(model.model.layers):
                if i in layers:
                    for name, param in layer.named_parameters():
                        if module_name in name:
                            param.requires_grad = param.requires_grad == False 

def freezing(model, config):
    give_flags_to_gradients(model, False)
    if 'mode' in config:
        if config['mode'] == 'inference':
            pass
        elif config['mode'] == 'train':
            give_flags_to_gradients(model, True)

    if 'layers' in config:
        layers = config['layers']
        if layers:
            for i, layer in enumerate(model.model.layers):
                if i in layers:
                    give_flags_to_gradients(layer, True)

    for module_name in [key for key in config.keys() if key not in ['layers', 'mode', 'other']]:
        freezing_layer_module_by_name(model, config, module_name)

    if 'other' in config and config['other']:
        for name, param in model.named_parameters():
            for other in config['other']:
                if other in name:
                    param.requires_grad = param.requires_grad == False 

    return get_freeze_dict(model)

def freeze_model(model, freeze_strategy):
    return freezing(model, freezing_configs[freeze_strategy])

freezing_configs = {
    FreezeStrategy.EMBS_MNORM_HEAD:
        {#1. учатся только эмбеды и голова + model.norm
            'other': ['model.norm', 'model.embed_tokens', 'lm_head']
        },
    FreezeStrategy.EMBS_HEAD:
        {#2. учатся только эмбеды и голова
            'other': ['model.embed_tokens', 'lm_head']
        },
    FreezeStrategy.LNS:
        {#3. учатся только лн 
            'layernorm': range(5),
            'other': ['model.norm']
        },
    FreezeStrategy.WITHOUT_LNS:
        {#4. учится все, кроме лн
            'layers': range(5),
            'layernorm': range(5),
            'other': ['model.embed_tokens', 'lm_head']
        },
    FreezeStrategy.MIXED:
        {#5. часть слоев заморожена полностью, часть разморожена полностью, часть слоев заморожена но у них разморожен лн
            'layers': range(1, 3),
            'layernorm': [4],
            'other': ['model.norm']
        },
    FreezeStrategy.SOME_LAYERS_ALL_LNS:
        {#6. часть слоев заморожена полностью, часть разморожена + заморожен лн
            'layers': range(3, 5),
            'layernorm': range(3, 5),
        },
    FreezeStrategy.PF_MLPS:
        {#7. учится только mlp
            'mlp': range(5)
        },
    FreezeStrategy.PF_WITHOUT_MLPS:
        {#8. учится все, кроме mlp
            'layers': range(5),
            'mlp': range(5),
            'other': ['model.embed_tokens', 'model.norm', 'lm_head']
        },
    FreezeStrategy.PF_ALL_LNS_OTHER_MIXED:
        {#9. учится все ln, слои заморожены / разморожены / частично
            'layernorm': range(5),
            'self_attn': [2, 4],
            'mlp': [3, 4],
            'other': ['model.norm', 'lm_head']
        },
    FreezeStrategy.PF_ALL_MIXED:
        {#10. все частично замороженно
            'layernorm': range(3),
            'self_attn': [2],
            'mlp': [3],
            'layers': [4],
            'other': ['lm_head']
        },
    FreezeStrategy.MM_PROJ:
        {#11. учатся только proj
            'other': ['projector']
        },
    FreezeStrategy.MM_TOWERS:
        {#12. учатся только башни
            'other': ['tower']
        },
    FreezeStrategy.MM_ALL_WITHOUT_LLM_WITH_LNS:
        {#13. учится все, кроме ллм, в ллм учатся ln
            'mode': 'train',
            'other': ['llm', 'layernorm']
        },
    FreezeStrategy.INFERENCE:
        {# инференс мод
            'mode': 'inference',
        },
    FreezeStrategy.FULL_TRAINABLE:
        {# учится все - трейн мод
            'mode': 'train',
        },
}