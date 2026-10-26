import enum
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm
import torch
import torch.nn as nn
from pathlib import Path


CONFIG_PATH = Path(__file__).parent / "config.json"

class Models(enum.Enum):
    TEXT = "llama_like"
    VISION = "Vision modality are enabled"
    TTS = "TTS modality are enabled"
    OMNIMODEL = "All the modalities are enabled"

def prep_data(args, rank):
    all_data = {
        "input_ids": torch.randint(0, 128, (args.batch_size, args.seq_len), dtype=torch.long, device=rank),
        "labels": torch.randint(0, 128, (args.batch_size, args.seq_len), dtype=torch.long, device=rank),
        "vision_ids": torch.randn((args.batch_size, args.seq_len, 16), requires_grad=False, device=rank, dtype=args.param_dtype),
        "audio_ids": torch.randn((args.batch_size, args.seq_len, 16), requires_grad=False, device=rank, dtype=args.param_dtype)
    }
    keys = DATA_FN[args.model_type]
    data = {key:all_data[key] for key in keys}
    return data

DATA_FN = {
    Models.TEXT: ["input_ids", "labels"],
    Models.VISION: ["input_ids", "vision_ids"],
    Models.TTS: ["input_ids"],
    Models.OMNIMODEL: ["input_ids", "audio_ids", "vision_ids"]
}

class VisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = nn.Sequential(nn.Linear(16, 16, bias=False), 
                                            nn.Linear(16, 16, bias=False),
                                            nn.Linear(16, 16, bias=False))
        self.vision_projector = nn.Linear(16, 16, bias=False)

    def forward(self, input_ids):
        return self.vision_projector(self.vision_encoder(input_ids))
    
class AudioTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_encoder = nn.Sequential(nn.Linear(16, 16, bias=False), 
                                           nn.Linear(16, 16, bias=False))
        self.audio_projector = nn.Linear(16, 16, bias=False)

    def forward(self, input_ids):
        return self.audio_projector(self.audio_encoder(input_ids))

class LLMBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Sequential(nn.Linear(16, 16, bias=False), 
                                       nn.Linear(16, 16, bias=False),
                                       nn.Linear(16, 16, bias=False))
        self.layernorm = LlamaRMSNorm(16)
    
    def forward(self, input_ids):
        return self.layernorm(self.self_attn(input_ids))

class TTS(nn.Module):
    def __init__(self):
        super().__init__()
        self.tts_projector = nn.Linear(16, 16, bias=False)
        self.tts_adapter = nn.Sequential(nn.Linear(16, 16, bias=False), 
                                         nn.Linear(16, 16, bias=False),
                                         nn.Linear(16, 16, bias=False))
    def forward(self, input_embs):
        return self.tts_adapter(self.tts_projector(input_embs))

class MultimodalModel(nn.Module):
    def __init__(self, add_vision=False, add_audio=False, add_tts=False):
        super().__init__()
        
        self.add_vision = add_vision
        self.add_audio = add_audio
        self.add_tts = add_tts
        
        self.embed_tokens = nn.Embedding(128, 16)

        if add_vision:
            self.vision_tower = VisionTower()
        if add_audio:
            self.audio_tower = AudioTower()

        self.llm = nn.ModuleList([LLMBlock() for _ in range(5)])
        self.lm_head = nn.Linear(16, 16, bias=False)

        if add_tts:
            self.tts_tower = TTS()
        
    def forward(self, input_ids, vision_ids=None, audio_ids=None):
        embeds = []

        if self.add_vision:
            embeds.append(self.vision_tower(vision_ids))
        if self.add_audio:
            embeds.append(self.audio_tower(audio_ids))
        embeds.append(self.embed_tokens(input_ids))

        embs = torch.cat(embeds, dim=1)
        for module in self.llm:
            embs = module(embs)
        output = self.lm_head(embs)

        if self.add_tts:
            output = output + self.tts_tower(embs)

        return output.sum(), None

def get_text_model():
    config = LlamaConfig.from_json_file(CONFIG_PATH)
    model = LlamaForCausalLM(config)
    return model

MODELS_FN = {
    Models.TEXT: get_text_model,
    Models.VISION: {'add_vision':True},
    Models.TTS: {'add_tts':True},
    Models.OMNIMODEL: {'add_vision':True, 'add_audio':True, 'add_tts':True}
}

def get_model(args):
    if args.model_type == Models.TEXT:
        return get_text_model()
    else:
        return MultimodalModel(**MODELS_FN[args.model_type])


if __name__ == "__main__":
    pass
