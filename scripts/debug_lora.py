import sys, os
sys.path.insert(0, '/share/home/u19666033/dhj/dped-vln/habitat-baselines/habitat_baselines/rl/ddppo/policy/streamvln')

import torch
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser
from peft import LoraConfig, get_peft_model, PeftModel
import transformers
from dataclasses import dataclass, field

@dataclass
class ModelArgs:
    model_name_or_path: str = "/share/home/u19666033/dhj/dped-vln/pretrained_model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln"
    version: str = "qwen_1_5"
    mm_vision_tower: str = "google/siglip-so400m-patch14-384"
    mm_projector_type: str = "mlp2x_gelu"
    mm_vision_select_layer: int = -2
    mm_vision_select_feature: str = "patch"
    mm_use_im_start_end: bool = False
    mm_use_im_patch_token: bool = False
    image_aspect_ratio: str = "anyres_max_9"
    mm_spatial_pool_stride: int = 2
    mm_newline_position: str = "grid"
    mm_patch_merge_type: str = "flat"
    num_future_steps: int = 4
    num_history: int = 8
    model_max_length: int = 32768
    token_compression: str = "none"

@dataclass
class DataArgs:
    is_multimodal: bool = True
    mm_use_im_start_end: bool = False

@dataclass
class TrainingArgs:
    lora_enable: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    bf16: bool = True
    fp16: bool = False
    bits: int = 16
    gradient_checkpointing: bool = False
    cache_dir: str = None
    mm_tunable_parts: str = "mm_lora_layer"
    attn_implementation: str = "eager"

model_args = ModelArgs()
data_args = DataArgs()
training_args = TrainingArgs()

print("Loading model...", flush=True)
# Load the base model first, then apply LoRA
from streamvln.model.stream_video_vln import StreamVLNForCausalLM

model = StreamVLNForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    config=AutoConfig.from_pretrained(model_args.model_name_or_path),
    torch_dtype=torch.bfloat16,
    attn_implementation='eager',
)
print("Model loaded", flush=True)

# Check what PEFT would target
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["vision_tower", "mm_projector", "mem_projector", "point_projector", "vision_resampler", "mem_resampler", "pointnet"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)

targets = find_all_linear_names(model)
print(f"PEFT target modules: {targets}", flush=True)

lora_config = LoraConfig(
    r=training_args.lora_r,
    lora_alpha=training_args.lora_alpha,
    target_modules=targets,
    lora_dropout=training_args.lora_dropout,
    bias=training_args.lora_bias,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
print("LoRA applied", flush=True)

# Count trainable params
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total/1e6:.1f}M, Trainable: {trainable/1e6:.1f}M", flush=True)

# Check a few lora params
count = 0
for n, p in model.named_parameters():
    if 'lora' in n.lower() and p.requires_grad:
        print(f"  {n}: requires_grad=True, shape={p.shape}")
        count += 1
        if count >= 3:
            break
if count == 0:
    print("  NO LoRA params with requires_grad=True!")
    # Check if any lora params exist at all
    lora_params = [(n, p.requires_grad) for n, p in model.named_parameters() if 'lora' in n.lower()]
    print(f"  Total lora params: {len(lora_params)}")
    for n, r in lora_params[:5]:
        print(f"    {n}: requires_grad={r}")

print("\nTest forward pass...", flush=True)
# Simple test input
dummy_input = torch.randint(0, 1000, (1, 16))
dummy_images = torch.randn(1, 4, 3, 336, 336, dtype=torch.bfloat16)
try:
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output = model(input_ids=dummy_input.cuda(), images=dummy_images.cuda())
        loss = output.logits.sum()
        loss.backward()
        print("Forward + backward PASSED!", flush=True)
except Exception as e:
    print(f"Forward/backward FAILED: {e}", flush=True)
    import traceback
    traceback.print_exc()
