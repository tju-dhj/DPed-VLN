#!/usr/bin/env python3
"""NaVid import and model loading test script"""
import sys
import os

sys.path.insert(0, '/share/home/u19666033/dhj/dped-vln')
sys.path.insert(0, '/share/home/u19666033/dhj/dped-vln/habitat-lab')
sys.path.insert(0, '/share/home/u19666033/dhj/dped-vln/habitat-baselines')

print("=== Test 1: NaVid module imports ===")
from habitat_baselines.rl.ddppo.policy.navid.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, NAVIGATION_SPECIAL_TOKEN
)
print("OK: constants")

from habitat_baselines.rl.ddppo.policy.navid.conversation import conv_templates
print("OK: conversation, templates:", list(conv_templates.keys()))

from habitat_baselines.rl.ddppo.policy.navid.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
print("OK: mm_utils")

from habitat_baselines.rl.ddppo.policy.navid.model.language_model.llava_navid import (
    LlavaLlamaAttForCausalLM
)
print("OK: llava_navid model class")

from habitat_baselines.rl.ddppo.policy.navid.model.builder import load_pretrained_model
print("OK: builder/load_pretrained_model")

from habitat_baselines.rl.ddppo.policy.navid.action_parser import NaVidActionParser
p = NaVidActionParser()
a, n = p.parse_action("forward 50")
assert a == 1 and n == 2, f"Expected (1,2) got ({a},{n})"
print("OK: action_parser, forward 50 ->", a, n)

print()
print("=== Test 2: Model loading ===")
import torch
model_path = "/share/home/u19666033/dhj/dped-vln/pretrained_model/navid_checkpoint"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, "navid"
)
print(f"Model loaded: {type(model).__name__}")
print(f"Tokenizer: {type(tokenizer).__name__}")
print(f"Vision tower: {type(model.get_vision_tower()).__name__}")
print(f"Context length: {context_len}")

model = model.to("cuda")
model.eval()
print("Model on GPU OK")

print()
print("=== Test 3: Vision processing ===")
import numpy as np
from PIL import Image

# Create a dummy RGB image
dummy_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
dummy_img = Image.fromarray(dummy_rgb).convert("RGB")
img_tensor = image_processor.preprocess(dummy_img, return_tensors='pt')['pixel_values']
print(f"Image processed: {img_tensor.shape}")

# Move to GPU and process through vision tower
img_tensor = img_tensor.half().cuda()
with torch.inference_mode():
    vision_features = model.get_vision_tower()(img_tensor)
print(f"Vision features: {vision_features.shape}")

print()
print("=== ALL TESTS PASSED ===")
