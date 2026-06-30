import math
import torch
import torch.nn as nn
from math import ceil
from typing import List, Optional, Union, Tuple
import sys
import os
import logging
#utils路径修复
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers import Qwen2ForCausalLM
from llava.model.language_model.llava_qwen import LlavaQwenModel
from llava.model.llava_arch import LlavaMetaForCausalLM
from ..utils.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX, MEMORY_TOKEN_INDEX

logger = logging.getLogger(__name__)

class StreamVLNModel(LlavaQwenModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(StreamVLNModel, self).__init__(config)
        
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False

        self.num_history = getattr(config, 'num_history', None)
        

class StreamVLNForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(Qwen2ForCausalLM, self).__init__(config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None
        config.delay_load = True
        
        self.model = StreamVLNModel(config, **kwargs)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_model(self):
        return self.model
    
    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side # 27
        
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [ceil(height / stride), ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature
    
    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]
        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(3, 4)
        image_feature = torch.cat((image_feature, self.model.image_newline[:,None, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        if getattr(self.config, "add_faster_video", False):
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        image_feature = image_feature.flatten(2, 3).permute(1, 2, 0).contiguous()
        return image_feature
    
    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    def encode_rgbd(self, images, depths, poses, intrinsics, time_ids=None, task_ids=None, has_memory_token=False):
        batch_size, num_view, _, H, W = images.shape
        image_features = self.get_model().get_vision_tower()(images.flatten(0,1))
        
        num_patches_per_side = self.get_model().get_vision_tower().num_patches_per_side
        # (B, V, C, num_patch, num_patch)
        image_features = image_features.permute(0, 2, 1).reshape(batch_size, num_view, -1, num_patches_per_side, num_patches_per_side)
        
        # ✅ 修复：生成memory_features的逻辑（参考 streamvln_eval.py）
        # 关键：只有当num_view > 1时（有历史图像），才能生成memory_features
        # batch_size, num_view, H, W = depths.shape
        if num_view != 1:
            memory_features = []
            image_features_ = []
            for b in range(batch_size):
                if time_ids[b] is not None and len(time_ids[b]) > 0:
                    start_idx = time_ids[b][0]
                else:
                    start_idx = 0
                
                # ✅ 修复：即使start_idx == 0，如果有历史图像（num_view > 1），也应该生成memory_features
                # 因为memory token需要memory_features，而memory_features来自历史图像
                if start_idx == 0 and num_view == 1:
                    # 如果没有历史图像，无法生成memory_features
                    memory_features.append(None)
                    image_features_.append(image_features[b])
                    continue
                else:
                    # 有历史图像，生成memory_features
                    history_idx = self.model.num_history if hasattr(self.model, 'num_history') and self.model.num_history is not None else min(num_view - 1, 8)
                    # 确保history_idx不超过num_view
                    history_idx = min(history_idx, num_view - 1)
                    
                    if history_idx > 0:
                        # 历史图像：前history_idx个
                        his_image_feature = image_features[b, :history_idx].flatten(2,3).permute(0,2,1)
                        his_image_feature = self.get_model().mm_projector(his_image_feature)
                        his_image_feature = self.get_2dPool(his_image_feature, 2) # [N, 196, 1152]
                        memory_features.append(his_image_feature.flatten(0,1).unsqueeze(0))
                        
                        # 当前图像：从history_idx开始
                        image_features_.append(image_features[b, history_idx:])
                    else:
                        # 如果没有历史图像，无法生成memory_features
                        memory_features.append(None)
                        image_features_.append(image_features[b])
            image_features = image_features_
        else:
            # ✅ 修复：即使num_view == 1，如果有memory token，也需要尝试生成memory_features
            # 但是，如果num_view == 1，说明没有历史图像，所以memory_features应该是None
            # 只有在有历史图像的情况下（num_view > 1），才能生成memory_features
            if has_memory_token:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"[StreamVLNModel.encode_rgbd] ⚠️ 检测到memory token，但num_view=1，无法生成memory_features")
                logger.warning(f"[StreamVLNModel.encode_rgbd] 这可能是因为历史图像没有被正确添加，或者这是第一步")
                logger.warning(f"[StreamVLNModel.encode_rgbd] images shape: {images.shape}, 应该有历史图像时num_view > 1")
                logger.warning(f"[StreamVLNModel.encode_rgbd] 建议：检查历史图像的添加逻辑，确保在有memory token时添加历史图像")
            memory_features = [None] * batch_size
        
        image_features_=[]
        for j, image_feature in enumerate(image_features):
            image_feature = image_feature.flatten(2,3).permute(0,2,1)
            image_feature = self.get_model().mm_projector(image_feature)
            image_feature = self.get_2dPool(image_feature, 2)
            image_features_.append(image_feature)
        image_features = image_features_
        return image_features, memory_features
   
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, 
        images, image_sizes, depths, poses, intrinsics, time_ids=None, task_ids=None
    ):  
        import logging
        logger = logging.getLogger(__name__)
        
        vision_tower = self.get_vision_tower()
        # logger.info(f"[StreamVLNModel.prepare_inputs_labels_for_multimodal] 调用开始:")
        # logger.info(f"  - vision_tower is None: {vision_tower is None}")
        # logger.info(f"  - images is None: {images is None}")
        # if images is not None:
        #     logger.info(f"  - images shape: {images.shape}")
        # logger.info(f"  - input_ids shape: {input_ids.shape if input_ids is not None else None}")
        
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            logger.warning(f"[StreamVLNModel.prepare_inputs_labels_for_multimodal] ⚠️ 跳过视觉特征处理: "
                         f"vision_tower={vision_tower is None}, images={images is None}, "
                         f"input_ids.shape[1]={input_ids.shape[1] if input_ids is not None else None}")
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # logger.info(f"[StreamVLNModel.prepare_inputs_labels_for_multimodal] ✓✓✓ 开始提取视觉特征...")
        # logger.info(f"  - 输入images shape: {images.shape if images is not None else None}")
        # logger.info(f"  - 输入depths shape: {depths.shape if depths is not None else None}")
        
        # ✅ 修复：检查input_ids中是否有MEMORY_TOKEN_INDEX，如果有，需要生成memory_features
        has_memory_token = (input_ids == MEMORY_TOKEN_INDEX).any().item() if input_ids is not None else False
        # logger.info(f"[StreamVLNModel.prepare_inputs_labels_for_multimodal] 检查memory token: has_memory_token={has_memory_token}")
        # if has_memory_token:
        #     logger.info(f"[StreamVLNModel.prepare_inputs_labels_for_multimodal] ✓ 检测到MEMORY_TOKEN_INDEX，需要生成memory_features")
        
        image_features, memory_features = self.encode_rgbd(images, depths, poses, intrinsics, time_ids, task_ids, has_memory_token=has_memory_token)
        # logger.info(f"[StreamVLNModel.prepare_inputs_labels_for_multimodal] ✓✓✓ 视觉特征提取完成:")
        # logger.info(f"  - image_features 数量: {len(image_features) if image_features else 0}")
        # if image_features and len(image_features) > 0:
        #     logger.info(f"  - image_features[0] shape: {image_features[0].shape if image_features[0] is not None else None}")
        #     if len(image_features) > 1:
        #         logger.info(f"  - image_features[1] shape: {image_features[1].shape if image_features[1] is not None else None}")
        # logger.info(f"  - memory_features 数量: {len(memory_features) if memory_features else 0}")
        # if memory_features and len(memory_features) > 0:
        #     logger.info(f"  - memory_features[0] shape: {memory_features[0].shape if memory_features[0] is not None else None}")
        #     logger.info(f"  - memory_features[0] 包含的batch数量: {len(memory_features[0]) if isinstance(memory_features[0], list) else 1}")
        #     if len(memory_features) > 0 and isinstance(memory_features[0], list) and len(memory_features[0]) > 0:
        #         logger.info(f"  - memory_features[0][0] shape: {memory_features[0][0].shape if memory_features[0][0] is not None else None}")

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError
        
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        
        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_memories = (cur_input_ids == MEMORY_TOKEN_INDEX).sum()
            num_specials = num_images + num_memories
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            memory_token_indices = torch.where(cur_input_ids == MEMORY_TOKEN_INDEX)[0].tolist()
            special_token_indices = sorted(image_token_indices + memory_token_indices)
            special_tokens = [cur_input_ids[indice] for indice in special_token_indices]
            special_token_indices = [-1] + special_token_indices + [cur_input_ids.shape[0]]
            
            # # ✅ 详细调试：打印特殊token信息
            # logger.info(f"[StreamVLNModel.prepare_inputs_labels_for_multimodal] Batch {batch_idx} 特殊Token分析:")
            # logger.info(f"  - IMAGE_TOKEN数量: {num_images}, 位置: {image_token_indices}")
            # logger.info(f"  - MEMORY_TOKEN数量: {num_memories}, 位置: {memory_token_indices}")
            # logger.info(f"  - 总特殊Token数量: {num_specials}")
            
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            
            for i in range(len(special_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[special_token_indices[i]+1:special_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[special_token_indices[i]+1:special_token_indices[i+1]])
                
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            
            cur_img_id = 0
            cur_mem_id = 0
            
            for i in range(num_specials + 1):  # num_images = 1? [0, 1]
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_specials:
                    # print(f"Batch Index: {batch_idx}\n, Current Image Index: {cur_image_idx}\n, Num Images: {num_images}")
                    special_token = special_tokens[i]
                
                    if special_token == IMAGE_TOKEN_INDEX:
                        cur_image_feature = image_features[batch_idx][cur_img_id]
                        cur_img_id += 1
                        # import logging
                        # logger = logging.getLogger(__name__)
                        # logger.info(f"[StreamVLNModel.prepare_inputs_labels_for_multimodal] ✓✓✓ 插入图像特征: "
                        #           f"batch_idx={batch_idx}, img_id={cur_img_id-1}, "
                        #           f"feature shape={cur_image_feature.shape}, "
                        #           f"特征token数量={cur_image_feature.shape[0]}")
                        cur_new_input_embeds.append(cur_image_feature)
                        cur_new_labels.append(torch.full((cur_image_feature.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    elif special_token == MEMORY_TOKEN_INDEX:
                        cur_memory_feature = memory_features[batch_idx][cur_mem_id]
                        cur_mem_id += 1
                        # logger.info(f"[StreamVLNModel.prepare_inputs_labels_for_multimodal] ✓✓✓ 插入记忆特征: "
                        #           f"batch_idx={batch_idx}, mem_id={cur_mem_id-1}, "
                        #           f"feature shape={cur_memory_feature.shape}, "
                        #           f"特征token数量={cur_memory_feature.shape[0]}")
                        cur_new_input_embeds.append(cur_memory_feature)
                        cur_new_labels.append(torch.full((cur_memory_feature.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    else:
                        raise NotImplementedError
            
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            # assert len(cur_new_input_embeds) <= 4096
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
        
        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            
        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        
        # import logging
        # logger = logging.getLogger(__name__)
        # logger.info(f"[StreamVLNModel.prepare_inputs_labels_for_multimodal] ✓✓✓ 多模态输入准备完成:")
        # logger.info(f"  - 原始input_ids长度: {_input_ids.shape[1] if _input_ids is not None else 'unknown'}")
        # logger.info(f"  - 新的input_embeds长度: {new_input_embeds.shape[1]}")
        # logger.info(f"  - 长度增加: {new_input_embeds.shape[1] - (_input_ids.shape[1] if _input_ids is not None else 0)} tokens")
        # logger.info(f"  - new_input_embeds shape: {new_input_embeds.shape}")
        # logger.info(f"  - 包含视觉特征: {new_input_embeds.shape[1] > _input_ids.shape[1] if _input_ids is not None else 'unknown'}")
        
        # ✅ 统计最终嵌入中的视觉特征
        total_visual_tokens = 0
        for batch_idx in range(len(input_ids)):
            num_images = (input_ids[batch_idx] == IMAGE_TOKEN_INDEX).sum().item()
            num_memories = (input_ids[batch_idx] == MEMORY_TOKEN_INDEX).sum().item()
            # 估算视觉特征token数量（每个图像特征约196个token，每个记忆特征可能更多）
            if num_images > 0 and image_features and len(image_features) > batch_idx:
                for img_id in range(num_images):
                    if img_id < len(image_features[batch_idx]):
                        total_visual_tokens += image_features[batch_idx][img_id].shape[0]
            if num_memories > 0 and memory_features and len(memory_features) > batch_idx:
                for mem_id in range(num_memories):
                    if mem_id < len(memory_features[batch_idx]):
                        total_visual_tokens += memory_features[batch_idx][mem_id].shape[0]
        # logger.info(f"  - 估算的视觉特征token总数: {total_visual_tokens}")
        
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded
            
        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)
        
        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: torch.FloatTensor = None,
        depths: torch.FloatTensor = None,
        poses: torch.FloatTensor = None,
        intrinsics: torch.FloatTensor = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"[StreamVLNForCausalLM.forward] 调用: inputs_embeds is None={inputs_embeds is None}, "
                    f"images is None={images is None}")
        
        tokenizer = kwargs.get("tokenizer", None)
        input_ids_ = input_ids
        time_ids = kwargs.get("time_ids", None)
        task_ids = kwargs.get("task_type", None)
        if inputs_embeds is None:
            # logger.info(f"[StreamVLNForCausalLM.forward] inputs_embeds is None，调用 prepare_inputs_labels_for_multimodal")
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels, 
                images, 
                image_sizes,
                depths, 
                poses, 
                intrinsics,
                time_ids,
                task_ids
            )
            # logger.info(f"[StreamVLNForCausalLM.forward] ✓ prepare_inputs_labels_for_multimodal 完成: "
                    #   f"inputs_embeds shape={inputs_embeds.shape if inputs_embeds is not None else None}")
        else:
            logger.debug(f"[StreamVLNForCausalLM.forward] inputs_embeds 已提供，跳过 prepare_inputs_labels_for_multimodal")
    
        logger.debug(f"[StreamVLNForCausalLM.forward] ✓ 调用 Qwen2ForCausalLM.forward()")
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        depths: Optional[torch.FloatTensor] = None,
        poses: Optional[torch.FloatTensor] = None,
        intrinsics: Optional[torch.FloatTensor] = None,
        task_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # import logging
        # logger = logging.getLogger(__name__)
        # logger.info(f"[StreamVLNForCausalLM.generate] 调用开始:")
        # logger.info(f"  - inputs shape: {inputs.shape if inputs is not None else None}")
        # logger.info(f"  - images shape: {images.shape if images is not None else None}")
        # logger.info(f"  - depths shape: {depths.shape if depths is not None else None}")
        # logger.info(f"  - poses shape: {poses.shape if poses is not None else None}")
        # logger.info(f"  - intrinsics shape: {intrinsics.shape if intrinsics is not None else None}")
        
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        time_ids = kwargs.pop("time_ids", None)
        task_ids = kwargs.pop("task_type", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            # logger.info(f"[StreamVLNForCausalLM.generate] ✓ 检测到图像输入，调用 prepare_inputs_labels_for_multimodal")
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes,
                depths,
                poses,
                intrinsics,
                time_ids,
                task_ids
            )
            # logger.info(f"[StreamVLNForCausalLM.generate] ✓ prepare_inputs_labels_for_multimodal 完成:")
            # logger.info(f"  - inputs_embeds shape: {inputs_embeds.shape if inputs_embeds is not None else None}")
        else:
            # logger.info(f"[StreamVLNForCausalLM.generate] ⚠️ 没有图像输入，仅使用文本嵌入")
            inputs_embeds = self.get_model().embed_tokens(inputs)
            # logger.info(f"[StreamVLNForCausalLM.generate] ✓ 文本嵌入完成: shape={inputs_embeds.shape}")
        
        env_id = kwargs.pop("env_id", None)
        # logger.info(f"[StreamVLNForCausalLM.generate] ✓ 准备调用 super().generate()，inputs_embeds shape: {inputs_embeds.shape}")
        # 关键修复：
        # 旧逻辑会在每次 generate 时把 inputs_embeds 沿序列维拼接，导致序列长度每步增长（你看到的 ~330 增量），
        # 但 position_ids/attention_mask 仍然是“本次输入块”的长度，从而在 Qwen2 内部 reshape/view 时触发维度错误。
        #
        # 正确做法：不要手动累积 inputs_embeds，让 transformers 的 past_key_values/use_cache 机制管理缓存。
        # 因此这里每次都直接使用本次 inputs_embeds。
        if env_id is not None:
            # 保留字段以兼容其它逻辑，但不再累积
            self.cache[env_id]["inputs_embeds"] = inputs_embeds
            self.curr_t[env_id] += 1
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)

        # ===== 基本逻辑：对齐 sk_streamvln，允许使用 inputs_embeds 进行首步生成 =====

        # 计算 past_length（已生成的 token 数）
        past_length = 0
        if past_key_values is not None and len(past_key_values) > 0:
            try:
                past_length = int(past_key_values[0][0].shape[2])
            except Exception:
                past_length = 0

        # 如果 cache_position 为 None，根据 input_ids 或 inputs_embeds 构造
        if cache_position is None:
            if input_ids is not None and input_ids.shape[1] > 0:
                device = input_ids.device
                cache_position = torch.arange(past_length, past_length + input_ids.shape[1], device=device)
            elif inputs_embeds is not None and inputs_embeds.shape[1] > 0:
                device = inputs_embeds.device
                cache_position = torch.arange(past_length, past_length + inputs_embeds.shape[1], device=device)
            else:
                device = input_ids.device if input_ids is not None else (inputs_embeds.device if inputs_embeds is not None else torch.device("cpu"))
                cache_position = torch.arange(past_length, past_length + 1, device=device)

        # 如果有 past_key_values，切片 input_ids 只保留新 token
        if past_key_values is not None and input_ids is not None and input_ids.shape[1] > 1:
            input_ids = input_ids[:, -1:]

        # 计算当前序列长度（用于 position_ids）
        if inputs_embeds is not None:
            seq_length = inputs_embeds.shape[1]
        elif input_ids is not None:
            seq_length = input_ids.shape[1]
        else:
            seq_length = 0

        # 生成 position_ids
        if position_ids is None:
            if attention_mask is not None:
                # 从 attention_mask 推导 position_ids
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                # 如果有 past_key_values，只保留最后 seq_length 个位置
                if past_key_values is not None and seq_length > 0:
                    position_ids = position_ids[:, -seq_length:]
                    position_ids = position_ids.clone(memory_format=torch.contiguous_format)
            elif cache_position is not None and seq_length > 0:
                # 从 cache_position 构造 position_ids
                try:
                    cache_pos_len = cache_position.shape[0] if hasattr(cache_position, 'shape') else len(cache_position)
                    if cache_pos_len == seq_length:
                        # 直接使用 cache_position
                        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
                        position_ids = cache_position.long().unsqueeze(0).expand(batch_size, -1)
                    else:
                        # 如果长度不匹配，从 past_length 开始构造
                        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
                        position_ids = torch.arange(past_length, past_length + seq_length, device=cache_position.device).long().unsqueeze(0).expand(batch_size, -1)
                except Exception:
                    # 兜底：从 past_length 开始构造
                    batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
                    device = input_ids.device if input_ids is not None else inputs_embeds.device
                    position_ids = torch.arange(past_length, past_length + seq_length, device=device).long().unsqueeze(0).expand(batch_size, -1)
            else:
                # 最后的兜底：从 past_length 开始构造
                batch_size = input_ids.shape[0] if input_ids is not None else (inputs_embeds.shape[0] if inputs_embeds is not None else 1)
                device = input_ids.device if input_ids is not None else (inputs_embeds.device if inputs_embeds is not None else torch.device("cpu"))
                position_ids = torch.arange(past_length, past_length + max(seq_length, 1), device=device).long().unsqueeze(0).expand(batch_size, -1)

        # 确保 position_ids 的长度与 seq_length 匹配
        if position_ids is not None and seq_length > 0:
            if position_ids.shape[1] != seq_length:
                # 如果长度不匹配，重新构造
                batch_size = position_ids.shape[0]
                device = position_ids.device
                position_ids = torch.arange(past_length, past_length + seq_length, device=device).long().unsqueeze(0).expand(batch_size, -1)

        # 决定使用 inputs_embeds 还是 input_ids
        if inputs_embeds is not None and cache_position is not None:
            try:
                cache_len = cache_position.shape[0] if hasattr(cache_position, 'shape') else len(cache_position)
            except Exception:
                cache_len = 0
            first_pos = int(cache_position[0]) if cache_len > 0 else 0

            if first_pos == 0:
                # 首步：使用完整的 inputs_embeds
                model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
            elif cache_len > 0 and cache_len < inputs_embeds.shape[1]:
                # 后续步：只使用最后 cache_len 个 token
                model_inputs = {"inputs_embeds": inputs_embeds[:, -cache_len:], "input_ids": None}
                # 更新 position_ids 以匹配新的序列长度
                if position_ids is not None and position_ids.shape[1] != cache_len:
                    batch_size = position_ids.shape[0]
                    device = position_ids.device
                    position_ids = torch.arange(past_length, past_length + cache_len, device=device).long().unsqueeze(0).expand(batch_size, -1)
            else:
                # 回退到 input_ids
                if input_ids is not None and input_ids.shape[1] > 0:
                    model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}
                else:
                    # 如果 input_ids 也为空，仍然使用 inputs_embeds
                    model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # 使用 input_ids
            if input_ids is not None and input_ids.shape[1] > 0:
                model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}
            elif inputs_embeds is not None:
                # 如果 input_ids 为空但 inputs_embeds 不为空，使用 inputs_embeds
                model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
            else:
                raise ValueError("Both input_ids and inputs_embeds are None or empty")

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        if images is not None:
            model_inputs['images'] = images
        if image_sizes is not None:
            model_inputs['image_sizes'] = image_sizes
        return model_inputs
    
    def reset(self, env_num):
        self.curr_t = [0] * env_num
        self.cache = [dict()] * env_num
    
    def reset_for_env(self, env_idx):
        self.curr_t[env_idx] = 0
        self.cache[env_idx] = dict()