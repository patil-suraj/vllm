# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/19e6e80e10118f855137b90740936c0b11ac397f/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Tarsier2 model compatible with HuggingFace weights.

Note: In Tarsier2, videos are treated as multi-images rather than having separate 
video token handling. Video frames are processed as individual images and use 
image tokens for placeholder replacement.
"""
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Any, List, Optional, Union
import os
import decord

import torch
import torch.nn as nn
from PIL import Image, ImageSequence
from transformers import BatchFeature, ProcessorMixin, AutoImageProcessor, AutoTokenizer
from transformers.models.qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig
from vllm.model_executor.models.module_mapping import MultiModelKeys
# Import vision classes from qwen2_vl instead of reimplementing them
from vllm.model_executor.models.qwen2_vl import (
    Qwen2VisionTransformer,
    Qwen2VLImagePixelInputs,
    Qwen2VLImageEmbeddingInputs,
    Qwen2VLVideoPixelInputs,
    Qwen2VLVideoEmbeddingInputs,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (ImageItem, ModalityData,
                                    MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs, VideoItem)
from vllm.multimodal.parse import (DictEmbeddingItems, ImageSize,
                                   ModalityDataItems, MultiModalDataItems,
                                   MultiModalDataParser)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import uses_mrope
from vllm.transformers_utils.processor import cached_image_processor_from_config

from .interfaces import (MultiModalEmbeddings, SupportsLoRA,
                         SupportsMultiModal, SupportsPP)
from .utils import (AutoWeightsLoader, WeightsMapper,
                    init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)

logger = init_logger(__name__)

# For profile run
_MAX_FRAMES_PER_VIDEO = 16

# === Vision Inputs === #

# Use type aliases for the imported types to maintain API compatibility
Qwen2VLImageInputs = Union[Qwen2VLImagePixelInputs,
                           Qwen2VLImageEmbeddingInputs]

Qwen2VLVideoInputs = Union[Qwen2VLVideoPixelInputs,
                           Qwen2VLVideoEmbeddingInputs]


# === Tarsier Video/Image Processing Utilities === #

def sample_frame_indices(start_frame: int, total_frames: int, n_frames: int) -> List[int]:
    """Sample frame indices uniformly, always including first and last frames."""
    if n_frames == 1:
        return [0]
    if total_frames <= n_frames:
        return list(range(total_frames))
    sample_ids = [round(i * (total_frames - 1) / (n_frames - 1)) for i in range(n_frames)]
    sample_ids = [i + start_frame for i in sample_ids]
    return sample_ids


def get_visual_type(input_file: str) -> str:
    """Determine the visual media type from file extension."""
    ext = os.path.splitext(input_file)[-1].lower()
    if ext in {'.gif'}:
        return 'gif'
    elif ext in {'.mp4', '.avi', '.webm', '.mov', '.mkv', '.wmv'}:
        return 'video'
    elif ext in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}:
        return 'image'
    else:
        return 'unknown'


def sample_video_frames(video_path: str, n_frames: int = 8) -> List[Image.Image]:
    """Sample frames from video file."""
    assert os.path.exists(video_path), f"File not found: {video_path}"
    
    if video_path.endswith('.gif'):
        # Handle GIF files
        gif_frames = Image.open(video_path)
        total_frames = gif_frames.n_frames
        
        frame_indices = sample_frame_indices(0, total_frames, n_frames)
        frames = []
        i = 0
        for frame in ImageSequence.Iterator(gif_frames):
            if i in frame_indices:
                frames.append(frame.convert('RGB'))
            i += 1
        return frames
    else:
        # Handle video files
        vr = decord.VideoReader(video_path, num_threads=1, ctx=decord.cpu(0))
        vr.seek(0)
        total_frames = len(vr)
        
        frame_indices = sample_frame_indices(0, total_frames, n_frames)
        frames = vr.get_batch(frame_indices).asnumpy()
        frames = [Image.fromarray(f).convert('RGB') for f in frames]
        return frames


def load_image(image_path: str) -> Image.Image:
    """Load single image from file."""
    assert os.path.exists(image_path), f"File not found: {image_path}"
    return Image.open(image_path).convert('RGB')


def format_tarsier_sample(media_file: str = None, prompt: str = "Describe the video in detail.") -> dict:
    """Format input into Tarsier message structure."""
    sample = {"messages": []}
    
    user_content = {"role": "user", "content": []}
    
    if media_file is not None:
        media_type = get_visual_type(media_file)
        if media_type in ("video", "gif"):
            media_type = "video"
        elif media_type == "image":
            media_type = "image"
        else:
            raise ValueError(f"Unsupported media type: {media_type}")
            
        media_path_key = f"{media_type}_file"
        user_content["content"].append({
            "type": media_type,
            media_type: {
                media_path_key: media_file,
            }
        })
    
    user_content["content"].append({
        "type": "text",
        "text": prompt
    })
    
    assistant_content = {"role": "assistant", "content": []}
    
    sample["messages"].append(user_content)
    sample["messages"].append(assistant_content)
    
    if media_file is not None:
        sample["task"] = f"{media_type}/QA"
    else:
        sample["task"] = 'text-only'
    
    return sample


class TarsierProcessor(ProcessorMixin):
    """
    Tarsier processor that handles both images and videos with custom preprocessing.
    Inherits from ProcessorMixin to be compatible with transformers ecosystem.
    """
    
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = [
        "chat_template", "image_token", "patch_size", "merge_size", 
        "temporal_patch_size", "max_seq_len", "n_frames", "max_pixels", "min_pixels"
    ]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_token="<image>",
        patch_size=None,
        merge_size=1,
        temporal_patch_size=1,
        max_seq_len=8192,
        n_frames=16,
        max_pixels=460800,  # 1280 * 720 // 2
        min_pixels=0,
        max_pixels_per_sample=128 * 384 * 384,
        **kwargs,
    ):
        self.image_token = image_token
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.temporal_patch_size = temporal_patch_size
        self.max_seq_len = max_seq_len
        self.n_frames = n_frames
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_pixels_per_sample = max_pixels_per_sample

        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        
        # Initialize vision processor for custom preprocessing
        self.vision_processor = TarsierVisionProcessor(
            n_frames=n_frames,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            temporal_patch_size=temporal_patch_size,
            max_pixels_per_sample=max_pixels_per_sample
        )

    def __call__(self, messages, processing_config=None, **kwargs):
        """Process messages with Tarsier minimal preprocessing."""
        if processing_config is None:
            # Default config - only pixel resizing is used
            processing_config = {
                'max_pixels': self.max_pixels,
                'min_pixels': self.min_pixels
            }
        
        # Process messages using vision processor
        processed_messages = self.vision_processor.process_messages(messages, processing_config)
        
        # Convert to format expected by underlying image processor and tokenizer
        text_content = ""
        images = []
        videos = []
        
        for msg in processed_messages:
            for content in msg["content"]:
                if content["type"] == "text":
                    text_content += content["text"] + " "
                elif content["type"] == "image":
                    if isinstance(content["image"], list):
                        images.extend(content["image"])
                    else:
                        images.append(content["image"])
                elif content["type"] == "video":
                    if isinstance(content["video"], list):
                        videos.extend(content["video"])
                    else:
                        videos.append(content["video"])
        
        # Process with underlying processors
        result = {}
        
        if images:
            image_inputs = self.image_processor(images=images, return_tensors="pt")
            result.update(image_inputs)
        
        if videos:
            # For Tarsier2, videos are treated as images
            video_inputs = self.image_processor(images=videos, return_tensors="pt")
            # Rename to video format
            if "pixel_values" in video_inputs:
                result["pixel_values_videos"] = video_inputs["pixel_values"]
                if "image_grid_thw" in video_inputs:
                    result["video_grid_thw"] = video_inputs["image_grid_thw"]
        
        if text_content.strip():
            text_inputs = self.tokenizer(text_content.strip(), return_tensors="pt")
            result.update(text_inputs)
        
        return BatchFeature(result)


    def batch_decode(self, *args, **kwargs):
        """Forward to tokenizer's batch_decode."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Forward to tokenizer's decode."""
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


class TarsierVisionProcessor:
    """Handles vision processing for Tarsier2 with minimal preprocessing based on default config."""
    
    def __init__(self, 
                 n_frames: int = 16,
                 max_pixels: int = 460800,  # 1280 * 720 // 2
                 min_pixels: int = 0,
                 temporal_patch_size: int = 1,
                 max_pixels_per_sample: int = 128 * 384 * 384):
        self.n_frames = n_frames
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.temporal_patch_size = temporal_patch_size
        self.max_pixels_per_sample = max_pixels_per_sample
    
    def resize2pixels(self, pil_img: Image.Image, max_pixels: int = None, min_pixels: int = None) -> Image.Image:
        """Resize image based on pixel count using smart_resize - the only preprocessing used by default."""
        width, height = pil_img.size
        new_height, new_width = smart_resize(
            height, width, factor=1, 
            max_pixels=max_pixels or self.max_pixels,
            min_pixels=min_pixels or self.min_pixels
        )
        pil_img = pil_img.resize((new_width, new_height))
        return pil_img

    def preprocess_image(self, pil_img: Union[Image.Image, List[Image.Image]], 
                        processing_config: dict) -> Union[Image.Image, List[Image.Image]]:
        """Apply minimal preprocessing - only pixel resizing based on default config."""
        if processing_config is None:
            return pil_img
        
        images = pil_img if isinstance(pil_img, list) else [pil_img]
        
        # Based on default config, only max_pixels processing is used
        # (do_crop=false, do_padding=false, do_resize=false)
        if processing_config.get('max_pixels'):
            images = [self.resize2pixels(
                img, 
                int(processing_config['max_pixels']), 
                int(processing_config.get('min_pixels', 0))
            ) for img in images]
        
        return images[0] if isinstance(pil_img, Image.Image) else images
    
    def load_vision_item(self, vision_item: Union[dict, Image.Image, List[Image.Image]], vision_type: str) -> List[Image.Image]:
        """Load vision item supporting multiple formats:
        - Dict with 'image_file' or 'video_file' keys (file paths) 
        - PIL.Image directly (for single images)
        - List[PIL.Image] directly (for videos as frame sequences)
        """
        if vision_type == 'image':
            if isinstance(vision_item, dict):
                # File path case: {"image_file": "/path/to/image.jpg"}
                image_file = vision_item.get('image_file')
                if image_file:
                    return [load_image(image_file)]
            elif isinstance(vision_item, Image.Image):
                # PIL Image directly
                return [vision_item]
            elif isinstance(vision_item, list) and len(vision_item) > 0 and isinstance(vision_item[0], Image.Image):
                # List of PIL Images (treat first as single image)
                return [vision_item[0]]
                
        elif vision_type == 'video':
            if isinstance(vision_item, dict):
                # File path case: {"video_file": "/path/to/video.mp4"}
                video_file = vision_item.get('video_file')
                if video_file:
                    return sample_video_frames(video_file, self.n_frames)
            elif isinstance(vision_item, list) and len(vision_item) > 0 and isinstance(vision_item[0], Image.Image):
                # List of PIL Images directly (video frames)
                # Subsample if we have more frames than needed
                if len(vision_item) <= self.n_frames:
                    return vision_item
                else:
                    # Use the same uniform sampling strategy as for video files
                    frame_indices = sample_frame_indices(0, len(vision_item), self.n_frames)
                    return [vision_item[i] for i in frame_indices]
            elif isinstance(vision_item, Image.Image):
                # Single PIL Image treated as single-frame video
                return [vision_item]
        
        raise ValueError(f"Invalid vision item: {vision_item} (type: {type(vision_item)}) for vision_type: {vision_type}")
    
    def adjust_pixel_limits_for_multiple_items(self, messages: List[dict], processing_config: dict) -> dict:
        """Adjust max_pixels when there are multiple images/videos to fit within sample limits."""
        config = dict(processing_config)
        
        # Count total number of frames needed
        num_frames = 0
        for msg in messages:
            for content in msg['content']:
                if content['type'] == 'image':
                    num_frames += self.temporal_patch_size
                elif content['type'] == 'video':
                    # For video files, estimate frames needed
                    if isinstance(content['video'], dict) and 'video_file' in content['video']:
                        num_frames += self.n_frames
                    else:
                        num_frames += len(content['video']) if isinstance(content['video'], list) else 1
        
        # Adjust max_pixels if we have multiple items
        if num_frames > 0 and self.max_pixels_per_sample // num_frames < config['max_pixels']:
            config['max_pixels'] = self.max_pixels_per_sample // num_frames
            config['min_pixels'] = min(config['min_pixels'], config['max_pixels'])
        
        return config

    def process_messages(self, messages: List[dict], processing_config: dict) -> List[dict]:
        """Process messages to load and preprocess vision content."""
        # Adjust pixel limits for multiple items
        adjusted_config = self.adjust_pixel_limits_for_multiple_items(messages, processing_config)
        
        processed_messages = []
        
        for msg in messages:
            processed_msg = {"role": msg["role"], "content": []}
            
            for content in msg["content"]:
                if content["type"] == "text":
                    processed_msg["content"].append(content)
                elif content["type"] in ["image", "video"]:
                    # Load vision content
                    vision_item = content[content["type"]]  
                    images = self.load_vision_item(vision_item, content["type"])
                    
                    # Apply preprocessing
                    images = self.preprocess_image(images, adjusted_config)
                    
                    # Create processed content
                    processed_content = {
                        "type": content["type"],
                        content["type"]: images
                    }
                    processed_msg["content"].append(processed_content)
            
            processed_messages.append(processed_msg)
        
        return processed_messages


def _tarsier2_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
    image_grid_sizes = image_grid_thw.prod(-1)

    video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
    video_grid_sizes = video_grid_thw.prod(-1)

    return dict(
        pixel_values=MultiModalFieldConfig.flat_from_sizes(
            "image", image_grid_sizes),
        image_embeds=MultiModalFieldConfig.flat_from_sizes(
            "image", image_grid_sizes),
        image_grid_thw=MultiModalFieldConfig.batched("image"),
        pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
            "video", video_grid_sizes),
        video_embeds=MultiModalFieldConfig.flat_from_sizes(
            "video", video_grid_sizes),
        video_grid_thw=MultiModalFieldConfig.batched("video"),
    )


class Tarsier2MultiModalDataParser(MultiModalDataParser):

    def _parse_image_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[ImageItem]],
    ) -> Optional[ModalityDataItems[Any, Any]]:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="image",
                required_fields={"image_embeds", "image_grid_thw"},
                fields_factory=_tarsier2_field_config,
            )

        return super()._parse_image_data(data)

    def _parse_video_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[VideoItem]],
    ) -> Optional[ModalityDataItems[Any, Any]]:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="video",
                required_fields={"video_embeds", "video_grid_thw"},
                fields_factory=_tarsier2_field_config,
            )

        return super()._parse_video_data(data)


class Tarsier2ProcessingInfo(BaseProcessingInfo):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize Tarsier vision processor with hardcoded values
        # Based on tarsier2_default_config.yaml defaults
        self.vision_processor = TarsierVisionProcessor(
            n_frames=16,  # Default frame count from config
            max_pixels=460800,  # 1280 * 720 // 2 from config
            min_pixels=0,
            temporal_patch_size=1
        )

    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen2VLConfig)

    def get_hf_processor(
        self,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        **kwargs: object,
    ) -> TarsierProcessor:
        """Get TarsierProcessor instance with custom preprocessing."""
        image_processor = self.get_image_processor(
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            size=size,
            **kwargs
        )
        
        tokenizer = self.get_tokenizer()
        
        # Use hardcoded values based on tarsier2_default_config.yaml
        return TarsierProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            n_frames=16,  # Default frame count from config
            max_pixels=460800,  # 1280 * 720 // 2 from config
            min_pixels=0,
            temporal_patch_size=1,
            max_pixels_per_sample=128 * 384 * 384,  # Memory budget
            **kwargs
        )

    def process_tarsier_messages(self, messages: List[dict], processing_config: dict = None) -> List[dict]:
        """Process messages using Tarsier vision processor with default config."""
        if processing_config is None:
            # Default config from tarsier2_default_config.yaml - only max_pixels is used
            processing_config = {
                'max_pixels': 460800,  # 1280 * 720 // 2 from config
                'min_pixels': 0
            }
        
        return self.vision_processor.process_messages(messages, processing_config)

    def convert_to_tarsier2_format(self, prompt: str, images: Optional[list] = None, videos: Optional[list] = None):
        """Convert input to Tarsier2 message format
        Note: In Tarsier2, videos are treated as multi-images, so video frames are processed as individual images"""
        messages = []
        user_content = []
        
        # Add images
        if images:
            for img in images:
                user_content.append({
                    "type": "image",
                    "image": img
                })
        
        # Add videos (treated as multi-images in Tarsier2)
        if videos:
            for video in videos:
                user_content.append({
                    "type": "video", 
                    "video": video
                })
        
        # Add text prompt
        user_content.append({
            "type": "text",
            "text": prompt
        })
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return messages

    def _get_image_processor_kwargs(
        self,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        **kwargs: object,
    ):
        # Use hardcoded defaults instead of trying to read from config
        # Pass min_pixels and max_pixels directly without size dictionary issues
        if min_pixels is not None:
            kwargs["min_pixels"] = min_pixels
        else:
            kwargs["min_pixels"] = 0  # Hardcoded default

        if max_pixels is not None:
            kwargs["max_pixels"] = max_pixels
        else:
            kwargs["max_pixels"] = 460800  # 1280 * 720 // 2 from config

        # Only pass size if it was explicitly provided
        if size is not None:
            kwargs["size"] = size

        return kwargs

    def get_image_processor(
        self,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        **kwargs: object,
    ) -> Qwen2VLImageProcessor:
        # Use direct AutoImageProcessor creation to avoid config issues
        # with hardcoded safe parameters
        from transformers import AutoImageProcessor
        
        # Get essential parameters with hardcoded defaults
        processor_kwargs = {}
        processor_kwargs["min_pixels"] = min_pixels if min_pixels is not None else 0
        processor_kwargs["max_pixels"] = max_pixels if max_pixels is not None else 460800  # 1280 * 720 // 2
        
        # Add any additional kwargs
        processor_kwargs.update(kwargs)
        
        return AutoImageProcessor.from_pretrained(
            self.ctx.model_config.model,
            revision=self.ctx.model_config.revision,
            trust_remote_code=self.ctx.model_config.trust_remote_code,
            **processor_kwargs,
            size=None,
        )

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None, "video": None}

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 1,
        do_resize: bool = True,
        image_processor: Optional[Qwen2VLImageProcessor],
    ) -> tuple[ImageSize, int]:
        if image_processor is None:
            image_processor = self.get_image_processor()

        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size
        temporal_patch_size = vision_config.temporal_patch_size

        if do_resize:
            resized_height, resized_width = smart_resize(
                height=image_height,
                width=image_width,
                factor=patch_size * merge_size,
                min_pixels=image_processor.min_pixels,
                max_pixels=image_processor.max_pixels,
            )
            preprocessed_size = ImageSize(width=resized_width,
                                          height=resized_height)
        else:
            preprocessed_size = ImageSize(width=image_width,
                                          height=image_height)

        # NOTE: Frames are padded to be divisible by `temporal_patch_size`
        # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L294
        padded_num_frames = num_frames + num_frames % temporal_patch_size

        grid_t = max(padded_num_frames // temporal_patch_size, 1)
        grid_h = preprocessed_size.height // patch_size
        grid_w = preprocessed_size.width // patch_size

        num_patches = grid_t * grid_h * grid_w
        num_vision_tokens = num_patches // (merge_size**2)

        return preprocessed_size, num_vision_tokens

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        image_processor: Optional[Qwen2VLImageProcessor],
    ) -> int:
        _, num_image_tokens = self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            image_processor=image_processor,
        )
        return num_image_tokens

    def get_num_video_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int,
        image_processor: Optional[Qwen2VLImageProcessor],
    ) -> int:
        _, num_video_tokens = self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            num_frames=num_frames,
            image_processor=image_processor,
        )
        return num_video_tokens

    def get_image_size_with_most_features(self) -> ImageSize:
        max_image_size, _ = self._get_vision_info(
            image_width=9999999,
            image_height=9999999,
            image_processor=None,
        )
        return max_image_size

    def get_max_image_tokens(self) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
            image_processor=None,
        )

    def _get_max_video_frames(self, max_tokens: int) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        num_frames = 0

        while True:
            next_num_frames = num_frames + 1
            next_max_tokens = self.get_num_video_tokens(
                image_width=target_width,
                image_height=target_height,
                num_frames=next_num_frames,
                image_processor=None,
            )

            if next_max_tokens > max_tokens:
                break

            num_frames = next_num_frames

        return num_frames

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        max_images = mm_counts.get("image", 0)
        max_videos = mm_counts.get("video", 0)

        max_image_tokens = self.get_max_image_tokens() * max_images
        max_total_frames = self._get_max_video_frames(seq_len -
                                                      max_image_tokens)
        max_frames_per_video = min(max_total_frames // max(max_videos, 1),
                                   _MAX_FRAMES_PER_VIDEO)

        return max(max_frames_per_video, 1)

    def get_max_video_tokens(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_video_tokens(
            image_width=target_width,
            image_height=target_height,
            num_frames=self.get_num_frames_with_most_features(
                seq_len, mm_counts),
            image_processor=None,
        )


class Tarsier2DummyInputsBuilder(BaseDummyInputsBuilder[Tarsier2ProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        tarsier_processor = self.info.get_hf_processor()
        image_token: str = tarsier_processor.image_token

        # In Tarsier2, videos are treated as multi-images, so we use image tokens for both images and video frames
        # The total count will be dynamically calculated based on actual frame counts during processing
        return image_token * (num_images + num_videos)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        target_width, target_height = \
            self.info.get_image_size_with_most_features()
        target_num_frames = \
            self.info.get_num_frames_with_most_features(seq_len, mm_counts)

        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images),
            "video":
            self._get_dummy_videos(
                width=target_width,
                height=target_height,
                num_frames=target_num_frames,
                num_videos=num_videos,
            )
        }


class Tarsier2MultiModalProcessor(BaseMultiModalProcessor[Tarsier2ProcessingInfo]
                                 ):

    def _get_data_parser(self) -> MultiModalDataParser:
        return Tarsier2MultiModalDataParser()

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Use TarsierProcessor directly for processing."""
        # Get TarsierProcessor instance
        processor = self.info.get_hf_processor(**mm_kwargs)
        
        # Create Tarsier message format
        messages = []
        user_content = []
        
        # Process images
        if 'image' in mm_data:
            images = mm_data['image']
            if not isinstance(images, list):
                images = [images]
            
            for img in images:
                if isinstance(img, str):
                    # Image file path
                    user_content.append({
                        "type": "image",
                        "image": {"image_file": img}
                    })
                else:
                    # PIL Image (already loaded)
                    user_content.append({
                        "type": "image", 
                        "image": img
                    })
        
        # Process videos 
        if 'video' in mm_data:
            videos = mm_data['video']
            if not isinstance(videos, list):
                videos = [videos]
                
            for video in videos:
                if isinstance(video, str):
                    # Video file path
                    user_content.append({
                        "type": "video",
                        "video": {"video_file": video}
                    })
                else:
                    # List of PIL Images (frames)
                    user_content.append({
                        "type": "video",
                        "video": video
                    })
        
        # Add text prompt
        user_content.append({
            "type": "text", 
            "text": prompt
        })
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # Use simplified processing config based on tarsier2_default_config.yaml
        # Only max_pixels processing is used (other preprocessing disabled by default)
        processing_config = {
            'max_pixels': 460800,  # 1280 * 720 // 2 from config
            'min_pixels': 0
        }
        
        # Process using TarsierProcessor
        return processor(messages, processing_config=processing_config)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(
            **hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        # In Tarsier2, videos are treated as multi-images, so we only use image tokens 
        image_token_id = vocab[hf_processor.image_token]

        merge_length = image_processor.merge_size**2

        def get_replacement_tarsier2(item_idx: int, modality: str):
            grid_thw = out_mm_kwargs[f"{modality}_grid_thw"][item_idx]
            assert isinstance(grid_thw, torch.Tensor)

            num_tokens = int(grid_thw.prod()) // merge_length
            return [image_token_id] * num_tokens

        return [
            PromptReplacement(
                modality=modality,
                target=[image_token_id],
                replacement=partial(get_replacement_tarsier2,
                                    modality=modality),
            ) for modality in ("image", "video")
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _tarsier2_field_config(hf_inputs)


@MULTIMODAL_REGISTRY.register_processor(Tarsier2MultiModalProcessor,
                                        info=Tarsier2ProcessingInfo,
                                        dummy_inputs=Tarsier2DummyInputsBuilder)
class Tarsier2ForConditionalGeneration(nn.Module, SupportsMultiModal,
                                      SupportsLoRA, SupportsPP):
    """Tarsier2 model for conditional generation.
    
    Key difference from Qwen2VL: In Tarsier2, videos are treated as multi-images
    rather than having separate video processing. All video frames use image tokens
    and are processed through the same vision pipeline as images.
    """

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            # mapping for original checkpoint
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: Qwen2VLConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        self.visual = Qwen2VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=self._maybe_ignore_quant_config(quant_config),
            prefix=maybe_prefix(prefix, "visual"),
        )

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def _maybe_ignore_quant_config(self, quant_config: QuantizationConfig):
        # GPTQ configs do not have a list of ignored modules, however AutoGPTQ
        # seems to avoid vision encoder sections for some models.
        # See: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4
        if isinstance(quant_config, (GPTQConfig, GPTQMarlinConfig)):
            return None
        return quant_config

    def _validate_and_reshape_mm_tensor(self, mm_input: object,
                                        name: str) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. "
                             f"Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            if mm_input.ndim == 2:
                return mm_input
            if mm_input.ndim != 3:
                raise ValueError(f"{name} should be 2D or batched 3D tensor. "
                                 f"Got ndim: {mm_input.ndim} "
                                 f"(shape={mm_input.shape})")
            return torch.concat(list(mm_input))
        else:
            return torch.concat(mm_input)

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[Qwen2VLImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(
                pixel_values, "image pixel values")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return Qwen2VLImagePixelInputs(type="pixel_values",
                                           pixel_values=pixel_values,
                                           image_grid_thw=image_grid_thw)

        if image_embeds is not None:
            image_embeds = self._validate_and_reshape_mm_tensor(
                image_embeds, "image embeds")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")
            return Qwen2VLImageEmbeddingInputs(type="image_embeds",
                                               image_embeds=image_embeds,
                                               image_grid_thw=image_grid_thw)

    def _parse_and_validate_video_input(
            self, **kwargs: object) -> Optional[Qwen2VLVideoInputs]:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            pixel_values_videos = self._validate_and_reshape_mm_tensor(
                pixel_values_videos, "video pixel values")
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw")

            return Qwen2VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )

        if video_embeds is not None:
            video_embeds = self._validate_and_reshape_mm_tensor(
                video_embeds, "video embeds")
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw")

            if not isinstance(video_embeds, torch.Tensor):
                raise ValueError("Incorrect type of video embeddings. "
                                 f"Got type: {type(video_embeds)}")
            return Qwen2VLVideoEmbeddingInputs(type="video_embeds",
                                               video_embeds=video_embeds,
                                               video_grid_thw=video_grid_thw)

    def _process_image_input(
            self, image_input: Qwen2VLImageInputs) -> tuple[torch.Tensor, ...]:

        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"]
        else:
            pixel_values = image_input["pixel_values"]
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return image_embeds.split(sizes.tolist())

    def _process_video_input(
            self, video_input: Qwen2VLVideoInputs) -> tuple[torch.Tensor, ...]:

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"]
        else:
            pixel_values_videos = video_input["pixel_values_videos"]
            video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)

        # Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return video_embeds.split(sizes.tolist())

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("pixel_values",
                             "image_embeds") and "images" not in modalities:
                modalities["images"] = self._parse_and_validate_image_input(
                    **kwargs)
            if input_key in ("pixel_values_videos",
                             "video_embeds") and "videos" not in modalities:
                modalities["videos"] = self._parse_and_validate_video_input(
                    **kwargs)

        return modalities

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(self,
                                  **kwargs: object) -> MultiModalEmbeddings:

        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in modalities:
            if modality == "images":
                image_input = modalities["images"]
                vision_embeddings = self._process_image_input(image_input)
                multimodal_embeddings += vision_embeddings
            if modality == "videos":
                video_input = modalities["videos"]
                video_embeddings = self._process_video_input(video_input)
                multimodal_embeddings += video_embeddings

        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            # In Tarsier2, videos are treated as multi-images, so we only use image_token_id
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.config.image_token_id)
        return inputs_embeds

    def get_input_embeddings_v0(
        self,
        input_ids: torch.Tensor,
        image_input: Optional[Qwen2VLImagePixelInputs] = None,
        video_input: Optional[Qwen2VLVideoPixelInputs] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.get_input_embeddings(input_ids)
        if image_input is not None:
            image_embeds = self._process_image_input(image_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                image_embeds,
                placeholder_token_id=self.config.image_token_id,
            )

        if video_input is not None:
            video_embeds = self._process_video_input(video_input)
            # In Tarsier2, videos are treated as multi-images, so we use image_token_id
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                video_embeds,
                placeholder_token_id=self.config.image_token_id,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Run forward pass for Tarsier2.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Tarsier2
                models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
            pixel_values: Pixel values to be fed to a model.
                `None` if no images are passed.
            image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in LLM.
                `None` if no images are passed.
            pixel_values_videos: Pixel values of videos to be fed to a model.
                `None` if no videos are passed. Note: In Tarsier2, video frames
                are processed as multi-images using image tokens.
            video_grid_thw: Tensor `(n_videos, 3)` of video 3D grid in LLM.
                `None` if no videos are passed.
        """

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner from
        # `get_multimodal_embeddings` and `get_input_embeddings`, this
        # condition is only for v0 compatibility.
        elif inputs_embeds is None:
            image_input = self._parse_and_validate_image_input(**kwargs)
            video_input = self._parse_and_validate_video_input(**kwargs)

            if image_input is None and video_input is None:
                inputs_embeds = None
            else:
                if uses_mrope(self.config):
                    assert positions.ndim == 2 and positions.size(0) == 3, (
                        "multimodal section rotary embedding requires "
                        f"(3, seq_len) positions, but got {positions.size()}")
                inputs_embeds = self.get_input_embeddings_v0(
                    input_ids,
                    image_input=image_input,
                    video_input=video_input)
                input_ids = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:

        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="visual.merger.",
            tower_model="visual.",
        )
