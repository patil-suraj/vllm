#!/usr/bin/env python3
"""
Simple example for video captioning using Tarsier2 model with vLLM.

This demonstrates the basic usage of Tarsier2 multimodal model for video understanding
using vLLM's inference engine. The key insight is that Tarsier2 treats videos as 
multi-images, so video frames are processed using image tokens.
"""

import os
import sys
from typing import List
from PIL import Image
import cv2

# vLLM imports
from vllm import LLM, SamplingParams
from vllm.multimodal import MultiModalDataDict


def extract_video_frames(video_path: str, max_frames: int = 8) -> List[Image.Image]:
    """Extract frames from video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(i * total_frames / max_frames) for i in range(max_frames)]
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    return frames


def main():
    # Example usage
    model_path = "/blob/raw/huggingface_repos/Tarsier2-Recap-7b"  # Update this path
    video_path = "-q0kwjzxOlE_000004.mp4"  # Update this path
    
    # Check if paths exist
    if not os.path.exists(model_path):
        print(f"Please update model_path to point to your Tarsier2 model: {model_path}")
        return
    
    if not os.path.exists(video_path):
        print(f"Please update video_path to point to your video file: {video_path}")
        return
    
    # Initialize vLLM engine with Tarsier2
    print("Initializing vLLM with Tarsier2...")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,  # Required for Tarsier2
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    )
    
    # Extract video frames
    print("Extracting video frames...")
    video_frames = extract_video_frames(video_path, max_frames=8)
    print(f"Extracted {len(video_frames)} frames")
    
    # Prepare the prompt
    prompt = "Describe this video in detail."
    
    # Create multimodal data
    # Key insight: Tarsier2 treats videos as multi-images
    mm_data = MultiModalDataDict({
        "video": video_frames  # Pass frames as video data
    })
    
    # Set up generation parameters
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic
        max_tokens=512,
        top_p=1.0,
    )
    
    # Generate caption
    print("Generating caption...")
    
    # Prepare input in the correct format for vLLM
    inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data
    }
    
    outputs = llm.generate(
        inputs,
        sampling_params=sampling_params
    )
    
    # Display result
    if outputs and len(outputs) > 0:
        caption = outputs[0].outputs[0].text.strip()
        print(f"\n🎬 Video: {os.path.basename(video_path)}")
        print(f"📝 Caption: {caption}")
    else:
        print("No caption generated")


if __name__ == "__main__":
    main() 