#!/usr/bin/env python3
"""
Example script for video captioning using Tarsier2 model with vLLM.

This script demonstrates how to use the Tarsier2 multimodal model for video captioning
through vLLM's inference engine. The Tarsier2 model treats videos as multi-images,
processing video frames as individual images using image tokens.

Usage:
    python examples/tarsier2_video_captioning_vllm.py \
        --model_name_or_path /path/to/tarsier2/model \
        --video_path /path/to/video.mp4 \
        --prompt "Describe this video in detail."

Requirements:
    - vLLM installed with Tarsier2 support
    - Video files in supported formats (mp4, avi, mov, etc.)
    - GPU with sufficient memory for the model
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image

# Add the parent directory to the path to import vllm
sys.path.insert(0, str(Path(__file__).parent.parent))

from vllm import LLM, SamplingParams
from vllm.multimodal import MultiModalDataDict


def load_video_frames(video_path: str, max_frames: int = 16) -> List[Image.Image]:
    """
    Load video frames from a video file.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to extract
        
    Returns:
        List of PIL Image objects representing video frames
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for video processing. Install with: pip install opencv-python")
    
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to sample uniformly
    if total_frames <= max_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = [int(i * total_frames / max_frames) for i in range(max_frames)]
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
        else:
            print(f"Warning: Could not read frame {frame_idx}")
    
    cap.release()
    
    if not frames:
        raise ValueError(f"No frames could be extracted from video: {video_path}")
    
    print(f"Extracted {len(frames)} frames from video: {video_path}")
    return frames


def create_tarsier2_prompt(instruction: str) -> str:
    """
    Create a properly formatted prompt for Tarsier2 model.
    
    Args:
        instruction: The user instruction/question about the video
        
    Returns:
        Formatted prompt string
    """
    # Tarsier2 uses a simple format where the instruction follows the video content
    return instruction


def setup_vllm_engine(model_name_or_path: str, **kwargs) -> LLM:
    """
    Initialize the vLLM engine with Tarsier2 model.
    
    Args:
        model_name_or_path: Path to the Tarsier2 model
        **kwargs: Additional arguments for LLM initialization
        
    Returns:
        Initialized LLM engine
    """
    # Default engine arguments for Tarsier2
    engine_args = {
        "model": model_name_or_path,
        "task": "generate",
        "trust_remote_code": True,  # Required for custom models
        "max_model_len": kwargs.get("max_model_len", 4096),
        "gpu_memory_utilization": kwargs.get("gpu_memory_utilization", 0.9),
        "dtype": kwargs.get("dtype", "bfloat16"),
    }
    
    # Update with any additional kwargs
    engine_args.update(kwargs)
    
    print("Initializing vLLM engine with Tarsier2...")
    print(f"Model: {model_name_or_path}")
    print(f"Engine args: {engine_args}")
    
    try:
        llm = LLM(**engine_args)
        print("✓ vLLM engine initialized successfully!")
        return llm
    except Exception as e:
        print(f"✗ Failed to initialize vLLM engine: {e}")
        raise


def generate_video_caption(
    llm: LLM,
    video_frames: List[Image.Image],
    prompt: str,
    sampling_params: Optional[SamplingParams] = None
) -> str:
    """
    Generate a caption for video frames using the Tarsier2 model.
    
    Args:
        llm: Initialized vLLM engine
        video_frames: List of video frames as PIL Images  
        prompt: Text prompt/instruction
        sampling_params: Sampling parameters for generation
        
    Returns:
        Generated caption text
    """
    if sampling_params is None:
        sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic generation
            max_tokens=512,
            top_p=1.0,
            stop_token_ids=None,
        )
    
    # Prepare multimodal data - in Tarsier2, videos are treated as multi-images
    # So we pass the video frames as a list of images
    mm_data: MultiModalDataDict = {
        "video": video_frames,  # Tarsier2 will process these as multi-images
    }
    
    print(f"Generating caption for {len(video_frames)} video frames...")
    print(f"Prompt: {prompt}")
    
    try:
        # Generate response using vLLM
        outputs = llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": mm_data,
            },
            sampling_params=sampling_params
        )
        
        if outputs and len(outputs) > 0:
            generated_text = outputs[0].outputs[0].text.strip()
            print("✓ Caption generated successfully!")
            return generated_text
        else:
            raise ValueError("No output generated")
            
    except Exception as e:
        print(f"✗ Failed to generate caption: {e}")
        raise


def process_single_video(
    llm: LLM,
    video_path: str,
    prompt: str,
    max_frames: int = 16,
    sampling_params: Optional[SamplingParams] = None
) -> str:
    """
    Process a single video file and generate its caption.
    
    Args:
        llm: Initialized vLLM engine
        video_path: Path to video file
        prompt: Caption generation prompt
        max_frames: Maximum frames to extract
        sampling_params: Generation parameters
        
    Returns:
        Generated caption
    """
    print(f"\n{'='*60}")
    print(f"Processing video: {video_path}")
    print(f"{'='*60}")
    
    # Load video frames
    try:
        video_frames = load_video_frames(video_path, max_frames=max_frames)
    except Exception as e:
        print(f"✗ Failed to load video frames: {e}")
        return ""
    
    # Generate caption
    try:
        caption = generate_video_caption(llm, video_frames, prompt, sampling_params)
        
        print(f"\n🎬 Video: {os.path.basename(video_path)}")
        print(f"📝 Caption: {caption}")
        
        return caption
        
    except Exception as e:
        print(f"✗ Failed to generate caption for {video_path}: {e}")
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="Video captioning using Tarsier2 model with vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Caption a single video
    python examples/tarsier2_video_captioning_vllm.py \\
        --model_name_or_path /path/to/tarsier2/model \\
        --video_path /path/to/video.mp4 \\
        --prompt "Describe this video in detail."
    
    # Caption all videos in a directory
    python examples/tarsier2_video_captioning_vllm.py \\
        --model_name_or_path /path/to/tarsier2/model \\
        --video_path /path/to/videos/ \\
        --prompt "What is happening in this video?"
    
    # Custom generation parameters
    python examples/tarsier2_video_captioning_vllm.py \\
        --model_name_or_path /path/to/tarsier2/model \\
        --video_path /path/to/video.mp4 \\
        --prompt "Describe the video" \\
        --max_frames 32 \\
        --max_tokens 1024 \\
        --temperature 0.7
        """
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to the Tarsier2 model directory"
    )
    
    # Input arguments
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to video file or directory containing videos"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this video in detail.",
        help="Prompt for video captioning"
    )
    
    # Video processing arguments
    parser.add_argument(
        "--max_frames",
        type=int,
        default=16,
        help="Maximum number of frames to extract from each video"
    )
    
    # Generation arguments
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for deterministic)"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter"
    )
    
    # Engine arguments
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=4096,
        help="Maximum model context length"
    )
    
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0-1.0)"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model data type"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video_path):
        print(f"✗ Video path does not exist: {args.video_path}")
        sys.exit(1)
    
    # Setup sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        stop_token_ids=None,
    )
    
    # Initialize vLLM engine
    try:
        llm = setup_vllm_engine(
            args.model_name_or_path,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
        )
    except Exception as e:
        print(f"✗ Failed to setup vLLM engine: {e}")
        sys.exit(1)
    
    # Determine if input is a file or directory
    video_paths = []
    if os.path.isfile(args.video_path):
        video_paths = [args.video_path]
    elif os.path.isdir(args.video_path):
        # Find all video files in directory
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
        for filename in os.listdir(args.video_path):
            if any(filename.lower().endswith(ext) for ext in video_extensions):
                video_paths.append(os.path.join(args.video_path, filename))
        
        if not video_paths:
            print(f"✗ No video files found in directory: {args.video_path}")
            sys.exit(1)
    else:
        print(f"✗ Invalid video path: {args.video_path}")
        sys.exit(1)
    
    print(f"Found {len(video_paths)} video(s) to process")
    
    # Process each video
    results = []
    for i, video_path in enumerate(video_paths, 1):
        print(f"\n[{i}/{len(video_paths)}] Processing: {os.path.basename(video_path)}")
        
        try:
            caption = process_single_video(
                llm=llm,
                video_path=video_path,
                prompt=args.prompt,
                max_frames=args.max_frames,
                sampling_params=sampling_params
            )
            
            results.append({
                'video_path': video_path,
                'caption': caption,
                'success': bool(caption)
            })
            
        except KeyboardInterrupt:
            print("\n⚠️  Processing interrupted by user")
            break
        except Exception as e:
            print(f"✗ Error processing {video_path}: {e}")
            results.append({
                'video_path': video_path,
                'caption': "",
                'success': False
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"Total videos: {total}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {total - successful}")
    
    if successful > 0:
        print(f"\n✓ Video captioning completed!")
        
        # Show results
        for result in results:
            if result['success']:
                print(f"\n🎬 {os.path.basename(result['video_path'])}")
                print(f"📝 {result['caption']}")
    else:
        print(f"\n✗ No videos were successfully processed")
        sys.exit(1)


if __name__ == "__main__":
    main() 