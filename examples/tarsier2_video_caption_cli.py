#!/usr/bin/env python3
"""
Command-line interface for video captioning using Tarsier2 model with vLLM.

This script provides a convenient CLI for generating video captions using the 
Tarsier2 multimodal model through vLLM. Tarsier2 treats videos as multi-images,
processing video frames as individual images using image tokens.

Usage examples:
    python examples/tarsier2_video_caption_cli.py \\
        --model /path/to/tarsier2/model \\
        --video video.mp4 \\
        --prompt "Describe this video in detail."
        
    python examples/tarsier2_video_caption_cli.py \\
        --model /path/to/tarsier2/model \\
        --video /path/to/videos/ \\
        --batch-size 4
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional
from PIL import Image
import cv2

from vllm import LLM, SamplingParams


def extract_video_frames(
    video_path: str, 
    max_frames: int = 8,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
) -> List[Image.Image]:
    """
    Extract frames from video file with optional time range.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to extract
        start_time: Start time in seconds (optional)
        end_time: End time in seconds (optional)
        
    Returns:
        List of PIL Image objects
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Calculate frame range
    start_frame = int(start_time * fps) if start_time else 0
    end_frame = int(end_time * fps) if end_time else total_frames
    end_frame = min(end_frame, total_frames)
    
    available_frames = end_frame - start_frame
    
    # Calculate frame indices to sample
    if available_frames <= max_frames:
        frame_indices = list(range(start_frame, end_frame))
    else:
        # Uniform sampling
        step = available_frames / max_frames
        frame_indices = [int(start_frame + i * step) for i in range(max_frames)]
    
    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    
    if not frames:
        raise ValueError(f"No frames extracted from {video_path}")
    
    print(f"Extracted {len(frames)} frames from {video_path} "
          f"(duration: {duration:.1f}s)")
    return frames


def find_video_files(path: str) -> List[str]:
    """Find all video files in the given path."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    
    if os.path.isfile(path):
        return [path] if any(path.lower().endswith(ext) for ext in video_extensions) else []
    
    elif os.path.isdir(path):
        video_files = []
        for filename in os.listdir(path):
            if any(filename.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(path, filename))
        return sorted(video_files)
    
    return []


def create_tarsier2_prompt(instruction: str) -> str:
    """
    Create a properly formatted prompt for Tarsier2.
    Based on the run_tarsier function in vLLM examples.
    """
    return f"USER: <image>\n{instruction} ASSISTANT:"


def generate_caption(
    llm: LLM,
    video_frames: List[Image.Image],
    prompt: str,
    sampling_params: SamplingParams
) -> str:
    """Generate caption for video frames using Tarsier2."""
    
    # Prepare multimodal data
    # Key insight: Tarsier2 treats videos as multi-images
    mm_data = {"video": video_frames}
    
    # Prepare input for vLLM
    inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data
    }
    
    # Generate
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    
    if outputs and len(outputs) > 0:
        return outputs[0].outputs[0].text.strip()
    else:
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="Video captioning with Tarsier2 and vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video
  python tarsier2_video_caption_cli.py --model /path/to/model --video video.mp4
  
  # Directory of videos  
  python tarsier2_video_caption_cli.py --model /path/to/model --video /videos/
  
  # Custom prompt and parameters
  python tarsier2_video_caption_cli.py \\
    --model /path/to/model \\
    --video video.mp4 \\
    --prompt "What activities are shown in this video?" \\
    --max-frames 16 \\
    --temperature 0.3 \\
    --max-tokens 256
        """
    )
    
    # Model arguments
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to Tarsier2 model directory"
    )
    
    # Input arguments
    parser.add_argument(
        "--video", "-v", 
        required=True,
        help="Path to video file or directory"
    )
    
    parser.add_argument(
        "--prompt", "-p",
        default="Describe this video in detail.",
        help="Caption generation prompt"
    )
    
    # Video processing
    parser.add_argument(
        "--max-frames",
        type=int,
        default=8,
        help="Maximum frames to extract per video"
    )
    
    parser.add_argument(
        "--start-time",
        type=float,
        help="Start time in seconds (optional)"
    )
    
    parser.add_argument(
        "--end-time", 
        type=float,
        help="End time in seconds (optional)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float, 
        default=0.0,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling"
    )
    
    # vLLM engine parameters
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Model context length"
    )
    
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing multiple videos"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        print(f"Error: Model path not found: {args.model}")
        sys.exit(1)
        
    if not os.path.exists(args.video):
        print(f"Error: Video path not found: {args.video}")
        sys.exit(1)
    
    # Find video files
    video_files = find_video_files(args.video)
    if not video_files:
        print(f"Error: No video files found in: {args.video}")
        sys.exit(1)
    
    print(f"Found {len(video_files)} video(s) to process")
    
    # Initialize vLLM engine
    print(f"Loading Tarsier2 model: {args.model}")
    try:
        llm = LLM(
            model=args.model,
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            limit_mm_per_prompt={"video": 1}  # Limit one video per prompt
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        sys.exit(1)
    
    # Setup generation parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p
    )
    
    # Format prompt for Tarsier2
    formatted_prompt = create_tarsier2_prompt(args.prompt)
    
    # Process videos
    print(f"\n{'='*60}")
    print("Processing Videos")
    print(f"{'='*60}")
    
    results = []
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] {os.path.basename(video_path)}")
        
        try:
            # Extract frames
            frames = extract_video_frames(
                video_path,
                max_frames=args.max_frames,
                start_time=args.start_time,
                end_time=args.end_time
            )
            
            # Generate caption
            print("Generating caption...")
            caption = generate_caption(
                llm, frames, formatted_prompt, sampling_params
            )
            
            if caption:
                print(f"✓ Caption: {caption}")
                results.append({
                    'video': video_path,
                    'caption': caption,
                    'success': True
                })
            else:
                print("✗ No caption generated")
                results.append({
                    'video': video_path, 
                    'caption': "",
                    'success': False
                })
                
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({
                'video': video_path,
                'caption': "",
                'success': False
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"Total videos: {total}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {total - successful}")
    
    # Show all results
    if successful > 0:
        print(f"\n📝 Generated Captions:")
        print("-" * 40)
        for result in results:
            if result['success']:
                video_name = os.path.basename(result['video'])
                print(f"\n🎬 {video_name}")
                print(f"   {result['caption']}")
    
    print(f"\n✓ Processing complete!")


if __name__ == "__main__":
    main() 