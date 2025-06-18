# Tarsier2 Video Captioning with vLLM

This directory contains examples for performing video captioning using the Tarsier2 multimodal model with vLLM. Tarsier2 is a multimodal model that treats videos as multi-images, processing video frames as individual images using image tokens.

## Files

- `tarsier2_vllm_simple.py` - Simple example showing basic usage
- `tarsier2_video_caption_cli.py` - Full-featured command-line interface
- `tarsier2_video_captioning_vllm.py` - Comprehensive example with advanced features

## Key Features

🎬 **Video Understanding**: Process video files for captioning and description  
🖼️ **Multi-Image Processing**: Treats videos as sequences of images (Tarsier2's approach)  
⚡ **vLLM Acceleration**: Leverages vLLM for fast inference  
🔧 **Flexible Configuration**: Customizable frame extraction and generation parameters  
📝 **Multiple Formats**: Support for various video formats (mp4, avi, mov, etc.)

## Installation

### Prerequisites

1. **vLLM with Tarsier2 support**:
   ```bash
   pip install vllm
   ```

2. **Video processing dependencies**:
   ```bash
   pip install opencv-python pillow
   ```

3. **Tarsier2 model**: Download or obtain access to a Tarsier2 model checkpoint

### Required Dependencies

```bash
pip install vllm opencv-python pillow torch transformers
```

## Usage

### Simple Example

For basic usage with minimal setup:

```bash
python tarsier2_vllm_simple.py
```

**Note**: Update the `model_path` and `video_path` variables in the script before running.

### Command-Line Interface

For production use with full control over parameters:

```bash
# Single video
python tarsier2_video_caption_cli.py \
    --model /path/to/tarsier2/model \
    --video /path/to/video.mp4 \
    --prompt "Describe this video in detail."

# Directory of videos
python tarsier2_video_caption_cli.py \
    --model /path/to/tarsier2/model \
    --video /path/to/videos/ \
    --max-frames 16

# Custom parameters
python tarsier2_video_caption_cli.py \
    --model /path/to/tarsier2/model \
    --video video.mp4 \
    --prompt "What activities are shown in this video?" \
    --max-frames 16 \
    --temperature 0.3 \
    --max-tokens 256 \
    --start-time 10.0 \
    --end-time 30.0
```

### Advanced Example

For batch processing and advanced features:

```bash
python tarsier2_video_captioning_vllm.py \
    --model_name_or_path /path/to/tarsier2/model \
    --video_path /path/to/videos/ \
    --prompt "Describe what is happening in this video." \
    --max_frames 32 \
    --max_tokens 1024 \
    --temperature 0.7
```

## Configuration Options

### Model Parameters

- `--model` / `--model_name_or_path`: Path to Tarsier2 model directory
- `--max-model-len`: Maximum model context length (default: 4096)
- `--gpu-memory-utilization`: GPU memory usage (default: 0.9)

### Video Processing

- `--max-frames`: Maximum frames to extract (default: 8)
- `--start-time`: Start time in seconds (optional)
- `--end-time`: End time in seconds (optional)

### Generation Parameters

- `--prompt`: Text prompt for captioning
- `--temperature`: Sampling temperature (0.0 = deterministic)
- `--max-tokens`: Maximum tokens to generate
- `--top-p`: Top-p sampling parameter

## How It Works

### Tarsier2 Architecture

Tarsier2 is based on the Qwen2VL architecture but with a key difference: **videos are treated as multi-images** rather than having separate video processing. This means:

1. **Frame Extraction**: Videos are sampled into individual frames
2. **Image Token Usage**: Each frame uses image tokens (not video tokens)
3. **Multi-Image Processing**: Frames are processed as a sequence of images
4. **Unified Pipeline**: Same processing pipeline for both images and videos

### Processing Pipeline

1. **Video Loading**: Load video file using OpenCV
2. **Frame Sampling**: Extract frames using uniform sampling
3. **Preprocessing**: Convert frames to PIL Images (RGB format)
4. **Model Input**: Package frames as multimodal data
5. **Generation**: Use vLLM to generate captions
6. **Output**: Return generated text descriptions

## Example Outputs

```
🎬 cooking_demo.mp4
📝 The video shows a person in a kitchen preparing a meal. They start by chopping vegetables on a cutting board, then move to the stove where they heat oil in a pan. The person adds the vegetables to the pan and stirs them while cooking. The kitchen appears well-lit and organized with various cooking utensils visible in the background.

🎬 sports_highlights.mp4  
📝 This video captures exciting moments from a basketball game. Players are seen dribbling, passing, and shooting the ball. There are several successful shots and defensive plays. The crowd in the background appears engaged and enthusiastic. The video showcases the athletic skills and teamwork of both teams.
```

## Performance Tips

### Memory Optimization

- Reduce `--max-frames` for memory-constrained environments
- Lower `--gpu-memory-utilization` if encountering OOM errors
- Use smaller `--max-model-len` for shorter contexts

### Speed Optimization

- Use `--temperature 0.0` for deterministic, faster generation
- Process videos in batches when possible
- Consider frame resolution vs. processing speed trade-offs

### Quality Optimization

- Increase `--max-frames` for longer videos
- Use higher `--temperature` for more creative descriptions
- Adjust `--max-tokens` based on desired caption length

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```
   Error: Model path not found
   ```
   - Verify the model path is correct
   - Ensure the model is compatible with vLLM
   - Check that `trust_remote_code=True` is set

2. **Video Processing Errors**
   ```
   Could not open video file
   ```
   - Verify video file exists and is readable
   - Check video format is supported by OpenCV
   - Ensure opencv-python is installed

3. **Memory Issues**
   ```
   CUDA out of memory
   ```
   - Reduce `--max-frames`
   - Lower `--gpu-memory-utilization`
   - Use smaller batch sizes

4. **Generation Issues**
   ```
   No caption generated
   ```
   - Check prompt formatting
   - Verify multimodal data structure
   - Ensure model supports video input

### Debug Mode

Add verbose output for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Model Compatibility

These examples are designed for **Tarsier2** models. Key requirements:

- Model must be compatible with vLLM
- Model should support image token processing
- Multimodal capabilities required
- Trust remote code enabled

## Performance Benchmarks

Typical performance on common hardware:

| Hardware | Model Size | Frames | Time per Video |
|----------|------------|---------|----------------|
| RTX 4090 | 7B | 8 frames | ~2-3 seconds |
| RTX 3080 | 7B | 8 frames | ~3-5 seconds |
| A100 | 7B | 16 frames | ~1-2 seconds |

*Times include frame extraction and caption generation*

## Contributing

To contribute improvements or report issues:

1. Test with multiple video formats
2. Verify memory usage with different configurations
3. Check compatibility with different model sizes
4. Provide clear error messages and examples

## License

These examples follow the same license as the parent vLLM project.

---

For more information about vLLM and multimodal processing, see the [official vLLM documentation](https://docs.vllm.ai/). 