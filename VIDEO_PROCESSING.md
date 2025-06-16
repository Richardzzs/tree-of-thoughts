# Video Processing Integration

This document explains how to use the enhanced Tree of Thoughts (ToT) agent with video processing capabilities.

## Overview

The Tree of Thoughts repository has been enhanced with video processing functionality using Qwen VL models. This integration allows you to:

1. Process and analyze video content
2. Combine video analysis with structured Tree of Thoughts reasoning
3. Extract features and insights from videos using AI models

## Components

### VideoProcessor Class

The `VideoProcessor` class handles video content processing:

```python
from tree_of_thoughts import VideoProcessor

# Initialize processor
processor = VideoProcessor(
    api_base="http://localhost:8000/v1",
    model_name="Qwen/Qwen2.5-VL-32B-Instruct"
)

# Analyze video
result = processor.analyze_video_features(video_url)
```

### Enhanced TotAgent Class

The `TotAgent` class now supports video processing when enabled:

```python
from tree_of_thoughts import TotAgent

# Initialize with video capabilities
agent = TotAgent(
    enable_video_processing=True,
    video_api_base="http://localhost:8000/v1"
)

# Analyze video with ToT reasoning
result = agent.analyze_video(video_url, "分析视频中的商品特点")
```

## Setup Requirements

### Dependencies

The following packages are required for video processing:

- `numpy`: For numerical operations
- `Pillow`: For image processing
- `openai`: For API communication
- `qwen-vl-utils`: For video frame extraction

Install all dependencies:

```bash
pip install -r requirements.txt
```

### vLLM Server Setup

You need to run a vLLM server with Qwen VL model support:

```bash
# Example vLLM server startup
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-32B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

### Environment Variables

Set your OpenAI API key for text-based ToT reasoning:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## Usage Examples

### Basic Video Processing

```python
from tree_of_thoughts import VideoProcessor

processor = VideoProcessor()
result = processor.process_video(
    video_url="path/to/video.mp4",
    text_prompt="请用表格总结一下视频中的商品特点",
    fps=3.0
)
print(result["content"])
```

### Combined Video + ToT Analysis

```python
from tree_of_thoughts import TotAgent

agent = TotAgent(enable_video_processing=True)
result = agent.analyze_video(
    video_url="path/to/video.mp4",
    prompt="分析视频内容并提供结构化思考"
)

# Access video analysis
video_content = result["video_analysis"]["content"]

# Access ToT reasoning
tot_thoughts = result["tot_reasoning"]

# Access combined results
combined = result["combined_result"]
```

### Custom Video Analysis Parameters

```python
result = processor.process_video(
    video_url="path/to/video.mp4",
    text_prompt="Custom analysis prompt",
    total_pixels=20480 * 28 * 28,  # Adjust for quality
    min_pixels=16 * 28 * 2,
    fps=2.0,  # Frames per second
    system_message="Custom system prompt"
)
```

## Configuration Options

### VideoProcessor Configuration

- `api_key`: API key for vLLM server (default: "EMPTY")
- `api_base`: Base URL for vLLM API server
- `model_name`: Name of the video model to use

### TotAgent Video Configuration

- `enable_video_processing`: Enable/disable video capabilities
- `video_api_base`: vLLM server URL
- `video_model_name`: Video model name

### Video Processing Parameters

- `total_pixels`: Total pixels for video processing (affects quality)
- `min_pixels`: Minimum pixels for processing
- `fps`: Frames per second to extract from video
- `system_message`: System prompt for the model

## API Reference

### VideoProcessor Methods

#### `process_video(video_url, text_prompt, **kwargs)`
Process a video with a text prompt.

**Parameters:**
- `video_url` (str): URL or path to video file
- `text_prompt` (str): Analysis prompt
- `total_pixels` (int): Total pixels for processing
- `min_pixels` (int): Minimum pixels
- `fps` (float): Frames per second

**Returns:**
Dictionary with response, content, and video_kwargs

#### `analyze_video_features(video_url)`
Convenience method for product feature analysis.

### TotAgent Video Methods

#### `analyze_video(video_url, prompt, **kwargs)`
Combine video analysis with ToT reasoning.

**Returns:**
Dictionary with video_analysis, tot_reasoning, and combined_result

#### `analyze_video_features(video_url)`
Quick product feature analysis with ToT reasoning.

## Troubleshooting

### Common Issues

1. **Import Error for qwen_vl_utils**
   ```bash
   pip install qwen-vl-utils
   ```

2. **vLLM Server Not Running**
   - Ensure vLLM server is running on the specified port
   - Check the API base URL configuration

3. **OpenAI API Key Missing**
   - Set OPENAI_API_KEY environment variable for ToT reasoning

4. **Memory Issues with Large Videos**
   - Reduce `total_pixels` parameter
   - Lower the `fps` value
   - Use shorter video clips

### Performance Optimization

1. **Adjust Video Parameters**
   - Lower `fps` for faster processing
   - Reduce `total_pixels` for lower memory usage
   - Use appropriate `min_pixels` for quality threshold

2. **Server Configuration**
   - Ensure adequate GPU memory for vLLM server
   - Consider using model quantization for efficiency

## Integration with Existing Code

The video processing functionality is designed to be non-intrusive:

- Existing ToT functionality remains unchanged
- Video processing is optional and disabled by default
- Backward compatibility is maintained

You can gradually adopt video processing features without affecting existing workflows.

## Examples

See `examples/video_processing_example.py` for comprehensive usage examples covering:

- Basic video processing
- Combined ToT + video analysis
- Custom parameter configurations
- Error handling patterns
