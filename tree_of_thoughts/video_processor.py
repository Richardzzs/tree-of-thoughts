"""
Video processing module for Tree of Thoughts.
Provides functionality to process video content using Qwen VL models.
"""

import base64
import numpy as np
from PIL import Image
from io import BytesIO
from openai import OpenAI
from typing import List, Dict, Any, Tuple, Optional
import os
from dotenv import load_dotenv

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError("qwen_vl_utils is required for video processing. Please install it with: pip install qwen_vl_utils")

load_dotenv()


class VideoProcessor:
    """
    A class for processing video content using Qwen VL models via vLLM API server.
    
    This class handles video frame extraction, encoding, and communication with
    the vLLM API server for video understanding tasks.
    """
    
    def __init__(
        self,
        api_key: str = "EMPTY",
        api_base: str = "http://localhost:8000/v1",
        model_name: str = "Qwen/Qwen2.5-VL-32B-Instruct"
    ):
        """
        Initialize the VideoProcessor.
        
        Args:
            api_key: OpenAI API key (default "EMPTY" for vLLM)
            api_base: Base URL for the vLLM API server
            model_name: Name of the model to use
        """
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )
    
    def prepare_message_for_vllm(self, content_messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Prepare video messages for vLLM processing.
        
        The frame extraction logic for videos in `vLLM` differs from that of `qwen_vl_utils`.
        Here, we utilize `qwen_vl_utils` to extract video frames, with the `media_type` of 
        the video explicitly set to `video/jpeg`. By doing so, vLLM will no longer attempt 
        to extract frames from the input base64-encoded images.
        
        Args:
            content_messages: List of message dictionaries containing video content
            
        Returns:
            Tuple of (processed_messages, video_kwargs)
        """
        vllm_messages, fps_list = [], []
        
        for message in content_messages:
            message_content_list = message["content"]
            if not isinstance(message_content_list, list):
                vllm_messages.append(message)
                continue

            new_content_list = []
            for part_message in message_content_list:
                if 'video' in part_message:
                    video_message = [{'content': [part_message]}]
                    image_inputs, video_inputs, video_kwargs = process_vision_info(
                        video_message, return_video_kwargs=True
                    )
                    assert video_inputs is not None, "video_inputs should not be None"
                    video_input = (video_inputs.pop()).permute(0, 2, 3, 1).numpy().astype(np.uint8)
                    fps_list.extend(video_kwargs.get('fps', []))

                    # Encode images with base64
                    base64_frames = []
                    for frame in video_input:
                        img = Image.fromarray(frame)
                        output_buffer = BytesIO()
                        img.save(output_buffer, format="jpeg")
                        byte_data = output_buffer.getvalue()
                        base64_str = base64.b64encode(byte_data).decode("utf-8")
                        base64_frames.append(base64_str)

                    part_message = {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_frames)}"}
                    }
                new_content_list.append(part_message)
            
            message["content"] = new_content_list
            vllm_messages.append(message)
        
        return vllm_messages, {'fps': fps_list}
    
    def process_video(
        self, 
        video_url: str, 
        text_prompt: str,
        total_pixels: int = 20480 * 28 * 28,
        min_pixels: int = 16 * 28 * 2,
        fps: float = 3.0,
        system_message: str = "You are a helpful assistant."
    ) -> Dict[str, Any]:
        """
        Process a video with a text prompt using the Qwen VL model.
        
        Args:
            video_url: URL or path to the video file
            text_prompt: Text prompt describing what to analyze in the video
            total_pixels: Total pixels for video processing
            min_pixels: Minimum pixels for video processing
            fps: Frames per second for video processing
            system_message: System message for the model
            
        Returns:
            Dictionary containing the model's response
        """
        video_messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": [
                {"type": "text", "text": text_prompt},
                {
                    "type": "video",
                    "video": video_url,
                    "total_pixels": total_pixels,
                    "min_pixels": min_pixels,
                    'fps': fps
                }]
            },
        ]
        
        # Prepare messages for vLLM
        processed_messages, video_kwargs = self.prepare_message_for_vllm(video_messages)
        
        # Get chat completion
        chat_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=processed_messages,
            extra_body={
                "mm_processor_kwargs": video_kwargs
            }
        )
        
        return {
            "response": chat_response,
            "content": chat_response.choices[0].message.content if chat_response.choices else None,
            "video_kwargs": video_kwargs
        }
    
    def analyze_video_features(self, video_url: str) -> Dict[str, Any]:
        """
        Analyze and summarize features of a video using a table format.
        
        Args:
            video_url: URL or path to the video file
            
        Returns:
            Dictionary containing the analysis results
        """
        prompt = "请用表格总结一下视频中的商品特点"
        return self.process_video(video_url, prompt)


# Example usage function
def example_video_analysis():
    """
    Example function demonstrating how to use the VideoProcessor.
    """
    # Initialize the processor
    processor = VideoProcessor()
    
    # Example video URL
    video_url = "https://duguang-labelling.oss-cn-shanghai.aliyuncs.com/qiansun/video_ocr/videos/50221078283.mp4"
    
    # Analyze video features
    result = processor.analyze_video_features(video_url)
    
    print("Video analysis result:")
    print(result["content"])
    
    return result


if __name__ == "__main__":
    example_video_analysis()
