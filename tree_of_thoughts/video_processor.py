"""
思维树视频处理模块
为思维树提供使用Qwen VL模型处理视频内容的功能
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
    raise ImportError("视频处理需要qwen_vl_utils库。请使用以下命令安装: pip install qwen_vl_utils")

load_dotenv()


class VideoProcessor:
    """
    使用Qwen VL模型通过vLLM API服务器处理视频内容的类
    
    该类处理视频帧提取、编码以及与vLLM API服务器的通信，用于视频理解任务
    """
    
    def __init__(
        self,
        api_key: str = "EMPTY",
        api_base: str = "http://localhost:8000/v1",
        model_name: str = "Qwen/Qwen2.5-VL-32B-Instruct"
    ):
        """
        初始化视频处理器
        
        参数:
            api_key: OpenAI API密钥 (vLLM默认使用"EMPTY")
            api_base: vLLM API服务器的基础URL
            model_name: 要使用的模型名称
        """
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name
        
        self.client = OpenAI(
            api_key=self.api_key,            base_url=self.api_base,
        )
    
    def prepare_message_for_vllm(self, content_messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        为vLLM处理准备视频消息
        
        vLLM中视频的帧提取逻辑与qwen_vl_utils不同。
        这里我们使用qwen_vl_utils来提取视频帧，并将视频的media_type显式设置为video/jpeg。
        这样，vLLM就不会再尝试从输入的base64编码图像中提取帧。
        
        参数:
            content_messages: 包含视频内容的消息字典列表
            
        返回:
            (处理后的消息, 视频参数) 的元组
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
                    assert video_inputs is not None, "video_inputs不应为None"
                    video_input = (video_inputs.pop()).permute(0, 2, 3, 1).numpy().astype(np.uint8)
                    fps_list.extend(video_kwargs.get('fps', []))

                    # 使用base64编码图像
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
        system_message: str = "你是一个有帮助的助手。"
    ) -> Dict[str, Any]:
        """
        使用Qwen VL模型处理视频和文本提示
        
        参数:
            video_url: 视频文件的URL或路径
            text_prompt: 描述要分析视频内容的文本提示
            total_pixels: 视频处理的总像素数
            min_pixels: 视频处理的最小像素数
            fps: 视频处理的每秒帧数
            system_message: 模型的系统消息
            
        返回:
            包含模型响应的字典
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
                }]            },
        ]
        
        # 为vLLM准备消息
        processed_messages, video_kwargs = self.prepare_message_for_vllm(video_messages)
        
        # 获取聊天补全
        chat_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=processed_messages,
            extra_body={
                "mm_processor_kwargs": video_kwargs
            }
        )
        
        return {
            "response": chat_response,            "content": chat_response.choices[0].message.content if chat_response.choices else None,
            "video_kwargs": video_kwargs
        }
    
    def analyze_video_features(self, video_url: str) -> Dict[str, Any]:
        """
        使用表格格式分析和总结视频的特征
        
        参数:
            video_url: 视频文件的URL或路径
            
        返回:
            包含分析结果的字典
        """
        prompt = "请用表格总结一下视频中的商品特点"
        return self.process_video(video_url, prompt)


# 示例使用函数
def example_video_analysis():
    """
    演示如何使用VideoProcessor的示例函数
    """
    # 初始化处理器
    processor = VideoProcessor()
    
    # 示例视频URL
    video_url = "https://duguang-labelling.oss-cn-shanghai.aliyuncs.com/qiansun/video_ocr/videos/50221078283.mp4"
    
    # 分析视频特征
    result = processor.analyze_video_features(video_url)
    
    print("视频分析结果:")
    print(result["content"])
    
    return result


if __name__ == "__main__":
    example_video_analysis()
