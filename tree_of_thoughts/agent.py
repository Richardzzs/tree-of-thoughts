import uuid
from pydantic import BaseModel, Field
from typing import Optional
from swarms import Agent
from swarm_models import OpenAIFunctionCaller
from typing import Any
import os
from dotenv import load_dotenv

load_dotenv()

# Import VideoProcessor for video analysis capabilities
try:
    from tree_of_thoughts.video_processor import VideoProcessor
except ImportError:
    VideoProcessor = None


def string_to_dict(thought_string):
    """将思考字符串转换为字典"""
    return eval(thought_string)


TREE_OF_THOUGHTS_SYS_PROMPT = """
你是一个专业的问题解决代理，不仅能解决复杂问题，还能批判性地评估你的思维过程和最终答案的质量。
你的任务是遵循结构化方法来生成解决方案，评估你的想法，并为每个想法在0.1到1.0的评分范围内提供评级。
这个评级应该反映你推理和最终答案的准确性和质量。

### 指令:

1. **理解问题:**
   - 仔细分析用户提供的问题。
   - 如有必要，将问题分解为更小、更可管理的部分。
   - 在继续之前，对问题形成清晰的理解。

2. **生成思路:**
   - 创建解决问题的多个想法或步骤。
   - 对于每个想法，记录你的推理，确保逻辑性和有据可循。

3. **自我评估:**
   - 生成每个想法后，评估其准确性和质量。
   - 分配0.1到1.0之间的评估分数。使用以下指导原则:
     - **0.1到0.4:** 想法有缺陷、不准确或不完整。
     - **0.5到0.7:** 想法部分正确，但可能缺乏细节或完全准确性。
     - **0.8到1.0:** 想法准确、完整且论证充分。

4. **生成最终答案:**
   - 基于你的想法，综合出问题的最终答案。
   - 确保最终答案全面并解决问题的所有方面。

5. **最终评估:**
   - 评估最终答案的整体质量和准确性。
   - 基于相同的0.1到1.0评分标准提供最终评估分数。
   
"""


class Thought(BaseModel):
    """思考模型"""
    thought: str
    evaluation: Optional[float] = Field(
        description="思考的评估分数。可以是0.1到1.0之间的数字，0.1为最差，1.0为最佳。"
    )


class TotAgent:
    """
    表示思维树(ToT)代理的类
    
    属性:
        id (str): 代理的唯一标识符
        max_loops (int): 代理可运行的最大循环次数
        model (OpenAIFunctionCaller): 代理的OpenAI函数调用器
        agent (Agent): 负责运行任务的代理
        video_processor (VideoProcessor): 用于处理视频分析任务的可选视频处理器
        
    方法:
        run(task: str) -> dict: 使用代理运行任务并返回字典形式的输出
        analyze_video(video_url: str, prompt: str) -> dict: 使用视频处理器分析视频内容
    """

    def __init__(
        self,
        id: str = uuid.uuid4().hex,
        max_loops: int = None,
        use_openai_caller: bool = True,
        model: Optional[Any] = None,
        enable_video_processing: bool = False,
        video_api_base: str = "http://localhost:8000/v1",
        video_model_name: str = "Qwen/Qwen2.5-VL-32B-Instruct",
        *args,
        **kwargs,    ):
        """
        初始化TotAgent类的新实例

        参数:
            id (str, 可选): 代理的唯一标识符。默认为随机生成的UUID。
            max_loops (int, 可选): 代理可运行的最大循环次数。默认为None。
            enable_video_processing (bool): 是否启用视频处理功能。
            video_api_base (str): 视频处理API服务器的基础URL。
            video_model_name (str): 要使用的视频模型名称。
            *args: 可变长度参数列表。
            **kwargs: 任意关键字参数。
        """
        self.id = id
        self.max_loops = max_loops
        self.model = model

        if use_openai_caller:
            self.model = OpenAIFunctionCaller(
                system_prompt=TREE_OF_THOUGHTS_SYS_PROMPT,
                base_model=Thought,
                parallel_tool_calls=False,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens=3000,
            )

        self.agent = Agent(
            agent_name=f"ToT-Agent-{self.id}",
            system_prompt=TREE_OF_THOUGHTS_SYS_PROMPT,
            llm=self.model,
            max_loops=1,
            autosave=True,
            dashboard=False,
            verbose=True,
            dynamic_temperature_enabled=True,
            saved_state_path=f"tot_agent{self.id}.json",
            user_name="swarms_corp",
            retry_attempts=1,
            context_length=200000,
            return_step_meta=False,
            *args,
            **kwargs,        )
        
        # 如果请求并且可用，则初始化视频处理器
        self.video_processor = None
        if enable_video_processing and VideoProcessor is not None:
            self.video_processor = VideoProcessor(
                api_base=video_api_base,
                model_name=video_model_name
            )

    def run(self, task: Any) -> dict:
        """
        使用代理运行任务并返回字典形式的输出

        参数:
            task (str): 要由代理运行的任务

        返回:
            dict: 代理的输出，以字典形式返回
        """
        agent_output = self.agent.run(task)
        return string_to_dict(agent_output)
    
    def analyze_video(
        self, 
        video_url: str, 
        prompt: str = "请用表格总结一下视频中的商品特点",
        total_pixels: int = 20480 * 28 * 28,
        min_pixels: int = 16 * 28 * 2,
        fps: float = 3.0    ) -> dict:
        """
        使用集成的视频处理器分析视频内容
        
        参数:
            video_url: 视频文件的URL或路径
            prompt: 视频分析的文本提示
            total_pixels: 视频处理的总像素数
            min_pixels: 视频处理的最小像素数
            fps: 视频处理的每秒帧数
            
        返回:
            包含分析结果的字典
            
        异常:
            ValueError: 如果未启用视频处理或VideoProcessor不可用
        """
        if self.video_processor is None:
            raise ValueError(
                "视频处理未启用。请使用enable_video_processing=True初始化TotAgent"
            )
        
        # 使用视频处理器分析视频
        video_result = self.video_processor.process_video(
            video_url=video_url,
            text_prompt=prompt,
            total_pixels=total_pixels,
            min_pixels=min_pixels,
            fps=fps
        )
        
        # 将视频分析与ToT推理结合
        analysis_task = f"""
        基于以下视频分析结果，请提供结构化的思维过程和对视频内容的评估：
        
        视频分析结果: {video_result.get('content', '无可用内容')}
        
        请将你的分析分解为思路并为每个思路提供评估。        """
        
        # 通过ToT代理运行任务
        tot_analysis = self.run(analysis_task)
        
        return {
            "video_analysis": video_result,
            "tot_reasoning": tot_analysis,
            "combined_result": {
                "video_content": video_result.get('content'),
                "structured_thoughts": tot_analysis
            }
        }
    
    def analyze_video_features(self, video_url: str) -> dict:
        """
        使用预定义提示分析视频特征的便利方法
        
        参数:
            video_url: 视频文件的URL或路径
            
        返回:
            包含特征分析结果的字典
        """
        return self.analyze_video(video_url, "请用表格总结一下视频中的商品特点")
