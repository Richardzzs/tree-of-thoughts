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
    return eval(thought_string)


TREE_OF_THOUGHTS_SYS_PROMPT = """
You are an expert problem-solving agent designed to not only solve complex problems but also critically evaluate the quality of your thought process and final answers. 
Your task is to follow a structured approach to generate solutions, assess your thoughts, and provide a rating for each on a scale of 0.1 to 1.0. 
This rating should reflect the accuracy and quality of your reasoning and final answer.

### Instructions:

1. **Understand the Problem:**
   - Carefully analyze the problem provided by the user.
   - Break down the problem into smaller, manageable parts if necessary.
   - Formulate a clear understanding of the problem before proceeding.

2. **Generate Thoughts:**
   - Create multiple thoughts or steps toward solving the problem.
   - For each thought, document your reasoning, ensuring that it is logical and well-founded.

3. **Self-Evaluation:**
   - After generating each thought, evaluate its accuracy and quality.
   - Assign an evaluation score between 0.1 and 1.0. Use the following guidelines:
     - **0.1 to 0.4:** The thought is flawed, inaccurate, or incomplete.
     - **0.5 to 0.7:** The thought is partially correct but may lack detail or full accuracy.
     - **0.8 to 1.0:** The thought is accurate, complete, and well-reasoned.

4. **Generate Final Answer:**
   - Based on your thoughts, synthesize a final answer to the problem.
   - Ensure the final answer is comprehensive and addresses all aspects of the problem.

5. **Final Evaluation:**
   - Evaluate the overall quality and accuracy of your final answer.
   - Provide a final evaluation score based on the same 0.1 to 1.0 scale.
   
"""


class Thought(BaseModel):
    thought: str
    evaluation: Optional[float] = Field(
        description="The evaluation of the thought. It can be a number between 0.1 and 1.0 being 0.1 the worst and 1.0 the best."
    )


class TotAgent:
    """
    Represents a Tree of Thoughts (ToT) agent.

    Attributes:
        id (str): The unique identifier for the agent.
        max_loops (int): The maximum number of loops the agent can run.
        model (OpenAIFunctionCaller): The OpenAI function caller for the agent.
        agent (Agent): The agent responsible for running tasks.
        video_processor (VideoProcessor): Optional video processor for handling video analysis tasks.

    Methods:
        run(task: str) -> dict: Runs a task using the agent and returns the output as a dictionary.
        analyze_video(video_url: str, prompt: str) -> dict: Analyzes video content using video processor.
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
        **kwargs,
    ):
        """
        Initializes a new instance of the TotAgent class.

        Args:
            id (str, optional): The unique identifier for the agent. Defaults to a randomly generated UUID.
            max_loops (int, optional): The maximum number of loops the agent can run. Defaults to None.
            enable_video_processing (bool): Whether to enable video processing capabilities.
            video_api_base (str): Base URL for the video processing API server.
            video_model_name (str): Name of the video model to use.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
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
            **kwargs,
        )
        
        # Initialize video processor if requested and available
        self.video_processor = None
        if enable_video_processing and VideoProcessor is not None:
            self.video_processor = VideoProcessor(
                api_base=video_api_base,
                model_name=video_model_name
            )

    def run(self, task: Any) -> dict:
        """
        Runs a task using the agent and returns the output as a dictionary.

        Args:
            task (str): The task to be run by the agent.

        Returns:
            dict: The output of the agent as a dictionary.
        """
        agent_output = self.agent.run(task)
        return string_to_dict(agent_output)
    
    def analyze_video(
        self, 
        video_url: str, 
        prompt: str = "请用表格总结一下视频中的商品特点",
        total_pixels: int = 20480 * 28 * 28,
        min_pixels: int = 16 * 28 * 2,
        fps: float = 3.0
    ) -> dict:
        """
        Analyzes video content using the integrated video processor.
        
        Args:
            video_url: URL or path to the video file
            prompt: Text prompt for video analysis
            total_pixels: Total pixels for video processing
            min_pixels: Minimum pixels for video processing
            fps: Frames per second for video processing
            
        Returns:
            Dictionary containing the analysis results
            
        Raises:
            ValueError: If video processing is not enabled or VideoProcessor is not available
        """
        if self.video_processor is None:
            raise ValueError(
                "Video processing is not enabled. Initialize TotAgent with enable_video_processing=True"
            )
        
        # Use video processor to analyze the video
        video_result = self.video_processor.process_video(
            video_url=video_url,
            text_prompt=prompt,
            total_pixels=total_pixels,
            min_pixels=min_pixels,
            fps=fps
        )
        
        # Combine video analysis with ToT reasoning
        analysis_task = f"""
        Based on the following video analysis result, please provide a structured thought process 
        and evaluation of the video content:
        
        Video Analysis Result: {video_result.get('content', 'No content available')}
        
        Please break down your analysis into thoughts and provide evaluations for each thought.
        """
        
        # Run the task through the ToT agent
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
        Convenience method to analyze video features using a predefined prompt.
        
        Args:
            video_url: URL or path to the video file
            
        Returns:
            Dictionary containing the feature analysis results
        """
        return self.analyze_video(video_url, "请用表格总结一下视频中的商品特点")
