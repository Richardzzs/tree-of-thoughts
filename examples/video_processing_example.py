"""
Example usage of the enhanced Tree of Thoughts agent with video processing capabilities.
"""

import os
from tree_of_thoughts import TotAgent, VideoProcessor


def example_basic_video_processing():
    """
    Example of using VideoProcessor directly for video analysis.
    """
    print("=== Basic Video Processing Example ===")
    
    # Initialize video processor
    processor = VideoProcessor(
        api_base="http://localhost:8000/v1",
        model_name="Qwen/Qwen2.5-VL-32B-Instruct"
    )
    
    # Example video URL
    video_url = "https://duguang-labelling.oss-cn-shanghai.aliyuncs.com/qiansun/video_ocr/videos/50221078283.mp4"
    
    try:
        # Analyze video features
        result = processor.analyze_video_features(video_url)
        
        print("Video Analysis Result:")
        print(result["content"])
        print(f"Video processing kwargs: {result['video_kwargs']}")
        
    except Exception as e:
        print(f"Error processing video: {e}")
        print("Make sure the vLLM server is running at http://localhost:8000")
    
    return result


def example_tot_agent_with_video():
    """
    Example of using enhanced TotAgent with video processing capabilities.
    """
    print("\n=== Tree of Thoughts Agent with Video Processing Example ===")
    
    # Initialize ToT agent with video processing enabled
    agent = TotAgent(
        enable_video_processing=True,
        video_api_base="http://localhost:8000/v1",
        video_model_name="Qwen/Qwen2.5-VL-32B-Instruct"
    )
    
    # Example video URL
    video_url = "https://duguang-labelling.oss-cn-shanghai.aliyuncs.com/qiansun/video_ocr/videos/50221078283.mp4"
    
    try:
        # Analyze video using ToT reasoning
        result = agent.analyze_video(
            video_url=video_url,
            prompt="分析这个视频中的商品特点，并提供结构化的思考过程"
        )
        
        print("Combined Video + ToT Analysis:")
        print(f"Video Content: {result['combined_result']['video_content']}")
        print(f"Structured Thoughts: {result['combined_result']['structured_thoughts']}")
        
    except Exception as e:
        print(f"Error with ToT video analysis: {e}")
        print("Make sure both OpenAI API key is set and vLLM server is running")
    
    return result


def example_custom_video_analysis():
    """
    Example of custom video analysis with different parameters.
    """
    print("\n=== Custom Video Analysis Example ===")
    
    processor = VideoProcessor()
    
    # Custom video analysis with different parameters
    video_url = "https://duguang-labelling.oss-cn-shanghai.aliyuncs.com/qiansun/video_ocr/videos/50221078283.mp4"
    custom_prompt = "请详细描述视频中的场景、人物活动和物品，并分析其商业价值"
    
    try:
        result = processor.process_video(
            video_url=video_url,
            text_prompt=custom_prompt,
            total_pixels=15360 * 28 * 28,  # Different pixel settings
            fps=2.0  # Different FPS
        )
        
        print("Custom Analysis Result:")
        print(result["content"])
        
    except Exception as e:
        print(f"Error with custom analysis: {e}")
    
    return result


def example_tot_reasoning_only():
    """
    Example of using ToT agent for text-based reasoning.
    """
    print("\n=== Tree of Thoughts Text Reasoning Example ===")
    
    # Initialize basic ToT agent (without video processing)
    agent = TotAgent()
    
    # Text-based reasoning task
    task = """
    分析以下商品营销策略的有效性：
    1. 社交媒体推广
    2. 网红合作
    3. 限时优惠
    4. 客户推荐计划
    
    请为每个策略提供思考过程和评分。
    """
    
    try:
        result = agent.run(task)
        print("ToT Reasoning Result:")
        print(result)
        
    except Exception as e:
        print(f"Error with ToT reasoning: {e}")
        print("Make sure OpenAI API key is properly configured")
    
    return result


if __name__ == "__main__":
    print("Tree of Thoughts Video Processing Examples")
    print("=" * 50)
    
    # Check if required environment variables are set
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
        print("Text-based ToT reasoning may not work properly")
    
    # Run examples
    try:
        # Basic video processing
        example_basic_video_processing()
        
        # ToT agent with video
        example_tot_agent_with_video()
        
        # Custom video analysis
        example_custom_video_analysis()
        
        # Text-based ToT reasoning
        example_tot_reasoning_only()
        
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    print("\nAll examples completed!")
