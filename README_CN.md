# 思维树 (Tree of Thoughts) - 视频处理版本

[![Multi-Modality](images/agorabanner.png)](https://discord.gg/qUtxnK2NMf)

![Tree of Thoughts Banner](images/treeofthoughts.png)

一个强大且灵活的思维树算法实现，能够显著提升模型推理能力达70%。这个即插即用版本允许您连接自己的模型并体验超级智能！

## 🌟 特性

- **结构化推理**: 使用思维树算法进行复杂问题解决
- **🎥 视频处理能力**: 集成Qwen VL模型进行视频内容分析
- **多模态分析**: 结合视频理解和结构化推理
- **插件化设计**: 易于集成和扩展
- **中文优化**: 完整支持中文提示和分析

## 🎥 视频处理功能

本仓库现在包含增强的视频处理功能，使用Qwen VL模型！您可以：

- **分析视频内容** - 使用AI驱动的视频理解
- **结合视频分析与思维树推理** - 获得结构化见解  
- **自动提取特征和模式** - 从视频内容中自动提取
- **处理多种视频格式** - 支持可自定义参数

### 快速视频处理示例

```python
from tree_of_thoughts import TotAgent

# 初始化具有视频功能的代理
agent = TotAgent(enable_video_processing=True)

# 使用结构化推理分析视频
result = agent.analyze_video(
    video_url="path/to/video.mp4", 
    prompt="分析视频中的商品特点"
)
```

## 📦 安装

```bash
pip install -r requirements.txt
```

### 依赖项

主要依赖项包括：
- `swarms` - 核心代理框架
- `pydantic` - 数据验证
- `openai` - OpenAI API客户端
- `numpy` - 数值计算
- `Pillow` - 图像处理
- `qwen-vl-utils` - Qwen VL模型工具

## 🚀 快速开始

### 环境配置

在您的 `.env` 文件中，您需要以下变量：

```bash
WORKSPACE_DIR="artifacts"
OPENAI_API_KEY="your_openai_api_key"
```

### 基础使用示例

```python
from tree_of_thoughts import TotAgent, ToTDFSAgent
from dotenv import load_dotenv

load_dotenv()

# 创建TotAgent实例
tot_agent = TotAgent(use_openai_caller=False)

# 使用指定参数创建ToTDFSAgent实例
dfs_agent = ToTDFSAgent(
    agent=tot_agent,  # 使用TotAgent实例作为DFS算法的代理
    threshold=0.8,  # 设置评估思路质量的阈值
    max_loops=1,  # 设置DFS算法的最大循环次数
    prune_threshold=0.5,  # 评估低于0.5的分支将被剪枝
    number_of_agents=4,  # 设置DFS算法中使用的代理数量
)

# 定义DFS算法的初始状态
initial_state = """

你的任务：使用4个数字和基本算术运算(+-*/)在1个等式中得到24，只返回数学表达式

输入: 4 1 8 7
"""

# 运行DFS算法
solution = dfs_agent.run(initial_state)
print(solution)
```

### 视频处理示例

```python
from tree_of_thoughts import VideoProcessor, TotAgent

# 1. 基础视频处理
processor = VideoProcessor()
result = processor.analyze_video_features("video.mp4")
print("视频分析:", result["content"])

# 2. 结合思维树的视频分析
agent = TotAgent(enable_video_processing=True)
analysis = agent.analyze_video(
    video_url="video.mp4",
    prompt="详细分析视频中的商品特点和营销价值"
)
print("结构化分析:", analysis["combined_result"])
```

## 📋 配置选项

### VideoProcessor 配置

- `api_key`: vLLM服务器的API密钥（默认："EMPTY"）
- `api_base`: vLLM API服务器的基础URL
- `model_name`: 要使用的视频模型名称

### TotAgent 视频配置

- `enable_video_processing`: 启用/禁用视频功能
- `video_api_base`: vLLM服务器URL
- `video_model_name`: 视频模型名称

### 视频处理参数

- `total_pixels`: 视频处理的总像素数（影响质量）
- `min_pixels`: 处理的最小像素数
- `fps`: 从视频中提取的每秒帧数
- `system_message`: 模型的系统提示

## 🛠️ 服务器设置

### vLLM服务器设置

您需要运行支持Qwen VL模型的vLLM服务器：

```bash
# 示例vLLM服务器启动
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-32B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

## 📚 API参考

### VideoProcessor方法

#### `process_video(video_url, text_prompt, **kwargs)`
使用文本提示处理视频

**参数:**
- `video_url` (str): 视频文件的URL或路径
- `text_prompt` (str): 分析提示
- `total_pixels` (int): 处理的总像素数
- `min_pixels` (int): 最小像素数
- `fps` (float): 每秒帧数

**返回:**
包含响应、内容和video_kwargs的字典

#### `analyze_video_features(video_url)`
产品特征分析的便利方法

### TotAgent视频方法

#### `analyze_video(video_url, prompt, **kwargs)`
将视频分析与ToT推理结合

**返回:**
包含video_analysis、tot_reasoning和combined_result的字典

#### `analyze_video_features(video_url)`
使用ToT推理进行快速产品特征分析

## 🔧 故障排除

### 常见问题

1. **qwen_vl_utils导入错误**
   ```bash
   pip install qwen-vl-utils
   ```

2. **vLLM服务器未运行**
   - 确保vLLM服务器在指定端口运行
   - 检查API基础URL配置

3. **OpenAI API密钥缺失**
   - 为ToT推理设置OPENAI_API_KEY环境变量

4. **大视频的内存问题**
   - 减少`total_pixels`参数
   - 降低`fps`值
   - 使用较短的视频片段

### 性能优化

1. **调整视频参数**
   - 较低的`fps`以便更快处理
   - 减少`total_pixels`以降低内存使用
   - 为质量阈值使用适当的`min_pixels`

2. **服务器配置**
   - 确保vLLM服务器有足够的GPU内存
   - 考虑使用模型量化以提高效率

## 🎯 示例

查看 `examples/video_processing_example.py` 获取全面的使用示例，包括：

- 基础视频处理
- 结合ToT + 视频分析
- 自定义参数配置
- 错误处理模式

## 📖 详细文档

有关视频处理功能的详细文档，请参阅 [VIDEO_PROCESSING.md](VIDEO_PROCESSING.md)。

## 🤝 贡献

欢迎贡献！请提交Pull Request或创建Issue来报告问题或建议改进。

## 📄 许可证

请参阅LICENSE文件了解许可证详情。

## 📚 相关资源

- [原始论文](https://arxiv.org/pdf/2305.10601.pdf)
- [作者实现](https://github.com/princeton-nlp/tree-of-thought-llm)
- [Qwen VL模型](https://github.com/QwenLM/Qwen-VL)

---

💡 **提示**: 视频处理功能是可选的且默认禁用。现有的ToT功能保持不变，确保向后兼容性。
