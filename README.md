# ComfyUI Qwen3 GGUF Node

A custom node for ComfyUI to load and run Qwen3 GGUF models using `llama-cpp-python`. This node is designed to work with AMD GPUs via ROCm, as well as CPU backends.

## Features

- Load Qwen3 GGUF models from `models/llm/`
- Comprehensive inference settings (Temperature, Top-P, Top-K, etc.)
- ROCm/AMD GPU support (no CUDA dependencies)
- ChatML prompt formatting for Qwen3

## Installation

1. Clone this repository into your `ComfyUI/custom_nodes/` directory.
2. Install the dependencies. **Crucial for AMD GPU users:** You must install `llama-cpp-python` with ROCm support.

```bash
# Navigate to the custom node directory
cd ComfyUI/custom_nodes/comfy-llm

# Install llama-cpp-python with ROCm support
CMAKE_ARGS="-DLLAMA_HIPBLAS=on" pip install llama-cpp-python
```

3. Place your Qwen3 GGUF model files in `ComfyUI/models/llm/`. Create the folder if it doesn't exist.

## Supported Models

This node supports all Qwen GGUF models available on [Hugging Face](https://huggingface.co/Qwen/models?search=gguf). Below is a comprehensive list organized by category:

### Text Generation Models

#### Base Models
- **Qwen3-0.6B-GGUF** (0.6B parameters)
  - Lightweight base model for general text generation
  - Fast inference, low memory requirements
  - Suitable for simple tasks and resource-constrained environments

- **Qwen3-1.7B-GGUF** (2B parameters)
  - Small base model with improved capabilities over 0.6B
  - Balanced performance and efficiency
  - Good for general-purpose text generation

- **Qwen3-4B-GGUF** (4B parameters)
  - Mid-size base model with enhanced reasoning
  - Better quality outputs than smaller variants
  - Suitable for most text generation tasks

- **Qwen3-8B-GGUF** (8B parameters)
  - Popular mid-to-large base model
  - Strong performance across diverse tasks
  - Excellent balance of quality and resource usage

- **Qwen3-14B-GGUF** (15B parameters)
  - Large base model with advanced capabilities
  - Superior reasoning and comprehension
  - Requires more VRAM/RAM

- **Qwen3-32B-GGUF** (33B parameters)
  - Very large base model
  - High-quality outputs for complex tasks
  - Requires significant computational resources

#### Specialized Models
- **Qwen3-30B-A3B-GGUF** (31B parameters)
  - Advanced architecture variant (A3B)
  - Enhanced performance on specialized tasks
  - Optimized for specific use cases

- **Qwen3-235B-A22B-GGUF** (235B parameters)
  - Ultra-large model with advanced architecture
  - State-of-the-art performance
  - Requires substantial computational resources

#### Instruct Models
- **Qwen3-Next-80B-A3B-Instruct-GGUF** (80B parameters)
  - Instruction-tuned variant optimized for following user instructions
  - Better at task completion and following prompts
  - Enhanced safety and helpfulness alignment

- **Qwen3-Next-80B-A3B-Thinking-GGUF** (80B parameters)
  - Thinking/reasoning variant with extended reasoning capabilities
  - Shows intermediate reasoning steps
  - Excellent for complex problem-solving

### Vision-Language (VL) Models

#### Instruct Variants
- **Qwen3-VL-2B-Instruct-GGUF** (2B parameters)
  - Lightweight vision-language model
  - Image understanding and text generation
  - Low resource requirements

- **Qwen3-VL-4B-Instruct-GGUF** (4B parameters)
  - Mid-size vision-language model
  - Better image comprehension than 2B variant
  - Balanced performance for vision tasks

- **Qwen3-VL-8B-Instruct-GGUF** (8B parameters)
  - Popular vision-language model size
  - Strong image understanding and generation
  - Good quality-to-resource ratio

- **Qwen3-VL-32B-Instruct-GGUF** (33B parameters)
  - Large vision-language model
  - Advanced multimodal understanding
  - High-quality vision-text interactions

- **Qwen3-VL-30B-A3B-Instruct-GGUF** (31B parameters)
  - Advanced architecture vision-language model
  - Enhanced multimodal capabilities
  - Specialized for complex vision tasks

- **Qwen3-VL-235B-A22B-Instruct-GGUF** (235B parameters)
  - Ultra-large vision-language model
  - State-of-the-art multimodal performance
  - Requires significant resources

#### Thinking Variants
- **Qwen3-VL-2B-Thinking-GGUF** (2B parameters)
  - Vision-language model with reasoning capabilities
  - Shows thinking process for vision tasks
  - Lightweight reasoning variant

- **Qwen3-VL-4B-Thinking-GGUF** (4B parameters)
  - Mid-size reasoning vision model
  - Better reasoning than 2B variant
  - Enhanced step-by-step vision analysis

- **Qwen3-VL-8B-Thinking-GGUF** (8B parameters)
  - Popular reasoning vision model
  - Strong reasoning with visual inputs
  - Good balance for complex vision reasoning

- **Qwen3-VL-32B-Thinking-GGUF** (33B parameters)
  - Large reasoning vision model
  - Advanced multimodal reasoning
  - High-quality visual problem-solving

- **Qwen3-VL-30B-A3B-Thinking-GGUF** (31B parameters)
  - Advanced architecture reasoning vision model
  - Enhanced reasoning capabilities
  - Specialized for complex visual reasoning

- **Qwen3-VL-235B-A22B-Thinking-GGUF** (235B parameters)
  - Ultra-large reasoning vision model
  - State-of-the-art visual reasoning
  - Maximum performance for complex tasks

### Embedding Models

- **Qwen3-Embedding-0.6B-GGUF** (0.6B parameters)
  - Lightweight embedding model
  - Fast text embeddings generation
  - Low memory footprint

- **Qwen3-Embedding-4B-GGUF** (4B parameters)
  - Mid-size embedding model
  - Better embedding quality than 0.6B
  - Balanced performance for semantic search

- **Qwen3-Embedding-8B-GGUF** (8B parameters)
  - Large embedding model
  - High-quality semantic embeddings
  - Excellent for retrieval and similarity tasks

### Code Generation Models

- **Qwen2.5-Coder-1.5B-Instruct-GGUF** (2B parameters)
  - Lightweight code generation model
  - Fast code completion and generation
  - Suitable for simple coding tasks

- **Qwen2.5-Coder-7B-Instruct-GGUF** (8B parameters)
  - Popular code generation model
  - Strong coding capabilities
  - Good balance for most programming tasks

- **Qwen2.5-Coder-14B-Instruct-GGUF** (15B parameters)
  - Large code generation model
  - Advanced code understanding and generation
  - Better handling of complex codebases

- **Qwen2.5-Coder-32B-Instruct-GGUF** (33B parameters)
  - Very large code generation model
  - State-of-the-art coding performance
  - Excellent for complex software development

### Other Models

- **QwQ-32B-GGUF** (33B parameters)
  - Specialized reasoning model (QwQ = Qwen with Q*)
  - Advanced problem-solving capabilities
  - Extended context window (131K tokens)
  - Competitive with reasoning-focused models

## Model Selection Guide

- **For general text generation**: Start with Qwen3-8B-GGUF or Qwen3-14B-GGUF
- **For instruction following**: Use Qwen3-Next-80B-A3B-Instruct-GGUF
- **For vision tasks**: Use Qwen3-VL-8B-Instruct-GGUF or larger variants
- **For code generation**: Use Qwen2.5-Coder-7B-Instruct-GGUF or larger
- **For embeddings**: Use Qwen3-Embedding-4B-GGUF or Qwen3-Embedding-8B-GGUF
- **For reasoning tasks**: Use Thinking variants or QwQ-32B-GGUF
- **For resource-constrained systems**: Use 0.6B, 1.7B, or 2B variants

## Node Parameters

- **model_name**: Select from available GGUF models in `models/llm/`.
- **prompt**: The main text prompt for generation.
- **system_message**: The system instruction for the model (ChatML format).
- **n_ctx**: Context window size (default: 2048).
- **n_gpu_layers**: Number of layers to offload to GPU (ROCm). Set to 0 for CPU-only.
- **n_threads**: Number of CPU threads to use.
- **n_batch**: Batch size for prompt processing.
- **temperature**: Sampling temperature (higher = more creative).
- **top_p**: Nucleus sampling threshold.
- **top_k**: Top-K sampling limit.
- **max_tokens**: Maximum number of tokens to generate.
- **repeat_penalty**: Penalty for repeating tokens.
- **seed**: Random seed (-1 for random).
- **stop**: Comma-separated list of stop sequences.

## Compatibility

- **AMD GPU**: Fully supported via ROCm. Ensure `llama-cpp-python` is built with `LLAMA_HIPBLAS=on`.
- **CPU**: Works out of the box.
- **CUDA**: Not used directly, but `llama-cpp-python` can be built with CUDA support if needed on other systems. This node itself contains no CUDA-specific code.

## License

MIT

## Disclaimer

This project was developed with the assistance of Cursor AI. While the code has been reviewed and tested, please be aware that AI-generated code may contain errors or require adjustments for your specific use case. Use at your own discretion and always review the code before deploying in production environments.
