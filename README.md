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
