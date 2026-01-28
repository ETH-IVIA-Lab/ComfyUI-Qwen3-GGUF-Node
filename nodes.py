import os
import folder_paths
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ComfyLLM")

# Model cache to keep models loaded in memory
_model_cache = {}

try:
    from llama_cpp import Llama
except ImportError:
    logger.error("llama-cpp-python not found. Please install it with ROCm support.")
    Llama = None

def get_llm_models():
    """
    Scans the ComfyUI models/llm directory for .gguf files.
    """
    try:
        llm_path = folder_paths.get_folder_paths("llm")
    except Exception:
        llm_path = []
    
    if not llm_path:
        try:
            base_path = folder_paths.models_dir
            llm_path = [os.path.join(base_path, "llm")]
        except Exception:
            llm_path = []
    
    models = []
    for path in llm_path:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(".gguf"):
                        rel_path = os.path.relpath(os.path.join(root, file), path)
                        models.append(rel_path)
    
    return sorted(models) if models else ["No models found in models/llm/"]

class Qwen3GGUFNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (get_llm_models(), {
                    "tooltip": "Select the GGUF model file to use. Models are loaded from ComfyUI/models/llm/ directory."
                }),
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "Hello, how are you?",
                    "tooltip": "The main user prompt or question for the model to respond to."
                }),
                "system_message": ("STRING", {
                    "multiline": True, 
                    "default": "You are a helpful assistant.",
                    "tooltip": "System instruction that defines the model's behavior and role. This is formatted using ChatML for Qwen3."
                }),
                "n_ctx": ("INT", {
                    "default": 2048, 
                    "min": 1, 
                    "max": 32768,
                    "tooltip": "Context window size - the maximum number of tokens the model can consider at once. Larger values allow longer conversations but use more memory."
                }),
                "n_gpu_layers": ("INT", {
                    "default": 33, 
                    "min": 0, 
                    "max": 100,
                    "tooltip": "Number of model layers to offload to GPU (ROCm/HIP). Set to 0 for CPU-only inference. Higher values use more VRAM but are faster."
                }),
                "n_threads": ("INT", {
                    "default": 8, 
                    "min": 1, 
                    "max": 64,
                    "tooltip": "Number of CPU threads to use for inference. More threads can speed up generation but may not always improve performance."
                }),
                "n_batch": ("INT", {
                    "default": 512, 
                    "min": 1, 
                    "max": 2048,
                    "tooltip": "Batch size for prompt processing. Larger batches process prompts faster but use more memory."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.01,
                    "tooltip": "Sampling temperature. Lower values (0.1-0.5) make output more deterministic and focused. Higher values (0.7-1.5) make output more creative and diverse."
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Nucleus sampling threshold. Only tokens with cumulative probability up to this value are considered. Lower values (0.5-0.9) produce more focused output."
                }),
                "top_k": ("INT", {
                    "default": 40, 
                    "min": 0, 
                    "max": 100,
                    "tooltip": "Top-K sampling limit. Only the top K most likely tokens are considered. Set to 0 to disable. Lower values produce more focused output."
                }),
                "max_tokens": ("INT", {
                    "default": 8192, 
                    "min": 1, 
                    "max": 2147483647,
                    "tooltip": "Maximum number of tokens to generate. Set to a high value (e.g., 32768) for unlimited generation. The model will stop earlier if it hits a stop sequence."
                }),
                "repeat_penalty": ("FLOAT", {
                    "default": 1.1, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.01,
                    "tooltip": "Penalty for repeating tokens. Values > 1.0 reduce repetition. Values < 1.0 allow more repetition. Typical range: 1.0-1.3."
                }),
                "seed": ("INT", {
                    "default": -1, 
                    "min": -1, 
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducible outputs. Set to -1 for random (non-deterministic) generation. Use a fixed positive number for reproducible results."
                }),
                "stop": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "Comma-separated list of stop sequences. When the model generates any of these sequences, it stops generating. Example: 'END,STOP'. The <|im_end|> token is automatically added."
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If enabled, keeps the model loaded in memory between generations for faster subsequent runs. Disable to free memory after each generation."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "generate_text"
    CATEGORY = "LLM"
    OUTPUT_NODE = True

    def generate_text(self, model_name, prompt, system_message, n_ctx, n_gpu_layers, n_threads, n_batch, 
                      temperature, top_p, top_k, max_tokens, repeat_penalty, seed, stop, keep_model_loaded):
        
        if Llama is None:
            raise ImportError("llama-cpp-python is not installed. Please install it to use this node.")

        if model_name == "No models found in models/llm/":
            raise ValueError("No GGUF models found. Please place .gguf files in ComfyUI/models/llm/")

        # Resolve model path
        try:
            llm_paths = folder_paths.get_folder_paths("llm")
        except Exception:
            llm_paths = []

        if not llm_paths:
            base_path = folder_paths.models_dir
            llm_paths = [os.path.join(base_path, "llm")]
            
        model_path = None
        for path in llm_paths:
            full_path = os.path.join(path, model_name)
            if os.path.exists(full_path):
                model_path = full_path
                break
        
        if model_path is None:
            raise FileNotFoundError(f"Model {model_name} not found in any llm model path.")

        # Create cache key based on model path and key parameters
        cache_key = f"{model_path}_{n_ctx}_{n_gpu_layers}_{n_threads}_{n_batch}"
        
        # Check if model is already loaded in cache
        if keep_model_loaded and cache_key in _model_cache:
            logger.info(f"Using cached model: {model_path}")
            llm = _model_cache[cache_key]
        else:
            logger.info(f"Loading model: {model_path}")
            
            try:
                # Initialize Llama (ROCm compatible via llama-cpp-python standard API)
                llm = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    n_threads=n_threads,
                    n_batch=n_batch,
                    seed=seed if seed != -1 else None,
                    verbose=False
                )
                
                # Cache the model if keep_model_loaded is enabled
                if keep_model_loaded:
                    _model_cache[cache_key] = llm
                    logger.info(f"Model cached in memory: {cache_key}")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise RuntimeError(f"Failed to load model: {str(e)}")

        # Format prompt for Qwen (ChatML style)
        formatted_prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        # Handle stop sequences
        stop_sequences = [s.strip() for s in stop.split(",")] if stop else []
        if "<|im_end|>" not in stop_sequences:
            stop_sequences.append("<|im_end|>")

        logger.info("Generating text...")
        
        try:
            # Generate text
            output = llm(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stop=stop_sequences,
                echo=False
            )
            
            generated_text = output["choices"][0]["text"]
            logger.info("Generation complete.")
            return (generated_text,)
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise RuntimeError(f"Inference failed: {str(e)}")
