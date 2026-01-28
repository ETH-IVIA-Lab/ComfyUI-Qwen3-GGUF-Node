import os
import folder_paths
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ComfyLLM")

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
                "model_name": (get_llm_models(),),
                "prompt": ("STRING", {"multiline": True, "default": "Hello, how are you?"}),
                "system_message": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "n_ctx": ("INT", {"default": 2048, "min": 1, "max": 32768}),
                "n_gpu_layers": ("INT", {"default": 33, "min": 0, "max": 100}),
                "n_threads": ("INT", {"default": 8, "min": 1, "max": 64}),
                "n_batch": ("INT", {"default": 512, "min": 1, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 4096}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "stop": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "generate_text"
    CATEGORY = "LLM"

    def generate_text(self, model_name, prompt, system_message, n_ctx, n_gpu_layers, n_threads, n_batch, 
                      temperature, top_p, top_k, max_tokens, repeat_penalty, seed, stop):
        
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
