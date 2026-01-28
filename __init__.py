from .nodes import Qwen3GGUFNode

NODE_CLASS_MAPPINGS = {
    "Qwen3GGUFNode": Qwen3GGUFNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3GGUFNode": "Qwen3 GGUF Loader & Generator"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
