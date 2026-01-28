import sys
import os
import logging
from unittest.mock import MagicMock

# ==========================================
# MOCKING COMFYUI ENVIRONMENT
# ==========================================
# Since this script runs outside of ComfyUI, we need to mock the 
# 'folder_paths' module that the node expects.

mock_folder_paths = MagicMock()
# Mock the get_folder_paths method to return a local 'models' directory
mock_folder_paths.get_folder_paths.return_value = [os.path.abspath("models/llm")]
mock_folder_paths.models_dir = os.path.abspath("models")

# Inject the mock into sys.modules
sys.modules["folder_paths"] = mock_folder_paths

# Add parent directory to sys.path so we can import nodes.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from nodes import Qwen3GGUFNode
except ImportError as e:
    print(f"Error importing node: {e}")
    sys.exit(1)

# ==========================================
# SETUP
# ==========================================

# Create a dummy model file for valid path checking if it doesn't exist
# In a real scenario, you would have a real .gguf file here.
os.makedirs("models/llm", exist_ok=True)
dummy_model_name = "qwen3-test.gguf"
dummy_model_path = os.path.join("models/llm", dummy_model_name)

if not os.path.exists(dummy_model_path):
    print(f"Creating dummy model file at {dummy_model_path} for testing path logic...")
    with open(dummy_model_path, "wb") as f:
        f.write(b"GGUF_DUMMY_HEADER")

# ==========================================
# DEMONSTRATION
# ==========================================

def run_demo():
    print("Initializing Qwen3GGUFNode...")
    node = Qwen3GGUFNode()

    # Define inputs mimicking what ComfyUI would pass
    inputs = {
        "model_name": dummy_model_name,
        "prompt": "Explain quantum computing in one sentence.",
        "system_message": "You are a concise science educator.",
        "n_ctx": 2048,
        "n_gpu_layers": 0,  # 0 for CPU to be safe
        "n_threads": 4,
        "n_batch": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 100,
        "repeat_penalty": 1.1,
        "seed": 42,
        "stop": "STOP,END",
        "keep_model_loaded": False
    }

    print("\n--- Input Parameters ---")
    for k, v in inputs.items():
        print(f"{k}: {v}")

    print("\n--- Running Generation ---")
    try:
        # Note: This will attempt to load the model via llama-cpp-python.
        # Since we are using a dummy file, it will likely fail at the loading stage 
        # unless we also mock Llama, but this demonstrates the workflow call.
        
        # We Mock Llama for this demonstration to show success flow
        # If you have a real model and llama-cpp-python installed, remove this mock block.
        import nodes
        if nodes.Llama is None:
             print("llama-cpp-python not found, mocking Llama class for demonstration.")
             mock_llama_instance = MagicMock()
             mock_llama_instance.return_value = {"choices": [{"text": "Quantum computing uses quantum mechanics to process information."}]}
             nodes.Llama = MagicMock(return_value=mock_llama_instance)
        elif os.path.getsize(dummy_model_path) < 100: # It's our dummy file
             print("Using dummy file, mocking Llama class for demonstration.")
             mock_llama_instance = MagicMock()
             mock_llama_instance.return_value = {"choices": [{"text": "Quantum computing uses quantum mechanics to process information."}]}
             nodes.Llama = MagicMock(return_value=mock_llama_instance)

        result = node.generate_text(**inputs)
        
        print("\n--- Output ---")
        print(result[0])
        print("--------------")
        
    except Exception as e:
        print(f"\nExecution failed as expected (if no real model): {e}")

if __name__ == "__main__":
    run_demo()
