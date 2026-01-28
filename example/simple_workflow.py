import websocket # NOTE: websocket-client (pip install websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import time
import os

server_address = os.environ.get("COMFYUI_SERVER_ADDRESS", "127.0.0.1:8188")
client_id = str(uuid.uuid4())

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_ws_messages():
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    return ws

def run_workflow():
    # Define the workflow
    # Note: This requires the Qwen3 GGUF model to be present in ComfyUI/models/llm/
    # You can change the model_name to match your actual file.
    
    workflow = {
        "1": {
            "inputs": {
                "model_name": "qwen3-test.gguf",  # CHANGE THIS to your actual model filename
                "prompt": "Why is the sky blue?",
                "system_message": "You are a helpful assistant.",
                "n_ctx": 2048,
                "n_gpu_layers": 33,
                "n_threads": 8,
                "n_batch": 512,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_tokens": 100,
                "repeat_penalty": 1.1,
                "seed": -1,
                "stop": "",
                "keep_model_loaded": True
            },
            "class_type": "Qwen3GGUFNode"
        }
    }

    print("Connecting to ComfyUI at {}...".format(server_address))
    ws = get_ws_messages()
    
    print("Queueing prompt...")
    try:
        prompt_id = queue_prompt(workflow)['prompt_id']
        print("Prompt ID: {}".format(prompt_id))
        
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        print("Execution complete.")
                        break
                    elif data['prompt_id'] == prompt_id:
                        print("Executing node: {}".format(data['node']))
        
        # Fetch history to get results (if any output was saved/returned)
        history = get_history(prompt_id)
        print("\nHistory:")
        print(json.dumps(history, indent=2))
        
        # Since Qwen3GGUFNode is an OUTPUT_NODE, it runs.
        # But it doesn't return UI output in the standard format unless we use a Preview node.
        # However, the history might contain the output if it was an output node with results.
        
        outputs = history[prompt_id]['outputs']
        if '1' in outputs:
             node_output = outputs['1']
             if 'generated_text' in node_output:
                 print("\nGenerated Text:")
                 print(node_output['generated_text'][0])

    except urllib.error.URLError:
        print("Error: Could not connect to ComfyUI. Is it running on {}?".format(server_address))
    except Exception as e:
        print("Error: {}".format(e))
    finally:
        ws.close()

if __name__ == "__main__":
    run_workflow()
