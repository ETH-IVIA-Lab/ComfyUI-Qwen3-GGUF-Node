import websocket # NOTE: websocket-client (pip install websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import time
import os
import ssl
import base64

raw_address = os.environ.get("COMFYUI_SERVER_ADDRESS", "127.0.0.1:8188")
if not (raw_address.startswith("http://") or raw_address.startswith("https://")):
    raw_address = "http://" + raw_address

parsed = urllib.parse.urlparse(raw_address)
server_address = parsed.netloc
scheme = parsed.scheme

# Basic authentication (optional)
auth_username = os.environ.get("COMFYUI_USERNAME")
auth_password = os.environ.get("COMFYUI_PASSWORD")
auth_header = None
if auth_username and auth_password:
    credentials = base64.b64encode("{}:{}".format(auth_username, auth_password).encode()).decode()
    auth_header = "Basic {}".format(credentials)

# Create an SSL context that ignores certificate validation
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

client_id = str(uuid.uuid4())

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    url = "{}://{}/prompt".format(scheme, server_address)
    req = urllib.request.Request(url, data=data)
    if auth_header:
        req.add_header("Authorization", auth_header)
    return json.loads(urllib.request.urlopen(req, context=ssl_context).read())

def get_history(prompt_id):
    url = "{}://{}/history/{}".format(scheme, server_address, prompt_id)
    req = urllib.request.Request(url)
    if auth_header:
        req.add_header("Authorization", auth_header)
    with urllib.request.urlopen(req, context=ssl_context) as response:
        return json.loads(response.read())

def get_ws_messages():
    ws_scheme = "wss" if scheme == "https" else "ws"
    ws_url = "{}://{}/ws?clientId={}".format(ws_scheme, server_address, client_id)
    
    # Prepare headers for WebSocket connection
    headers = {}
    if auth_header:
        headers["Authorization"] = auth_header
    
    if scheme == "https":
        # For WSS connections, disable certificate validation
        ws = websocket.WebSocket(sslopt={"cert_reqs": ssl.CERT_NONE, "check_hostname": False})
    else:
        ws = websocket.WebSocket()
    
    ws.connect(ws_url, header=headers if headers else None)
    return ws

def run_workflow():
    # Define the workflow
    # Note: This requires the Qwen3 GGUF model to be present in ComfyUI/models/llm/
    # You can change the model_name to match your actual file.
    
    workflow = {
        "1": {
            "inputs": {
                "model_name": "Qwen3-8B-Q4_K_M.gguf",  # CHANGE THIS to your actual model filename
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
    
    try:
        print("Queueing prompt...")
        prompt_id = queue_prompt(workflow)['prompt_id']
        print("Prompt ID: {}".format(prompt_id))
        
        print("Connecting WebSocket...")
        ws = get_ws_messages()
        print("WebSocket connected. Waiting for execution...")
        
        execution_complete = False
        max_wait_time = 300  # 5 minutes timeout
        start_time = time.time()
        last_poll_time = 0
        poll_interval = 5  # Poll history every 5 seconds as fallback
        
        while not execution_complete and (time.time() - start_time) < max_wait_time:
            try:
                # Try to receive with a short timeout
                ws.settimeout(2.0)
                out = ws.recv()
                
                if isinstance(out, str):
                    try:
                        message = json.loads(out)
                        msg_type = message.get('type', 'unknown')
                        
                        if msg_type == 'executing':
                            data = message.get('data', {})
                            msg_prompt_id = data.get('prompt_id')
                            node = data.get('node')
                            
                            if node is None and msg_prompt_id == prompt_id:
                                print("Execution complete.")
                                execution_complete = True
                                break
                            elif msg_prompt_id == prompt_id and node is not None:
                                print("Executing node: {}".format(node))
                        elif msg_type == 'progress':
                            # Progress updates
                            data = message.get('data', {})
                            if data.get('prompt_id') == prompt_id:
                                value = data.get('value', 0)
                                max_value = data.get('max', 1)
                                if max_value > 0:
                                    print("Progress: {}/{}".format(value, max_value))
                        elif msg_type == 'status':
                            # Status updates
                            pass
                        else:
                            # Debug: print unknown message types
                            if os.environ.get('COMFYUI_DEBUG'):
                                print("DEBUG: Received message type: {}".format(msg_type))
                    except json.JSONDecodeError:
                        print("Warning: Received non-JSON message: {}".format(out[:100]))
                        
            except Exception as e:
                # On any error (timeout, connection issue, etc.), poll history as fallback
                current_time = time.time()
                if current_time - last_poll_time >= poll_interval:
                    last_poll_time = current_time
                    try:
                        history = get_history(prompt_id)
                        if prompt_id in history:
                            # Check if execution is complete
                            status = history[prompt_id].get('status', {})
                            if status.get('completed') or status.get('status_str') == 'success':
                                print("Execution complete (detected via history).")
                                execution_complete = True
                                break
                            # Also check if outputs exist (execution might be done)
                            if history[prompt_id].get('outputs'):
                                print("Execution complete (outputs found in history).")
                                execution_complete = True
                                break
                    except Exception as hist_err:
                        # If we can't get history, continue waiting
                        pass
        
        if not execution_complete:
            print("Warning: Execution may not have completed. Checking history anyway...")
        
        # Fetch history to get results
        print("\nFetching execution history...")
        history = get_history(prompt_id)
        
        if prompt_id not in history:
            print("Error: Prompt ID not found in history.")
            return
        
        # Extract outputs from history
        outputs = history[prompt_id].get('outputs', {})
        
        print("\n=== Execution Results ===")
        if '1' in outputs:
            node_output = outputs['1']
            if 'generated_text' in node_output:
                generated_text = node_output['generated_text']
                if isinstance(generated_text, (list, tuple)) and len(generated_text) > 0:
                    print("\nGenerated Text:")
                    print(generated_text[0])
                elif isinstance(generated_text, str):
                    print("\nGenerated Text:")
                    print(generated_text)
                else:
                    print("\nGenerated Text (raw):")
                    print(generated_text)
            else:
                print("\nNode output keys: {}".format(list(node_output.keys())))
                print("Full output: {}".format(json.dumps(node_output, indent=2)))
        else:
            print("\nNo output found for node '1'")
            print("Available nodes in outputs: {}".format(list(outputs.keys())))
            print("\nFull history:")
            print(json.dumps(history, indent=2))

    except urllib.error.URLError as e:
        print("Error: Could not connect to ComfyUI HTTP API. Is it running on {}?".format(server_address))
        print("Details: {}".format(e))
    except Exception as e:
        print("Error: {}".format(e))
        import traceback
        traceback.print_exc()
    finally:
        if 'ws' in locals():
            try:
                ws.close()
            except:
                pass

if __name__ == "__main__":
    run_workflow()
