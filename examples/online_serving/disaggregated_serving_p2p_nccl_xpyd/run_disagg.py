#!/usr/bin/env python3

# =============================================================================
# vLLM Disaggregated Serving Script (Python Version) - P2P NCCL XpYd Architecture
# =============================================================================
# This script, converted from Bash, demonstrates disaggregated prefill and
# decode serving using P2P NCCL communication.
#
# Configuration is customized via environment variables:
#   MODEL: Model to serve
#   PREFILL_GPUS: Comma-separated GPU IDs for prefill servers
#   DECODE_GPUS: Comma-separated GPU IDs for decode servers
#   PREFILL_PORTS: Comma-separated ports for prefill servers
#   DECODE_PORTS: Comma-separated ports for decode servers
#   PROXY_PORT: Proxy server port used to setup XpYd connection.
#   TIMEOUT_SECONDS: Server startup timeout
# =============================================================================

import os
import sys
import subprocess
import signal
import time
import shutil
import json
import importlib.util
from pathlib import Path
from urllib import request, error
import urllib

# --- Configuration ---
MODEL = os.getenv("MODEL", "meta-llama/Meta-Llama-3-70B-Instruct")
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", 1200))
PROXY_PORT = int(os.getenv("PROXY_PORT", 30001))


# --- Global State ---
# Store subprocess.Popen objects
CHILD_PROCESSES = []

# --- Helper Functions ---

def print_config():
    """Prints the current configuration."""
    print("Warning: P2P NCCL disaggregated prefill XpYd support for vLLM v1 is experimental and subject to change.")
    print("")
    print("Architecture Configuration:")
    print(f"  Model: {MODEL}")
    print(f"  Proxy Port: {PROXY_PORT}")
    print(f"  Timeout: {TIMEOUT_SECONDS}s")
    print("")

def check_required_files():
    """Checks if required script files exist."""
    files_to_check = ["disagg_proxy_p2p_nccl_xpyd.py"]
    for file in files_to_check:
        if not Path(file).exists():
            print(f"Required file {file} not found in {os.getcwd()}")
            sys.exit(1)

def check_hf_token():
    """Checks for a valid Hugging Face token."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN is not set. Please set it to your Hugging Face token.")
        print("Example: export HF_TOKEN=your_token_here")
        sys.exit(1)
    if not hf_token.startswith("hf_"):
        print("HF_TOKEN is not a valid Hugging Face token. Please set it to your Hugging Face token.")
        sys.exit(1)
    print("HF_TOKEN is set and valid.")

def check_num_gpus():
    """Checks if at least 2 GPUs are available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        num_gpus = len(result.stdout.strip().split('\n'))
        if num_gpus < 2:
            print("You need at least 2 GPUs to run disaggregated prefill.")
            sys.exit(1)
        else:
            print(f"Found {num_gpus} GPUs.")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Error running nvidia-smi: {e}. Cannot verify GPU count.")
        sys.exit(1)

def ensure_python_library_installed(library_name):
    """Checks if a Python library is installed."""
    print(f"Checking if {library_name} is installed...")
    if importlib.util.find_spec(library_name) is None:
        print(f"{library_name} is not installed. Please install it via pip install {library_name}.")
        sys.exit(1)
    else:
        print(f"{library_name} is installed.")

def cleanup(signum=None, frame=None):
    """Cleans up all child processes."""
    print("\nStopping everything…")
    
    # Prevent re-entrancy
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    
    # Terminate all child processes
    for proc in reversed(CHILD_PROCESSES):
        if proc.poll() is None: # Check if process is still running
            try:
                proc.terminate()
            except ProcessLookupError:
                pass # Process already ended
                
    # Wait for a moment for processes to terminate
    time.sleep(2)

    # Force kill any remaining processes
    for proc in CHILD_PROCESSES:
        if proc.poll() is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    # Replicate pkill for the proxy
    try:
        subprocess.run(["pkill", "-9", "-f", "disagg_proxy_p2p_nccl_xpyd.py"], check=False)
    except FileNotFoundError:
        print("pkill command not found. Could not force-kill proxy.")

    # Replicate kill -- -$$ (kill process group)
    # This is more robust for stopping all descendants.
    try:
        os.killpg(os.getpgrp(), signal.SIGTERM)
    except Exception as e:
        print(f"Could not kill process group: {e}")
    
    time.sleep(10)

    sys.exit(0)

# def wait_for_server(port):
#     """Waits for a vLLM server to become ready."""
#     start_time = time.time()
#     url = f"http://localhost:{port}/v1/completions"
#     print(f"Waiting for server on port {port}...")

#     while True:
#         try:
#             with request.urlopen(url, timeout=1) as response:
#                 print(f"Response from server on port {port}: {response.status}")
#                 if response.status == 200 or 400 <= response.status < 500:
#                     print(f"Server on port {port} is ready.")
#                     return True
#         except (error.URLError, ConnectionRefusedError, TimeoutError):
#             pass # Server not ready yet

#         if time.time() - start_time >= TIMEOUT_SECONDS:
#             print(f"Timeout waiting for server on port {port}")
#             return False
        
#         time.sleep(1)

def wait_for_server(port):
    """
    Waits for a server to be ready on a specific port by polling an endpoint.
    
    Reads TIMEOUT_SECONDS from the environment, defaulting to 60 seconds.

    Args:
        port (int or str): The port number to check.

    Returns:
        bool: True if the server becomes ready, False if it times out.
    """
    # Get timeout from environment variable, default to 60 seconds
    timeout_seconds = 300

    start_time = time.time()
    
    url = f"http://localhost:{port}/v1/completions"
    print(f"Waiting for server on port {port} (timeout: {timeout_seconds}s)...")

    while True:
        try:
            # Try to open the URL. This is the equivalent of the curl command.
            # We set a short timeout (1s) for the request itself so the loop
            # doesn't hang on a non-responsive server.
            with urllib.request.urlopen(url, timeout=1) as response:
                # If we get here, the server responded (HTTP 2xx, 3xx).
                # This is a successful check.
                print(f"Server on port {port} is ready.")
                return True
        # 1. Catch the specific HTTPError first, as it has a .code
        except urllib.error.HTTPError as e:
            if e.code == 405:
                print(f"Server on port {port} is ready (received 405 Method Not Allowed).")
                return True
            
            # Other HTTP errors (e.g., 500, 503), server is up but not ready
            # print(f"Server responding with HTTP {e.code}, retrying...")
            pass # Continue to timeout check
        
        # 2. Catch the more general URLError (connection refused, timeouts)
        except urllib.error.URLError as e:
            # This catches non-HTTP errors (e.g., "Connection refused")
            # print(f"Server not reachable: {e.reason}, retrying...")
            pass  # Server not ready, continue to the timeout check
        
        except Exception as e:
            # Catch other potential errors
            print(f"An unexpected error occurred: {e}")
            pass # Continue trying

        now = time.time()
        if (now - start_time) >= timeout_seconds:
            print(f"Timeout waiting for server on port {port}")
            return False

        time.sleep(1)

def launch_proxy_server(output_dir):
    """Launches the disaggregated proxy server."""
    print(f"Starting proxy server on port {PROXY_PORT}...")
    log_file = open(output_dir / "proxy.log", "w+")
    cmd = ["python3", "disagg_proxy_p2p_nccl_xpyd.py"]
    
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file
    )
    CHILD_PROCESSES.append(proc)
    # Give proxy a moment to start
    time.sleep(1)
    print(f"Proxy server started on port {PROXY_PORT}")

def launch_vllm_server(
    role, gpu_id, port, kv_port, log_file, proxy_port, gpu_mem_util, kv_buffer_size, kv_send_type, tp_size
):
    """Launches a single vLLM server instance (prefill or decode)."""
    
    # 1. Create a copy of the current environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_id))

    # 2. Build the KV transfer config JSON
    kv_config = {
        "kv_connector": "P2pNcclConnector",
        "kv_role": role,
        "kv_buffer_size": str(kv_buffer_size),
        "kv_port": str(kv_port),
        "kv_connector_extra_config": {
            "proxy_ip": "0.0.0.0",
            "proxy_port": str(proxy_port),
            "http_port": str(port),
            "send_type": kv_send_type,
            "nccl_num_channels": "16"
        }
    }
    kv_config_str = json.dumps(kv_config)

    # 3. Build the full command
    cmd = [
        "vllm", "serve", MODEL,
        "--enforce-eager",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--tensor-parallel-size", str(tp_size),
        "--seed", "1024",
        "--max-num-seqs", "256",
        "--trust-remote-code",
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--kv-transfer-config", kv_config_str
    ]

    # 4. Launch the process
    log_file_handle = open(log_file, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file_handle,
        stderr=log_file_handle,
        env=env
    )
    CHILD_PROCESSES.append(proc)
    
def run_benchmark(log_path):
    """Runs the vLLM benchmark and tees output to a log file."""
    print("All servers are up. Starting benchmark...")
    
    # The benchmark script is assumed to be in ../../../benchmarks/
    benchmark_dir = Path(__file__).resolve().parent.parent.parent.parent / "benchmarks"
    if not benchmark_dir.is_dir():
        print(f"Benchmark directory not found at: {benchmark_dir}")
        print("Please check the relative path.")
        return

    cmd = [
        "vllm", "bench", "serve",
        "--port", "10001",
        "--seed", str(int(time.time())),
        "--model", MODEL,
        "--dataset-name", "random",
        "--random-input-len", "512",
        "--random-output-len", "512",
        "--num-prompts", "128",
        "--max-concurrency", "64",
        # "--burstiness", "100",
        # "--request-rate", "2"
    ]

    log_file_path = log_path / "benchmark.log"
    
    try:
        # Use Popen to stream output, mimicking `tee`
        process = subprocess.Popen(
            cmd,
            cwd=benchmark_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line-buffered
        )
        
        with open(log_file_path, "w+") as f:
            # Read stdout line by line
            for line in iter(process.stdout.readline, ''):
                sys.stdout.write(line) # Print to console
                f.write(line)          # Write to log file
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            print(f"Benchmark command failed with exit code {return_code}")
            
    except Exception as e:
        print(f"Failed to run benchmark: {e}")
    finally:
        print(f"Benchmark log saved to: {log_file_path}")

# --- Main Execution ---

def main(p_tp, p_dp, d_tp, d_dp):
    # Set process group ID so we can kill all children later
    try:
        os.setpgrp()
    except OSError:
        print("Could not set process group. Cleanup might be incomplete.")

    # Register signal handlers for cleanup
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    # Change to the script's directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    config_prefix = f"P_tp{p_tp}_dp{p_dp}_D_tp{d_tp}_dp{d_dp}"

    output_dir = script_dir / "benchmark_output" / config_prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"output_dir: {output_dir}")

    try:
        # 1. Print config and run checks
        print_config()
        check_required_files()
        check_hf_token()
        check_num_gpus()
        for lib in ["pandas", "datasets", "vllm", "quart"]:
            ensure_python_library_installed(lib)

        print("Launching disaggregated serving components...")
        print("Please check the log files for detailed output:")
        print("  - prefill*.log: Prefill server logs")
        print("  - decode*.log: Decode server logs")
        print("  - proxy.log: Proxy server log")
        
        # 2. Launch Proxy Server
        print("")
        launch_proxy_server(output_dir)


        prefill_gpus = [] # Each element is a list of gpus e.g. for tp=2 and dp=2, [0,1], [2,3]]
        prefill_ports = [] # each element corresponds to the port of the prefill server (one DP group)
        for dp_group in range(p_dp):
            dp_group_gpus = []
            for tp_group in range(p_tp):
                dp_group_gpus.append(tp_group + dp_group * p_tp)
            prefill_gpus.append(dp_group_gpus)
            prefill_ports.append(20001 + dp_group)

        decode_gpus = [] # Each element is a list of gpus, starting after the last prefill gpu e.g. for tp=2 and dp=2, [4,5], [6,7]]
        decode_ports = [] # each element corresponds to the port of the decode server (one DP group)
        total_prefill_gpus = p_tp * p_dp
        for dp_group in range(d_dp):
            dp_group_gpus = []
            for tp_group in range(d_tp):
                dp_group_gpus.append(tp_group + total_prefill_gpus + dp_group * d_tp)
            decode_gpus.append(dp_group_gpus)
            decode_ports.append(20001 + total_prefill_gpus + dp_group)
        
        print(f"prefill_gpus: {prefill_gpus}, ports: {prefill_ports}")
        print(f"decode_gpus: {decode_gpus}, ports: {decode_ports}")

        # 4. Launch Prefill Servers (X Producers)
        print("")
        print(f"Starting {len(prefill_gpus)} prefill server(s)...")
        for i, (gpu_id, port) in enumerate(zip(prefill_gpus, prefill_ports)):
            kv_port = 21001 + i * p_tp
            log_file = output_dir / f"prefill{i+1}.log"
            print(f"  Prefill server {i+1}: GPU {gpu_id}, Port {port}, KV Port {kv_port}")
            assert len(gpu_id) == p_tp
            launch_vllm_server(
                role="kv_producer",
                gpu_id=gpu_id,
                port=port,
                kv_port=kv_port,
                log_file=log_file,
                proxy_port=PROXY_PORT,
                gpu_mem_util=0.9,
                kv_buffer_size="1e1", # Small buffer for producer
                kv_send_type="PUT_ASYNC",
                tp_size=p_tp
            )
            
        # 5. Launch Decode Servers (Y Decoders)
        print("")
        print(f"Starting {len(decode_gpus)} decode server(s)...")
        for i, (gpu_id, port) in enumerate(zip(decode_gpus, decode_ports)):
            kv_port = 23001 + i * d_tp
            log_file = output_dir / f"decode{i+1}.log"
            print(f"  Decode server {i+1}: GPU {gpu_id}, Port {port}, KV Port {kv_port}")
            assert len(gpu_id) == d_tp
            launch_vllm_server(
                role="kv_consumer",
                gpu_id=gpu_id,
                port=port,
                kv_port=kv_port,
                log_file=log_file,
                proxy_port=PROXY_PORT,
                gpu_mem_util=0.9,
                kv_buffer_size="5e9", # Large buffer for consumer: 5e9 bytes = 5GB.
                kv_send_type="PUT_ASYNC",
                tp_size=d_tp
            )

        # 6. Wait for All Servers to Start
        print("")
        print("Waiting for all servers to start...")
        all_servers = prefill_ports + decode_ports
        for port in all_servers:
            if not wait_for_server(port):
                print(f"Failed to start server on port {port}")
                raise RuntimeError("Server startup failed")

        print("")

        # 7. Run Benchmark
        run_benchmark(output_dir)
        print("Benchmarking done.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Initiating cleanup...")
    finally:
        # 8. Cleanup
        cleanup()

if __name__ == "__main__":
    all_configs = [
        (2, 1, 2, 3),
        (2, 2, 2, 2),
        (4, 1, 4, 1),
        (2, 3, 2, 1),
    ]
    for p_tp, p_dp, d_tp, d_dp in all_configs:
        main(p_tp, p_dp, d_tp, d_dp)
    
    # main(p_tp=2, p_dp=1, d_tp=2, d_dp=3)