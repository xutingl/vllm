#!/bin/bash

# =============================================================================
# vLLM Disaggregated Serving Script - P2P NCCL XpYd Architecture
# =============================================================================
# This script demonstrates disaggregated prefill and decode serving using
# P2P NCCL communication. The architecture supports various XpYd configurations:
#
# - 1P3D: 1 Prefill server + 3 Decode servers (current default)
# - 3P1D: 3 Prefill servers + 1 Decode server
# - etc.
#
# Configuration can be customized via environment variables:
#   MODEL: Model to serve
#   PREFILL_GPUS: Comma-separated GPU IDs for prefill servers
#   DECODE_GPUS: Comma-separated GPU IDs for decode servers
#   PREFILL_PORTS: Comma-separated ports for prefill servers
#   DECODE_PORTS: Comma-separated ports for decode servers
#   PROXY_PORT: Proxy server port used to setup XpYd connection.
#   TIMEOUT_SECONDS: Server startup timeout
# =============================================================================

# Configuration - can be overridden via environment variables
MODEL=${MODEL:-meta-llama/Meta-Llama-3-70B-Instruct}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
PROXY_PORT=${PROXY_PORT:-30001}

# Default 1P1D logical ports (original multi-port vars are no longer used)
PREFILL_PORT="20003"
DECODE_PORT="20005"
PREFILL_KV_PORT="21001"
DECODE_KV_PORT="22001"

echo "Warning: P2P NCCL disaggregated prefill XpYd support for vLLM v1 is experimental and subject to change."
echo ""
echo "Architecture Configuration:"
echo "  Model: $MODEL"
echo "  Proxy Port: $PROXY_PORT"
echo "  Timeout: ${TIMEOUT_SECONDS}s"
echo ""

PIDS=()

# Switch to the directory of the current script
cd "$(dirname "${BASH_SOURCE[0]}")"

check_required_files() {
    local files=("disagg_proxy_p2p_nccl_xpyd.py")
    for file in "${files[@]}"; do
        if [[ ! -f "$file" ]]; then
            echo "Required file $file not found in $(pwd)"
            exit 1
        fi
    done
}

check_hf_token() {
    if [ -z "$HF_TOKEN" ]; then
        echo "HF_TOKEN is not set. Please set it to your Hugging Face token."
        echo "Example: export HF_TOKEN=your_token_here"
        exit 1
    fi
    if [[ "$HF_TOKEN" != hf_* ]]; then
        echo "HF_TOKEN is not a valid Hugging Face token. Please set it to your Hugging Face token."
        exit 1
    fi
    echo "HF_TOKEN is set and valid."
}

check_num_gpus() {
    # Check if the number of GPUs are >=8 via nvidia-smi
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$num_gpus" -lt 8 ]; then
        echo "You need at least 8 GPUs to run this benchmark script."
        exit 1
    else
        echo "Found $num_gpus GPUs."
    fi
}

ensure_python_library_installed() {
    echo "Checking if $1 is installed..."
    if ! python3 -c "import $1" > /dev/null 2>&1; then
        echo "$1 is not installed. Please install it via pip install $1."
        exit 1
    else
        echo "$1 is installed."
    fi
}

# Partial cleanup for in-between loop iterations
partial_cleanup() {
    echo "Stopping components for this run..."
    trap - INT TERM USR1 # Disable main trap temporarily
    for pid in "${PIDS[@]}"; do
        kill $pid > /dev/null 2>&1
    done
    pkill -f "disagg_proxy_p2p_nccl_xpyd.py"
    wait
    PIDS=()
    trap main_cleanup INT TERM USR1 # Re-enable main trap
    sleep 5 # Give ports time to free up
}

# Main cleanup trap to stop everything
main_cleanup() {
    echo "Stopping everything…"
    trap - INT TERM USR1       # prevent re-entrancy
    pkill -9 -f "disagg_proxy_p2p_nccl_xpyd.py"
    kill -- -$$            # negative PID  ==  "this whole process-group"
    wait                   # reap children so we don't leave zombies
    exit 0
}

wait_for_server() {
  local port=$1
  local timeout_seconds=$TIMEOUT_SECONDS
  local start_time=$(date +%s)

  echo "Waiting for server on port $port..."

  while true; do
    if curl -s "localhost:${port}/v1/completions" > /dev/null; then
      echo "Server on port $port is ready."
      return 0
    fi

    local now=$(date +%s)
    if (( now - start_time >= timeout_seconds )); then
      echo "Timeout waiting for server on port $port"
      return 1
    fi

    sleep 1
  done
}

main() {
    check_required_files
    check_hf_token
    check_num_gpus
    ensure_python_library_installed pandas
    ensure_python_library_installed datasets
    ensure_python_library_installed vllm
    ensure_python_library_installed quart

    # Set main traps
    trap main_cleanup INT
    trap main_cleanup USR1
    trap main_cleanup TERM

    # Define all benchmark configurations
    # Columns: P_GPUS | D_GPUS | P_TP | D_TP | LOG_SUFFIX
    # DP is inferred by vLLM (num_gpus / TP)
    read -r -d '' BENCH_CONFIGS << EOM
0,1|2,3,4,5,6,7|2|2|2|6|2p6d_p-tp2_d-tp2
0,1,2,3|4,5,6,7|2|2|4|4|4p4d_p-tp2_d-tp2
0,1,2,3|4,5,6,7|2|4|4|4|4p4d_p-tp2_d-tp4
0,1,2,3|4,5,6,7|4|2|4|4|4p4d_p-tp4_d-tp2
0,1,2,3|4,5,6,7|4|4|4|4|4p4d_p-tp4_d-tp4
0,1,2,3,4,5|6,7|2|2|6|2|6p2d_p-tp2_d-tp2
EOM

    # Iterate over each configuration
    echo "$BENCH_CONFIGS" | while IFS='|' read -r P_GPUS D_GPUS P_TP D_TP NUM_P_GPUS NUM_D_GPUS LOG_SUFFIX; do
        P_DP=$((NUM_P_GPUS / P_TP))
        D_DP=$((NUM_D_GPUS / D_TP))
        echo ""
        echo "======================================================================"
        echo "Starting Benchmark for Config: $LOG_SUFFIX"
        echo "  Prefill GPUs: $P_GPUS (TP=$P_TP, DP=$P_DP)"
        echo "  Decode GPUs: $D_GPUS (TP=$D_TP, DP=$D_DP)"
        echo "======================================================================"
        echo ""

        PIDS=() # Reset PIDS for this run

        # =============================================================================
        # Launch Proxy Server
        # =============================================================================
        echo "Starting proxy server on port $PROXY_PORT..."
        python3 disagg_proxy_p2p_nccl_xpyd.py > "proxy_${LOG_SUFFIX}.log" 2>&1 &
        PIDS+=($!)
        sleep 3 # Give proxy a moment to start

        # =============================================================================
        # Launch Prefill Server (1 logical service)
        # =============================================================================
        echo "Starting 1 prefill server... (TP=$P_TP, DP=$P_DP)"
        echo "  Prefill server: GPUs $P_GPUS, Port $PREFILL_PORT, KV Port $PREFILL_KV_PORT"

        local PREFILL_KV_CONFIG
        PREFILL_KV_CONFIG="{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"1e1\",\"kv_port\":\"$PREFILL_KV_PORT\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$PREFILL_PORT\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}"

        CUDA_VISIBLE_DEVICES=$P_GPUS vllm serve $MODEL \
        --enforce-eager \
        --host 0.0.0.0 \
        --port $PREFILL_PORT \
        --tensor-parallel-size $P_TP \
        --data-parallel-size $P_DP \
        --seed 1024 \
        --max-num-seqs 64 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --kv-transfer-config "$PREFILL_KV_CONFIG" > "prefill_${LOG_SUFFIX}.log" 2>&1 &
        PIDS+=($!)

        # =============================================================================
        # Launch Decode Server (1 logical service)
        # =============================================================================
        echo ""
        echo "Starting 1 decode server... (TP=$D_TP, DP=$D_DP)"
        echo "  Decode server: GPUs $D_GPUS, Port $DECODE_PORT, KV Port $DECODE_KV_PORT"

        local DECODE_KV_CONFIG
        DECODE_KV_CONFIG="{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"8e9\",\"kv_port\":\"$DECODE_KV_PORT\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$DECODE_PORT\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}"

        CUDA_VISIBLE_DEVICES=$D_GPUS vllm serve $MODEL \
        --enforce-eager \
        --host 0.0.0.0 \
        --port $DECODE_PORT \
        --tensor-parallel-size $D_TP \
        --data-parallel-size $D_DP \
        --seed 1024 \
        --max-num-seqs 64 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --kv-transfer-config "$DECODE_KV_CONFIG" > "decode_${LOG_SUFFIX}.log" 2>&1 &
        PIDS+=($!)

        # =============================================================================
        # Wait for All Servers to Start
        # =============================================================================
        echo ""
        echo "Waiting for all servers to start..."
        if ! wait_for_server $PREFILL_PORT; then
            echo "Failed to start prefill server for config $LOG_SUFFIX"
            partial_cleanup
            continue
        fi
        if ! wait_for_server $DECODE_PORT; then
            echo "Failed to start decode server for config $LOG_SUFFIX"
            partial_cleanup
            continue
        fi

        echo ""
        echo "All servers are up. Starting benchmark for $LOG_SUFFIX..."

        # =============================================================================
        # Run Benchmark
        # =============================================================================
        # Assuming the original script was in a dir one level below 'benchmarks'
        # Adjust path if needed
        cd ../../../benchmarks/
        if [ $? -ne 0 ]; then
            echo "Could not cd to benchmark directory. Please check path."
            partial_cleanup
            continue
        fi

        RESULT_FILENAME="disagg_P_tp${P_TP}_dp${P_DP}_D_tp${D_TP}_dp${D_DP}_concurrency64.json"
        
        vllm bench serve --port 10001 --seed $(date +%s) \
            --model $MODEL \
            --dataset-name random --random-input-len 1024 --random-output-len 2048 \
            --save-result \
            --save-detailed \
            --result-dir "disagg_log_concurrency_64" \
            --result-filename "$RESULT_FILENAME" \
            --num-prompts 256 --max-concurrency 64 | tee "benchmark_${LOG_SUFFIX}.log"
        
        cd - # Go back to original script dir

        echo "Benchmarking for $LOG_SUFFIX done. Cleaning up for next run..."
        partial_cleanup

    done

    echo ""
    echo "All benchmark runs completed."
}

main