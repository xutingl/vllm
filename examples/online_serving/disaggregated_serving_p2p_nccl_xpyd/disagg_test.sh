#!/bin/bash

# =============================================================================
# vLLM Disaggregated Serving Script - P2P NCCL XpYd Architecture
#
# This script iterates through all valid disaggregated configurations
# on an 8-GPU system and benchmarks them.
#
# Constraint:
# prefill_tp * num_prefill_server + decode_tp * num_decode_server = 8
#
# =============================================================================

# Configuration - can be overridden via environment variables
MODEL=${MODEL:-meta-llama/Meta-Llama-3-70B-Instruct}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
PROXY_PORT=${PROXY_PORT:-30001}
TOTAL_GPUS=8
MIN_TP_SIZE=4

echo "Warning: P2P NCCL disaggregated prefill XpYd support for vLLM v1 is experimental and subject to change."
echo "Running full benchmark sweep for $TOTAL_GPUS GPUs."
echo "Constraint: prefill_tp * num_prefill + decode_tp * num_decode = $TOTAL_GPUS"
echo "Constraint: prefill_tp >= $MIN_TP_SIZE, decode_tp >= $MIN_TP_SIZE"
echo ""

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
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$num_gpus" -ne $TOTAL_GPUS ]; then
        echo "Error: This script is hardcoded for $TOTAL_GPUS GPUs, but found $num_gpus."
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

cleanup() {
    echo "Stopping everything…"
    trap - INT TERM        # prevent re-entrancy
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
      return 1 # Return failure
    fi

    sleep 1
  done
}

run_benchmark_config() {
    local p_tp=$1
    local p_num=$2
    local d_tp=$3
    local d_num=$4
    
    local config_name="${p_num}p${p_tp}tp_${d_num}d${d_tp}tp"
    echo "======================================================================"
    echo "Running benchmark for config: $config_name"
    echo "  Prefill Servers: $p_num, Prefill TP: $p_tp"
    echo "  Decode Servers: $d_num, Decode TP: $d_tp"
    echo "======================================================================"

    local PIDS=()
    local all_gpus=( $(seq 0 $((TOTAL_GPUS - 1))) )
    local current_gpu_idx=0

    # --- Generate Prefill Config ---
    local PREFILL_GPUS_LIST=()
    local PREFILL_PORTS_LIST=()
    for i in $(seq 1 $p_num); do
        local port=$((20001 + i * 2)) # 20003, 20005, ...
        PREFILL_PORTS_LIST+=($port)
        
        local gpus_for_this_server=()
        for j in $(seq 1 $p_tp); do
            gpus_for_this_server+=(${all_gpus[$current_gpu_idx]})
            current_gpu_idx=$((current_gpu_idx + 1))
        done
        # Create comma-separated string for this server's CUDA_VISIBLE_DEVICES
        PREFILL_GPUS_LIST+=("$(IFS=,; echo "${gpus_for_this_server[*]}")")
    done
    local PREFILL_PORTS=$(IFS=,; echo "${PREFILL_PORTS_LIST[*]}")

    # --- Generate Decode Config ---
    local DECODE_GPUS_LIST=()
    local DECODE_PORTS_LIST=()
    for i in $(seq 1 $d_num); do
        local port=$((20001 + (p_num + i) * 2)) # Continue ports
        DECODE_PORTS_LIST+=($port)

        local gpus_for_this_server=()
        for j in $(seq 1 $d_tp); do
            gpus_for_this_server+=(${all_gpus[$current_gpu_idx]})
            current_gpu_idx=$((current_gpu_idx + 1))
        done
        DECODE_GPUS_LIST+=("$(IFS=,; echo "${gpus_for_this_server[*]}")")
    done
    local DECODE_PORTS=$(IFS=,; echo "${DECODE_PORTS_LIST[*]}")

    echo "Generated Configuration:"
    echo "  Model: $MODEL"
    echo "  Prefill Ports: $PREFILL_PORTS"
    echo "  Decode Ports: $DECODE_PORTS"
    echo "  Proxy Port: $PROXY_PORT"
    echo ""

    # =============================================================================
    # Launch Proxy Server
    # =============================================================================
    echo "Starting proxy server on port $PROXY_PORT..."
    python3 disagg_proxy_p2p_nccl_xpyd.py > "proxy_${config_name}.log" 2>&1 &
    PIDS+=($!)
    sleep 3 # Give proxy time to start

    # Parse port arrays
    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"

    # =============================================================================
    # Launch Prefill Servers (X Producers)
    # =============================================================================
    echo "Starting $p_num prefill server(s)..."
    for i in "${!PREFILL_GPUS_LIST[@]}"; do
        local gpu_ids=${PREFILL_GPUS_LIST[$i]}
        local port=${PREFILL_PORT_ARRAY[$i]}
        local kv_port=$((21001 + i))
        local log_file="prefill$((i+1))_${config_name}.log"

        echo "  Prefill server $((i+1)): GPUs $gpu_ids, Port $port, TP $p_tp"
        CUDA_VISIBLE_DEVICES=$gpu_ids vllm serve $MODEL \
        --enforce-eager \
        --host 0.0.0.0 \
        --port $port \
        --tensor-parallel-size $p_tp \
        --seed 1024 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --no-enable-prefix-caching \
        --kv-transfer-config \
        "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"1e1\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$port\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" > $log_file 2>&1 &
        PIDS+=($!)
    done

    # =============================================================================
    # Launch Decode Servers (Y Decoders)
    # =============================================================================
    echo ""
    echo "Starting $d_num decode server(s)..."
    for i in "${!DECODE_GPUS_LIST[@]}"; do
        local gpu_ids=${DECODE_GPUS_LIST[$i]}
        local port=${DECODE_PORT_ARRAY[$i]}
        local kv_port=$((22001 + i))
        local log_file="decode$((i+1))_${config_name}.log"

        echo "  Decode server $((i+1)): GPUs $gpu_ids, Port $port, TP $d_tp"
        CUDA_VISIBLE_DEVICES=$gpu_ids vllm serve $MODEL \
        --enforce-eager \
        --host 0.0.0.0 \
        --port $port \
        --tensor-parallel-size $d_tp \
        --seed 1024 \
        --max-num-seqs 128 \
        --trust-remote-code \
        --gpu-memory-utilization 0.7 \
        --no-enable-prefix-caching \
        --kv-transfer-config \
        "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"8e9\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$port\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" > $log_file 2>&1 &
        PIDS+=($!)
    done

    # =============================================================================
    # Wait for All Servers to Start
    # =============================================================================
    echo ""
    echo "Waiting for all servers to start..."
    for port in "${PREFILL_PORT_ARRAY[@]}" "${DECODE_PORT_ARRAY[@]}"; do
        if ! wait_for_server $port; then
            echo "Failed to start server on port $port for config $config_name"
            echo "Cleaning up this run and moving to next config..."
            pkill -9 -f "disagg_proxy_p2p_nccl_xpyd.py"
            for pid in "${PIDS[@]}"; do kill $pid > /dev/null 2>&1; done
            wait
            return 1 # Signal failure
        fi
    done

    echo ""
    echo "All servers are up. Starting benchmark for $config_name..."

    # =============================================================================
    # Run Benchmark
    # =============================================================================
    cd ../../../benchmarks/
    local bench_log="benchmark_${config_name}.log"
    local RESULT_FILENAME="disagg_P_tp${p_tp}_dp${p_num}_D_tp${d_tp}_dp${d_num}_concurrency64.json"
        
    vllm bench serve --port 10001 --seed $(date +%s) \
        --model $MODEL \
        --dataset-name random --random-input-len 1024 --random-output-len 2048 \
        --save-result \
        --save-detailed \
        --result-filename "$RESULT_FILENAME" \
        --request-rate 10 \
        --num-prompts 256 | tee $bench_log
    
    cd - # Go back to original directory

    echo "Benchmarking for $config_name done. Cleaning up..."

    # =============================================================================
    # Local Cleanup for this run
    # =============================================================================
    pkill -9 -f "disagg_proxy_p2p_nccl_xpyd.py"
    for pid in "${PIDS[@]}"; do
        kill $pid > /dev/null 2>&1
    done
    wait # Reap children
    echo "Cleanup for $config_name complete."
    sleep 5 # Grace period before next run
}


main() {
    check_required_files
    check_hf_token
    check_num_gpus
    ensure_python_library_installed pandas
    ensure_python_library_installed datasets
    ensure_python_library_installed vllm
    ensure_python_library_installed quart

    # Global trap for Ctrl+C
    trap cleanup INT
    trap cleanup USR1
    trap cleanup TERM

    echo "Starting benchmark sweep..."

    # Iterate over all combinations
    # Max P_tp = 8 - (2*1) = 6
    for p_tp in $(seq $MIN_TP_SIZE $((TOTAL_GPUS - MIN_TP_SIZE))); do
        # Max P_num = (8 - (2*1)) / p_tp
        max_p_num=$(( (TOTAL_GPUS - MIN_TP_SIZE) / p_tp ))
        for p_num in $(seq 1 $max_p_num); do
            gpus_used_by_prefill=$((p_tp * p_num))
            gpus_remaining=$((TOTAL_GPUS - gpus_used_by_prefill))

            if [ $gpus_remaining -lt $MIN_TP_SIZE ]; then
                continue
            fi

            # Max D_tp = gpus_remaining
            for d_tp in $(seq $MIN_TP_SIZE $gpus_remaining); do
                # Max D_num = gpus_remaining / d_tp
                if [ $((gpus_remaining % d_tp)) -ne 0 ]; then
                    continue # d_tp must evenly divide remaining gpus
                fi
                
                d_num=$((gpus_remaining / d_tp))
                
                # Final check (d_num must be >= 1)
                if [ $d_num -ge 1 ]; then
                    # This is a valid combination
                    # Call the benchmark function
                    run_benchmark_config $p_tp $p_num $d_tp $d_num
                fi
            done
        done
    done

    echo "All benchmark combinations complete."
    exit 0
}

# Run the script
main