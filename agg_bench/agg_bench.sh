#!/bin/bash

# --- Configuration ---
# Set the model identifier for the Llama 3 70B model.
MODEL_ID="meta-llama/Meta-Llama-3-70B-Instruct"

# Define the different maximum concurrency levels you want to test.
CONCURRENCY_LEVELS=(4 8 16 32 64)

# Define the Tensor Parallel (TP) and "Data Parallel" (DP) combinations.
# The product of TP and DP should equal your total number of GPUs (8).
# NOTE: For single-node model parallelism, vLLM uses Pipeline Parallelism (PP).
# Since the server entrypoint does not have a `--data-parallel-size` flag, this
# script uses the functional `--pipeline-parallel-size` flag with the DP value.
PARALLEL_CONFIGS=(
    "4 2"  # TP=4, DP=2
    "2 4"  # TP=2, DP=4
    # "1 8"  # TP=1, DP=8
    "8 1"  # TP=8, DP=1
)

# --- Benchmark Environment Setup ---
# Create a directory to store the log files and results from the benchmark runs.
LOG_DIR="benchmark_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Ensure vLLM is installed in your Python environment. If you installed from source,
# it's recommended to use `pip install -e .` from the repo root.
echo "Starting vLLM benchmark for model: $MODEL_ID"
echo "Log files and results will be saved in the '$LOG_DIR' directory."
echo "--------------------------------------------------------"

# --- Benchmarking Loop ---
# This script uses a server-client model. It starts a vLLM server with a
# specific parallel config, benchmarks it, and then restarts it with a new config.

for config in "${PARALLEL_CONFIGS[@]}"; do
    # Read the TP and DP values from the configuration string.
    read -r TP DP <<< "$config"

    SERVER_LOG_FILE="${LOG_DIR}/server_tp${TP}_dp${DP}.log"
    echo "Starting server for TP=${TP}, DP=${DP} ... Log: ${SERVER_LOG_FILE}"

    # Start the vLLM OpenAI-compatible server in the background.
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_ID" \
        --tensor-parallel-size "$TP" \
        --data-parallel-size "$DP" \
        --max-num-seqs 128 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        > "$SERVER_LOG_FILE" 2>&1 &
    
    # Store the Process ID (PID) of the server.
    SERVER_PID=$!

    # Wait for the server to initialize. This may need adjustment.
    echo "Waiting for server (PID: $SERVER_PID) to start..."
    sleep 45

    # Check if the server started successfully before running clients.
    if ! kill -0 $SERVER_PID > /dev/null 2>&1; then
        echo "Server failed to start. Check server log: ${SERVER_LOG_FILE}"
        continue
    fi
    echo "Server started. Running benchmarks..."

    # Loop through each concurrency level and run the benchmark client.
    for concurrency in "${CONCURRENCY_LEVELS[@]}"; do
        
        CLIENT_LOG_FILE="${LOG_DIR}/client_tp${TP}_dp${DP}_concurrency${concurrency}.log"
        RESULT_FILENAME="benchmark_tp${TP}_dp${DP}_concurrency${concurrency}.json"

        echo "  Running benchmark: Concurrency=${concurrency}... Log: ${CLIENT_LOG_FILE}"

        # Execute the benchmark client against the running server.
        vllm bench serve \
            --backend openai \
            --model "$MODEL_ID" \
            --tokenizer "$MODEL_ID" \
            --dataset-name random \
            --random-input-len 1024 \
            --random-output-len 2048 \
            --num-prompts 256 \
            --max-concurrency "$((concurrency * DP))" \
            --save-result \
            --save-detailed \
            --result-dir "$LOG_DIR" \
            --result-filename "$RESULT_FILENAME" \
            > "$CLIENT_LOG_FILE" 2>&1
        
        echo "  Benchmark finished for Concurrency=${concurrency}."
        sleep 5 # Small pause between client runs.
    done

    # Stop the server after all concurrency tests for this config are done.
    echo "Stopping server (PID: $SERVER_PID)..."
    kill $SERVER_PID
    wait $SERVER_PID
    echo "Server stopped."
    echo "--------------------------------------------------------"
    sleep 10 # Pause before starting the next server config.
done

echo "All benchmarking tasks are completed."
echo "Check the '$LOG_DIR' directory for detailed results."

