from vllm import LLM, SamplingParams
import json
import pandas as pd
import time

def run_batch_inference(llm, input_file, output_file):
    prompts = []
    with open(input_file, "r") as f:
        for line in f.readlines():
            prompts.append(json.loads(line)["text"])

    max_tokens = 50
    sampling_params = SamplingParams(seed=42, max_tokens=max_tokens)

    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    input_prompt = []
    generated_text = []
    arrival_time = []
    last_token_time = []
    first_scheduled_time = []
    first_token_time = []
    time_in_queue = []
    finished_time = []
    scheduler_time = []
    model_forward_time = []
    model_execute_time = []
    for output in outputs:
        input_prompt.append(output.prompt)
        generated_text.append(output.outputs[0].text)
        metrics = output.metrics
        arrival_time.append(metrics.arrival_time)
        last_token_time.append(metrics.last_token_time)
        first_scheduled_time.append(metrics.first_scheduled_time)
        first_token_time.append(metrics.first_token_time)
        time_in_queue.append(metrics.time_in_queue)
        finished_time.append(metrics.finished_time)
        scheduler_time.append(metrics.scheduler_time)
        model_forward_time.append(metrics.model_forward_time)
        model_execute_time.append(metrics.model_execute_time)
    df = pd.DataFrame({
        "input_prompt": input_prompt,
        "generated_text": generated_text,
        "arrival_time": arrival_time,
        "last_token_time": last_token_time,
        "first_scheduled_time": first_scheduled_time,
        "first_token_time": first_token_time,
        "time_in_queue": time_in_queue,
        "finished_time": finished_time,
        "scheduler_time": scheduler_time,
        "model_forward_time": model_forward_time,
        "model_execute_time": model_execute_time
    })
    # df.to_csv(output_file, index=False)

    print(f"Total time taken: {end_time - start_time} seconds. Throughput: {len(prompts) * max_tokens / (end_time - start_time)} tokens per second")

if __name__ == "__main__":
    llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", max_model_len=1024, enforce_eager=True, max_num_batched_tokens=4096, tensor_parallel_size=2)
    run_batch_inference(llm, "lmsys_10.jsonl", "lmsys_10_ee.csv")