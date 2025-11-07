#!/usr/bin/env python3
"""
Parse benchmark logs and extract metrics to CSV.

This script:
1. Takes a directory as input (e.g., "benchmark_output")
2. Iterates through subdirectories named like "P_tp2_dp1_D_tp2_dp3"
3. Extracts configuration (p_tp, p_dp, d_tp, d_dp) from directory names
4. Reads benchmark.log from each subdirectory to extract metrics
5. Saves results to benchmark.csv in the input directory
"""

import os
import re
import csv
import sys
import argparse


def parse_directory_name(dirname):
    """
    Parse directory name like "P_tp2_dp1_D_tp2_dp3" to extract:
    p_tp, p_dp, d_tp, d_dp
    
    Returns tuple (p_tp, p_dp, d_tp, d_dp) or None if pattern doesn't match
    """
    pattern = r'P_tp(\d+)_dp(\d+)_D_tp(\d+)_dp(\d+)'
    match = re.match(pattern, dirname)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
    return None


def extract_metrics_from_log(log_path):
    """
    Extract metrics from benchmark.log file.
    
    Returns dict with:
    - total_token_throughput (tok/s)
    - mean_ttft (ms)
    - median_ttft (ms)
    - mean_tpot (ms)
    - median_tpot (ms)
    """
    metrics = {}
    
    if not os.path.exists(log_path):
        print(f"Warning: {log_path} does not exist", file=sys.stderr)
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract Total Token throughput
    match = re.search(r'Total Token throughput \(tok/s\):\s+([\d.]+)', content)
    if match:
        metrics['total_token_throughput'] = float(match.group(1))
    else:
        print(f"Warning: Could not find Total Token throughput in {log_path}", file=sys.stderr)
        return None
    
    # Extract Mean TTFT
    match = re.search(r'Mean TTFT \(ms\):\s+([\d.]+)', content)
    if match:
        metrics['mean_ttft'] = float(match.group(1))
    else:
        print(f"Warning: Could not find Mean TTFT in {log_path}", file=sys.stderr)
        return None
    
    # Extract Median TTFT
    match = re.search(r'Median TTFT \(ms\):\s+([\d.]+)', content)
    if match:
        metrics['median_ttft'] = float(match.group(1))
    else:
        print(f"Warning: Could not find Median TTFT in {log_path}", file=sys.stderr)
        return None
    
    # Extract Mean TPOT
    match = re.search(r'Mean TPOT \(ms\):\s+([\d.]+)', content)
    if match:
        metrics['mean_tpot'] = float(match.group(1))
    else:
        print(f"Warning: Could not find Mean TPOT in {log_path}", file=sys.stderr)
        return None
    
    # Extract Median TPOT
    match = re.search(r'Median TPOT \(ms\):\s+([\d.]+)', content)
    if match:
        metrics['median_tpot'] = float(match.group(1))
    else:
        print(f"Warning: Could not find Median TPOT in {log_path}", file=sys.stderr)
        return None
    
    return metrics


def process_benchmark_directory(benchmark_dir):
    """
    Process all subdirectories in benchmark_dir and extract metrics.
    
    Returns list of dicts, each containing configuration and metrics.
    """
    results = []
    
    if not os.path.isdir(benchmark_dir):
        print(f"Error: {benchmark_dir} is not a directory", file=sys.stderr)
        return results
    
    # Iterate through all subdirectories
    for item in os.listdir(benchmark_dir):
        item_path = os.path.join(benchmark_dir, item)
        
        # Skip if not a directory
        if not os.path.isdir(item_path):
            continue
        
        # Parse directory name to extract configuration
        config = parse_directory_name(item)
        if config is None:
            print(f"Warning: Skipping directory '{item}' - name doesn't match expected pattern", file=sys.stderr)
            continue
        
        p_tp, p_dp, d_tp, d_dp = config
        
        # Read benchmark.log from this directory
        log_path = os.path.join(item_path, 'benchmark.log')
        metrics = extract_metrics_from_log(log_path)
        
        if metrics is None:
            print(f"Warning: Skipping {item} - could not extract metrics", file=sys.stderr)
            continue
        
        # Combine configuration and metrics
        result = {
            'p_tp': p_tp,
            'p_dp': p_dp,
            'd_tp': d_tp,
            'd_dp': d_dp,
            **metrics
        }
        results.append(result)
    
    return results


def write_csv(results, output_path):
    """
    Write results to CSV file.
    """
    if not results:
        print("Warning: No results to write", file=sys.stderr)
        return
    
    # Define column order
    fieldnames = [
        'p_tp', 'p_dp', 'd_tp', 'd_dp',
        'total_token_throughput',
        'mean_ttft',
        'median_ttft',
        'mean_tpot',
        'median_tpot'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Parse benchmark logs and extract metrics to CSV'
    )
    parser.add_argument(
        '--benchmark_dir',
        type=str,
        default='benchmark_output',
        help='Directory containing benchmark subdirectories (e.g., "benchmark_output")'
    )
    
    args = parser.parse_args()
    
    # Process benchmark directory
    results = process_benchmark_directory(args.benchmark_dir)
    
    if not results:
        print("Error: No valid benchmark results found", file=sys.stderr)
        sys.exit(1)
    
    # Write CSV to the benchmark directory
    output_path = os.path.join(args.benchmark_dir, 'benchmark.csv')
    write_csv(results, output_path)
    
    print(f"Processed {len(results)} benchmark configurations")


if __name__ == '__main__':
    main()

