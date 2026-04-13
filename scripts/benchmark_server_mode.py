#!/usr/bin/env python3
"""
Benchmark using vLLM OpenAI server mode to force concurrent request processing.

Unlike LLM.generate() which allows conservative batching, the server mode
receives requests continuously and MUST handle concurrency.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path
import subprocess
import sys
import signal

import aiohttp
import numpy as np


async def send_request(session, url, prompt, model, max_tokens, request_id):
    """Send a single completion request to the vLLM server."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False
    }

    start = time.time()
    async with session.post(f"{url}/v1/completions", json=payload) as response:
        result = await response.json()
        latency = (time.time() - start) * 1000  # ms

        return {
            "request_id": request_id,
            "latency_ms": latency,
            "output_tokens": len(result['choices'][0]['text'].split()),
            "finish_reason": result['choices'][0]['finish_reason']
        }


async def run_benchmark(server_url, prompts, model, max_tokens, concurrent_requests):
    """
    Run benchmark with controlled concurrency.

    Sends batches of concurrent_requests at a time to saturate the server.
    """
    print(f"Running benchmark: {len(prompts)} prompts, {concurrent_requests} concurrent")

    async with aiohttp.ClientSession() as session:
        all_results = []

        # Process in batches of concurrent_requests
        for i in range(0, len(prompts), concurrent_requests):
            batch = prompts[i:i+concurrent_requests]

            # Send all requests in this batch concurrently
            tasks = [
                send_request(session, server_url, prompt, model, max_tokens, i+j)
                for j, prompt in enumerate(batch)
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions
            valid_results = [r for r in batch_results if not isinstance(r, Exception)]
            all_results.extend(valid_results)

            print(f"Completed batch {i//concurrent_requests + 1}, "
                  f"{len(valid_results)}/{len(batch)} succeeded")

        return all_results


def start_vllm_server(model, gpu_mem_util, max_model_len, kv_policy, port=8000):
    """Start vLLM server with KV transfer config."""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--max-model-len", str(max_model_len),
        "--port", str(port),
        "--kv-connector", "OffloadingConnector",
        "--kv-role", "kv_both",
        "--kv-connector-extra-config",
        json.dumps({
            "cpu_bytes_to_use": 8_000_000_000,
            "block_size": 48,
            "eviction_policy": kv_policy,
            "log_evictions": True
        })
    ]

    print(f"Starting vLLM server: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for server to be ready
    print("Waiting for server to start...")
    time.sleep(30)  # Give it time to load model

    return process


def stop_server(process):
    """Stop the vLLM server gracefully."""
    print("Stopping server...")
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


async def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM in server mode")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--policies", nargs="+", default=["lru", "attention", "hybrid"])
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--concurrent-requests", type=int, default=50,
                       help="Number of concurrent requests to send (THIS IS KEY!)")
    parser.add_argument("--gpu-mem-util", type=float, default=0.45)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    # Load prompts
    with open(args.dataset_path) as f:
        data = json.load(f)
    prompts = [item['prompt'] for item in data[:args.num_prompts]]

    print(f"Loaded {len(prompts)} prompts")
    print(f"Concurrent requests: {args.concurrent_requests}")
    print(f"GPU memory: {args.gpu_mem_util}")
    print()

    all_results = []

    for policy in args.policies:
        print(f"\n{'='*70}")
        print(f"Testing policy: {policy}")
        print(f"{'='*70}\n")

        # Start server with this policy
        server_process = start_vllm_server(
            args.model,
            args.gpu_mem_util,
            args.max_model_len,
            policy,
            args.port
        )

        try:
            # Run benchmark
            server_url = f"http://localhost:{args.port}"

            start_time = time.time()
            results = await run_benchmark(
                server_url,
                prompts,
                args.model,
                args.max_tokens,
                args.concurrent_requests
            )
            total_time = time.time() - start_time

            # Calculate metrics
            latencies = [r['latency_ms'] for r in results]
            total_output_tokens = sum(r['output_tokens'] for r in results)

            metrics = {
                "policy": policy,
                "model": args.model,
                "num_prompts": len(results),
                "concurrent_requests": args.concurrent_requests,
                "total_time_seconds": total_time,
                "tokens_per_second": total_output_tokens / total_time,
                "requests_per_second": len(results) / total_time,
                "avg_latency_ms": float(np.mean(latencies)),
                "p50_latency_ms": float(np.percentile(latencies, 50)),
                "p95_latency_ms": float(np.percentile(latencies, 95)),
                "p99_latency_ms": float(np.percentile(latencies, 99)),
                "config": {
                    "gpu_memory_utilization": args.gpu_mem_util,
                    "max_model_len": args.max_model_len,
                    "max_tokens": args.max_tokens
                }
            }

            # TODO: Get eviction stats from server
            # Would need to add an endpoint to expose this
            metrics["total_evictions"] = 0  # Placeholder

            all_results.append(metrics)

            print(f"\nResults for {policy}:")
            print(f"  Throughput: {metrics['tokens_per_second']:.1f} tok/s")
            print(f"  Avg latency: {metrics['avg_latency_ms']:.1f} ms")
            print(f"  P95 latency: {metrics['p95_latency_ms']:.1f} ms")

        finally:
            stop_server(server_process)
            time.sleep(5)  # Cool down before next policy

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
