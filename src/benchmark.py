#!/usr/bin/env python3
"""Cerebras Gemma 4 31B throughput/latency benchmark.
Non-streaming, uses server time_info. Outputs total e2e latency including network.
"""

from __future__ import annotations
import argparse, os, time, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras
load_dotenv()

RTT_MS = 150  # measured network round-trip

@dataclass
class BenchRow:
    ctx: int; think: str; conc: int
    ttft_ms: float = 0.0
    decode_ts: float = 0.0
    e2e_ts: float = 0.0
    agg_ts: float = 0.0
    total_ms: float = 0.0  # server + network estimate

def run_bench(client, ctx=512, think="off", conc=1, output_toks=160):
    prompt = " ".join(["benchmark"] * (max(ctx, 100) // 2))
    body: dict[str, Any] = {"model": "gemma-4-31b", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}
    if think == "low":
        body["reasoning_effort"] = "low"
        body["max_completion_tokens"] = output_toks * 2 + 64
    else:
        body["max_completion_tokens"] = output_toks

    def _single(_):
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(**body)
            wall = (time.perf_counter() - t0) * 1000
            ti = resp.time_info
            usage = resp.usage
            r_tok = usage.completion_tokens_details.reasoning_tokens if usage.completion_tokens_details else 0
            return {
                "wall_ms": wall, "ttft_ms": (ti.queue_time + ti.prompt_time) * 1000,
                "model_ms": ti.total_time * 1000,
                "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                "reasoning_tokens": r_tok, "output_tokens": usage.completion_tokens - r_tok,
            }
        except Exception as e:
            return {"error": str(e)}

    t_start = time.perf_counter()
    if conc == 1: results = [_single(0)]
    else:
        with ThreadPoolExecutor(max_workers=conc) as ex:
            futs = [ex.submit(_single, i) for i in range(conc)]
            results = [f.result() for f in as_completed(futs)]
    total_wall = (time.perf_counter() - t_start) * 1000

    valid = [r for r in results if "error" not in r]
    if not valid: return BenchRow(ctx=ctx, think=think, conc=conc)

    avg_ttft = sum(r["ttft_ms"] for r in valid) / len(valid)
    total_out = sum(r["output_tokens"] for r in valid)
    decode_rates = []
    for r in valid:
        gen = r["model_ms"] - r["ttft_ms"]
        ot = r["output_tokens"]
        if gen > 0 and ot > 0: decode_rates.append(ot / (gen / 1000))
    avg_decode = sum(decode_rates) / len(decode_rates) if decode_rates else 0

    # Total e2e = max(server_time across batch) + network RTT
    max_server = max(r["model_ms"] for r in valid)
    total_e2e = max_server + RTT_MS

    return BenchRow(
        ctx=ctx, think=think, conc=conc,
        ttft_ms=round(avg_ttft, 1),
        decode_ts=round(avg_decode),
        e2e_ts=round(total_out / (max_server / 1000)) if max_server > 0 else 0,
        agg_ts=round(total_out / (total_wall / 1000)) if total_wall > 0 else 0,
        total_ms=round(total_e2e),
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", nargs="+", type=int, default=[512, 2048])
    p.add_argument("--think", nargs="+", default=["off", "low"])
    p.add_argument("--conc", nargs="+", type=int, default=[1, 4, 8])
    p.add_argument("--output", type=int, default=160)
    args = p.parse_args()
    client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])

    print(f"\nThroughput: gemma-4-31b (context x thinking x concurrency)")
    print(f"(output capped at {args.output} tok; network RTT ~{RTT_MS}ms)")
    print(f"  total e2e = max server time in batch + {RTT_MS}ms network\n")
    print(f"  {'ctx':>5} {'think':>6} {'conc':>4} {'TTFTms':>8} {'dec t/s':>8} {'server ms':>9} {'net ms':>7} {'TOTAL ms':>9} {'agg t/s':>8}")
    print(f"  {'-'*65}")

    results = []
    for ctx in args.ctx:
        for think in args.think:
            for conc in args.conc:
                print(f"  > ctx={ctx} think={think} conc={conc}...", end=" ", flush=True)
                row = run_bench(client, ctx=ctx, think=think, conc=conc, output_toks=args.output)
                results.append(row)
                print(f"TTFT={row.ttft_ms}ms decode={row.decode_ts}t/s total={row.total_ms}ms")
                time.sleep(0.5)

    print(f"\n  {'ctx':>5} {'think':>6} {'conc':>4} {'TTFTms':>8} {'dec t/s':>8} {'server ms':>9} {'net ms':>7} {'TOTAL ms':>9} {'agg t/s':>8}")
    print(f"  {'-'*65}")
    for r in results:
        # Estimate server compute time per request
        server_ms = r.ttft_ms + (args.output / max(r.decode_ts, 1) * 1000)
        print(f"  {r.ctx:>5} {r.think:>6} {r.conc:>4} {r.ttft_ms:>7.0f}ms {r.decode_ts:>8} {server_ms:>7.0f}ms {RTT_MS:>5}ms {r.total_ms:>7.0f}ms {r.agg_ts:>8}")
    print(f"\n  --- {len(results)} configs ---")

if __name__ == "__main__":
    main()
