# eval/run_eval.py
# quick-n-dirty eval on a JSONL test set

from __future__ import annotations
import argparse, json, time, statistics, pathlib
from agent.researcher import research

def _kw_hits(text: str, kws: list[str]) -> int:
    t = (text or "").lower()
    return sum(1 for k in kws if k.lower() in t)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="eval/testset.jsonl")
    p.add_argument("--model", default="hf:distilbart")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--max-chars", dest="max_chars", type=int, default=900)
    p.add_argument("--n", type=int, default=0, help="limit #rows (0 = all)")
    args = p.parse_args()

    path = pathlib.Path(args.file)
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if args.n and len(rows) >= args.n:
                break

    times = []
    any_hits = 0
    print(f"# eval on {len(rows)} queries | model={args.model} k={args.k} max_chars={args.max_chars}\n")

    for i, r in enumerate(rows, 1):
        q = r["query"]
        kws = r.get("keywords", [])
        t0 = time.time()
        out = research(q, k=args.k, model=args.model, max_chars=args.max_chars)
        dt = time.time() - t0
        times.append(dt)
        hits = _kw_hits(out, kws)
        any_hits += 1 if hits > 0 else 0
        print(f"{i:02d} | {dt:5.1f}s | hits={hits} | {q}")

    avg = sum(times)/len(times)
    med = statistics.median(times)
    p95 = statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times)
    acc = any_hits/len(rows)

    print("\n--- summary ---")
    print(f"accuracy(any keyword) : {acc:.2f}")
    print(f"latency avg/median/p95: {avg:.1f}s / {med:.1f}s / {p95:.1f}s")

if __name__ == "__main__":
    main()
