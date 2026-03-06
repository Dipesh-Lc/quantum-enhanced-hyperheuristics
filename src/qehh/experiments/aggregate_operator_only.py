from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def steps_to_within_pct(hist: list[float], pct: float = 0.01) -> int:
    final = float(hist[-1])
    thr = (1.0 + pct) * final
    for i, v in enumerate(hist):
        if float(v) <= thr:
            return i
    return len(hist) - 1


def auc(hist: list[float]) -> float:
    return float(np.sum(np.array(hist, dtype=float)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, default="results/operator_only")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--outfile", type=str, default="results/operator_only/summary.json")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    indir = Path(args.indir)

    # op -> list of metrics dicts across seeds
    by_op = {}

    for seed in seeds:
        f = indir / f"operator_only_seed{seed}.json"
        data = json.loads(f.read_text())
        for run in data["runs"]:
            op = run["operator"]
            hist = run["cmax_hist"]
            rec = {
                "seed": seed,
                "final": float(run["final_makespan"]),
                "steps_to_1pct": int(steps_to_within_pct(hist, pct=0.01)),
                "auc": float(auc(hist)),
                "runtime_sec": float(run["runtime_sec"]),
            }
            by_op.setdefault(op, []).append(rec)

    summary = {}
    for op, rows in by_op.items():
        finals = np.array([r["final"] for r in rows], dtype=float)
        steps = np.array([r["steps_to_1pct"] for r in rows], dtype=float)
        aucs = np.array([r["auc"] for r in rows], dtype=float)
        rts = np.array([r["runtime_sec"] for r in rows], dtype=float)

        summary[op] = {
            "n": int(len(rows)),
            "final_mean": float(finals.mean()),
            "final_std": float(finals.std(ddof=1)) if len(rows) > 1 else 0.0,
            "steps_to_1pct_mean": float(steps.mean()),
            "auc_mean": float(aucs.mean()),
            "runtime_mean_sec": float(rts.mean()),
        }

    out = {"seeds": seeds, "summary": summary}
    outpath = Path(args.outfile)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2))
    print("Wrote:", outpath)

    # Print a human table
    ops = sorted(summary.keys())
    print("\nOperator summary (mean over seeds):")
    for op in ops:
        s = summary[op]
        print(
            f"- {op:22s} final={s['final_mean']:.3f}±{s['final_std']:.3f} "
            f"steps@1%={s['steps_to_1pct_mean']:.1f} auc={s['auc_mean']:.1f} "
            f"rt={s['runtime_mean_sec']:.3f}s"
        )


if __name__ == "__main__":
    main()