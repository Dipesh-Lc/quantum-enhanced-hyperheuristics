from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def steps_to_within_pct(hist: list[float], final: float, pct: float = 0.01) -> int:
    thr = (1.0 + pct) * float(final)
    for i, v in enumerate(hist):
        if float(v) <= thr:
            return i
    return len(hist) - 1


def auc(hist: list[float]) -> float:
    return float(np.sum(np.array(hist, dtype=float)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, default="results/static_hybrid")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--outfile", type=str, default="results/static_hybrid/summary.json")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    indir = Path(args.indir)

    rows = []
    for seed in seeds:
        f = indir / f"static_hybrid_seed{seed}.json"
        data = json.loads(f.read_text())
        hist = data["cmax_hist"]
        final = float(data["final_makespan"])
        rows.append({
            "seed": seed,
            "final": final,
            "steps_to_1pct": steps_to_within_pct(hist, final, pct=0.01),
            "auc": auc(hist),
            "runtime_sec": float(data["runtime_sec"]),
        })

    finals = np.array([r["final"] for r in rows], dtype=float)
    steps = np.array([r["steps_to_1pct"] for r in rows], dtype=float)
    aucs  = np.array([r["auc"] for r in rows], dtype=float)
    rts   = np.array([r["runtime_sec"] for r in rows], dtype=float)

    summary = {
        "n": int(len(rows)),
        "final_mean": float(finals.mean()),
        "final_std": float(finals.std(ddof=1)) if len(rows) > 1 else 0.0,
        "steps_to_1pct_mean": float(steps.mean()),
        "auc_mean": float(aucs.mean()),
        "runtime_mean_sec": float(rts.mean()),
        "rows": rows,
    }

    outpath = Path(args.outfile)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(summary, indent=2))
    print("Wrote:", outpath)

    print("\nStatic hybrid summary (mean over seeds):")
    print(
        f"- static_hybrid final={summary['final_mean']:.3f}±{summary['final_std']:.3f} "
        f"steps@1%={summary['steps_to_1pct_mean']:.1f} auc={summary['auc_mean']:.1f} "
        f"rt={summary['runtime_mean_sec']:.3f}s"
    )


if __name__ == "__main__":
    main()