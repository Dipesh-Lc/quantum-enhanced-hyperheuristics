from __future__ import annotations
import argparse, json
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


def load_bandit(indir: Path, seeds: list[int], policy: str):
    rows = []
    for seed in seeds:
        f = indir / f"bandit_{policy}_seed{seed}.json"
        d = json.loads(f.read_text())
        hist = d["bandit"]["cmax_hist"]
        rows.append({
            "seed": seed,
            "final": float(d["bandit"]["final_makespan"]),
            "steps_to_1pct": steps_to_within_pct(hist, pct=0.01),
            "auc": auc(hist),
        })
    finals = np.array([r["final"] for r in rows], float)
    steps  = np.array([r["steps_to_1pct"] for r in rows], float)
    aucs   = np.array([r["auc"] for r in rows], float)
    return {
        "n": len(rows),
        "final_mean": float(finals.mean()),
        "final_std": float(finals.std(ddof=1)) if len(rows) > 1 else 0.0,
        "steps_to_1pct_mean": float(steps.mean()),
        "auc_mean": float(aucs.mean()),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--operator_only_summary", type=str, default="results/operator_only/summary.json")
    ap.add_argument("--static_hybrid_summary", type=str, default="results/static_hybrid/summary.json")
    ap.add_argument("--bandit_dir", type=str, default="results/bandit_hh")
    ap.add_argument("--bandit_policy", type=str, default="ucb1")
    ap.add_argument("--outfile", type=str, default="results/final_comparison.json")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    op_sum = json.loads(Path(args.operator_only_summary).read_text())["summary"]
    hy_sum = json.loads(Path(args.static_hybrid_summary).read_text())
    bd_sum = load_bandit(Path(args.bandit_dir), seeds, args.bandit_policy)

    out = {
        "operator_only": op_sum,
        "static_hybrid": hy_sum,
        "bandit": bd_sum,
    }

    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    Path(args.outfile).write_text(json.dumps(out, indent=2))
    print("Wrote:", args.outfile)

    print("\n=== Final comparison (mean over seeds) ===")
    print("Operator-only:")
    for k, v in op_sum.items():
        print(f"- {k:22s} final={v['final_mean']:.3f}±{v['final_std']:.3f} "
              f"steps@1%={v['steps_to_1pct_mean']:.1f} auc={v['auc_mean']:.1f} rt={v['runtime_mean_sec']:.3f}s")

    print("\nStatic hybrid:")
    print(f"- static_hybrid         final={hy_sum['final_mean']:.3f}±{hy_sum['final_std']:.3f} "
          f"steps@1%={hy_sum['steps_to_1pct_mean']:.1f} auc={hy_sum['auc_mean']:.1f} rt={hy_sum['runtime_mean_sec']:.3f}s")

    print("\nBandit HH:")
    print(f"- bandit_{args.bandit_policy:14s} final={bd_sum['final_mean']:.3f}±{bd_sum['final_std']:.3f} "
          f"steps@1%={bd_sum['steps_to_1pct_mean']:.1f} auc={bd_sum['auc_mean']:.1f}")


if __name__ == "__main__":
    main()