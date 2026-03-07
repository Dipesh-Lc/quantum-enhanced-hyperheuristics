from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=str, default="results/operator_only/operator_only_seed0.json")
    ap.add_argument("--outfile", type=str, default="results/operator_only/plot_seed0.png")
    args = ap.parse_args()

    data = json.loads(Path(args.infile).read_text())
    runs = data["runs"]

    plt.figure()
    for r in runs:
        y = np.array(r["cmax_hist"], dtype=float)
        plt.plot(y, label=r["operator"])
    plt.xlabel("step")
    plt.ylabel("makespan")
    plt.legend()
    outpath = Path(args.outfile)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Wrote: {outpath}")


if __name__ == "__main__":
    main()