#!/usr/bin/env python3
"""Score the eu_leaf_balance sweep. PRIMARY readout = f-stage (iter_01): did balanced
leaf loss lift the weak eu leaf states off ~0.25? Secondary = g-stage (iter_02)."""
import glob
import json
import statistics


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx <= 1e-12 or sy <= 1e-12:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy) ** 0.5


def cell_r(seed_dir, it):
    f = glob.glob(f"{seed_dir}/fno/leafq016/prediction_records/{it}_post_eval.jsonl")
    if not f:
        return None
    p, g = [], []
    for line in open(f[0]):
        r = json.loads(line)
        if r.get("split") != "test":
            continue
        p.append(r["prediction"])
        g.append(r["expert_score"])
    return pearson(p, g)


def agg(seed_dirs, it):
    rs = [cell_r(s, it) for s in seed_dirs]
    v = [r for r in rs if r is not None]
    if not v:
        return None, None
    return statistics.mean(v), (statistics.pstdev(v) if len(v) > 1 else 0.0)


def main():
    # eu f-stage baseline (pw=1, mean merge) ~0.25; g-stage ~0.30 (from merge-arch sweep).
    print(f"{'dim':10}{'pos_w':7}{'f-stage(PRIMARY)':>20}{'g-stage':>16}")
    for dim in ["eu", "economic"]:
        for pw in ["1", "5", "20"]:
            cells = sorted(glob.glob(f"outputs/eu_leaf_balance/{dim}/pw{pw}"))
            if not cells:
                print(f"{dim:10}{pw:7}{'(pending)':>20}")
                continue
            sds = sorted(d for d in glob.glob(f"{cells[0]}/seed_*") if "." not in d.split("/")[-1])
            fm, fs = agg(sds, "iter_01")
            gm, gs = agg(sds, "iter_02")
            fstr = f"{fm:+.3f}±{fs:.3f}" if fm is not None else "--"
            gstr = f"{gm:+.3f}±{gs:.3f}" if gm is not None else "--"
            print(f"{dim:10}{pw:7}{fstr:>20}{gstr:>16}")
        print()
    print("Reference (pw=1, no balance): eu f≈0.25 g≈0.30 | econ f≈0.54 g≈0.73")


if __name__ == "__main__":
    main()
