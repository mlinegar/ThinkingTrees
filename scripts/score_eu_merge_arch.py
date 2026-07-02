#!/usr/bin/env python3
"""Score the eu_merge_arch sweep: g-stage TEST Pearson, gated vs mean, 3-seed mean+/-pstdev.

Reads outputs/eu_merge_arch/<dim>/mm_<mode>_rw<rw>/seed_<s>/fno/leafq016/
prediction_records/iter_0{1,2}_post_eval.jsonl, computes root-pred vs expert_score Pearson
per doc on the TEST split, and reports f-stage (iter_01) and g-stage (iter_02) per cell.
"""
import glob
import json
import statistics


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx <= 1e-12 or sy <= 1e-12:
        return 0.0  # constant predictor -> collapse
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / (sx * sy) ** 0.5


def cell_seed_r(seed_dir, iter_tag):
    fs = glob.glob(f"{seed_dir}/fno/leafq016/prediction_records/{iter_tag}_post_eval.jsonl")
    if not fs:
        return None
    preds, golds = [], []
    for line in open(fs[0]):
        r = json.loads(line)
        if r.get("split") != "test":
            continue
        preds.append(r["prediction"])
        golds.append(r["expert_score"])
    return pearson(preds, golds)


def agg(seed_dirs, iter_tag):
    rs = [cell_seed_r(sd, iter_tag) for sd in seed_dirs]
    valid = [r for r in rs if r is not None]
    if not valid:
        return None, None, 0, rs
    n_collapse = sum(1 for r in valid if abs(r) < 1e-9)
    mean = statistics.mean(valid)
    std = statistics.pstdev(valid) if len(valid) > 1 else 0.0
    return mean, std, n_collapse, rs


def main():
    base = "outputs/eu_merge_arch"
    # supmix reference (merge_supervision=none, same recipe) for sanity-check.
    ref = {("eu", "mean"): 0.328, ("economic", "mean"): 0.727}
    print(f"{'dim':10} {'mode':6} {'rw':>3}  {'f-stage':>16}  {'g-stage':>16}  {'ref(g,supmix)':>13}")
    for dim in ["eu", "economic"]:
        for mode in ["mean", "gated"]:
            cells = sorted(glob.glob(f"{base}/{dim}/mm_{mode}_rw*"))
            if not cells:
                continue
            cell = cells[0]
            rw = cell.split("_rw")[-1]
            seed_dirs = sorted(d for d in glob.glob(f"{cell}/seed_*") if "." not in d.split("/")[-1])
            fm, fs_, fc, _ = agg(seed_dirs, "iter_01")
            gm, gs, gc, grs = agg(seed_dirs, "iter_02")
            fstr = f"{fm:+.3f}±{fs_:.3f}" if fm is not None else "    --    "
            gstr = f"{gm:+.3f}±{gs:.3f}" if gm is not None else "    --    "
            ref_s = f"{ref.get((dim,mode),'')}" if (dim, mode) in ref else ""
            print(f"{dim:10} {mode:6} {rw:>3}  {fstr:>16}  {gstr:>16}  {ref_s:>13}"
                  + (f"  collapse {gc}/3" if gc else ""))
        print()


if __name__ == "__main__":
    main()
