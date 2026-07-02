#!/usr/bin/env python3
"""Score eu_encoder sweep. PRIMARY = f-stage (iter_01): does a stronger multilingual
encoder lift the eu leaf states off ~0.25? Secondary = g-stage."""
import glob, json, statistics
def pear(xs,ys):
    n=len(xs)
    if n<2: return None
    mx,my=sum(xs)/n,sum(ys)/n
    sx=sum((x-mx)**2 for x in xs); sy=sum((y-my)**2 for y in ys)
    if sx<=1e-12 or sy<=1e-12: return 0.0
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/(sx*sy)**0.5
def cr(sd,it):
    f=glob.glob(f"{sd}/fno/leafq016/prediction_records/{it}_post_eval.jsonl")
    if not f: return None
    p,g=[],[]
    for ln in open(f[0]):
        r=json.loads(ln)
        if r.get("split")!="test": continue
        p.append(r["prediction"]); g.append(r["expert_score"])
    return pear(p,g)
def agg(sds,it):
    rs=[cr(s,it) for s in sds]; v=[r for r in rs if r is not None]
    return (statistics.mean(v),(statistics.pstdev(v) if len(v)>1 else 0)) if v else (None,None)
print(f"{'dim':10}{'encoder':12}{'f-stage(PRIMARY)':>20}{'g-stage':>16}{'n_seed':>8}")
for dim in ["eu","economic"]:
    for enc in ["gemma300m","qwen0_6b","qwen4b"]:
        cells=sorted(glob.glob(f"outputs/eu_encoder/{dim}/{enc}"))
        if not cells: print(f"{dim:10}{enc:12}{'(pending)':>20}"); continue
        sds=sorted(d for d in glob.glob(f"{cells[0]}/seed_*") if "." not in d.split("/")[-1])
        fm,fs=agg(sds,"iter_01"); gm,gs=agg(sds,"iter_02")
        fstr=f"{fm:+.3f}±{fs:.3f}" if fm is not None else "--"
        gstr=f"{gm:+.3f}±{gs:.3f}" if gm is not None else "--"
        nv=sum(1 for s in sds if cr(s,"iter_02") is not None)
        print(f"{dim:10}{enc:12}{fstr:>20}{gstr:>16}{nv:>8}")
    print()
print("Baseline (gemma300m): eu f≈0.25 g≈0.30 | econ f≈0.54 g≈0.73")
