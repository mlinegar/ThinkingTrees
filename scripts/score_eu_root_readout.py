#!/usr/bin/env python3
"""Score eu_root_readout sweep. PRIMARY = g-stage (iter_02) doc Pearson; f-stage shown too.
The headline test: does a top-k leaf-score readout beat the mean-composed root on eu?"""
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
print(f"{'dim':10}{'readout':12}{'f-stage':>15}{'g-stage':>15}{'n':>4}")
for dim in ["eu","economic"]:
    for arm in ["meanroot","top1","top3","softmax02"]:
        cells=sorted(glob.glob(f"outputs/eu_root_readout/{dim}/{arm}"))
        if not cells: print(f"{dim:10}{arm:12}{'(pending)':>15}"); continue
        sds=sorted(d for d in glob.glob(f"{cells[0]}/seed_*") if "." not in d.split("/")[-1])
        fm,fs=agg(sds,"iter_01"); gm,gs=agg(sds,"iter_02")
        fstr=f"{fm:+.3f}±{fs:.3f}" if fm is not None else "--"
        gstr=f"{gm:+.3f}±{gs:.3f}" if gm is not None else "--"
        nv=sum(1 for s in sds if cr(s,"iter_02") is not None)
        print(f"{dim:10}{arm:12}{fstr:>15}{gstr:>15}{nv:>4}")
    print()
print("Baseline (mean_root): eu g≈0.30-0.37 | econ g≈0.73 | eu top1-leaf CEILING=0.79")
