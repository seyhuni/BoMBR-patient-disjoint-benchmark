import sys, json, numpy as np, pandas as pd
rng=np.random.default_rng(20260929); out={}
for tag in sys.argv[1:]:
    p=pd.read_csv(f'seg_per_image_{tag}.csv'); cls=[c for c in p.columns if c.startswith('dice_')]
    p['mean_dice']=p[cls].mean(axis=1)
    U=p.patient.unique(); res={}
    for c in cls+['mean_dice']:
        pm=p.groupby('patient')[c].mean()        # patient-weighted mean of per-field Dice (NaN for absent class skipped)
        bs=[pm.loc[rng.choice(U,len(U))].mean() for _ in range(3000)]
        res[c]=(float(pm.mean()),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5)))
    res['by_grade_mean_dice']={int(g):float(v) for g,v in p.groupby('mf_grade').mean_dice.mean().items()}
    res['by_grade_ret_dice']={int(g):float(v) for g,v in p.groupby('mf_grade').dice_Reticulin.mean().items()}
    out[tag]=res; print(tag, {k:tuple(round(x,3) for x in v) if isinstance(v,tuple) else {a:round(b,3) for a,b in v.items()} for k,v in res.items()})
json.dump(out,open('seg_summary_'+'_'.join(sys.argv[1:])+'.json','w'),indent=1)
