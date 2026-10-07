"""Unified field-grade evaluation: locked test + repeated patient-grouped CV (+ optional field-level 'source-style' CV)."""
import sys, json, numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore')
import models as M, features as F
from sklearn.model_selection import GroupKFold
man=pd.read_csv('patient_safe_manifest.csv')
DEV=man[man.subset!='test'].filename.tolist(); TEST=man[man.subset=='test'].filename.tolist()
meta=man.set_index('filename')
LEGACY=['pct_bone','pct_fat','pct_cell','pct_reticulin','ret_component_count','ret_mean_component_area','ret_median_elongation',
        'ret_elongation_sd','ret_mean_perimeter','ret_to_cell_ratio','ret_lacunarity_proxy','ret_fractal_proxy']
def emb(name):
    d=pd.read_csv(f'emb/{name}.csv').set_index('filename'); return d[[c for c in d.columns if c.startswith('e')]]
def desc(tag, cols=None):
    if tag=='delivered':
        d=pd.concat([pd.read_csv('../derived_data/results/submitted_predicted_morphometrics_development.csv'),pd.read_csv('../derived_data/results/submitted_predicted_morphometrics_test.csv')]).set_index('filename'); return d[LEGACY]
    d=pd.read_csv(f'feat_{tag}.csv').set_index('filename'); return d[cols or F.COLS]
def mats(blocks):
    X=pd.concat(blocks,axis=1); return X.loc[DEV].to_numpy(float), X.loc[TEST].to_numpy(float)
y_dev=meta.loc[DEV,'mf_grade'].to_numpy(); y_te=meta.loc[TEST,'mf_grade'].to_numpy()
g_dev=meta.loc[DEV,'patient'].to_numpy(); g_te=meta.loc[TEST,'patient'].to_numpy()
def run(label, Xd, Xt, est, grid, repeats=10, leaky=False, scoring=M.QWK, save=None):
    t=time.time(); m,bp=M.fit_select(est,grid,Xd,y_dev,g_dev,GroupKFold(5),scoring=scoring); p=m.predict(Xt)
    r={'model':label,'n_dev':len(y_dev),'best_params':str(bp),'test':M.metrics(y_te,p),'test_ci':M.cluster_boot(y_te,p,g_te)}
    try: r['pca_components']=int(m.named_steps['pca'].n_components_)
    except Exception: pass
    if repeats:
        cv,_=M.repeated_cv(est,grid,Xd,y_dev,g_dev,repeats=repeats)
        r['cv_grouped']={k:(float(np.mean([c[k] for c in cv])),float(np.std([c[k] for c in cv]))) for k in cv[0]}
    if leaky:
        cvl,_=M.repeated_cv(est,grid,Xd,y_dev,g_dev,repeats=repeats,leaky=True)
        r['cv_fieldlevel']={k:(float(np.mean([c[k] for c in cvl])),float(np.std([c[k] for c in cvl]))) for k in cvl[0]}
    r['test_pred']=p.tolist(); r['seconds']=round(time.time()-t,1)
    print(f"{label:60s} test acc={r['test']['accuracy']:.3f} qwk={r['test']['qwk']:.3f} mae={r['test']['mae']:.3f} mf1rec={r['test']['recall_mf1']:.2f}"
          + (f" | gCV qwk={r['cv_grouped']['qwk'][0]:.3f}±{r['cv_grouped']['qwk'][1]:.3f}" if repeats else '')
          + (f" | fieldCV qwk={r['cv_fieldlevel']['qwk'][0]:.3f}±{r['cv_fieldlevel']['qwk'][1]:.3f}" if leaky else '') + f" [{r['seconds']}s]",flush=True)
    return r
