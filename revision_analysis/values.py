import json, numpy as np, glob, pandas as pd
KR=json.load(open('key_runs.json')); ALL={}
for f in glob.glob('res_*.json'):
    d=json.load(open(f)); runs=d['runs'] if isinstance(d,dict) and 'runs' in d else d
    for r in runs:
        if r['model'] in ALL and len(ALL[r['model']])>len(r): continue
        ALL[r['model']]=r
V={}
rep=json.load(open('res_repro.json'))['pca_loading']; V['PCA_PCS']=rep['retained_pcs']; V['PCA_MASS']=f"{rep['descriptor_share_of_input_variance']*0+rep['descriptor_share_of_retained_variance']*100:.2f}"
leak=[abs(r['cv_fieldlevel']['qwk'][0]-r['cv_grouped']['qwk'][0]) for r in ALL.values() if r.get('cv_fieldlevel') and r.get('cv_grouped')]
V['LEAK_MAX']=f"{max(leak):.3f}"; V['LEAK_N']=len(leak)
pred=[r['test']['qwk'] for k,r in ALL.items() if (k.startswith('Predicted masks') or k.startswith('Reference-trained')) ]
V['PRED_DESC_RANGE']=f"{min(pred):.3f}–{max(pred):.3f}"
S=json.load(open('seg_test_ft.json'))['metrics']; V['FT_RET_DICE']=f"{S['Reticulin']['dice']:.3f}"; V['FT_MEAN_DICE']=f"{S['mean_dice']:.3f}"; V['FT_RET_REC']=f"{S['Reticulin']['recall']:.3f}"
A=pd.read_csv('agreement_pred_original_pred_reinhard_pred_ft.csv'); a=A[A.masks=='pred_ft'].set_index('descriptor')['ccc']
V['FT_CCC_RET']=f"{a['pct_reticulin']:.3f}"; comp=['ret_component_density','ret_mean_component_area','ret_median_elongation','ret_elongation_sd','ret_mean_perimeter','ret_lacunarity','ret_fractal_dim']
V['FT_CCC_RANGE']=f"{a[comp].min():.3f}–{a[comp].max():.3f}"
e=np.array(KR['emb_r50']['test_pred']); man=pd.read_csv('patient_safe_manifest.csv'); te=man[man.subset=='test']; y=te.mf_grade.to_numpy(); g=te.patient.to_numpy()
for k in ['early_pred_r50','early_gt_r50','pcaapp_pred_r50','pcaapp_gt_r50','late_pred_r50','late_gt_r50']:
    V['CHG_'+k]=int((np.array(KR[k]['test_pred'])!=e).sum())
V['BIN01_SENS']=f"{((e[y>=1])>=1).mean():.2f}"; V['BIN23_SENS']=f"{((e[y>=2])>=2).mean():.2f}"
cnn=np.array(KR['cnn']['test_pred']); V['BIN01_SENS_CNN']=f"{((cnn[y>=1])>=1).mean():.2f}"
for k in ['dino_svm','phk_svm']:
    p=np.array(KR[k]['test_pred']); V['BIN01_SENS_'+k]=f"{((p[y>=1])>=1).mean():.2f}"
pp=[(np.array(KR['emb_r50']['test_pred'])[g==u]==y[g==u]).mean() for u in sorted(set(g))]; V['PP_MIN']=f"{min(pp):.1f}"; V['PP_MAX']=f"{max(pp):.1f}"
V['LATE_W_pred']=KR['late_pred_r50']['best_params']; V['LATE_W_gt']=KR['late_gt_r50']['best_params']
json.dump(V,open('values.json','w'),indent=1); print(json.dumps(V,indent=1))
