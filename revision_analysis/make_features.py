import sys, numpy as np, pandas as pd, cv2, features as F, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'../original_pipeline'); from src.data import extract_morphometrics as legacy
man=pd.read_csv('patient_safe_manifest.csv'); src=sys.argv[1]   # gt | pred_<mode> | legacyrepro
rec=[]
for r in man.itertuples():
    if src=='gt': m=cv2.imread(f'data/{r.split}/masks/{r.mask_file}',0)
    else:
        mode=src.split('_',1)[1]; m=cv2.imread(f'pred/{mode}/{r.filename}',0)
        if m is None: continue
    if src=='legacyrepro_original':   # submitted code + submitted protocol (dev resized to 512, test native), for verification
        d=legacy(cv2.resize(m,(512,512),interpolation=cv2.INTER_NEAREST) if m.shape!=(512,512) else m)
    else:
        d=F.field_features(m,native=(len(sys.argv)<3 or sys.argv[2]!='legacyscale'))
    d.update(filename=r.filename,patient=r.patient,mf_grade=r.mf_grade,subset=r.subset); rec.append(d)
tag=src+('_legacyscale' if len(sys.argv)>2 else '')
pd.DataFrame(rec).to_csv(f'feat_{tag}.csv',index=False); print(tag,len(rec))
