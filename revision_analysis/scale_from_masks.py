"""Estimate the test/development pixel-scale ratio from reference-mask morphology (Section 2.5):
(i) equivalent diameter of complete (border-free) fat components >= 30 px; (ii) mean reticulin width = 2 x mean distance transform."""
import cv2, numpy as np, pandas as pd
from skimage.measure import label, regionprops
man=pd.read_csv('patient_safe_manifest.csv'); D={'dev':[],'test':[]}; W={'dev':[],'test':[]}
for r in man.itertuples():
    m=cv2.imread(f'data/{r.split}/masks/{r.mask_file}',0); s='test' if r.subset=='test' else 'dev'; H,Wd=m.shape
    D[s]+=[p.equivalent_diameter_area for p in regionprops(label(m==1,connectivity=1)) if p.area>=30 and p.bbox[0]>0 and p.bbox[1]>0 and p.bbox[2]<H and p.bbox[3]<Wd]
    b=(m==0).astype(np.uint8)
    if b.sum()>=50: W[s].append(2*cv2.distanceTransform(b,cv2.DIST_L2,5)[b>0].mean())
d={k:float(np.median(v)) for k,v in D.items()}; w={k:float(np.median(v)) for k,v in W.items()}
print('adipocyte diameter (px)',d,'ratio %.2f'%(d['test']/d['dev'])); print('reticulin width (px)',w,'ratio %.2f'%(w['test']/w['dev']))
