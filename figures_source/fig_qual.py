"""Qualitative panel: one locked-test field per MF grade (input / reference mask / predicted mask) + row-normalised pixel confusion."""
import sys, json, numpy as np, pandas as pd, cv2, matplotlib.pyplot as plt
from matplotlib.patches import Patch
from style import *
R='../revision_analysis/'; man=pd.read_csv(R+'patient_safe_manifest.csv'); te=man[man.subset=='test']
pred=sys.argv[1]; label=sys.argv[2]
per=pd.read_csv(R+f'seg_per_image_{pred}.csv')
def col(m): 
    out=np.zeros(m.shape+(3,),np.uint8)
    for k,c in TISSUE.items(): out[m==k]=c
    return out
# choose per grade the field with median per-field mean Dice (representative, not best-case)
per['mean_dice']=per[[c for c in per.columns if c.startswith('dice_')]].mean(1)
sel=[]
for g in range(4):
    s=per[per.mf_grade==g].sort_values('mean_dice'); sel.append(s.iloc[len(s)//2].filename)
fig,axs=plt.subplots(4,3,figsize=(5.2,7.0))
for i,f in enumerate(sel):
    r=te[te.filename==f].iloc[0]
    im=cv2.cvtColor(cv2.imread(R+f'data/test/images/{f}'),cv2.COLOR_BGR2RGB); gt=cv2.imread(R+f'data/test/masks/{r.mask_file}',0); pr=cv2.imread(R+f'pred/{pred}/{f}',0)
    d=per[per.filename==f].iloc[0]
    for j,(a,t) in enumerate([(im,'Input field'),(col(gt),'Reference mask'),(col(pr),f'Prediction ({label})')]):
        ax=axs[i,j]; ax.imshow(a,interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(0.4)
        if i==0: ax.set_title(t,fontsize=7.5)
    axs[i,0].set_ylabel(f'F{r.mf_grade} · {r.patient}',fontsize=7.5)
    axs[i,2].text(1.02,0.5,f"Reticulin Dice {d['dice_Reticulin']:.2f}\nmean Dice {d['mean_dice']:.2f}",transform=axs[i,2].transAxes,fontsize=6.3,va='center')
fig.legend(handles=[Patch(fc=np.array(TISSUE[k])/255,ec='k',lw=0.3,label=f'{k} {TISSUE_NAMES[k]}') for k in range(4)],loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(0.45,-0.01),fontsize=7)
fig.subplots_adjust(wspace=0.04,hspace=0.06,bottom=0.05,right=0.84); save(fig,f'Figure_qualitative_{pred}')
print(sel)
