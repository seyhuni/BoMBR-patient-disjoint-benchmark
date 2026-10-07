import sys, json, numpy as np, matplotlib.pyplot as plt
from style import *
R='../revision_analysis/'; runs=[(a.split(':')[0],a.split(':')[1]) for a in sys.argv[1:]]
fig,axs=plt.subplots(1,len(runs),figsize=(2.35*len(runs),2.5))
for ax,(tag,title) in zip(np.atleast_1d(axs),runs):
    d=json.load(open(R+f'seg_test_{tag}.json')); CM=np.array(d['pixel_confusion'],float); N=CM/CM.sum(1,keepdims=True)
    ax.imshow(N,cmap='Blues',vmin=0,vmax=1)
    for i in range(4):
        for j in range(4): ax.text(j,i,f'{N[i,j]:.2f}\n{int(CM[i,j]/1000):,}k',ha='center',va='center',fontsize=5.8,color='white' if N[i,j]>0.55 else INK)
    names=d['order']; ax.set_xticks(range(4)); ax.set_xticklabels(names,rotation=35,ha='right'); ax.set_yticks(range(4)); ax.set_yticklabels(names if ax is np.atleast_1d(axs)[0] else [])
    ax.set_xlabel('Predicted class'); ax.set_title(title,fontsize=7.5)
    for s in ax.spines.values(): s.set_visible(False)
np.atleast_1d(axs)[0].set_ylabel('Reference class')
fig.tight_layout(); save(fig,'Figure_pixel_confusion')
