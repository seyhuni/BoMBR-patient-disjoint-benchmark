import sys; sys.path.insert(0,'../revision_analysis')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler; from sklearn.decomposition import PCA
from style import *
import os; os.chdir('../revision_analysis'); from run_grade import emb, desc, DEV, TEST, meta
E=emb('resnet50_ft_original'); D=desc(sys.argv[1] if len(sys.argv)>1 else 'delivered')
fig,axs=plt.subplots(1,2,figsize=(7.0,3.0),sharex=False)
for ax,(blocks,title) in zip(axs,[([E],'a  Embedding only (2,048 features)'),([E,D],'b  Embedding + 12 descriptors (2,060 features)')]):
    X=pd.concat(blocks,axis=1); sc=StandardScaler().fit(X.loc[DEV]); p=PCA(2).fit(sc.transform(X.loc[DEV]))
    for sub,idx,mk in [('development',DEV,'o'),('locked test',TEST,'D')]:
        Z=p.transform(sc.transform(X.loc[idx])); gr=meta.loc[idx,'mf_grade'].to_numpy()
        for g in range(4):
            s=gr==g
            ax.scatter(Z[s,0],Z[s,1],s=10 if sub=='development' else 20,marker=mk,facecolor=GRADE[g] if sub=='locked test' else 'none',
                       edgecolor=GRADE[g],linewidth=0.8,alpha=0.9 if sub=='locked test' else 0.6,label=f'F{g} ({sub})' if True else None)
    ax.set_title(title,loc='left',weight='bold'); ax.set_xlabel(f'PC1 ({p.explained_variance_ratio_[0]*100:.1f}%)'); ax.set_ylabel(f'PC2 ({p.explained_variance_ratio_[1]*100:.1f}%)')
    ax.grid(True,lw=0.4)
h,l=axs[1].get_legend_handles_labels()
fig.legend(h,l,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(0.5,-0.13),fontsize=6.8,handletextpad=0.3,columnspacing=0.9)
fig.tight_layout(); os.chdir('../figures_source'); save(fig,'Figure_PCA_embedding')
