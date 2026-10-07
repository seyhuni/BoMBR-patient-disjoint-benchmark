import json, numpy as np, pandas as pd, cv2, matplotlib.pyplot as plt
from style import *
R='../revision_analysis/'; F=json.load(open(R+'final_models.json')); man=pd.read_csv(R+'patient_safe_manifest.csv'); te=man[man.subset=='test'].reset_index(drop=True)
y=te.mf_grade.to_numpy(); g=te.patient.to_numpy()
FAM={'image':BLUE,'desc_pred':ORANGE,'desc_gt':AQUA,'fusion':VIOLET,'frozen':'#2f2f2f'}
# ---------- Figure 4: forest ----------
rows=[m for m in F if m.get('forest')]
fig,ax=plt.subplots(figsize=(7.0,0.26*len(rows)+0.9))
for i,m in enumerate(rows[::-1]):
    c=FAM[m['family']]; q=m['test']['qwk']; lo,hi=m['test_ci']['qwk']
    ax.plot([lo,hi],[i+0.12,i+0.12],color=c,lw=1.2); ax.plot(q,i+0.12,'o',color=c,ms=4.5)
    if m.get('cv_grouped'):
        mu,sd=m['cv_grouped']['qwk']; ax.plot([mu-sd,mu+sd],[i-0.18,i-0.18],color=c,lw=0.8,alpha=0.7); ax.plot(mu,i-0.18,'o',mfc='white',mec=c,ms=4,mew=0.9)
    ax.text(1.005,i,f"{q:.3f} ({lo:.2f}–{hi:.2f})"+(f"   {m['cv_grouped']['qwk'][0]:.3f} ± {m['cv_grouped']['qwk'][1]:.3f}" if m.get('cv_grouped') else '   n/a'),
            transform=ax.get_yaxis_transform(),va='center',fontsize=6.6)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([m['short'] for m in rows[::-1]],fontsize=6.8)
ax.set_xlim(0,1.02); ax.set_xlabel('Quadratic weighted kappa'); ax.grid(axis='x',lw=0.4)
ax.text(1.005,len(rows)-0.3,'Test QWK (95% CI)   grouped CV mean ± SD',transform=ax.get_yaxis_transform(),fontsize=6.6,weight='bold')
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],marker='o',color=INK,lw=1.2,ms=4,label='locked test (patient-clustered 95% CI)'),Line2D([],[],marker='o',mfc='white',mec=INK,color=INK,lw=0.8,ms=4,label='repeated grouped CV (mean ± SD)')]+
          [Line2D([],[],color=v,lw=3,label=k) for k,v in [('image network / embedding',BLUE),('frozen encoder',FAM['frozen']),('descriptors, predicted masks',ORANGE),('descriptors, reference masks',AQUA),('fusion',VIOLET)]],
          loc='upper left',bbox_to_anchor=(0,-0.09),ncol=4,frameon=False,fontsize=6.4)
save(fig,'Figure_forest')
# ---------- Figure 5: confusion matrices ----------
cms=[m for m in F if m.get('cm')]
fig,axs=plt.subplots(1,len(cms),figsize=(1.75*len(cms),2.1))
for k,(ax,m) in enumerate(zip(axs,cms)):
    p=np.array(m['test_pred']); C=np.zeros((4,4),int)
    for a,b in zip(y,p): C[a,b]+=1
    ax.imshow(C/C.sum(1,keepdims=True),cmap='Blues',vmin=0,vmax=1)
    for i in range(4):
        for j in range(4): ax.text(j,i,str(C[i,j]),ha='center',va='center',fontsize=7,color='white' if C[i,j]/C[i].sum()>0.55 else INK)
    ax.set_xticks(range(4)); ax.set_xticklabels([f'F{i}' for i in range(4)]); ax.set_yticks(range(4)); ax.set_yticklabels([f'F{i}' for i in range(4)] if k==0 else [])
    ax.set_title(f"{chr(97+k)}  {m['cmtitle']}",fontsize=7,loc='left'); ax.set_xlabel('Predicted')
    for s in ax.spines.values(): s.set_visible(False)
axs[0].set_ylabel('Reference'); fig.tight_layout(w_pad=0.6); save(fig,'Figure_grade_confusion')
# ---------- Figure 8: per-patient strip ----------
pp=[m for m in F if m.get('pp')]
fig,axs=plt.subplots(2,1,figsize=(7.0,3.6),sharex=True)
P=sorted(set(g),key=lambda s:int(s.split('_')[1])); x=np.arange(len(P)); w=0.8/len(pp)
for k,m in enumerate(pp):
    p=np.array(m['test_pred']); acc=[(p[g==u]==y[g==u]).mean() for u in P]; mae=[np.abs(p[g==u]-y[g==u]).mean() for u in P]
    c=FAM[m['family']] if m['family']!='image' or k==0 else '#7fb2ea'
    axs[0].bar(x-0.4+w/2+k*w,acc,w*0.9,color=m.get('color',c),label=m['short']); axs[1].bar(x-0.4+w/2+k*w,mae,w*0.9,color=m.get('color',c))
axs[0].set_ylabel('Accuracy'); axs[1].set_ylabel('MAE (grades)'); axs[0].set_ylim(0,1.05); axs[0].grid(axis='y',lw=0.4); axs[1].grid(axis='y',lw=0.4)
axs[1].set_xticks(x); axs[1].set_xticklabels([f"{u}\n"+','.join(f'F{v}' for v in sorted(set(y[g==u]))) for u in P],fontsize=6.5)
axs[0].legend(ncol=len(pp),loc='lower center',bbox_to_anchor=(0.5,1.0),frameon=False,fontsize=6.5)
fig.tight_layout(); save(fig,'Figure_per_patient')
# ---------- Figure 7: F1 errors ----------
f1=te[te.mf_grade==1].reset_index(drop=True); em=[m for m in F if m.get('f1panel')]
gt=pd.read_csv(R+'feat_gt.csv').set_index('filename')
fig,axs=plt.subplots(2,len(f1),figsize=(7.0,3.35))
for j,r in f1.iterrows():
    im=cv2.cvtColor(cv2.imread(R+f'data/test/images/{r.filename}'),cv2.COLOR_BGR2RGB); msk=cv2.imread(R+f'data/test/masks/{r.mask_file}',0)
    col=np.zeros(msk.shape+(3,),np.uint8)
    for k,c in TISSUE.items(): col[msk==k]=c
    axs[0,j].imshow(im); axs[1,j].imshow(col,interpolation='nearest')
    idx=int(np.where(te.filename==r.filename)[0][0])
    lab='  '.join(f"{m['f1lab']}: F{m['test_pred'][idx]}" for m in em)
    axs[0,j].set_title(f"{r.patient} · {r.filename.split('_')[-1][:-4]}",fontsize=6.8)
    axs[1,j].set_xlabel(lab.replace('  ','\n',1) if len(lab)>28 else lab,fontsize=6.2)
    axs[1,j].text(0.02,0.97,f"reticulin {gt.loc[r.filename,'pct_reticulin']*100:.1f}%",transform=axs[1,j].transAxes,va='top',fontsize=6.2,color='white',bbox=dict(fc='#00000080',ec='none',pad=1))
    for a in axs[:,j]: a.set_xticks([]); a.set_yticks([])
axs[0,0].set_ylabel('Input'); axs[1,0].set_ylabel('Reference mask')
fig.tight_layout(h_pad=0.3,w_pad=0.2); save(fig,'Figure_F1_errors')
