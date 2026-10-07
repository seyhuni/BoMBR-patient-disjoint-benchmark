"""R1.3/R1.4/R1.12: reproduction of the submitted fusion on the complete development set (n=201) and
decomposition of the difference between the submitted hybrid and the matched sensitivity analysis."""
from run_grade import *
from sklearn.metrics import make_scorer, f1_score
F1=make_scorer(lambda a,b: f1_score(a,b,average='macro'))
E=emb('resnet50_ft_original'); D=desc('delivered')
R=[]
for bal,sc,nm in [(True,F1,'submitted settings (class-balanced SVM, macro-F1 selection)'),(False,M.QWK,'matched-analysis settings (unweighted SVM, QWK selection)')]:
    for blocks,bn in [([E],'embedding-only'),([E,D],'early-fusion hybrid (2,048+12 -> PCA)')]:
        Xd,Xt=mats(blocks); est,grid=M.svm_pca(balanced=bal)
        R.append(run(f'{bn} | {nm}',Xd,Xt,est,grid,scoring=sc,repeats=0))
# PCA loading mass of the 12 descriptors in the early-fusion PCA
from sklearn.preprocessing import StandardScaler; from sklearn.decomposition import PCA
Xd,Xt=mats([E,D]); Z=StandardScaler().fit_transform(Xd); p=PCA(0.95,svd_solver='full').fit(Z)
L=p.components_; lam=p.explained_variance_
mass=float((lam[:,None]*L[:,-12:]**2).sum()/lam.sum())
per_feat_captured=((L[:,-12:]**2)*lam[:,None]).sum(0)/1.0   # variance of each standardised descriptor retained (max 1)
top=[int(np.argmax(np.abs(L[:,-12:][:,j]))) for j in range(12)]
info={'retained_pcs':int(p.n_components_),'n_dev':int(len(Xd)),'descriptor_share_of_retained_variance':mass,
      'descriptor_share_of_input_variance':12/Z.shape[1],'retained_variance_per_descriptor':dict(zip(LEGACY,[float(v) for v in per_feat_captured])),
      'pc_with_max_loading_per_descriptor':dict(zip(LEGACY,top))}
print(json.dumps(info,indent=1))
json.dump({'runs':R,'pca_loading':info},open('res_repro.json','w'),default=float,indent=1)
