"""R1.3: fusion designs in which the 12 descriptors can survive; identical selection protocol for every variant."""
import sys
from run_grade import *
dtag=sys.argv[1]; btag=sys.argv[2]; R=[]
E=emb(btag); D=desc(dtag); n=E.shape[1]
Xd,Xt=mats([E]); R.append(run(f'[{btag}] embedding-only | PCA-SVM',Xd,Xt,*M.svm_pca()))
Xd,Xt=mats([E,D])
R.append(run(f'[{btag}+{dtag}] early fusion (concat -> PCA) | SVM',Xd,Xt,*M.svm_pca()))
R.append(run(f'[{btag}+{dtag}] PCA(embedding) + raw descriptors | SVM',Xd,Xt,*M.pca_append(n,'svm')))
R.append(run(f'[{btag}+{dtag}] late fusion (ordinal probabilities)',Xd,Xt,*M.late_fusion(n)))
json.dump(R,open(f'res_fusion_{dtag}_{btag}.json','w'),default=float,indent=1)
