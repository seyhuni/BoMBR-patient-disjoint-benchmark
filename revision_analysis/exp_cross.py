"""Descriptor models trained on reference-mask development descriptors and applied to predicted-mask test descriptors
(train-on-reference / test-on-prediction), which removes the in-sample mask-quality advantage of development predictions."""
from run_grade import *
R=[]
Dgt=desc('gt')
for tag,nm in [('pred_original','submitted segmenter'),('pred_reinhard','submitted segmenter + CN'),('pred_ft','corrected segmenter')]:
    Dp=desc(tag); Xd=Dgt.loc[DEV].to_numpy(float); Xt=Dp.loc[TEST].to_numpy(float)
    R.append(run(f'Reference-trained descriptors, test masks from {nm} | HGB (12)',Xd,Xt,*M.hgb_desc(),repeats=0))
    R.append(run(f'Reference-trained descriptors, test masks from {nm} | ordinal logit (12)',Xd,Xt,*M.ordlogit_desc(),repeats=0))
    Xd=Dgt.loc[DEV,['pct_reticulin']].to_numpy(float); Xt=Dp.loc[TEST,['pct_reticulin']].to_numpy(float)
    R.append(run(f'Reference-trained descriptors, test masks from {nm} | reticulin area fraction proxy',Xd,Xt,*M.ordlogit_desc(log=False),repeats=0))
E=emb('resnet50_ft_original'); n=E.shape[1]
for tag in ['pred_ft']:
    Xd=pd.concat([E,Dgt],axis=1).loc[DEV].to_numpy(float); Xt=pd.concat([E,desc(tag)],axis=1).loc[TEST].to_numpy(float)
    R.append(run(f'[resnet50_ft_original+gt->{tag}] PCA(embedding) + raw descriptors | SVM',Xd,Xt,*M.pca_append(n,'svm'),repeats=0))
json.dump(R,open('res_cross.json','w'),default=float,indent=1)
