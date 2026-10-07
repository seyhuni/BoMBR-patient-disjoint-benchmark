"""R1.1: agreement between reference-mask (oracle) and predicted-mask descriptors on the locked test fields."""
import sys, numpy as np, pandas as pd, features as F
from scipy.stats import spearmanr
def ccc(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float); mx,my=x.mean(),y.mean(); vx,vy=x.var(),y.var(); c=((x-mx)*(y-my)).mean()
    return 2*c/(vx+vy+(mx-my)**2) if (vx+vy)>0 else np.nan
g=pd.read_csv('feat_gt.csv').set_index('filename'); out=[]
for tag in sys.argv[1:]:
    p=pd.read_csv(f'feat_{tag}.csv').set_index('filename'); te=[f for f in p.index if f in g.index and g.loc[f,'subset']=='test']
    for c in F.COLS:
        x,y=g.loc[te,c],p.loc[te,c]
        out.append(dict(masks=tag,descriptor=c,spearman=spearmanr(x,y)[0],ccc=ccc(x,y),median_gt=x.median(),median_pred=y.median(),
                        median_rel_bias=np.median((y-x)/np.where(np.abs(x)>1e-9,np.abs(x),np.nan))))
o=pd.DataFrame(out); o.to_csv('agreement_'+'_'.join(sys.argv[1:])+'.csv',index=False)
print(o.pivot(index='descriptor',columns='masks',values='ccc').loc[F.COLS].round(3)); print(o.pivot(index='descriptor',columns='masks',values='spearman').loc[F.COLS].round(3))
