"""Field-grade models and evaluation utilities used in the revision re-analysis."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import cohen_kappa_score, accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, recall_score
K=4
def qwk(y,p): return cohen_kappa_score(y,p,weights='quadratic',labels=list(range(K)))
class OrdinalLogit(BaseEstimator, ClassifierMixin):
    """Proportional-odds (cumulative-logit) model with L2 penalty: P(y<=k|x)=sigmoid(theta_k - w.x).
    Linear, rank-consistent ordinal head (same parametric form as a linear CORAL head)."""
    def __init__(self, alpha=1.0): self.alpha=alpha
    def _unpack(self,z,d):
        w=z[:d]; t0=z[d]; inc=np.exp(z[d+1:]); return w, np.concatenate([[t0],t0+np.cumsum(inc)])
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y,int); n,d=X.shape; self.classes_=np.arange(K)
        def nll(z):
            w,th=self._unpack(z,d); eta=X@w
            cum=expit(th[None,:]-eta[:,None]); cum=np.hstack([np.zeros((n,1)),cum,np.ones((n,1))])
            pr=np.clip(cum[np.arange(n),y+1]-cum[np.arange(n),y],1e-12,1)
            return -np.log(pr).sum()+self.alpha*0.5*(w@w)
        z0=np.concatenate([np.zeros(d),[-1.0],np.zeros(K-2)])
        self.z_=minimize(nll,z0,method='L-BFGS-B').x; self.w_,self.th_=self._unpack(self.z_,d); return self
    def predict_proba(self,X):
        eta=np.asarray(X,float)@self.w_; cum=expit(self.th_[None,:]-eta[:,None])
        cum=np.hstack([np.zeros((len(eta),1)),cum,np.ones((len(eta),1))]); return np.clip(np.diff(cum,axis=1),0,1)
    def predict(self,X): return self.predict_proba(X).argmax(1)
def metrics(y,p):
    y=np.asarray(y); p=np.asarray(p); r=recall_score(y,p,labels=list(range(K)),average=None,zero_division=0)
    return dict(accuracy=accuracy_score(y,p), balanced_accuracy=balanced_accuracy_score(y,p),
                macro_f1=f1_score(y,p,labels=list(range(K)),average='macro',zero_division=0), qwk=qwk(y,p),
                mae=np.abs(y-p).mean(), within1=(np.abs(y-p)<=1).mean(),
                recall_mf0=r[0], recall_mf1=r[1], recall_mf2=r[2], recall_mf3=r[3])
def cluster_boot(y,p,g,B=3000,seed=20260929,ref=None):
    """Patient-clustered percentile bootstrap. If ref is given, returns intervals of (p - ref) differences."""
    rng=np.random.default_rng(seed); y,p,g=map(np.asarray,(y,p,g)); U=np.unique(g); idx={u:np.where(g==u)[0] for u in U}
    out={'accuracy':[],'qwk':[],'mae':[]}
    for _ in range(B):
        s=np.concatenate([idx[u] for u in rng.choice(U,len(U),replace=True)])
        a=[accuracy_score(y[s],p[s]), qwk(y[s],p[s]), np.abs(y[s]-p[s]).mean()]
        if ref is not None:
            r=np.asarray(ref); a=[a[0]-accuracy_score(y[s],r[s]), a[1]-qwk(y[s],r[s]), a[2]-np.abs(y[s]-r[s]).mean()]
        for k,v in zip(out,a): out[k].append(v)
    return {k:(float(np.nanpercentile(v,2.5)),float(np.nanpercentile(v,97.5))) for k,v in out.items()}
# ---------- model factories ----------
def log1p_signed(X): return np.sign(X)*np.log1p(np.abs(X))
def svm_pca(balanced=False, pca=0.95):
    pipe=Pipeline([('sc',StandardScaler()),('pca',PCA(n_components=pca,svd_solver='full',random_state=0)),('svm',SVC(kernel='rbf',class_weight='balanced' if balanced else None))])
    grid={'svm__C':[0.1,1,10,100],'svm__gamma':['scale',0.01,0.001]}; return pipe,grid
def ordlogit_pca():
    pipe=Pipeline([('sc',StandardScaler()),('pca',PCA(n_components=10,random_state=0)),('sc2',StandardScaler()),('ol',OrdinalLogit())])
    grid={'pca__n_components':[5,10,20],'ol__alpha':[1.0,10.0,100.0]}; return pipe,grid
def ordlogit_desc(log=True):
    steps=([('log',FunctionTransformer(log1p_signed))] if log else [])+[('sc',StandardScaler()),('ol',OrdinalLogit())]
    return Pipeline(steps),{'ol__alpha':[0.1,1.0,10.0]}
def hgb_desc(seed=20260812):
    return Pipeline([('sc',StandardScaler()),('hgb',HistGradientBoostingClassifier(max_iter=250,learning_rate=0.05,max_leaf_nodes=15,random_state=seed))]),{}
class PCAThenAppend(BaseEstimator, ClassifierMixin):
    """PCA is fitted on the embedding block only; the standardised raw descriptors are appended afterwards
    (optionally block-weighted so that both blocks carry equal total variance), then the head is fitted."""
    def __init__(self, n_emb, head='svm', pca=0.95, block_equal=True, C=10, gamma='scale', alpha=10.0):
        self.n_emb=n_emb; self.head=head; self.pca=pca; self.block_equal=block_equal; self.C=C; self.gamma=gamma; self.alpha=alpha
    def _tr(self,X,fit=False):
        E,D=X[:,:self.n_emb],log1p_signed(X[:,self.n_emb:])
        if fit:
            self.sce_=StandardScaler().fit(E); self.p_=PCA(n_components=self.pca,svd_solver='full',random_state=0).fit(self.sce_.transform(E))
            self.scd_=StandardScaler().fit(D)
        Z=self.p_.transform(self.sce_.transform(E)); Dz=self.scd_.transform(D)
        if self.block_equal: Dz=Dz*np.sqrt(self.p_.explained_variance_.sum()/Dz.shape[1])
        return np.hstack([Z,Dz])
    def fit(self,X,y):
        X=np.asarray(X,float); Z=self._tr(X,fit=True); self.classes_=np.arange(K)
        self.m_=SVC(kernel='rbf',C=self.C,gamma=self.gamma) if self.head=='svm' else Pipeline([('sc',StandardScaler()),('ol',OrdinalLogit(self.alpha))])
        self.m_.fit(Z,y); return self
    def predict(self,X): return self.m_.predict(self._tr(np.asarray(X,float)))
    def predict_proba(self,X): return self.m_.predict_proba(self._tr(np.asarray(X,float)))
def pca_append(n_emb, head='svm'):
    m=PCAThenAppend(n_emb,head=head)
    grid={'C':[0.1,1,10,100],'gamma':['scale',0.01,0.001]} if head=='svm' else {'pca':[5,10,20],'alpha':[1.0,10.0,100.0]}
    return m,grid
class LateFusion(BaseEstimator, ClassifierMixin):
    """Weighted average of class probabilities from an embedding ordinal model and a descriptor ordinal model."""
    def __init__(self, n_emb, w=0.5, alpha_e=10.0, k=10, alpha_d=1.0):
        self.n_emb=n_emb; self.w=w; self.alpha_e=alpha_e; self.k=k; self.alpha_d=alpha_d
    def fit(self,X,y):
        X=np.asarray(X,float); self.classes_=np.arange(K)
        self.e_=Pipeline([('sc',StandardScaler()),('pca',PCA(n_components=self.k,random_state=0)),('sc2',StandardScaler()),('ol',OrdinalLogit(self.alpha_e))]).fit(X[:,:self.n_emb],y)
        self.d_=Pipeline([('log',FunctionTransformer(log1p_signed)),('sc',StandardScaler()),('ol',OrdinalLogit(self.alpha_d))]).fit(X[:,self.n_emb:],y); return self
    def predict_proba(self,X):
        X=np.asarray(X,float); return (1-self.w)*self.e_.predict_proba(X[:,:self.n_emb])+self.w*self.d_.predict_proba(X[:,self.n_emb:])
    def predict(self,X): return self.predict_proba(X).argmax(1)
def late_fusion(n_emb): return LateFusion(n_emb),{'w':[0.0,0.25,0.5,0.75],'k':[5,10,20],'alpha_e':[10.0,100.0]}
# ---------- selection / evaluation ----------
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold
from sklearn.metrics import make_scorer
QWK=make_scorer(qwk)
def fit_select(est,grid,X,y,g,cv,scoring=QWK):
    if not grid: return clone(est).fit(X,y), {}
    gs=GridSearchCV(clone(est),grid,scoring=scoring,cv=list(cv.split(X,y,g)),n_jobs=1,refit=True,error_score=0.0).fit(X,y)
    return gs.best_estimator_, gs.best_params_
def shuffled_group_folds(g,n_splits,seed):
    """Random patient-to-fold assignment (balanced by field count) for repeated grouped CV."""
    rng=np.random.default_rng(seed); U=np.array(sorted(set(g))); rng.shuffle(U)
    sizes={u:(g==u).sum() for u in U}; load=np.zeros(n_splits); fold={}
    for u in sorted(U,key=lambda u:-sizes[u]+rng.random()*1e-3): f=int(np.argmin(load)); fold[u]=f; load[f]+=sizes[u]
    fid=np.array([fold[u] for u in g]); return [(np.where(fid!=k)[0],np.where(fid==k)[0]) for k in range(n_splits)]
class FixedFolds:
    def __init__(self,folds): self.folds=folds
    def split(self,X=None,y=None,g=None): return iter(self.folds)
def repeated_cv(est,grid,X,y,g,repeats=10,n_splits=5,leaky=False,seed=0):
    """Outer: repeated patient-grouped (or, if leaky, field-level stratified) 5-fold CV over development fields.
    Inner hyperparameter selection mirrors the outer scheme (grouped GroupKFold(4) or stratified field-level 4-fold)."""
    res=[]; preds=[]
    for r in range(repeats):
        folds=(list(StratifiedKFold(n_splits,shuffle=True,random_state=seed+r).split(X,y)) if leaky else shuffled_group_folds(g,n_splits,seed+r))
        oof=np.full(len(y),-1)
        for tr,te in folds:
            inner=StratifiedKFold(4,shuffle=True,random_state=seed+r) if leaky else GroupKFold(4)
            m,_=fit_select(est,grid,X[tr],y[tr],g[tr],inner if leaky else inner)
            oof[te]=m.predict(X[te])
        res.append(metrics(y,oof)); preds.append(oof)
    return res, np.array(preds)
