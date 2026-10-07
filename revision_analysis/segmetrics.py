import numpy as np, pandas as pd, cv2, json, sys
# Corrected semantic mapping of the raw BoMBR mask indices (verified by colour statistics and visual check)
NAMES={0:'Reticulin',1:'Fat',2:'Cell',3:'Bone'}
man=pd.read_csv('patient_safe_manifest.csv'); te=man[man.subset=='test']
def evaluate(mode):
    CM=np.zeros((4,4),np.int64); per=[]
    for r in te.itertuples():
        g=cv2.imread(f'data/test/masks/{r.mask_file}',0).astype(int); p=cv2.imread(f'pred/{mode}/{r.filename}',0).astype(int)
        cm=np.bincount(g.ravel()*4+p.ravel(),minlength=16).reshape(4,4); CM+=cm
        d={'filename':r.filename,'patient':r.patient,'mf_grade':r.mf_grade}
        for k in range(4):
            tp=cm[k,k]; fp=cm[:,k].sum()-tp; fn=cm[k,:].sum()-tp
            d[f'dice_{NAMES[k]}']=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else np.nan
        per.append(d)
    res={}
    for k in range(4):
        tp=CM[k,k]; fp=CM[:,k].sum()-tp; fn=CM[k,:].sum()-tp; tn=CM.sum()-tp-fp-fn
        res[NAMES[k]]=dict(dice=2*tp/(2*tp+fp+fn), iou=tp/(tp+fp+fn), precision=tp/(tp+fp), recall=tp/(tp+fn), specificity=tn/(tn+fp), support=int(CM[k].sum()))
    res['pixel_accuracy']=np.trace(CM)/CM.sum(); res['mean_dice']=np.mean([res[NAMES[k]]['dice'] for k in range(4)])
    res['mean_iou']=np.mean([res[NAMES[k]]['iou'] for k in range(4)])
    return res, CM, pd.DataFrame(per)
if __name__=='__main__':
    for mode in sys.argv[1:]:
        res,CM,per=evaluate(mode)
        print(mode, json.dumps({k:(round(v,3) if not isinstance(v,dict) else {a:round(b,3) for a,b in v.items()}) for k,v in res.items()},indent=0))
        print(pd.DataFrame(CM,index=[f'GT {NAMES[k]}' for k in range(4)],columns=[f'P {NAMES[k]}' for k in range(4)]))
        per.to_csv(f'seg_per_image_{mode}.csv',index=False)
        json.dump({'metrics':{k:v for k,v in res.items()},'pixel_confusion':CM.tolist(),'order':[NAMES[k] for k in range(4)]},open(f'seg_test_{mode}.json','w'),default=float,indent=1)
