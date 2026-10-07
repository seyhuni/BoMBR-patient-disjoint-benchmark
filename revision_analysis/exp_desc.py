from run_grade import *
R=[]
for tag,nm in [('delivered','Delivered descriptors (submitted; index-swapped)'),('pred_original_legacyscale','Predicted masks, corrected mapping, submitted scale'),('gt_legacyscale','GT masks (oracle), submitted scale'),('gt','GT masks (oracle), matched scale')]:
    cols=LEGACY if tag=='delivered' else F.COLS
    Xd,Xt=mats([desc(tag)])
    R.append(run(f'{nm} | HGB (12)',Xd,Xt,*M.hgb_desc()))
    R.append(run(f'{nm} | ordinal logit (12)',Xd,Xt,*M.ordlogit_desc()))
    if tag!='delivered':
        Xd,Xt=mats([desc(tag,['pct_reticulin'])]); R.append(run(f'{nm} | reticulin area fraction proxy',Xd,Xt,*M.ordlogit_desc(log=False)))
        Xd,Xt=mats([desc(tag,['ret_cell_fraction'])]); R.append(run(f'{nm} | reticulin/(reticulin+cell) proxy',Xd,Xt,*M.ordlogit_desc(log=False)))
json.dump(R,open('res_desc.json','w'),default=float,indent=1)
