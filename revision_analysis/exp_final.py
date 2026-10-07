from run_grade import *
R=[]
for tag,nm in [('pred_ft','Predicted masks (corrected segmenter), matched scale'),('gt','GT masks (oracle), matched scale')]:
    Xd,Xt=mats([desc(tag)])
    R.append(run(f'{nm} | HGB (12)',Xd,Xt,*M.hgb_desc(),leaky=True))
    R.append(run(f'{nm} | ordinal logit (12)',Xd,Xt,*M.ordlogit_desc(),leaky=True))
    Xd,Xt=mats([desc(tag,['pct_reticulin'])]); R.append(run(f'{nm} | reticulin area fraction proxy',Xd,Xt,*M.ordlogit_desc(log=False),leaky=True))
json.dump(R,open('res_final_desc.json','w'),default=float,indent=1)
