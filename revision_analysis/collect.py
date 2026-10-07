"""Collect every result into values.json (manuscript placeholders), final_models.json (figures) and tables.json."""
import json, glob, numpy as np, pandas as pd
import models as M
man=pd.read_csv('patient_safe_manifest.csv'); te=man[man.subset=='test'].reset_index(drop=True)
y=te.mf_grade.to_numpy(); g=te.patient.to_numpy()
RUNS={}
for f in sorted(glob.glob('res_*.json')):
    d=json.load(open(f)); runs=d['runs'] if isinstance(d,dict) and 'runs' in d else d
    for r in runs:
        if r['model'] in RUNS and len(RUNS[r['model']])>len(r): continue
        RUNS[r['model']]=r
def get(k):
    if k not in RUNS: raise KeyError(k)
    return RUNS[k]
# end-to-end CNN and submitted hybrid predictions
gp=pd.read_csv('../derived_data/results/submitted_grade_test_predictions.csv').set_index('filename').loc[te.filename]
def mk(label,p):
    p=np.asarray(p); return {'model':label,'test':M.metrics(y,p),'test_ci':M.cluster_boot(y,p,g),'test_pred':p.tolist()}
cnn=mk('Image-only ResNet50 (end-to-end, submitted)',gp.cnn_prediction); sub_h=mk('Submitted hybrid (class-balanced SVM, n = 201)',gp.hybrid_svm_prediction)
sub_morph=mk('Submitted morphometrics-only HGB (index-interchanged)',gp.morph_hgb_prediction)
RUNS[cnn['model']]=cnn; RUNS[sub_h['model']]=sub_h; RUNS[sub_morph['model']]=sub_morph
K={ # short keys -> run labels
 'cnn':cnn['model'],'sub_hybrid':sub_h['model'],'sub_morph':sub_morph['model'],
 'emb_r50':'[resnet50_ft_original] embedding-only | PCA-SVM',
 'early_pred_r50':'[resnet50_ft_original+pred_ft] early fusion (concat -> PCA) | SVM',
 'pcaapp_pred_r50':'[resnet50_ft_original+pred_ft] PCA(embedding) + raw descriptors | SVM',
 'late_pred_r50':'[resnet50_ft_original+pred_ft] late fusion (ordinal probabilities)',
 'early_gt_r50':'[resnet50_ft_original+gt] early fusion (concat -> PCA) | SVM',
 'pcaapp_gt_r50':'[resnet50_ft_original+gt] PCA(embedding) + raw descriptors | SVM',
 'late_gt_r50':'[resnet50_ft_original+gt] late fusion (ordinal probabilities)',
 'pred_hgb':'Predicted masks (corrected segmenter), matched scale | HGB (12)',
 'pred_ol':'Predicted masks (corrected segmenter), matched scale | ordinal logit (12)',
 'pred_proxy':'Predicted masks (corrected segmenter), matched scale | reticulin area fraction proxy',
 'gt_hgb':'GT masks (oracle), matched scale | HGB (12)','gt_ol':'GT masks (oracle), matched scale | ordinal logit (12)',
 'gt_proxy':'GT masks (oracle), matched scale | reticulin area fraction proxy',
 'dino_svm':'DINOv2 ViT-S/14 (frozen), submitted scale | PCA-RBF-SVM','phk_svm':'Phikon ViT-B/16 (frozen), submitted scale | PCA-RBF-SVM',
 'phk_ol':'Phikon ViT-B/16 (frozen), submitted scale | PCA-ordinal logit','dino_ol':'DINOv2 ViT-S/14 (frozen), submitted scale | PCA-ordinal logit',
 'r18_svm':'ResNet18 ImageNet (frozen), submitted scale | PCA-RBF-SVM',
 'r50_ol':'ResNet50 fine-tuned (submitted CNN), submitted scale | PCA-ordinal logit',
 'r50_svm_cv':'ResNet50 fine-tuned (submitted CNN), submitted scale | PCA-RBF-SVM',
 'cross_ol':'Reference-trained descriptors, test masks from corrected segmenter | ordinal logit (12)',
 'cross_hgb':'Reference-trained descriptors, test masks from corrected segmenter | HGB (12)',
 'cross_proxy':'Reference-trained descriptors, test masks from corrected segmenter | reticulin area fraction proxy',
 'dino_svm_m':'DINOv2 ViT-S/14 (frozen), matched scale | PCA-RBF-SVM','phk_svm_m':'Phikon ViT-B/16 histopathology FM (frozen), matched scale | PCA-RBF-SVM',
 'r18_svm_m':'ResNet18 ImageNet (frozen), matched scale | PCA-RBF-SVM',
 'early_gt_r50':'[resnet50_ft_original+gt] early fusion (concat -> PCA) | SVM',
}
# final_models.json for figures
SHORT={'cnn':'Image-only ResNet50 CNN (end-to-end)','sub_hybrid':'Baseline hybrid (balanced SVM)','emb_r50':'ResNet50 embedding, PCA–SVM',
 'early_pred_r50':'Early fusion, predicted descriptors','pcaapp_pred_r50':'PCA(emb.) + predicted descriptors','late_pred_r50':'Late fusion, predicted descriptors',
 'pcaapp_gt_r50':'PCA(emb.) + reference descriptors (oracle)','pred_ol':'Descriptors, predicted masks (ordinal)','pred_hgb':'Descriptors, predicted masks (HGB)',
 'pred_proxy':'Reticulin area fraction, predicted masks','gt_ol':'Descriptors, reference masks (ordinal)','gt_hgb':'Descriptors, reference masks (HGB)',
 'gt_proxy':'Reticulin area fraction, reference masks','r18_svm':'ResNet18-IN frozen, PCA–SVM','dino_svm':'DINOv2 frozen, PCA–SVM','phk_svm':'Phikon frozen, PCA–SVM',
 'phk_ol':'Phikon frozen, ordinal logit','sub_morph':'Baseline-configuration descriptors (interchanged), HGB'}
FAM={'cnn':'image','sub_hybrid':'fusion','emb_r50':'image','early_pred_r50':'fusion','pcaapp_pred_r50':'fusion','late_pred_r50':'fusion','pcaapp_gt_r50':'fusion',
 'pred_ol':'desc_pred','pred_hgb':'desc_pred','pred_proxy':'desc_pred','gt_ol':'desc_gt','gt_hgb':'desc_gt','gt_proxy':'desc_gt','r18_svm':'frozen','dino_svm':'frozen','phk_svm':'frozen','phk_ol':'frozen','sub_morph':'desc_pred'}
forest=['cnn','sub_hybrid','emb_r50','early_pred_r50','pcaapp_pred_r50','late_pred_r50','pcaapp_gt_r50','sub_morph','pred_hgb','pred_ol','pred_proxy','gt_hgb','gt_ol','gt_proxy','r18_svm','dino_svm','phk_svm','phk_ol']
cms={'cnn':'Image-only CNN','emb_r50':'Embedding PCA–SVM','early_pred_r50':'Early-fusion hybrid','pcaapp_gt_r50':'Oracle fusion','phk_svm':'Phikon PCA–SVM'}
f1p={'cnn':'CNN','emb_r50':'Emb','pcaapp_gt_r50':'Oracle','phk_ol':'Phikon-OL'}
pp={'cnn':'#7fb2ea','emb_r50':'#2a78d6','pcaapp_gt_r50':'#1baf7a','phk_svm':'#2f2f2f'}
FM=[]
for k in forest+[k for k in cms if k not in forest]:
    r=get(K[k]); e={'key':k,'model':r['model'],'short':SHORT[k],'family':FAM[k],'test':r['test'],'test_ci':r['test_ci'],'test_pred':r['test_pred'],
       'cv_grouped':r.get('cv_grouped'),'forest':k in forest,'cm':k in cms,'cmtitle':cms.get(k),'f1panel':k in f1p,'f1lab':f1p.get(k),'pp':k in pp,'color':pp.get(k)}
    FM.append(e)
json.dump(FM,open('final_models.json','w'),default=float,indent=1)
json.dump({k:RUNS[v] for k,v in K.items()},open('key_runs.json','w'),default=float,indent=1)
print('ok',len(FM))
