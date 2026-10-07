from run_grade import *
R=[]
for b,nm in [('resnet18_in_original','ResNet18 ImageNet (frozen), submitted scale'),('dinov2_vits14_original','DINOv2 ViT-S/14 (frozen), submitted scale'),('phikon_original','Phikon ViT-B/16 (frozen), submitted scale')]:
    Xd,Xt=mats([emb(b)])
    R.append(run(f'{nm} | PCA-RBF-SVM',Xd,Xt,*M.svm_pca(),leaky=True))
    R.append(run(f'{nm} | PCA-ordinal logit',Xd,Xt,*M.ordlogit_pca(),leaky=True))
json.dump(R,open('res_emb_orig.json','w'),default=float,indent=1)
