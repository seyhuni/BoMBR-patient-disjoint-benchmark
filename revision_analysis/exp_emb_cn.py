from run_grade import *
R=[]
for b,nm in [('resnet18_in_matchedcn','ResNet18 ImageNet (frozen), matched scale + CN'),('dinov2_vits14_matchedcn','DINOv2 ViT-S/14 (frozen), matched scale + CN'),('phikon_matchedcn','Phikon ViT-B/16 (frozen), matched scale + CN')]:
    Xd,Xt=mats([emb(b)])
    R.append(run(f'{nm} | PCA-RBF-SVM',Xd,Xt,*M.svm_pca(),leaky=True))
    R.append(run(f'{nm} | PCA-ordinal logit',Xd,Xt,*M.ordlogit_pca(),leaky=True))
json.dump(R,open('res_emb_cn.json','w'),default=float,indent=1)
