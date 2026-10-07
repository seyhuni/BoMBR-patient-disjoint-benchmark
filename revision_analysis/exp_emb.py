"""R1.7/R1.8/R1.6: backbone comparison at matched scale; ordinal heads; repeated patient-grouped CV;
field-level (source-style) CV to quantify patient-leakage optimism."""
from run_grade import *
R=[]
BACK=[('resnet50_ft_original','ResNet50 fine-tuned (submitted CNN), submitted scale'),('resnet50_ft_matched','ResNet50 fine-tuned (submitted CNN), matched scale'),
      ('resnet18_in_matched','ResNet18 ImageNet (frozen), matched scale'),('dinov2_vits14_matched','DINOv2 ViT-S/14 (frozen), matched scale'),('phikon_matched','Phikon ViT-B/16 histopathology FM (frozen), matched scale')]
for b,nm in BACK:
    Xd,Xt=mats([emb(b)])
    R.append(run(f'{nm} | PCA-RBF-SVM',Xd,Xt,*M.svm_pca(),leaky=True))
    R.append(run(f'{nm} | PCA-ordinal logit',Xd,Xt,*M.ordlogit_pca(),leaky=True))
json.dump(R,open('res_emb.json','w'),default=float,indent=1)
