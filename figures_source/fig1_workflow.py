import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from style import *
fig,ax=plt.subplots(figsize=(7.2,4.6)); ax.set_xlim(0,100); ax.set_ylim(0,70); ax.axis('off')
def box(x,y,w,h,title,body='',fc='white',ec=INK,lw=0.7):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.2,rounding_size=0.8',fc=fc,ec=ec,lw=lw))
    ax.text(x+w/2,y+h-1.2,title,ha='center',va='top',fontsize=7.4,weight='bold',color=INK)
    if body: ax.text(x+w/2,y+h-4.1,body,ha='center',va='top',fontsize=6.4,color='#333333',linespacing=1.3)
def arrow(x1,y1,x2,y2,c=INK,ls='-'):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',lw=0.7,color=c,ls=ls,shrinkA=0,shrinkB=0,mutation_scale=7))
box(30,60,40,9,'BoMBR resource: 251 fields, 29 patients','indexed masks (verified): 0 Reticulin · 1 Fat · 2 Cell · 3 Bone')
box(1,45,66,10,'Development set: 201 fields, 19 patients','training 161 fields / 15 patients · validation 40 fields / 4 patients\nper-image colour normalisation; resampled ×0.65 to the test pixel scale')
box(73,45,26,10,'Locked test set','n = 50 fields · 10 new patients\nnative 512×512 crops',fc='#fdecea',ec=RED,lw=1.1)
arrow(42,59.6,34,55.4); arrow(58,59.6,86,55.4)
ax.text(76,61.5,'patient-disjoint split',ha='center',fontsize=6.3,style='italic',color='#333333')
box(1,26,32,14,'A · Deep image features','ResNet50 (fine-tuned)\nResNet18-IN · DINOv2 ViT-S/14\nPhikon ViT-B/16 (frozen)\ntile-averaged embeddings',fc='#eaf2fb',ec=BLUE,lw=1.0)
box(35,26,32,14,'B · Mask-derived descriptors','UNet++–ResNet34 predicted masks\nor reference masks (oracle)\n12 descriptors (Table 3)\nreticulin-area proxy',fc='#fdf0ea',ec=ORANGE,lw=1.0)
arrow(17,44.6,17,40.4); arrow(51,44.6,51,40.4)
box(1,9,32,12,'Embedding-only models','PCA–RBF-SVM · PCA–ordinal logit\nselected by repeated patient-grouped CV\n(n = 201 development fields)',fc='#eaf2fb',ec=BLUE,lw=0.7)
box(35,9,32,12,'Descriptor-only and fusion models','HGB · ordinal logit · early fusion\nPCA(emb.) + descriptors · late fusion\nsame grouped-CV selection (n = 201)',fc='#fdf0ea',ec=ORANGE,lw=0.7)
arrow(17,25.6,17,21.4); arrow(51,25.6,51,21.4); arrow(33,31,38,21.4,c=GRAY)
box(20,-1.5,60,6.2,'Single evaluation on the locked test set (50 fields, 10 patient clusters)','',fc='#fdecea',ec=RED,lw=1.1)
arrow(17,8.6,30,5.8); arrow(51,8.6,50,5.8); arrow(86,44.6,86,26); ax.plot([86,86],[26,8],color=RED,lw=0.8,ls=(0,(3,2))); arrow(86,8,70,5.8,c=RED)
ax.text(87.5,26,'never used for\ntraining, selection,\nscaling, PCA or\nnormalisation\nstatistics',ha='left',va='center',fontsize=6.2,color=RED)
save(fig,'Figure1_workflow')
