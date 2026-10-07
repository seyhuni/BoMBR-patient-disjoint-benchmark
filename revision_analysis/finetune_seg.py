"""Corrected segmentation training: fine-tunes the submitted UNet++-ResNet34 checkpoint with
 (i) the corrected class mapping (0=Reticulin,1=Fat,2=Cell,3=Bone) so that the intended class weights
     (Reticulin 2.2, Fat 1.0, Cell 0.7, Bone 0.8) are applied to the intended tissues,
 (ii) development fields resampled to the pixel scale of the locked test crops (factor 0.65), and
 (iii) per-image Reinhard colour normalisation to development-set LAB statistics (applied identically to all subsets).
Only training-patient fields are used for weight updates; validation-patient tiles are used for checkpoint selection."""
from features import to_test_scale
import numpy as np, pandas as pd, cv2, torch, json, time, torch.nn.functional as Fnn, segmentation_models_pytorch as smp, sys
sys.path.insert(0,'../original_pipeline'); from src.models import CompoundSegmentationLoss
torch.set_num_threads(2); torch.manual_seed(20260929); rng=np.random.default_rng(20260929)
MEAN=np.array([0.485,0.456,0.406]); STD=np.array([0.229,0.224,0.225]); ST=json.load(open('dev_lab_stats.json'))
def reinhard(bgr):
    lab=cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB).astype(np.float32); mu,sd=lab.reshape(-1,3).mean(0),lab.reshape(-1,3).std(0)+1e-6
    return cv2.cvtColor(np.clip((lab-mu)/sd*np.array(ST['std'])+np.array(ST['mean']),0,255).astype(np.uint8),cv2.COLOR_LAB2RGB)
def norm(rgb): return ((rgb/255.0-MEAN)/STD).transpose(2,0,1).astype(np.float32)
man=pd.read_csv('patient_safe_manifest.csv')
def load(sub):
    out=[]
    for r in man[man.subset==sub].itertuples():
        out.append((to_test_scale(reinhard(cv2.imread(f'data/{r.split}/images/{r.filename}')),False), to_test_scale(cv2.imread(f'data/{r.split}/masks/{r.mask_file}',0),True).astype(np.int64)))
    return out
tr=load('train'); va=load('val'); print('loaded',len(tr),len(va),flush=True)
vt=[]
for im,m in va:
    H,W=m.shape
    for _ in range(1):
        y,x=rng.integers(0,H-511),rng.integers(0,W-511); vt.append((norm(im[y:y+512,x:x+512]),m[y:y+512,x:x+512]))
vt=vt[:40]
ck=torch.load('ckpt/segmentation_unetpp_resnet34_best.pt',map_location='cpu',weights_only=False)
net=smp.UnetPlusPlus(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=4); net.load_state_dict(ck['state_dict'])
crit=CompoundSegmentationLoss([2.2,1.0,0.7,0.8],{'cross_entropy':0.4,'dice':0.4,'focal':0.2})
opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4)
STEPS=int(sys.argv[1]) if len(sys.argv)>1 else 1200; BS=4; CROP=256; EVAL=200
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS)
def batch():
    X,Y=[],[]
    for _ in range(BS):
        im,m=tr[rng.integers(len(tr))]; H,W=m.shape; y,x=rng.integers(0,H-CROP+1),rng.integers(0,W-CROP+1)
        a,b=im[y:y+CROP,x:x+CROP],m[y:y+CROP,x:x+CROP]; k=rng.integers(4); a,b=np.rot90(a,k),np.rot90(b,k)
        if rng.random()<.5: a,b=a[:,::-1],b[:,::-1]
        a=np.clip(a.astype(np.float32)*rng.uniform(.9,1.1)+rng.uniform(-10,10),0,255)
        X.append(norm(np.ascontiguousarray(a))); Y.append(np.ascontiguousarray(b))
    return torch.from_numpy(np.stack(X)),torch.from_numpy(np.stack(Y))
@torch.no_grad()
def validate():
    net.eval(); CM=np.zeros((4,4),np.int64)
    for x,m in vt:
        p=net(torch.from_numpy(x[None]))[0].argmax(0).numpy(); CM+=np.bincount(m.ravel()*4+p.ravel(),minlength=16).reshape(4,4)
    d=[2*CM[k,k]/(CM[k].sum()+CM[:,k].sum()) for k in range(4)]; net.train(); return float(np.mean(d)), [float(v) for v in d]
best,hist=validate(); hist=[{'step':0,'val_mean_dice':best[0] if isinstance(best,tuple) else best}]
b0=validate(); best=b0[0]; print('step 0 val',b0,flush=True); torch.save({'state_dict':net.state_dict(),'step':0,'val':b0},'ckpt_ft_best.pt')
hist=[{'step':0,'val_mean_dice':b0[0],'val_class_dice':b0[1]}]; net.train(); t=time.time()
for s in range(1,STEPS+1):
    X,Y=batch(); loss=crit(net(X),Y); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if s%50==0: print(s,round(loss.item(),4),round(time.time()-t),flush=True)
    if s%EVAL==0 or s==STEPS:
        v=validate(); hist.append({'step':s,'val_mean_dice':v[0],'val_class_dice':v[1]}); print('val',s,v,flush=True)
        if v[0]>best: best=v[0]; torch.save({'state_dict':net.state_dict(),'step':s,'val':v},'ckpt_ft_best.pt')
json.dump(hist,open('finetune_history.json','w'),indent=1)
