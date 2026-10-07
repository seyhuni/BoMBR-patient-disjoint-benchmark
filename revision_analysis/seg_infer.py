"""UNet++-ResNet34 inference (existing checkpoint) under three input-scale protocols.
Raw model output indices are saved unchanged; the corrected class mapping is applied downstream.
  original : field resized to 512x512 (as in training / the submitted run)
  native   : native-resolution 512x512 tiles stitched to the full field (test fields are native 512x512 crops)
  matched  : (test only) test crop downsampled to the dev-equivalent input scale, predicted, upsampled back
"""
import sys, os, numpy as np, pandas as pd, cv2, torch, segmentation_models_pytorch as smp
torch.set_num_threads(2)
MEAN=np.array([0.485,0.456,0.406]); STD=np.array([0.229,0.224,0.225])
mode=sys.argv[1]
ck=torch.load('ckpt_ft_best.pt' if mode=='ft' else 'ckpt/segmentation_unetpp_resnet34_best.pt',map_location='cpu',weights_only=False)
net=smp.UnetPlusPlus(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=4); net.load_state_dict(ck['state_dict']); net.eval()
def prep(rgb): return torch.from_numpy(((rgb/255.0-MEAN)/STD).transpose(2,0,1)[None].astype(np.float32))
@torch.no_grad()
def logits(rgb): return net(prep(rgb))[0].numpy()
def pred_resized(rgb,size=(512,512)):
    return logits(cv2.resize(rgb,size,interpolation=cv2.INTER_LINEAR)).argmax(0).astype(np.uint8)
def starts(L,t=512):
    s=list(range(0,max(L-t,0)+1,t))
    if s[-1]+t<L: s.append(L-t)
    return s
def pred_native(rgb):
    H,W=rgb.shape[:2]; acc=np.zeros((4,H,W),np.float32)
    for y in starts(H):
        for x in starts(W):
            acc[:,y:y+512,x:x+512]+=logits(rgb[y:y+512,x:x+512])
    return acc.argmax(0).astype(np.uint8)
man=pd.read_csv('patient_safe_manifest.csv')
mode=sys.argv[1]; subset=sys.argv[2]
rows=man[man.subset=='test'] if subset=='test' else man[man.subset.isin(['train','val'])]
out=f'pred/{mode}'; os.makedirs(out,exist_ok=True)
for i,r in enumerate(rows.itertuples()):
    op=f'{out}/{r.filename}'
    if os.path.exists(op): continue
    rgb=cv2.cvtColor(cv2.imread(f'data/{r.split}/images/{r.filename}'),cv2.COLOR_BGR2RGB)
    H,W=rgb.shape[:2]
    if mode=='original': p=cv2.resize(pred_resized(rgb),(W,H),interpolation=cv2.INTER_NEAREST)
    elif mode=='native': p=pred_native(rgb) if (H>512 or W>512) else pred_resized(rgb,(W,H))
    elif mode=='reinhard':
        # unsupervised per-image Reinhard colour normalisation to development-set LAB statistics (no test labels used)
        import json; st=json.load(open('dev_lab_stats.json'))
        lab=cv2.cvtColor(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),cv2.COLOR_BGR2LAB).astype(np.float32)
        mu,sd=lab.reshape(-1,3).mean(0),lab.reshape(-1,3).std(0)+1e-6
        lab=np.clip((lab-mu)/sd*np.array(st['std'])+np.array(st['mean']),0,255).astype(np.uint8)
        rgbn=cv2.cvtColor(cv2.cvtColor(lab,cv2.COLOR_LAB2BGR),cv2.COLOR_BGR2RGB)
        p=pred_resized(rgbn,(W,H)) if (H<=512 and W<=512) else pred_native(rgbn)
    elif mode=='ft':
        # corrected segmenter: colour-normalised input at the test pixel scale; prediction returned at native size
        import json; from features import to_test_scale
        st=json.load(open('dev_lab_stats.json'))
        lab=cv2.cvtColor(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),cv2.COLOR_BGR2LAB).astype(np.float32)
        mu,sd=lab.reshape(-1,3).mean(0),lab.reshape(-1,3).std(0)+1e-6
        rgbn=cv2.cvtColor(cv2.cvtColor(np.clip((lab-mu)/sd*np.array(st['std'])+np.array(st['mean']),0,255).astype(np.uint8),cv2.COLOR_LAB2BGR),cv2.COLOR_BGR2RGB)
        if H<=512 and W<=512: p=pred_resized(rgbn,(W,H))
        else:
            small=to_test_scale(rgbn,False); p=cv2.resize(pred_native(small),(W,H),interpolation=cv2.INTER_NEAREST)
    elif mode=='matched':
        # dev fields (mean 1840x1200 native) are shrunk to 512x512: factors ~3.6 (x) and 2.34 (y)
        w,h=int(round(W/3.6/32))*32, int(round(H/2.34/32))*32
        p=cv2.resize(pred_resized(rgb,(w,h)),(W,H),interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(op,p)
    print(mode,subset,i,r.filename,flush=True)
