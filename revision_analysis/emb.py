"""Frozen (or original fine-tuned) image embeddings at matched native scale.
Every field is represented by native-resolution 512x512 tiles (test fields = one native crop; development
fields = edge-aligned 512x512 grid), each tile resized to the backbone input size; tile embeddings are averaged."""
import sys, os, numpy as np, pandas as pd, cv2, torch, torch.nn as nn
torch.set_num_threads(2)
from features import grid, to_test_scale
MEAN=np.array([0.485,0.456,0.406]); STD=np.array([0.229,0.224,0.225])
def to_t(rgb,size): 
    x=cv2.resize(rgb,(size,size),interpolation=cv2.INTER_AREA if rgb.shape[0]>size else cv2.INTER_LINEAR)/255.0
    return torch.from_numpy(((x-MEAN)/STD).transpose(2,0,1).astype(np.float32))
import json
ST=json.load(open('dev_lab_stats.json'))
def reinhard(bgr):   # per-image Reinhard normalisation to development LAB statistics (label-free)
    lab=cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB).astype(np.float32); mu,sd=lab.reshape(-1,3).mean(0),lab.reshape(-1,3).std(0)+1e-6
    return cv2.cvtColor(np.clip((lab-mu)/sd*np.array(ST['std'])+np.array(ST['mean']),0,255).astype(np.uint8),cv2.COLOR_LAB2RGB)
def load(name):
    if name=='resnet50_ft':   # the submitted image-grade CNN (ImageNet-initialised, fine-tuned on the training patients)
        from torchvision.models import resnet50
        b=resnet50(weights=None); fe=nn.Sequential(*list(b.children())[:-1])
        sd=torch.load('ckpt/image_grade_resnet50_best.pt',map_location='cpu',weights_only=False)['state_dict']
        fe.load_state_dict({k[len('feature_extractor.'):]:v for k,v in sd.items() if k.startswith('feature_extractor.')})
        return (lambda x: torch.flatten(fe(x),1)), fe, 384
    if name=='resnet18_in':   # ImageNet-1k, frozen
        from torchvision.models import resnet18
        b=resnet18(weights=None); b.load_state_dict(torch.load('weights/resnet18-5c106cde.pth',map_location='cpu'))
        fe=nn.Sequential(*list(b.children())[:-1]); return (lambda x: torch.flatten(fe(x),1)), fe, 224
    if name=='dinov2_vits14':  # self-supervised LVD-142M, frozen
        import timm
        from timm.models.vision_transformer import checkpoint_filter_fn
        m=timm.create_model('vit_small_patch14_dinov2',pretrained=False,img_size=224,num_classes=0)
        sd=torch.load('weights/dinov2_vits14_pretrain.pth',map_location='cpu',weights_only=True)
        print(m.load_state_dict(checkpoint_filter_fn(sd,m),strict=False)); return (lambda x: m(x)), m, 224
    if name=='phikon':         # histopathology foundation model (iBOT ViT-B/16 on TCGA), frozen, CLS token
        import timm, functools
        from safetensors.torch import load_file
        hf=load_file('weights/phikon_model.safetensors'); sd={}
        sd['cls_token']=hf['embeddings.cls_token']; sd['pos_embed']=hf['embeddings.position_embeddings']
        sd['patch_embed.proj.weight']=hf['embeddings.patch_embeddings.projection.weight']; sd['patch_embed.proj.bias']=hf['embeddings.patch_embeddings.projection.bias']
        for i in range(12):
            p=f'encoder.layer.{i}.'; q=f'blocks.{i}.'
            for t in ['weight','bias']:
                sd[q+'norm1.'+t]=hf[p+'layernorm_before.'+t]; sd[q+'norm2.'+t]=hf[p+'layernorm_after.'+t]
                sd[q+'attn.qkv.'+t]=torch.cat([hf[p+'attention.attention.'+k+'.'+t] for k in ['query','key','value']],0)
                sd[q+'attn.proj.'+t]=hf[p+'attention.output.dense.'+t]; sd[q+'mlp.fc1.'+t]=hf[p+'intermediate.dense.'+t]; sd[q+'mlp.fc2.'+t]=hf[p+'output.dense.'+t]
        sd['norm.weight']=hf['layernorm.weight']; sd['norm.bias']=hf['layernorm.bias']
        m=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0,global_pool='token',norm_layer=functools.partial(nn.LayerNorm,eps=1e-12))
        print(m.load_state_dict(sd,strict=True)); return (lambda x: m(x)), m, 224
name=sys.argv[1]; protocol=sys.argv[2] if len(sys.argv)>2 else 'matched'
f,mod,size=load(name); mod.eval()
man=pd.read_csv('patient_safe_manifest.csv'); rec=[]
with torch.no_grad():
    for i,r in enumerate(man.itertuples()):
        bgr=cv2.imread(f'data/{r.split}/images/{r.filename}')
        rgb=reinhard(bgr) if protocol.endswith('cn') else cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); H,W=rgb.shape[:2]
        if protocol=='original': tiles=[to_t(rgb,size)]            # whole field resized (submitted protocol)
        elif H>512 or W>512:
            rgb=to_test_scale(rgb,False); H,W=rgb.shape[:2]; tiles=[to_t(rgb[y:y+512,x:x+512],size) for y,x in grid(H,W)]
        else: tiles=[to_t(rgb,size)]
        e=f(torch.stack(tiles)).numpy().mean(0)
        rec.append([r.filename,r.patient,r.mf_grade,r.subset,len(tiles)]+e.tolist())
        if i%25==0: print(name,protocol,i,flush=True)
cols=['filename','patient','mf_grade','subset','n_tiles']+[f'e{j}' for j in range(len(e))]
os.makedirs('emb',exist_ok=True); pd.DataFrame(rec,columns=cols).to_csv(f'emb/{name}_{protocol}.csv',index=False)
