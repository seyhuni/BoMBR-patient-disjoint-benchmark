"""Twelve predicted-/reference-mask descriptors with the CORRECTED BoMBR index mapping
(0=Reticulin, 1=Fat, 2=Cell, 3=Bone) and a bounded elongation definition.
Descriptors are computed on 512x512 tiles at native acquisition resolution (the test fields are
native 512x512 crops; development fields are split into a non-overlapping 512x512 grid) and averaged per field."""
import numpy as np
from skimage.measure import label, regionprops
RET,FAT,CELL,BONE=0,1,2,3
COLS=['pct_reticulin','pct_fat','pct_cell','pct_bone','ret_component_density','ret_mean_component_area',
      'ret_median_elongation','ret_elongation_sd','ret_mean_perimeter','ret_cell_fraction','ret_lacunarity','ret_fractal_dim']
def lacunarity(b,box=16):
    h,w=b.shape; h2,w2=h-h%box,w-w%box
    blk=b[:h2,:w2].reshape(h2//box,box,w2//box,box).sum(axis=(1,3)).ravel().astype(float)
    m=blk.mean(); return float(blk.var()/(m*m)+1.0) if m>0 else 1.0
def fractal(b):
    if b.sum()==0: return 0.0
    s,c=[],[]
    for k in [2,4,8,16,32]:
        h2,w2=b.shape[0]-b.shape[0]%k,b.shape[1]-b.shape[1]%k
        n=int((b[:h2,:w2].reshape(h2//k,k,w2//k,k).sum(axis=(1,3))>0).sum())
        if n>0: s.append(k); c.append(n)
    return float(np.polyfit(np.log(1/np.array(s)),np.log(np.array(c)),1)[0]) if len(c)>=2 else 0.0
def tile_features(m, legacy_elongation=False):
    f={}; n=m.size
    f['pct_reticulin']=(m==RET).sum()/n; f['pct_fat']=(m==FAT).sum()/n; f['pct_cell']=(m==CELL).sum()/n; f['pct_bone']=(m==BONE).sum()/n
    b=(m==RET).astype(np.uint8); props=regionprops(label(b,connectivity=2))
    big=[p for p in props if p.area>=3]
    if legacy_elongation: el=np.array([p.axis_major_length/max(p.axis_minor_length,1e-6) for p in big])
    else: el=np.array([p.axis_major_length/max(p.axis_minor_length,1.0) for p in big])   # minor axis floored at 1 px
    f['ret_component_density']=len(props)/(n/1e6)          # components per megapixel
    f['ret_mean_component_area']=float(np.mean([p.area for p in props])) if props else 0.0
    f['ret_median_elongation']=float(np.median(el)) if len(el) else 0.0
    f['ret_elongation_sd']=float(np.std(el)) if len(el) else 0.0
    f['ret_mean_perimeter']=float(np.mean([p.perimeter for p in big])) if big else 0.0
    tot=f['pct_reticulin']+f['pct_cell']; f['ret_cell_fraction']=f['pct_reticulin']/tot if tot>0 else 0.0
    f['ret_lacunarity']=lacunarity(b); f['ret_fractal_dim']=fractal(b)
    return f
def _starts(L,t=512):
    s=list(range(0,max(L-t,0)+1,t))
    if s[-1]+t<L: s.append(L-t)          # edge-aligned final tile -> full field coverage
    return s
def grid(H,W,t=512):
    return [(y,x) for y in _starts(H,t) for x in _starts(W,t)]
DEV_TO_TEST_SCALE=0.65   # empirical: adipocyte diameter ratio 0.63, reticulin fibre thickness ratio 0.69 (test / dev native)
def to_test_scale(a, is_mask):
    """Resample a development field to the pixel scale of the locked test crops."""
    import cv2
    H,W=a.shape[:2]; size=(int(round(W*DEV_TO_TEST_SCALE)),int(round(H*DEV_TO_TEST_SCALE)))
    return cv2.resize(a,size,interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA)
def field_features(mask, native=True):
    """native=True: average over native 512 tiles. native=False: legacy protocol (whole field resized to 512x512 first)."""
    import cv2
    if not native:
        return tile_features(cv2.resize(mask,(512,512),interpolation=cv2.INTER_NEAREST))
    H,W=mask.shape
    if H<=512 and W<=512: return tile_features(mask)
    mask=to_test_scale(mask,True); H,W=mask.shape
    fs=[tile_features(mask[y:y+512,x:x+512]) for y,x in grid(H,W)]
    return {k:float(np.mean([f[k] for f in fs])) for k in COLS}
