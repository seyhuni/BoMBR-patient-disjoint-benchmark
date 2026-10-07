"""Visual/colour unit test for the BoMBR mask encoding (0=Reticulin, 1=Fat, 2=Cell, 3=Bone).
Reticulin (silver-impregnated fibres) must be the darkest, least saturated tissue class;
bone (orange-brown trabeculae) must be the most saturated class with R > G > B; fat must be the brightest.
Run:  python test_mask_encoding.py <Myelofibrozis_PNG root> [n_fields]"""
import sys, glob, os, numpy as np, cv2
def class_stats(root, n=40):
    V={k:[] for k in range(4)}; S={k:[] for k in range(4)}; RGB={k:[] for k in range(4)}
    ims=sorted(glob.glob(os.path.join(root,'*','images','*.png')))[::max(1,len(glob.glob(os.path.join(root,'*','images','*.png')))//n)]
    for f in ims:
        im=cv2.imread(f); m=cv2.imread(f.replace(os.sep+'images'+os.sep,os.sep+'masks'+os.sep)[:-4]+'_mask.png',0); hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        for k in range(4):
            s=m==k
            if s.sum()>200: V[k].append(hsv[...,2][s].mean()); S[k].append(hsv[...,1][s].mean()); RGB[k].append(im[s].mean(0)[::-1])
    return {k:(np.mean(V[k]),np.mean(S[k]),np.mean(RGB[k],0)) for k in range(4)}
def test_encoding(root, n=40):
    st=class_stats(root,n); v={k:st[k][0] for k in st}; s={k:st[k][1] for k in st}; rgb=st[3][2]
    assert min(v,key=v.get)==0, f'index 0 is not the darkest class: {v}'
    assert max(s,key=s.get)==3 and rgb[0]>rgb[1]>rgb[2], f'index 3 is not orange-brown bone: S={s}, RGB={rgb}'
    assert max(v,key=v.get)==1, f'index 1 is not the brightest (fat) class: {v}'
    return st
if __name__=='__main__':
    st=test_encoding(sys.argv[1], int(sys.argv[2]) if len(sys.argv)>2 else 40)
    for k,n in enumerate(['Reticulin','Fat','Cell','Bone']): print(k,n,'V=%.0f S=%.0f RGB=%s'%(st[k][0],st[k][1],np.round(st[k][2]).astype(int)))
    print('PASS: 0=Reticulin, 1=Fat, 2=Cell, 3=Bone')
