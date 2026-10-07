import matplotlib as mpl, matplotlib.font_manager as fm
for f in ['/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf','/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-bold.otf',
          '/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-italic.otf','/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-bolditalic.otf']:
    fm.fontManager.addfont(f)
mpl.rcParams.update({'font.family':'TeX Gyre Termes','font.size':8,'axes.titlesize':8.5,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
    'legend.fontsize':7.5,'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':600,'figure.dpi':150,'axes.edgecolor':'#333333','axes.labelcolor':'#111111',
    'xtick.color':'#333333','ytick.color':'#333333','grid.color':'#e6e6e6','grid.linewidth':0.5})
BLUE='#2a78d6'; ORANGE='#eb6834'; AQUA='#1baf7a'; VIOLET='#4a3aa7'; GRAY='#6b6b6b'; LIGHT='#f2f2f0'; INK='#111111'; RED='#e34948'
GRADE=['#8fc1e3','#3f8fcb','#1a5aa0','#0a2a5e']   # sequential single hue MF-0..MF-3
# tissue colours for masks (BoMBR source figure convention: reticulin blue, fat green, cell pink, bone red)
TISSUE={0:(33,102,172),1:(120,198,121),2:(244,165,194),3:(203,24,29)}
TISSUE_NAMES={0:'Reticulin',1:'Fat',2:'Cell',3:'Bone'}
def save(fig,name):
    for ext in ['pdf','svg','png']: fig.savefig(f'{name}.{ext}',bbox_inches='tight',pad_inches=0.02)
    fig.savefig(f'{name}.tiff',bbox_inches='tight',pad_inches=0.02,dpi=600,pil_kwargs={'compression':'tiff_lzw'})
