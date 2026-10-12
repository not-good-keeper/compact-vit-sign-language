import json, csv, numpy as np
from pathlib import Path
W=Path('C:/isl_gan_work'); R=Path('runs/gan_study')
def hits(path_tpl, fam='frozen_int4'):
    out=None
    for s in range(3):
        j=json.load(open(path_tpl.format(s=s)))
        h=dict(zip(j['video_paths'], j['hits_masked_tta']))
        out=h if out is None else {k:out[k]+h[k] for k in h}
    return out  # summed over 3 seeds (0..3)
clean=hits('runs/f16_262w_lm_s{s}_qat4/masked50_clean.json')
paths=sorted(clean)
def cov(lm):  # per-clip genuine-hand fraction (either hand), from landmark presence
    p=np.load(W/lm/'hand_src.npy')>0   # (N,T,2)
    idx=[r['video_path'] for r in csv.DictReader(open(W/lm.replace('_lm','')/'index.csv'))]
    return dict(zip(idx,p.mean(axis=(1,2))))
refp=np.load('cache_lm_f32_test/hand_src.npy')>0
refidx=[r['video_path'] for r in csv.DictReader(open('cache128_f32_test/index.csv'))]
refcov=dict(zip(refidx,refp.mean(axis=(1,2))))
print('clean acc', np.mean([clean[p]/3 for p in paths]))
for lvl in ('L2','L3'):
  for arm,fam in (('bicubic',''),('srl1',''),('srgan','')):
    h=hits(str(R/f'frozen_int4/{lvl}_{arm}_s{{s}}.json'))
    c=cov(f'test_{lvl}_{arm}_lm')
    a=np.array([h[p]/3 for p in paths]); k=np.array([c[p] for p in paths]); r=np.array([refcov[p] for p in paths]); cl=np.array([clean[p]/3 for p in paths])
    ratio=k/np.maximum(r,1e-6)
    print(f'\n{lvl} {arm}: acc {a.mean():.3f}  hand-presence {k.mean():.3f} (clean {r.mean():.3f})')
    # accuracy by retained fraction of the clean hand presence
    for lo,hi in ((0,0.5),(0.5,0.8),(0.8,0.95),(0.95,9)):
        m=(ratio>=lo)&(ratio<hi)&(r>0.2)
        print(f'   retained {lo:.2f}-{min(hi,1.0):.2f}: n={m.sum():3d} acc {a[m].mean():.3f} (same clips on clean {cl[m].mean():.3f})')
    lost=(cl-a); print(f'   clips losing >=1/3 accuracy vs clean: {(lost>=0.34).sum()}, of which retained<0.8: {((lost>=0.34)&(ratio<0.8)).sum()}')
    print('   corr(acc loss, hand-presence loss) =', round(float(np.corrcoef(lost, r-k)[0,1]),3))
