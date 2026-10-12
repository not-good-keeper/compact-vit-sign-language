cd /d/nn_dl_prac
export PYTHONPATH=/d/nn_dl_prac PYTHONDONTWRITEBYTECODE=1
PYX=/c/Users/ADMIN/anaconda3/envs/islvit/python.exe
D=C:/isl_gan_work/diag
# baseline = the study's own 180p SR-GAN landmarks, re-labelled so they can sit beside the clean crops
mkdir -p $D/L3srgan_base; for f in hands hand_src pose done; do cp C:/isl_gan_work/test_L3_srgan_lm/$f.npy $D/L3srgan_base/; done
$PYX - <<'PY'
import sys; sys.path.insert(0,'.')
from pathlib import Path
from islvit.sr.diag_detector import write_meta
write_meta(Path('C:/isl_gan_work/diag/L3srgan_base'))
PY
for v in base all; do $PYX -m islvit.sr.diag_detector interp --src $D/L3srgan_$v --out $D/L3srgan_${v}_interp --gap 4; done
$PYX -m islvit.sr.diag_detector score $D/L3srgan_base $D/L3srgan_lowconf $D/L3srgan_bigroi $D/L3srgan_multiroi $D/L3srgan_all $D/L3srgan_base_interp $D/L3srgan_all_interp
echo SCORED
