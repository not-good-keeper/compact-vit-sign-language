cd "$(dirname "$0")/../../.." || exit 1
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1
PYX=${PY:-python}
D=${GAN_WORK:-gan_work}/diag
# baseline = the study's own 180p SR-GAN landmarks, re-labelled so they can sit beside the clean crops
mkdir -p $D/L3srgan_base; for f in hands hand_src pose done; do cp ${GAN_WORK:-gan_work}/test_L3_srgan_lm/$f.npy $D/L3srgan_base/; done
$PYX - <<'PY'
import os, sys; sys.path.insert(0,'.')
from pathlib import Path
from islvit.sr.diag_detector import write_meta
write_meta(Path(os.environ.get('GAN_WORK', 'gan_work'), 'diag', 'L3srgan_base'))
PY
for v in base all; do $PYX -m islvit.sr.diag_detector interp --src $D/L3srgan_$v --out $D/L3srgan_${v}_interp --gap 4; done
$PYX -m islvit.sr.diag_detector score $D/L3srgan_base $D/L3srgan_lowconf $D/L3srgan_bigroi $D/L3srgan_multiroi $D/L3srgan_all $D/L3srgan_base_interp $D/L3srgan_all_interp
echo SCORED
