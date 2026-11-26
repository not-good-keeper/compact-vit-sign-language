# Records from the GAN study's work directory

The GAN study kept its large intermediates outside the repository, in `$GAN_WORK`
(`C:/isl_gan_work` on the machine that ran it; the scripts now default to `gan_work/`
in the repository root). This folder keeps every small file from that directory, at
its original relative path:

- `FRONTEND.json`: the front end (upscaler, degradation level, arm) each extraction cache was built with;
- `index.csv`: the clip list of each extraction cache;
- `score_s{0,1,2}.json`: per-seed scores of the detector-lever diagnostics (`diag/`);
- `hr_*.json`, `baselines.json`, `*_paths.csv`: manifests of the patch banks and clip lists.

The large files are not kept: the patch banks (`hr_train.npy`, `hr_train_v3.npy`,
`hr_heldout.npy`, `hands_train_v3.npy`, `hp_train.npz`, `hp_val.npz`, about 7 GB) and the
crop and landmark caches (`crops.npy`, `hands.npy`, `pose.npy`, about 6 GB). All of them are
rebuilt from INCLUDE by the commands in `run_gan_study.sh`, `run_gan_study_v2.sh` and
`run_gan_v3.sh` (`islvit.data.crops`, `islvit.data.landmarks`), except the three banks the v3
generator reads, which no script builds:

```bash
export ISLVIT_CACHE=cache128_f32 GAN_WORK=gan_work
python -m islvit.sr.patches --split train --split-file splits/full263__session-disjoint.csv \
  --out $GAN_WORK/hr_train_v3.npy                       # manifest: hr_train_v3.json
python -m islvit.sr.handproxy build --split train --out $GAN_WORK/hp_train.npz
python -m islvit.sr.handproxy build --split val   --out $GAN_WORK/hp_val.npz --per-clip 6
# hands_train_v3.npy is the images array of hp_train.npz (checked equal, 43,682 crops)
python -c "import numpy as np; np.save('gan_work/hands_train_v3.npy', np.load('gan_work/hp_train.npz')['images'])"
```
