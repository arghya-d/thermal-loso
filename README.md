# thermal-loso

Code and results for the paper

> **How frame-level partitioning inflates accuracy in thermal intoxication detection: a controlled study on two cohorts**
> Arghya Dev Mondal and Harkeerat Kaur, Indian Institute of Technology Jammu (submitted)

Thermal face images taken in short bursts give many near-identical frames per
recording. If the frames are split into training and test sets at random, a
test frame has its own near-copies in training. This repository measures how
much that changes the reported accuracy. Everything else stays fixed: the
network, the frames, the folds and the seed.

| Dataset | P1, by participant | P2, by frame |
|---|---|---|
| Patras Sober-Drunk (34 participants, periorbital) | 76.2% | 99.7% |
| PUCV-DTF (46 participants, face) | 67.4% | 100.0% |

Three more partition rules (P3, P4, P4c) separate the near-duplicate frames
from the person being seen in training. A one-feature logistic regression is
used as a baseline. See the paper for details.

## Layout

```
configs/      patras.yaml, pucv.yaml (all settings used in the paper)
src/
  data.py       reading both datasets, frame sampling, normalisation, transforms
  model.py      ResNet-18 / MobileNet-V2 with one output
  train.py      training loop with early stopping, prediction
  splits.py     the partition rules P1, P2, P3 and checks on each split
  run.py        one rule end to end, writes the result CSVs
  stats.py      fold metrics and bootstrap intervals
  baseline.py   nested one-feature baseline
notebooks/
  01_data_check.ipynb     what is in each archive, why 7 Patras participants are excluded
  02_patras_runs.ipynb    every Patras network run (GPU)
  03_pucv_runs.ipynb      the PUCV-DTF runs (GPU)
  04_baseline.ipynb       one-feature baseline and image measurements (CPU, needs the images)
  05_analysis.ipynb       every number, table and figure in the results (CPU, no images needed)
outputs/
  runs/         predictions of the runs reported in the paper
  manifests/    frames and test folds of each run
  tables/       tables and per-recording values
  figures/      Figures 1 and 2
```

## Running the code

The notebooks run in order:

1. `01_data_check` lists what is in each archive and why seven Patras
   participants are excluded.
2. `02_patras_runs` and `03_pucv_runs` train the networks (GPU, about 1.5 to
   2 hours per run on a P100 or T4) and write the predictions to
   `outputs/runs/`.
3. `04_baseline` fits the one-feature baseline and measures the images
   (CPU, needs the images). It writes to `outputs/runs/` and `outputs/tables/`.
4. `05_analysis` prints every number in the results section from those files,
   writes Tables 2 to 4 and draws Figures 1 and 2 (CPU, no images needed).

The predictions used in the paper are included, so `05_analysis` can be run
on its own without the data or a GPU.

Notes:

- GPU training is not bit-for-bit repeatable, so a new training run gives
  numbers within the seed-to-seed spread (MCC sd 0.033 on Patras), not
  identical ones.
- The Patras runs drew frames in the order the file system listed the files,
  which differs between machines. The manifests in `outputs/manifests/` fix
  the frames and test folds used.
- PUCV-DTF files are read in sorted order, so frames and folds are fixed by
  the seed.
- The frames added in P4 and P4c are drawn by the code and were not saved.
- The predictions in `outputs/runs/` came from an earlier version of this
  code. On a small CPU test on PUCV-DTF, the current code gives identical
  predictions for P1, P2, P3, the normalisation control and MobileNet-V2.

## Data

Neither dataset is redistributed here. Put them under `data/` (ignored by git):

```
data/
  patras/            one folder per participant, NN_<name>/
    NN_<name>_<acq>_<site>_<sex>_<age>_<weight>.tif     50-frame burst
  pucv/              one folder per condition, 1 to 5
    Sujeto_<subject>_<condition>_<frame>_.jpg
```

**University of Patras Sober-Drunk.** G. Koukiou and V. Anastassopoulos,
Electronics Laboratory, Physics Department, University of Patras. The
laboratory provides a download link at <https://old.physics.upatras.gr/sober/>;
when it did not work for us, the authors sent the data by email. Their terms
require citing their papers (listed in the paper). The file names hold
personal details of the participants; the code only reads the participant
number, acquisition and site from them.

**PUCV-DTF.** G. Hermosilla et al., "Face recognition and drunk
classification using infrared face images", Journal of Sensors, 2018,
[doi:10.1155/2018/5813514](https://doi.org/10.1155/2018/5813514). The link
given in that paper no longer works. We used the public copy at
<https://www.kaggle.com/datasets/lymingchen/pucv-drunk-database>, which has
11,500 images matching the published description.

## Setup

Python 3.10 or newer. Install PyTorch for your machine first
(<https://pytorch.org/get-started/locally/>), then

```bash
pip install -r requirements.txt
```

The paper's runs used PyTorch 2.5.1 (CUDA 12.1) on a P100 and PyTorch 2.11.0
(CUDA 12.8) on a T4. CUDA is used if available, then Apple MPS, then CPU. On
macOS or Windows set `num_workers: 0` in the config.

## Output files

`outputs/runs/<dataset>/<run>_preds.csv`: one row per test frame
(`fold, sample_id, label, probability, prediction`).

`outputs/runs/<dataset>/<run>_folds.csv`: one row per fold (accuracy,
sensitivity, specificity, MCC, ROC-AUC, best epoch, best validation loss).
Under P1, P4 and P4c the fold is the participant number.

A `sample_id` such as `25_e_4_f22` is participant 25, eye site, acquisition 4,
frame 22 of the burst.

| Run | What it is |
|---|---|
| `P1`, `P2`, `P3`, `P4`, `P4c` | the partition rules |
| `P1_s43`, `P1_s44`, `P1_s45` | P1 with seeds 43 to 45 |
| `P1G` | P1 with normalisation over all participants |
| `P1_mnv2`, `P2_mnv2` | MobileNet-V2 under P1 and P2 |
| `D1`, `D2` | sober against the 20 and 50 min acquisitions |
| `N36` | P1 with participants 9 and 24 put back |
| `baseline` | the one-feature baseline on the P1 folds |

## Citation

If you use this code, please cite the paper (citation will be updated once
published) and the two datasets.

## Licence

MIT for the code, see `LICENSE`. The datasets have their own terms.
