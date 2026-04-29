# Recognizer-Crop Experiment (Idea 1)

## Why

HiGAN+'s `Recognizer (R)` is a strong, frozen OCR teacher.  Its CTC loss against the *whole* word is what keeps the generated image legible, but it is also what pushes `G` toward neat, OCR-friendly handwriting and away from the messy style supplied by `E`.

The hypothesis behind this branch: relax `R`'s grip by only asking it to read a *slice* of each fake word.  If `G` only has to keep some portion of the word readable on each iter, it has more freedom to honour the writer's style — including ugly styles — without `R` flagging the whole image as garbage.

This is the same crop-as-regularisation trick VATr+ uses on its discriminator, repurposed for the recognizer.

## What changed in code

* `HiGAN+/networks/recog_crop.py` — new module: `CropConfig`, `crop_for_recognizer`, `should_apply`.
* `HiGAN+/networks/model.py` — in the G-step of `GlobalLocalAdversarialModel.train()`, the three fake-CTC pathways now route their inputs through `crop_for_recognizer` when `training.recog_crop` is enabled.  Disabled / missing config falls straight back to the original full-image pathway.
* `HiGAN+/lib/wandb_logger.py` — optional logger.  Off by default in this branch (`wandb.enabled: false` in the experiment YAMLs).  No-ops cleanly when wandb is missing, the key is unset, or `enabled: false`, and re-enabling later is a one-line YAML flip.
* `HiGAN+/networks/model.py` — also adds a dedicated `training.eval_fid_every` cadence so FID/KID/IS gets logged every N epochs regardless of the legacy `start_save_epoch_val + save_epoch_val` block.

Three experiment configs, plus a baseline, **all train G/D/E from random init** (the upstream paper's `gan_iam.yml` recipe).  Only the auxiliary teachers `R`, `W`, and `B` are warm-started from the author's released checkpoints (`ocr_iam_new.pth` and `wid_iam_new.pth`).  This is the correct setup for measuring whether the recognizer-crop trick actually changes how `G` learns to draw, rather than just nudging an already-trained `G` into a slightly different attractor.

| File                                      | `pretrained_ckpt` | `recog_crop.mode`     | What it does                                                  |
| ----------------------------------------- | ----------------- | --------------------- | ------------------------------------------------------------- |
| `configs/gan_iam_crop_baseline.yml`       | _empty_           | _(none)_              | Reference run -- upstream R-CTC behaviour, no crop.           |
| `configs/gan_iam_crop_left_half.yml`      | _empty_           | `left_half`           | Always keep left 50 % of the image and first ⌈L/2⌉ chars.     |
| `configs/gan_iam_crop_left_3q.yml`        | _empty_           | `left_three_quarter`  | Always keep left 75 % and first ⌈3L/4⌉ chars.                 |
| `configs/gan_iam_crop_char_aligned.yml`   | _empty_           | `char_aligned`        | Per-sample uniform random `[i:j]` slice, ≥ 1 char.            |

A smoke config exists too: `configs/gan_iam_crop_smoke.yml` (2 epochs × 5 iters, `prob: 1.0` to force the crop branch on every iter).

> **Why not start from `deploy_HiGAN+.pth`?**  That checkpoint is the result of 70 epochs trained with the full-image CTC loss.  Loading it and then fine-tuning with crop would only let `G` "unlearn" a small amount before settling near the previous attractor.  Any FID gap we measure would be dominated by the warm-start, not by the crop trick.  Random init keeps the comparison clean -- baseline and crop runs see exactly the same starting state.

## Geometry refresher

`G` produces images of width `lb_len * char_width` (default `char_width = 32`).  `R` downsamples width by `len_scale = 16`, so each character occupies `32 px = 2 CTC time-steps`.  Cropping at character boundaries therefore keeps the `input_lengths = img_len // len_scale` math intact.

Pad value for the cropped tensor is `-1`, matching `nn.ConstantPad2d(2, -1)` inside `Recognizer` and the rest of the pipeline.

## Crop probability and where the crop applies

* `recog_crop.prob` (default `0.5`): probability of drawing the crop branch this iter.  When skipped, the full image flows to `R` exactly as in baseline.  Setting `prob: 1.0` in the smoke config is intentional — it exercises the new code path on every iter.
* The crop is applied **only to the three fake streams** (`fake_imgs`, `style_imgs`, `recn_imgs`).  Real images that go to `D` and `R` are untouched, so `R` keeps seeing well-formed ground truth and never drifts.

## Sort guarantee

The recognizer's BLSTM uses `pack_padded_sequence(..., enforce_sorted=True)`.  The dataset's collate already sorts every batch by length DESC, but random char-aligned crops can break that order.  `crop_for_recognizer` re-sorts its outputs by `cropped_img_lens` DESC before returning, so callers do not have to think about it.

## FID / KID / IS per epoch

Set `training.eval_fid_every: N` in the YAML to compute FID/KID/IS every N epochs (the experiment configs use `1`).  The legacy `start_save_epoch_val + save_epoch_val` cadence still gates the `best.pth` save, so existing configs without the new key behave exactly as before.

`scores` returned by `validate()` is mirrored to TensorBoard (`valid/<key>`).

## wandb (disabled by default)

The repo ships an optional wandb shim at `HiGAN+/lib/wandb_logger.py`.  All four experiment configs currently set `wandb.enabled: false`, and the four Kaggle notebooks do not include a wandb-login cell — running them produces TensorBoard logs only, no external account required.

To re-enable wandb later (for example to share a comparison dashboard across the four runs), flip `enabled: true` in the YAML and set `WANDB_API_KEY` in the environment before running:

```yaml
wandb:
  enabled: true
  project: 'higanplus-recog-crop'
  group: 'recog-crop-v1'
  tags: ['char_aligned']
  # entity: null
  # mode: online        # online | offline | disabled
```

The shim no-ops cleanly when the `wandb` Python package is missing, the key is unset, or `enabled: false`, so the trainer never crashes because of a wandb misconfiguration.

## How to run

Local smoke (verifies the crop path with real data on GPU, ≈ 10 iters):

```bash
cd HiGAN+
python train.py --config configs/gan_iam_crop_smoke.yml
```

Full experiment, one of:

```bash
python train.py --config configs/gan_iam_crop_baseline.yml
python train.py --config configs/gan_iam_crop_left_half.yml
python train.py --config configs/gan_iam_crop_left_3q.yml
python train.py --config configs/gan_iam_crop_char_aligned.yml
```

Kaggle notebooks: one per experiment so each gets its own 9-hour session budget without sharing GPU or accidentally cross-patching configs.

| Experiment            | Notebook                              | Config                              |
| --------------------- | ------------------------------------- | ----------------------------------- |
| baseline (no crop)    | `docs/kaggle_train_baseline.ipynb`    | `gan_iam_crop_baseline.yml`         |
| left_half             | `docs/kaggle_train_left_half.ipynb`   | `gan_iam_crop_left_half.yml`        |
| left_three_quarter    | `docs/kaggle_train_left_3q.ipynb`     | `gan_iam_crop_left_3q.yml`          |
| char_aligned          | `docs/kaggle_train_char_aligned.ipynb`| `gan_iam_crop_char_aligned.yml`     |

Each notebook clones the `feat/recog-random-crop` branch, sets up the dataset, runs the smoke config once, then trains its single experiment.  The resume helper inside each notebook is scoped to that experiment's `runs/<config-name>-*` prefix only, so running them in parallel under different Kaggle accounts is safe.

### Time budget

Because every experiment trains G/D/E from scratch, expect **~30-35 hours per experiment** at `batch_size: 4` on a Kaggle T4.  Kaggle sessions cap at 9 hours, so plan on 4-5 sessions per experiment, using the in-notebook resume helper to chain them.  Sample images written every 500 G-steps; do not expect clean handwriting before ~epoch 5-10.

## Limits / known gotchas

* The `gp_ctc` gradient-balance term is computed from `grad(fake_ctc_loss_rand, fake_ctc_rand)`.  When the crop branch is taken on a step, `fake_ctc_rand` are logits from the cropped image, and `gp_ctc` therefore reflects the *cropped* CTC's gradient magnitude.  This is intentional — we want the balance term to track whatever signal `R` is currently producing — but the per-iter scale of `gp_ctc` will jitter more than in the baseline when `prob ∈ (0, 1)`.  Worth keeping an eye on in TensorBoard.
* `min_chars` defaults to `1` so no batch ever ends up with an empty CTC target.  Increase it (e.g. `min_chars: 2`) only if the recognizer struggles with single-char crops.
* `RecognizeModel` and `WriterIdentifyModel` were intentionally left unchanged.  The crop trick is GAN-specific; OCR/WID training still wants the full image.
