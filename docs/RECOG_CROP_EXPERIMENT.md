# Recognizer-Crop Experiment (Idea 1)

## Why

HiGAN+'s `Recognizer (R)` is a strong, frozen OCR teacher.  Its CTC loss against the *whole* word is what keeps the generated image legible, but it is also what pushes `G` toward neat, OCR-friendly handwriting and away from the messy style supplied by `E`.

The hypothesis behind this branch: relax `R`'s grip by only asking it to read a *slice* of each fake word.  If `G` only has to keep some portion of the word readable on each iter, it has more freedom to honour the writer's style — including ugly styles — without `R` flagging the whole image as garbage.

This is the same crop-as-regularisation trick VATr+ uses on its discriminator, repurposed for the recognizer.

## What changed in code

* `HiGAN+/networks/recog_crop.py` — new module: `CropConfig`, `crop_for_recognizer`, `should_apply`.
* `HiGAN+/networks/model.py` — in the G-step of `GlobalLocalAdversarialModel.train()`, the three fake-CTC pathways now route their inputs through `crop_for_recognizer` when `training.recog_crop` is enabled.  Disabled / missing config falls straight back to the original full-image pathway.
* `HiGAN+/lib/wandb_logger.py` — new optional logger.  Mirrors every TensorBoard scalar (loss/* and valid/*) to wandb when the `wandb:` block in the YAML enables it.  No-ops when wandb is missing or `enabled: false`.
* `HiGAN+/networks/model.py` — also adds a dedicated `training.eval_fid_every` cadence so FID/KID/IS gets logged every N epochs regardless of the legacy `start_save_epoch_val + save_epoch_val` block.

Three experiment configs, plus a baseline, all derived from `gan_iam_kaggle.yml`:

| File                                      | `recog_crop.mode`     | What it does                                                  |
| ----------------------------------------- | --------------------- | ------------------------------------------------------------- |
| `configs/gan_iam_crop_baseline.yml`       | _(none)_              | Upstream behaviour, FID/KID logged every epoch.               |
| `configs/gan_iam_crop_left_half.yml`      | `left_half`           | Always keep left 50 % of the image and first ⌈L/2⌉ chars.     |
| `configs/gan_iam_crop_left_3q.yml`        | `left_three_quarter`  | Always keep left 75 % and first ⌈3L/4⌉ chars.                 |
| `configs/gan_iam_crop_char_aligned.yml`   | `char_aligned`        | Per-sample uniform random `[i:j]` slice, ≥ 1 char.            |

A smoke config exists too: `configs/gan_iam_crop_smoke.yml` (2 epochs × 5 iters, `prob: 1.0` to force the crop branch on every iter).

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

`scores` returned by `validate()` is mirrored both to TensorBoard (`valid/<key>`) and to wandb when enabled.

## wandb

Add this block to the YAML:

```yaml
wandb:
  enabled: true
  project: 'higanplus-recog-crop'
  group: 'recog-crop-v1'
  tags: ['char_aligned']
  # entity: null      # default account
  # mode: online      # online | offline | disabled
```

Set `WANDB_API_KEY` in the environment (or via `wandb login`) before running.  When the package is missing, the key is unset, or `enabled: false`, the trainer logs a single info line and continues with TensorBoard only.

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

Kaggle notebook: `docs/kaggle_train_recog_crop.ipynb` clones the `feat/recog-random-crop` branch, sets up the dataset, optionally logs into wandb, smoke-tests, then runs each experiment.

## Limits / known gotchas

* The `gp_ctc` gradient-balance term is computed from `grad(fake_ctc_loss_rand, fake_ctc_rand)`.  When the crop branch is taken on a step, `fake_ctc_rand` are logits from the cropped image, and `gp_ctc` therefore reflects the *cropped* CTC's gradient magnitude.  This is intentional — we want the balance term to track whatever signal `R` is currently producing — but the per-iter scale of `gp_ctc` will jitter more than in the baseline when `prob ∈ (0, 1)`.  Worth keeping an eye on in the wandb plot.
* `min_chars` defaults to `1` so no batch ever ends up with an empty CTC target.  Increase it (e.g. `min_chars: 2`) only if the recognizer struggles with single-char crops.
* `RecognizeModel` and `WriterIdentifyModel` were intentionally left unchanged.  The crop trick is GAN-specific; OCR/WID training still wants the full image.
