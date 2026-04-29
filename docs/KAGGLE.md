# Run on Kaggle

Notebook recipe to clone and train this fork on Kaggle.

```bash
# In a Kaggle notebook cell (Internet ON, GPU enabled)
!git clone https://github.com/ks2n/higanplus.git
%cd higanplus
!pip install -r requirements.txt
!bash scripts/setup_data.sh
%cd HiGAN+
!python train.py --config ./configs/gan_iam.yml
```

Notes:
- Kaggle ships PyTorch already, so `requirements.txt` skips torch. Install a
  matching torch build only if needed.
- Outputs land in `HiGAN+/runs/` (tensorboard logs) and `HiGAN+/ckpts/`
  (checkpoints). Save them as a Kaggle dataset / output to persist.
- Smoke test before training:
  ```bash
  !python run_demo.py --text "hello world" --out out_rand.png
  ```
