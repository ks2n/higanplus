"""Non-interactive HiGAN+ demo runner.

Generates handwriting images for a given text using a pretrained checkpoint
and saves them as PNG (no GUI / plt.show required).

Usage:
    python run_demo.py --text "hello world" --out out.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import torch

from lib.utils import yaml2config
from networks import get_model
from networks.rand_dist import prepare_z_dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gan_iam.yml")
    parser.add_argument("--ckpt", default="./pretrained/deploy_HiGAN+.pth")
    parser.add_argument("--text", default="hello world",
                        help="Text to render. Multiple words space-separated.")
    parser.add_argument("--nrow", type=int, default=8,
                        help="Number of style samples (rows).")
    parser.add_argument("--out", default="out.png")
    parser.add_argument("--device", default=None,
                        help="Override device (e.g. cpu, cuda:0).")
    args = parser.parse_args()

    cfg = yaml2config(args.config)
    if args.device:
        cfg.device = args.device

    model = get_model(cfg.model)(cfg, args.config)
    model.load(args.ckpt, cfg.device)
    model.set_mode("eval")

    device = torch.device(cfg.device)

    texts = args.text.split(" ")
    nrow = args.nrow
    ncol = len(texts)

    if ncol == 1:
        fake_lbs = model.label_converter.encode(texts)
        fake_lbs = torch.LongTensor(fake_lbs)
        fake_lb_lens = torch.IntTensor([len(texts[0])])
    else:
        fake_lbs, fake_lb_lens = model.label_converter.encode(texts)

    fake_lbs = fake_lbs.repeat(nrow, 1).to(device)
    fake_lb_lens = fake_lb_lens.repeat(nrow,).to(device)

    rand_z = prepare_z_dist(nrow, cfg.EncModel.style_dim, device)
    rand_z.sample_()
    rand_styles = rand_z.unsqueeze(1).repeat(1, ncol, 1).view(
        nrow * ncol, cfg.GenModel.style_dim
    )

    with torch.no_grad():
        gen_imgs = model.models.G(rand_styles, fake_lbs, fake_lb_lens)

    gen_imgs = (1 - gen_imgs).squeeze(1).cpu().numpy() * 127

    fig = plt.figure(figsize=(2 * ncol, nrow), dpi=120)
    for i in range(nrow):
        for j in range(ncol):
            ax = plt.subplot(nrow, ncol, i * ncol + 1 + j)
            ax.imshow(gen_imgs[i * ncol + j], cmap="gray", vmin=-128, vmax=127)
            ax.axis("off")
    plt.tight_layout()
    out_path = os.path.abspath(args.out)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}  shape={gen_imgs.shape}  text={texts}")


if __name__ == "__main__":
    main()
