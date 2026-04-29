"""Sanity check: load deploy ckpt, generate one handwritten word image, save to PNG."""
import sys, os
sys.path.insert(0, '.')
import torch
import numpy as np
from PIL import Image
from lib.utils import yaml2config
from networks import get_model
from networks.rand_dist import prepare_z_dist


def main():
    cfg = yaml2config('configs/gan_iam.yml')
    model = get_model(cfg.model)(cfg, './tmp_sanity')
    model.load('./pretrained/deploy_HiGAN+.pth', cfg.device)
    model.set_mode('eval')

    text = 'hello'
    fake_lbs = model.label_converter.encode([text])
    fake_lbs = torch.LongTensor(fake_lbs)
    fake_lb_lens = torch.IntTensor([len(text)])

    nrow = 4
    fake_lbs = fake_lbs.repeat(nrow, 1).to(cfg.device)
    fake_lb_lens = fake_lb_lens.repeat(nrow,).to(cfg.device)
    rand_z = prepare_z_dist(nrow, cfg.EncModel.style_dim, cfg.device)
    rand_z.sample_()

    with torch.no_grad():
        gen_imgs = model.models.G(rand_z, fake_lbs, fake_lb_lens)
    print('gen_imgs shape:', tuple(gen_imgs.shape), 'min/max:',
          gen_imgs.min().item(), gen_imgs.max().item())

    arr = ((1 - gen_imgs) * 127).clamp(0, 255).squeeze(1).cpu().numpy().astype(np.uint8)
    big = np.concatenate(list(arr), axis=0)  # vertical stack
    Image.fromarray(big).save('sanity_hello.png')
    print('saved -> sanity_hello.png')


if __name__ == '__main__':
    main()
