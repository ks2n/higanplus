import os
# Reduce CUDA memory fragmentation. Must be set before any `import torch`.
# HiGAN+'s G-step concats fake+style+recn into a 3x super-batch and the
# patch discriminator processes a variable number of patches per word, so
# the peak allocation oscillates a lot between iters. expandable_segments
# lets the allocator return memory to the pool between peaks.
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
# Older alias still honored on torch <2.4.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from datetime import datetime
import argparse

from lib.utils import yaml2config
from networks import get_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/gan_iam.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()
    cfg = yaml2config(args.config)
    run_id = datetime.strftime(datetime.now(), '%m-%d-%H-%M')
    logdir = os.path.join("runs", os.path.basename(args.config)[:-4] + '-' + str(run_id))

    model = get_model(cfg.model)(cfg, logdir)
    model.train()

