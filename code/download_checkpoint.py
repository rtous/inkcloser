"""Download the released InkCloser weights (CODE BLOCK 13b of the notebook).

    python code/download_checkpoint.py [--out /content/checkpoints/inkcloser_paper.pt]

The file is a full training checkpoint (model, optimizer, scheduler, scaler,
history, loss_log, global_step, epochs_done), so it can be passed to
evaluate.py --checkpoint or train.py --resume.
"""
import argparse
import os

CKPT_URL = "https://drive.google.com/file/d/1li1ZjYNWHQfcrC2lq19JLoKHxM9KuW77/view?usp=drive_link"
DEFAULT_OUT = "/content/checkpoints/inkcloser_paper.pt"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    import gdown
    gdown.download(CKPT_URL, args.out, quiet=False, fuzzy=True)
    print(f"{args.out}: {os.path.getsize(args.out) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
