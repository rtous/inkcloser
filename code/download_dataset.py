"""Download and unpack the InkRepair dataset (CODE BLOCK 1 of lineartanimdata3_closer_v12.ipynb).

Downloads the zip built by the dataset generator notebook from Google Drive,
unzips it into SPLIT_ROOT (outside the repo) and runs the same sanity checks
as the notebook.

Usage:
    python code/download_dataset.py [--split-root /content/split] [--zip /content/inkrepair_dataset.zip]
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inkcloser.data import load_split  # noqa: E402  (also usable from other scripts)

DRIVE_URL = "https://drive.google.com/file/d/1NO8i_J5mSlh7YYuwh2ZyFqkIW7ISrGuk/view?usp=drive_link"
DEFAULT_ZIP = "/content/inkrepair_dataset.zip"
DEFAULT_SPLIT_ROOT = "/content/split"


def download(local_zip):
    import gdown
    t0 = time.perf_counter()
    gdown.download(DRIVE_URL, local_zip, quiet=False, fuzzy=True)
    print(f"zip ready: {os.path.getsize(local_zip) / 1e9:.2f} GB "
          f"in {time.perf_counter() - t0:.0f} s")


def unpack(local_zip, split_root, keep_zip=False):
    tmp_dir = os.path.join(os.path.dirname(split_root), "_unzip")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(split_root, ignore_errors=True)
    t0 = time.perf_counter()
    subprocess.run(["unzip", "-q", local_zip, "-d", tmp_dir], check=True)

    found = glob.glob(os.path.join(tmp_dir, "**", "split.json"), recursive=True)
    assert found, "split.json not found in the zip. Was it built by the dataset generator notebook?"
    shutil.move(os.path.dirname(found[0]), split_root)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if not keep_zip:
        os.remove(local_zip)  # free disk space
    print(f"unzipped in {time.perf_counter() - t0:.0f} s -> {split_root}")


def check(s):
    for d in (s["test_clean_dir"], s["test_corr_dir"], s["pre_clean"], s["pre_corr"]):
        assert os.path.isdir(d), f"Missing folder: {d}"

    n_test_clean = len(os.listdir(s["test_clean_dir"]))
    n_test_corr = len(os.listdir(s["test_corr_dir"]))
    assert n_test_clean == n_test_corr == len(s["test_files"]), \
        f"test pairs mismatch: {n_test_clean} clean / {n_test_corr} corrupted / {len(s['test_files'])} listed"
    assert all(os.path.exists(os.path.join(s["test_clean_dir"], f)) for f in s["sample_files"]), \
        "preview files missing from clean_test"

    pre_manifest_path = os.path.join(s["pre_root"], "manifest.json")
    if os.path.exists(pre_manifest_path):
        with open(pre_manifest_path) as fh:
            pm = json.load(fh)
        if not pm.get("complete", False):
            print("WARNING: pre-corrupted generation was not marked complete")
        print(f"pre-corrupted: {pm.get('n_variants')} variants/image (as configured)")

    n_pre_clean = len([f for f in os.listdir(s["pre_clean"]) if f.endswith(".png")])
    n_pre_corr = len([f for f in os.listdir(s["pre_corr"])
                      if f.endswith(".png") and not f.endswith(".tmp.png")])
    print(f"IMAGE_SIZE {s['IMAGE_SIZE']} | train {len(s['train_files'])} "
          f"(pre-clean {n_pre_clean}, pre-corrupted pairs {n_pre_corr}) | "
          f"test {len(s['test_files'])} | sample {s['sample_files']}")
    if n_pre_clean < len(s["train_files"]):
        print(f"WARNING: {len(s['train_files']) - n_pre_clean} training images missing from precorrupted/clean")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-root", default=DEFAULT_SPLIT_ROOT)
    ap.add_argument("--zip", default=DEFAULT_ZIP)
    ap.add_argument("--keep-zip", action="store_true", help="do not delete the zip after unpacking")
    ap.add_argument("--skip-download", action="store_true", help="reuse an existing zip")
    ap.add_argument("--check-only", action="store_true", help="only run sanity checks on an existing split")
    args = ap.parse_args()

    if not args.check_only:
        if not (args.skip_download and os.path.exists(args.zip)):
            download(args.zip)
        unpack(args.zip, args.split_root, keep_zip=args.keep_zip)
    check(load_split(args.split_root))


if __name__ == "__main__":
    main()
