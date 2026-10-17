import pathlib
import sys

import cv2
import numpy as np

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def to_luma(img):
    # matches image crate's into_luma8: (2126 r + 7152 g + 722 b) / 10000, rounded
    if img.ndim == 2:
        return img
    b = img[:, :, 0].astype(np.uint32)
    g = img[:, :, 1].astype(np.uint32)
    r = img[:, :, 2].astype(np.uint32)
    return ((2126 * r + 7152 * g + 722 * b + 5000) // 10000).astype(np.uint8)


def compress_once(img):
    # same as get_compressed_img in lib.rs: 3x3 [1 2 1] x [1 2 1] / 16 kernel with clamped
    # edges, sampled at every even pixel, so output is (w/2, h/2)
    h, w = img.shape
    p = np.pad(img.astype(np.uint32), 1, mode="edge")
    ys = np.arange(h // 2) * 2 + 1
    xs = np.arange(w // 2) * 2 + 1
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    center = p[yy, xx]
    edges = p[yy, xx - 1] + p[yy, xx + 1] + p[yy - 1, xx] + p[yy + 1, xx]
    corners = p[yy - 1, xx - 1] + p[yy + 1, xx - 1] + p[yy - 1, xx + 1] + p[yy + 1, xx + 1]
    # integer form of (0.25 c + 0.125 e + 0.0625 k).round(); all values are >= 0
    return ((4 * center + 2 * edges + corners + 8) // 16).astype(np.uint8)


if len(sys.argv) != 2:
    print("usage: python compress.py <num_compressions>")
    sys.exit(1)
num_compressions = int(sys.argv[1])
if num_compressions < 0:
    print("num_compressions must be >= 0; exiting")
    sys.exit(1)

paths = sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() in IMG_EXTS)
if not paths:
    print("no images found in " + str(DATA_DIR) + "; exiting")
    sys.exit(1)

n = 0
for path in paths:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print("failed to read " + str(path) + "; skipping")
        continue
    compressed = to_luma(img)
    for _ in range(num_compressions):
        compressed = compress_once(compressed)
    if not cv2.imwrite(str(path), compressed):
        print("failed to write " + str(path) + "; exiting")
        sys.exit(1)
    n += 1
    if n % 50 == 0:
        print("compressed " + str(n) + " images")

print("done; compressed " + str(n) + " of " + str(len(paths)) + " images "
      + str(num_compressions) + " times")
