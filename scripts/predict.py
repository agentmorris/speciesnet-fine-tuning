"""
predict.py

Run a fine-tuned SpeciesNet classifier on new images. You run MegaDetector
separately; this script reads the MegaDetector results file, classifies each
animal box above a confidence threshold (using exactly the same crop, resize,
and normalization as training), and writes predictions to a CSV.

Usage:
  python scripts/predict.py RUN_FOLDER/model_best.pt md_results.json \\
      --image-root IMAGES --output-csv predictions.csv [--topk 5]
"""

import argparse
import csv
import json
import os
import sys

import timm
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import build_transforms, crop_resize    # noqa: E402


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = timm.create_model(ckpt["timm_model"], pretrained=False,
                              num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, ckpt


def animal_boxes(md_path, conf_threshold):
    with open(md_path, encoding="utf-8") as f:
        md = json.load(f)
    det_cats = md.get("detection_categories") or {}
    animal_ids = [k for k, v in det_cats.items() if v == "animal"]
    if not animal_ids:
        raise SystemExit("ERROR: MD file has no detection category named 'animal'.")
    aid = animal_ids[0]
    boxes = []
    for im in md.get("images", []):
        for d in (im.get("detections") or []):
            if d.get("category") == aid and (d.get("conf") or 0.0) >= conf_threshold:
                boxes.append((im["file"], tuple(d["bbox"]), float(d.get("conf") or 0.0)))
    return boxes


def main(argv=None):
    args = parse_args(argv)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, ckpt = load_model(args.checkpoint, device)
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 480)
    mean = tuple(ckpt.get("norm_mean", (0.5, 0.5, 0.5)))
    std = tuple(ckpt.get("norm_std", (0.5, 0.5, 0.5)))
    transform = build_transforms(img_size, train=False, mean=mean, std=std)

    boxes = animal_boxes(args.md_results, args.conf_threshold)
    topk = max(1, min(args.topk, len(classes)))

    fieldnames = ["filename", "x", "y", "w", "h", "detection_conf"]
    for r in range(topk):
        fieldnames += ["pred%d_class" % (r + 1), "pred%d_score" % (r + 1)]

    out_rows = []
    batch_tensors, batch_meta = [], []
    cache_name, cache_img = None, None

    def flush():
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)
        scores, idx = probs.topk(topk, dim=1)
        for (fname, bbox, conf), sc, ix in zip(batch_meta, scores.tolist(), idx.tolist()):
            row = {"filename": fname, "x": bbox[0], "y": bbox[1], "w": bbox[2],
                   "h": bbox[3], "detection_conf": conf}
            for r in range(topk):
                row["pred%d_class" % (r + 1)] = classes[ix[r]]
                row["pred%d_score" % (r + 1)] = sc[r]
            out_rows.append(row)
        batch_tensors.clear()
        batch_meta.clear()

    for fname, bbox, conf in boxes:
        if fname != cache_name:
            path = fname if os.path.isabs(fname) else os.path.join(args.image_root, fname)
            try:
                cache_img = Image.open(path).convert("RGB")
            except Exception:
                cache_img = None
            cache_name = fname
        if cache_img is None:
            continue
        batch_tensors.append(transform(crop_resize(cache_img, bbox, img_size)))
        batch_meta.append((fname, bbox, conf))
        if len(batch_tensors) >= args.batch_size:
            flush()
    flush()

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print("Wrote %d predictions to %s" % (len(out_rows), args.output_csv))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="model_best.pt from a training run")
    p.add_argument("md_results", help="MegaDetector results .json")
    p.add_argument("--image-root", required=True, help="folder the MD filenames are relative to")
    p.add_argument("--output-csv", required=True)
    p.add_argument("--conf-threshold", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--topk", type=int, default=1)
    p.add_argument("--device", default="auto", help="'auto', 'cpu', or 'cuda'")
    return p.parse_args(argv)


if __name__ == "__main__":
    main()
