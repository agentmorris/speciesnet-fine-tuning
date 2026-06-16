"""
predict.py

Run a fine-tuned SpeciesNet classifier on new images. You run MegaDetector
separately; this script reads the MegaDetector results file, classifies each
animal box above a confidence threshold (using exactly the same crop, resize,
and normalization as training), and writes the predictions.

By default it writes a MegaDetector-format results file: a copy of the input
with our classifications added to each above-threshold animal detection. Every
original detection is preserved (below-threshold and non-animal detections are
left untouched), so the output drops straight into MegaDetector's postprocessing
tools (analyze_classification_results.py, postprocess_batch_results.py, etc.).

Pass --csv-output to instead write a flat CSV with one row per classified box.

Usage:
  python scripts/predict.py RUN_FOLDER/model_best.pt md_results.json \\
      --image-root IMAGES --output predictions.json [--topk 5]
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


def main(argv=None):
    args = parse_args(argv)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, ckpt = load_model(args.checkpoint, device)
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 480)
    mean = tuple(ckpt.get("norm_mean", (0.0, 0.0, 0.0)))
    std = tuple(ckpt.get("norm_std", (1.0, 1.0, 1.0)))
    transform = build_transforms(img_size, train=False, mean=mean, std=std)
    topk = max(1, min(args.topk, len(classes)))

    with open(args.md_results, encoding="utf-8") as f:
        md = json.load(f)
    det_cats = md.get("detection_categories") or {}
    animal_ids = [k for k, v in det_cats.items() if v == "animal"]
    if not animal_ids:
        raise SystemExit("ERROR: MD file has no detection category named 'animal'.")
    animal_id = animal_ids[0]

    # Collect the detections we will classify (above-threshold animal boxes),
    # keeping a reference to each detection dict so we can write results back.
    work = []  # (filename, bbox, detection_dict)
    for im in md.get("images", []):
        for det in (im.get("detections") or []):
            if det.get("category") == animal_id and (det.get("conf") or 0.0) >= args.conf_threshold:
                work.append((im["file"], tuple(det["bbox"]), det))

    csv_rows = []
    buf = []  # (filename, bbox, det, tensor)
    cache_name, cache_img = None, None

    def flush():
        if not buf:
            return
        x = torch.stack([t for _, _, _, t in buf]).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)
        scores, idx = probs.topk(topk, dim=1)
        for (fname, bbox, det, _), sc, ix in zip(buf, scores.tolist(), idx.tolist()):
            if args.csv_output:
                row = {"filename": fname, "x": bbox[0], "y": bbox[1], "w": bbox[2],
                       "h": bbox[3], "detection_conf": det.get("conf")}
                for r in range(topk):
                    row["pred%d_class" % (r + 1)] = classes[ix[r]]
                    row["pred%d_score" % (r + 1)] = sc[r]
                csv_rows.append(row)
            else:
                det["classifications"] = [[str(ix[r]), sc[r]] for r in range(topk)]
        buf.clear()

    for fname, bbox, det in work:
        if fname != cache_name:
            path = fname if os.path.isabs(fname) else os.path.join(args.image_root, fname)
            try:
                cache_img = Image.open(path).convert("RGB")
            except Exception:
                cache_img = None
            cache_name = fname
        if cache_img is None:
            continue
        buf.append((fname, bbox, det, transform(crop_resize(cache_img, bbox, img_size))))
        if len(buf) >= args.batch_size:
            flush()
    flush()

    if args.csv_output:
        fieldnames = ["filename", "x", "y", "w", "h", "detection_conf"]
        for r in range(topk):
            fieldnames += ["pred%d_class" % (r + 1), "pred%d_score" % (r + 1)]
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print("Wrote %d classified boxes (CSV) to %s" % (len(csv_rows), args.output))
    else:
        # MegaDetector-format output: add our classification label map and keep
        # every original detection.
        md["classification_categories"] = {str(i): c for i, c in enumerate(classes)}
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(md, f, indent=1)
        n_classified = sum(1 for _, _, det in work if "classifications" in det)
        print("Wrote MegaDetector-format results to %s (classified %d animal boxes; "
              "all original detections preserved)" % (args.output, n_classified))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="model_best.pt from a training run")
    p.add_argument("md_results", help="MegaDetector results .json")
    p.add_argument("--image-root", required=True, help="folder the MD filenames are relative to")
    p.add_argument("--output", required=True, help="output path (MD-format .json by default)")
    p.add_argument("--csv-output", action="store_true",
                   help="write a flat CSV (one row per classified box) instead of MD format")
    p.add_argument("--conf-threshold", type=float, default=0.1,
                   help="classify animal boxes at or above this MD confidence (default 0.1)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--topk", type=int, default=1)
    p.add_argument("--device", default="auto", help="'auto', 'cpu', or 'cuda'")
    return p.parse_args(argv)


if __name__ == "__main__":
    main()
