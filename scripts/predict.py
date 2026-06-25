#%% Header

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
      IMAGES predictions.json [--topk 5]
"""


#%% Imports and environment

import argparse
import csv
import json
import os
import sys

import timm
import torch
from PIL import Image
from tqdm import tqdm

file_dir = os.path.dirname(os.path.abspath(__file__))
if file_dir not in sys.path:
    sys.path.insert(0, file_dir)

from dataset import build_transforms, crop_resize


#%% Support functions

def load_model(checkpoint_path, device):
    """
    Load a fine-tuned model from a self-describing inference checkpoint.

    Args:
        checkpoint_path (str): path to a model_best.pt written by train.py, with
            keys "timm_model", "num_classes", "state_dict", "classes", and the
            preprocessing fields
        device (str): the torch device to load the model onto, e.g. "cpu" or "cuda"

    Returns:
        tuple: a 2-tuple (model, ckpt). model is the timm model in eval mode on
        [device]; ckpt (dict) is the full loaded checkpoint, including "classes",
        "img_size", "norm_mean", and "norm_std"
    """

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = timm.create_model(ckpt["timm_model"], pretrained=False,
                              num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, ckpt


#%% Core inference function

def predict(checkpoint,
            md_results,
            output_file,
            image_root,
            csv_output=False,
            conf_threshold=0.1,
            batch_size=32,
            topk=1,
            device='auto'):
    """
    Classify the animal detections in a MegaDetector results file with a
    fine-tuned model, and write the predictions.

    By default this writes a MegaDetector-format results file (a copy of
    [md_results] with our classifications added to each animal detection at or
    above [conf_threshold]; all original detections are preserved). If
    [csv_output] is True, it writes a flat CSV with one row per classified box
    instead.

    Args:
        checkpoint (str): path to a fine-tuned model_best.pt (see load_model())
        md_results (str): path to a MegaDetector results .json for the images to
            classify
        output_file (str): path to write results to; a MegaDetector-format .json, or a
            CSV if [csv_output] is set
        image_root (str): base folder the MegaDetector "file" paths are relative to
        csv_output (bool, optional): if True, write a flat per-box CSV instead of
            MegaDetector-format output (default False)
        conf_threshold (float, optional): only classify animal boxes at or above
            this MegaDetector confidence (default 0.1)
        batch_size (int, optional): number of crops to classify per batch
            (default 32)
        topk (int, optional): how many top predictions, with scores, to record per
            box (default 1)
        device (str, optional): torch device to run on: "auto" (default), "cpu",
            or "cuda"
    """

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, ckpt = load_model(checkpoint, device)
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 480)
    mean = tuple(ckpt.get("norm_mean", (0.0, 0.0, 0.0)))
    std = tuple(ckpt.get("norm_std", (1.0, 1.0, 1.0)))
    transform = build_transforms(img_size, train=False, mean=mean, std=std)
    topk = max(1, min(topk, len(classes)))

    with open(md_results, encoding="utf-8") as f:
        md = json.load(f)
    det_cats = md.get("detection_categories") or {}
    animal_ids = [k for k, v in det_cats.items() if v == "animal"]
    if not animal_ids:
        raise SystemExit("ERROR: MD file has no detection category named 'animal'.")
    animal_id = animal_ids[0]

    # Collect the detections we will classify (above-threshold animal boxes),
    # keeping a reference to each detection dict so we can write results back.
    detections_to_classify = []  # (filename, bbox, detection_dict)
    for im in md.get("images", []):
        for det in (im.get("detections") or []):
            if det.get("category") == animal_id and (det.get("conf") or 0.0) >= conf_threshold:
                detections_to_classify.append((im["file"], tuple(det["bbox"]), det))

    csv_rows = []
    current_batch = []  # (filename, bbox, det, tensor)
    cache_name, cache_img = None, None

    # Run the model on the current batch, record each box's top-k predictions
    # (in the MD detections or in csv_rows, depending on the output format), and
    # clear the buffer.
    def _process_current_batch():

        if not current_batch:
            return

        # Pulls just the crop tensor out of each tuple, ignoring the other three elements
        # (the _ are throwaway names for filename, bbox, and det). The result is a plain
        # Python list of N tensors, each [3, H, W] (typically [3, 480, 480]), where N is
        # the size of the current batch (typically the batch size, but might be smaller if
        # this is the last batch).
        crop_tensors = [t for _, _, _, t in current_batch]

        # Stack the tensors for each crop in the current batch into a single tensor with a new leading
        # dimension, i.e., turn N separate [3, H, W] tensors into one [N, 3, H, W] tensor. That new
        # first dimension is the batch dimension.  This is exactly the shape the model expects (batch,
        # channels, height, width).
        x = torch.stack(crop_tensors).to(device)

        # Run the model on the current batch
        with torch.no_grad():
            # [probs] will have size [n_crops, num_classes]
            probs = torch.softmax(model(x), dim=1)

        scores, idx = probs.topk(topk, dim=1)

        # Loop over the results for each crop in the batch and assemble them into the
        # right output format.
        for (fname, bbox, det, _), sc, ix in zip(current_batch, scores.tolist(), idx.tolist()):

            if csv_output:
                row = {"filename": fname, "x": bbox[0], "y": bbox[1], "w": bbox[2],
                       "h": bbox[3], "detection_conf": det.get("conf")}
                for r in range(topk):
                    row["pred%d_class" % (r + 1)] = classes[ix[r]]
                    row["pred%d_score" % (r + 1)] = sc[r]
                csv_rows.append(row)
            else:
                det["classifications"] = [[str(ix[r]), sc[r]] for r in range(topk)]

        # ...for each crop in the current batch

        current_batch.clear()

    # ...def _process_current_batch()

    for fname, bbox, det in tqdm(detections_to_classify):

        # We are often processing multiple detections from the same image, don't re-load
        # the image for every detection.
        if fname != cache_name:
            path = fname if os.path.isabs(fname) else os.path.join(image_root, fname)
            try:
                cache_img = Image.open(path).convert("RGB")
            except Exception:
                cache_img = None
            cache_name = fname
        if cache_img is None:
            continue
        transformed_crop = transform(crop_resize(cache_img, bbox, img_size))
        current_batch.append((fname, bbox, det, transformed_crop))

        # Is it time to process a batch?
        if len(current_batch) >= batch_size:
            _process_current_batch()

    # ...for each detection that needs classifying

    _process_current_batch()

    if csv_output:
        fieldnames = ["filename", "x", "y", "w", "h", "detection_conf"]
        for r in range(topk):
            fieldnames += ["pred%d_class" % (r + 1), "pred%d_score" % (r + 1)]
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print("Wrote %d classified boxes (CSV) to %s" % (len(csv_rows), output_file))
    else:
        # MegaDetector-format output: add our classification label map and keep
        # every original detection.
        md["classification_categories"] = {str(i): c for i, c in enumerate(classes)}
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(md, f, indent=1)
        n_classified = sum(1 for _, _, det in detections_to_classify if "classifications" in det)
        print("Wrote MegaDetector-format results to %s (classified %d animal boxes; "
              "all original detections preserved)" % (output_file, n_classified))

# ...def predict(...)


#%% Command-line driver

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="model_best.pt from a training run")
    p.add_argument("md_results", help="MegaDetector results .json")
    p.add_argument("image_root", help="folder the MegaDetector filenames are relative to")
    p.add_argument("output_file", help="output path (MD-format .json by default; CSV with --csv-output)")
    p.add_argument("--csv-output", action="store_true",
                   help="write a flat CSV (one row per classified box) instead of MD format")
    p.add_argument("--conf-threshold", type=float, default=0.1,
                   help="classify animal boxes at or above this MD confidence (default 0.1)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--topk", type=int, default=1)
    p.add_argument("--device", default="auto", help="'auto', 'cpu', or 'cuda'")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    predict(checkpoint=args.checkpoint,
            md_results=args.md_results,
            output_file=args.output_file,
            image_root=args.image_root,
            csv_output=args.csv_output,
            conf_threshold=args.conf_threshold,
            batch_size=args.batch_size,
            topk=args.topk,
            device=args.device)


if __name__ == "__main__":
    main()
