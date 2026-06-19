"""
create_split_coco_file.py

Given a COCO Camera Traps file and a splits.csv (location,split) produced during
training, write a new COCO file containing only the images, and their
annotations, whose location belongs to a chosen split (typically "val").

This is most often used to make a validation-only ground-truth file to hand to an
evaluation tool such as MegaDetector's analyze_classification_results.py,
alongside the predictions written by predict.py.

Usage:
  python scripts/create_split_coco_file.py labels.json split.csv val_gt.json --split val
"""

import argparse
import csv
import json
import os
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def fail(message):
    eprint("ERROR: " + message)
    sys.exit(1)


def load_split_locations(splits_csv, split_name):
    """Return (locations_in_split, all_locations) from a splits.csv (location,split)."""
    in_split = set()
    all_locations = set()
    all_split_names = set()
    with open(splits_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = {(n or "").strip().lower(): n for n in (reader.fieldnames or [])}
        for req in ("location", "split"):
            if req not in fields:
                fail("splits file '%s' must have columns 'location' and 'split' "
                     "(found: %s)" % (splits_csv, reader.fieldnames))
        for row in reader:
            loc = (row.get(fields["location"]) or "").strip()
            sp = (row.get(fields["split"]) or "").strip()
            all_locations.add(loc)
            all_split_names.add(sp)
            if sp == split_name:
                in_split.add(loc)
    if not in_split:
        fail("no locations are assigned to split '%s' in %s (available splits: %s)"
             % (split_name, splits_csv, ", ".join(sorted(all_split_names)) or "none"))
    return in_split, all_locations


def main(argv=None):
    args = parse_args(argv)
    in_split, all_split_locations = load_split_locations(args.splits_csv, args.split)

    with open(args.input_coco, encoding="utf-8") as f:
        coco = json.load(f)
    images = coco.get("images", [])

    missing_location = [im.get("id", im.get("file_name", "?")) for im in images
                        if im.get("location") is None
                        or (isinstance(im.get("location"), str) and not im["location"].strip())]
    if missing_location:
        examples = ", ".join(str(x) for x in missing_location[:5])
        fail("%d image(s) in '%s' have no 'location' field (e.g. %s); this script "
             "requires a location on every image." % (len(missing_location), args.input_coco, examples))

    def loc_key(im):
        return str(im["location"]).strip()

    kept_images = [im for im in images if loc_key(im) in in_split]
    kept_ids = {im["id"] for im in kept_images}
    kept_anns = [a for a in coco.get("annotations", []) if a.get("image_id") in kept_ids]

    # Informational: images whose location is in no split at all (possible mismatch).
    not_in_any_split = sum(1 for im in images if loc_key(im) not in all_split_locations)

    out = {k: v for k, v in coco.items() if k not in ("images", "annotations")}
    out["images"] = kept_images
    out["annotations"] = kept_anns

    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    eprint("Split '%s': %d locations." % (args.split, len(in_split)))
    eprint("Kept %d of %d images and %d of %d annotations -> %s"
           % (len(kept_images), len(images), len(kept_anns),
              len(coco.get("annotations", [])), args.output_json))
    if not_in_any_split:
        eprint("Note: %d image(s) have a location that is not in the splits file at all "
               "(neither train nor val); they were excluded." % not_in_any_split)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_coco", help="COCO Camera Traps .json (every image needs a 'location')")
    p.add_argument("splits_csv", help="splits.csv (columns: location, split) from a training run")
    p.add_argument("output_json", help="output COCO .json for the chosen split")
    p.add_argument("--split", default="val", help="split name to extract (default: val)")
    return p.parse_args(argv)


if __name__ == "__main__":
    main()
