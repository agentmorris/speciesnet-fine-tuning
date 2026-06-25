#%% Header

"""
create_split_results_file.py

Given a MegaDetector results file and a split source, write a new results file
containing only the images that belong to a chosen split (typically "val"),
preserving all detections of those images.

The split source may be either:

* a per-image split .json (file name -> "train"/"val"/"excluded", as written by
  train.py), in which case images are selected by file name, or
* a location splits.csv (columns location,split). MegaDetector results files have
  no location field, so when splitting by location you must also pass a COCO
  Camera Traps file (--coco-file) to map file names to locations.

This is most often used to make a validation-only MegaDetector results file that
can be handed to predict.py to replicate the validation split during inference.

Usage:
  python scripts/create_split_results_file.py md.json image_splits.json val_md.json --split val
  python scripts/create_split_results_file.py md.json split.csv         val_md.json --split val --coco-file labels.json
"""


#%% Imports

import argparse
import json
import os
import sys

file_dir = os.path.dirname(os.path.abspath(__file__))
if file_dir not in sys.path:
    sys.path.insert(0, file_dir)

from create_split_coco_file import eprint, fail, load_split_locations, load_image_splits


#%% Support functions

def _load_locations_from_coco(coco_file):
    """
    Return a dict mapping image file name -> location (as a string), read from a
    COCO Camera Traps file. Used to split a MegaDetector results file (which has
    no location field) by location.
    """

    with open(coco_file, encoding="utf-8") as f:
        coco = json.load(f)
    file_to_location = {}
    for im in coco.get("images", []):
        loc = im.get("location")
        if loc is None or (isinstance(loc, str) and not loc.strip()):
            continue
        file_to_location[im["file_name"]] = str(loc).strip()
    return file_to_location


#%% Main function

def create_split_results_file(input_file,
                              output_file,
                              split_source,
                              split='val',
                              coco_file=None):
    """
    Given a MegaDetector results file and a split source, write a new results file
    containing only the images that belong to a chosen split (typically "val").
    All detections of the kept images are preserved.

    [split_source] may be either a per-image split .json (file name -> split, as
    written by train.py), in which case images are selected by file name, or a
    location splits.csv (columns location,split). MegaDetector results files have
    no location field, so when splitting by location, [coco_file] (a COCO Camera
    Traps file with a "location" on every image) is also required, to map file
    names to locations.

    Args:
        input_file (str): path to the MegaDetector results .json file to filter
        split_source (str): path to the split source. A name ending in ".json" is
            read as a per-image split file (file name -> split) and images are
            selected by file name; anything else is read as a location splits.csv
            (columns location,split) and images are selected by location, which
            also requires [coco_file]
        output_file (str): path of the MegaDetector results .json to write;
            parent directories are created if needed
        split (str, optional): split name to extract, e.g. "train" or "val" (and,
            for a per-image split file, "excluded") (default "val")
        coco_file (str, optional): COCO Camera Traps .json with a "location" on
            every image, used to map file names to locations; required only when
            [split_source] is a location splits.csv (default None)
    """

    use_image_splits = str(split_source).lower().endswith(".json")

    with open(input_file, encoding="utf-8") as f:
        results = json.load(f)
    images = results.get("images", [])

    if use_image_splits:
        target_files = load_image_splits(split_source, split)
        kept_images = [im for im in images if im.get("file") in target_files]
        source_desc = "%d file names in split '%s'" % (len(target_files), split)
    else:
        if not coco_file:
            fail("splitting a results file by a location splits.csv requires --coco-file "
                 "(MegaDetector results files have no 'location' field).")
        in_split, _all_locations = load_split_locations(split_source, split)
        file_to_location = _load_locations_from_coco(coco_file)
        kept_images = [im for im in images if file_to_location.get(im.get("file")) in in_split]
        source_desc = "%d locations in split '%s'" % (len(in_split), split)

    out = {k: v for k, v in results.items() if k != "images"}
    out["images"] = kept_images

    out_dir = os.path.dirname(os.path.abspath(output_file))
    if out_dir and (not os.path.isdir(out_dir)):
        os.makedirs(out_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    eprint("Split source '%s' (%s)." % (split_source, source_desc))
    eprint("Kept %d of %d images -> %s" % (len(kept_images), len(images), output_file))


#%% Command-line driver

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_file", help="MegaDetector results .json")
    p.add_argument("split_source", help="a location splits.csv (columns location,split) or a "
                   "per-image split .json (filename -> split) from a training run")
    p.add_argument("output_file", help="output MegaDetector results .json for the chosen split")
    p.add_argument("--split", default="val", help="split name to extract (default: val)")
    p.add_argument("--coco-file", default=None,
                   help="COCO Camera Traps .json providing locations; required only when "
                   "split_source is a location splits.csv")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    create_split_results_file(input_file=args.input_file,
                              output_file=args.output_file,
                              split_source=args.split_source,
                              split=args.split,
                              coco_file=args.coco_file)

if __name__ == "__main__":
    main()
