#%% Header

"""
create_split_coco_file.py

Given a COCO Camera Traps file and a split source, write a new COCO file
containing only the images, and their annotations, that belong to a chosen split
(typically "val").

The split source may be either:

* a location splits.csv (columns location,split), in which case images are
  selected by their "location" field, or
* a per-image split .json (file name -> "train"/"val"/"excluded", as written by
  train.py), in which case images are selected by file name (and the split may be
  "excluded").

This is most often used to make a validation-only ground-truth file to hand to an
evaluation tool such as MegaDetector's analyze_classification_results.py,
alongside the predictions written by predict.py.

Usage:
  python scripts/create_split_coco_file.py labels.json split.csv          val_gt.json --split val
  python scripts/create_split_coco_file.py labels.json image_splits.json  val_gt.json --split val
"""


#%% Imports

import argparse
import csv
import json
import os
import sys

file_dir = os.path.dirname(os.path.abspath(__file__))
if file_dir not in sys.path:
    sys.path.insert(0, file_dir)

from mapping import load_mapping, apply_mapping, mapping_warnings


#%% Support functions

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def fail(message):
    eprint("ERROR: " + message)
    sys.exit(1)


def load_split_locations(splits_csv, split_name):
    """
    Return (locations_in_split, all_locations) from a splits.csv (location,split).
    """

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


def load_image_splits(split_json, split_name):
    """
    Return the set of file names assigned to [split_name] in a per-image split
    JSON file (a mapping from file name to "train"/"val"/"excluded", as written
    by train.py).
    """

    with open(split_json, encoding="utf-8") as f:
        image_splits = json.load(f)
    files = {fn for fn, sp in image_splits.items() if sp == split_name}
    if not files:
        available = ", ".join(sorted(set(image_splits.values()))) or "none"
        fail("no images are assigned to split '%s' in %s (available splits: %s)"
             % (split_name, split_json, available))
    return files


def apply_category_mapping(coco, mapping_file):
    """
    Remap the category names in a COCO object in place, using a training-style
    mapping CSV.

    Renames, merges, or drops categories per [mapping_file] (see mapping.py),
    regenerates category IDs (merging collapses several source categories into a
    single output category), rewrites each annotation's category_id to match, and
    drops annotations whose category was removed. [coco] is modified in place.

    Args:
        coco (dict): a parsed COCO Camera Traps object; its "categories" and
            "annotations" are replaced
        mapping_file (str): path to a category mapping CSV (columns input,output)
    """

    mapping = load_mapping(mapping_file)
    cats = coco.get("categories", [])

    for w in mapping_warnings(mapping, {c["name"] for c in cats}):
        eprint("WARNING: " + w)

    # Assign a fresh id to each surviving output category, in order of first
    # appearance among the original categories, and record how each old id maps
    # to a new one (None for categories dropped via "remove")
    new_name_to_id = {}
    new_categories = []
    old_id_to_new_id = {}
    for c in cats:
        new_name = apply_mapping(c["name"], mapping)
        if new_name is None:
            old_id_to_new_id[c["id"]] = None
            continue
        if new_name not in new_name_to_id:
            new_id = len(new_categories)
            new_name_to_id[new_name] = new_id
            new_categories.append({"id": new_id, "name": new_name})
        old_id_to_new_id[c["id"]] = new_name_to_id[new_name]

    # Rewrite annotations to the new category ids, dropping any whose category was
    # removed (mapped to None, hence a None new id)
    new_annotations = []
    n_dropped = 0
    for a in coco.get("annotations", []):
        new_id = old_id_to_new_id.get(a.get("category_id"))
        if new_id is None:
            n_dropped += 1
            continue
        a = dict(a)
        a["category_id"] = new_id
        new_annotations.append(a)

    coco["categories"] = new_categories
    coco["annotations"] = new_annotations

    eprint("Applied mapping '%s': %d categories -> %d, dropped %d annotation(s) in "
           "removed categories." % (mapping_file, len(cats), len(new_categories), n_dropped))


#%% Main function

def create_split_coco_file(input_coco,
                           split_source,
                           output_json,
                           split='val',
                           mapping_file=None):
    """
    Given a COCO Camera Traps file and a split source, write a new COCO file
    containing only the images, and their annotations, that belong to a chosen
    split (typically "val").

    [split_source] may be either a location splits.csv (columns location,split),
    in which case images are selected by their "location" field, or a per-image
    split .json (file name -> "train"/"val"/"excluded", as written by train.py),
    in which case images are selected by file name and [split] may be "excluded".

    If [mapping_file] is given, category names are renamed, merged, or dropped
    first (using the same logic as training), so the output's labels match the
    classes the model was trained on. Category IDs are regenerated, since merging
    collapses several source categories into one.

    Args:
        input_coco (str): path to the COCO Camera Traps .json file to filter
        split_source (str): path to the split source. A name ending in ".json" is
            read as a per-image split file (file name -> split) and images are
            selected by file name; anything else is read as a location splits.csv
            (columns location,split) and images are selected by their "location"
            field
        output_json (str): path of the COCO .json to write; parent directories are
            created if needed
        split (str, optional): split name to extract, e.g. "train" or "val" (and,
            for a per-image split file, "excluded") (default "val")
        mapping_file (str, optional): path to a category mapping CSV (columns
            input,output), in the same format as train.py's --mapping; when given,
            category names are remapped and annotations of removed categories are
            dropped before the split is applied (default None, no remapping)
    """

    use_image_splits = str(split_source).lower().endswith(".json")

    with open(input_coco, encoding="utf-8") as f:
        coco = json.load(f)

    if mapping_file:
        apply_category_mapping(coco, mapping_file)

    images = coco.get("images", [])

    not_in_any_split = 0
    if use_image_splits:
        target_files = load_image_splits(split_source, split)
        kept_images = [im for im in images if im.get("file_name") in target_files]
        source_desc = "%d file names in split '%s'" % (len(target_files), split)
    else:
        in_split, all_split_locations = load_split_locations(split_source, split)
        missing_location = [im.get("id", im.get("file_name", "?")) for im in images
                            if im.get("location") is None
                            or (isinstance(im.get("location"), str) and not im["location"].strip())]
        if missing_location:
            examples = ", ".join(str(x) for x in missing_location[:5])
            fail("%d image(s) in '%s' have no 'location' field (e.g. %s); a location is "
                 "required on every image when splitting by location."
                 % (len(missing_location), input_coco, examples))

        def loc_key(im):
            return str(im["location"]).strip()

        kept_images = [im for im in images if loc_key(im) in in_split]
        not_in_any_split = sum(1 for im in images if loc_key(im) not in all_split_locations)
        source_desc = "%d locations in split '%s'" % (len(in_split), split)

    kept_ids = {im["id"] for im in kept_images}
    kept_anns = [a for a in coco.get("annotations", []) if a.get("image_id") in kept_ids]

    out = {k: v for k, v in coco.items() if k not in ("images", "annotations")}
    out["images"] = kept_images
    out["annotations"] = kept_anns

    out_dir = os.path.dirname(os.path.abspath(output_json))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    eprint("Split source '%s' (%s)." % (split_source, source_desc))
    eprint("Kept %d of %d images and %d of %d annotations -> %s"
           % (len(kept_images), len(images), len(kept_anns),
              len(coco.get("annotations", [])), output_json))
    if not_in_any_split:
        eprint("Note: %d image(s) have a location that is not in the splits file at all "
               "(neither train nor val); they were excluded." % not_in_any_split)


#%% Command-line driver

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_coco", help="COCO Camera Traps .json")
    p.add_argument("split_source", help="a location splits.csv (columns location,split) or a "
                   "per-image split .json (filename -> split) from a training run")
    p.add_argument("output_json", help="output COCO .json for the chosen split")
    p.add_argument("--split", default="val", help="split name to extract (default: val)")
    p.add_argument("--mapping-file", default=None,
                   help="optional category mapping CSV (columns input,output, same format as "
                   "train.py's --mapping); remaps category names before splitting")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    create_split_coco_file(input_coco=args.input_coco,
                           split_source=args.split_source,
                           output_json=args.output_json,
                           split=args.split,
                           mapping_file=args.mapping_file)

if __name__ == "__main__":
    main()
