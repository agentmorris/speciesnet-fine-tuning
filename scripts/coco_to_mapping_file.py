#%% Header

"""
coco_to_mapping_file.py

Create a starting-point "mapping file" from a COCO Camera Traps .json file.

A mapping file is a CSV with three columns:

    input    a category name from your data
    output   left blank here, for you to fill in later
    count    the number of images that contain that category

There is one row per category, plus a row for the special category "unlabeled"
(images with no annotations), which is always present.

This script does not make any decisions for you; it just lays out every class in
your data, sorted from most to least common, so you can decide how to rename,
merge, or drop classes before training. You fill in the "output" column later
and feed the completed mapping file to the training-preparation steps. (For
example, you might map "lion", "lion_male", and "lion_female" all to "lion", or
map a vague "animal" class to "remove".)

The "count" column is there to inform those decisions: very rare classes are
often worth merging into a coarser class or dropping entirely.

Run with --help for usage.
"""

#%% Imports and constants

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

# The category name used for images that have no annotations. A row for this
# category is always present in the output.
UNLABELED_CATEGORY = "unlabeled"


#%% Support functions

def eprint(*args, **kwargs):
    """
    Print to stderr (used for warnings and the final summary).
    """

    print(*args, file=sys.stderr, **kwargs)


def fail(message):
    """
    Print an error and exit non-zero, without writing any output.
    """

    eprint("ERROR: " + message)
    sys.exit(1)


def build_mapping_rows(coco):
    """
    Return a list of (input, output, count) rows, sorted by count descending.

    There is one row per distinct category name, plus an UNLABELED_CATEGORY row
    for images with no annotations (always present).

    Args:
        coco (dict): a parsed COCO Camera Traps object, with "images",
            "annotations", and "categories" lists

    Returns:
        list: a list of (input, output, count) tuples, one per output row, sorted
        by count descending (ties broken alphabetically). input (str) is a category
        name; output (str) is always the empty string (the blank column for you to
        fill in); count (int) is the number of distinct images containing that
        category. There is one tuple per distinct category, plus an
        UNLABELED_CATEGORY tuple.
    """

    cat_id_to_name = {}
    for c in coco.get("categories", []):
        cat_id_to_name[c["id"]] = c["name"]

    # Number of distinct images that contain each category
    images_per_category = defaultdict(set)
    annotated_image_ids = set()
    for ann in coco.get("annotations", []):
        cid = ann["category_id"]
        name = cat_id_to_name.get(cid)
        if name is None:
            fail("annotation '{}' references unknown category id {}".format(
                ann.get("id", "<unknown>"), cid))
        images_per_category[name].add(ann["image_id"])
        annotated_image_ids.add(ann["image_id"])

    # Count is the number of distinct images containing the category. We start
    # from the full category list (so categories with zero images still appear).
    counts = {name: 0 for name in cat_id_to_name.values()}
    for name, image_ids in images_per_category.items():
        counts[name] = len(image_ids)

    # Images with no annotations at all become the "unlabeled" category
    all_image_ids = {im["id"] for im in coco.get("images", [])}
    n_unlabeled = len(all_image_ids - annotated_image_ids)

    if UNLABELED_CATEGORY in counts:
        # Very unlikely: a real category is already named "unlabeled". Don't
        # invent a second row with the same name; warn instead.
        eprint(
            "WARNING: a category named '{}' already exists in the data, so its "
            "count reflects annotated images. {} image(s) with no annotations "
            "are NOT separately represented.".format(UNLABELED_CATEGORY, n_unlabeled)
        )
    else:
        counts[UNLABELED_CATEGORY] = n_unlabeled

    # Sort by count descending, breaking ties alphabetically
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(name, "", count) for name, count in rows]

# ...def build_mapping_rows(...)


def write_mapping_file(rows, output_csv):
    out_dir = os.path.dirname(os.path.abspath(output_csv))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "output", "count"])
        writer.writerows(rows)


#%% Command-line driver

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input_json", help="path to a COCO Camera Traps .json file")
    p.add_argument("output_csv", help="path to the mapping CSV to create")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(args.input_json, encoding="utf-8") as f:
        coco = json.load(f)

    rows = build_mapping_rows(coco)
    write_mapping_file(rows, args.output_csv)

    eprint("")
    eprint("Wrote mapping template: {}".format(args.output_csv))
    eprint("  {} category row(s) (including '{}').".format(len(rows), UNLABELED_CATEGORY))
    eprint("  Fill in the 'output' column, then use it in the training-prep steps.")

if __name__ == "__main__":
    main()
