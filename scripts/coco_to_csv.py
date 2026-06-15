"""
coco_to_csv.py

Convert a COCO Camera Traps .json file into the flat CSV format used by the
SpeciesNet fine-tuning tutorial.

The tutorial does not ask you to reorganize your images on disk. Instead, it
describes your dataset with a single CSV file with three columns:

    filename   the path to one image
    category   the species (or other class label, e.g. "blank") for that image
    location   the camera / deployment the image came from

This CSV is meant to be a literal representation of your labels. Decisions about
renaming, merging, or dropping classes are made later, when you prepare for
training (see scripts/coco_to_mapping_file.py for a starting point).

The "location" column matters: when we later split the data into training and
validation sets, we keep all images from a given camera on the same side of the
split, so the model is evaluated on cameras it never saw during training.

About "filename": this script writes whatever path is in the COCO "file_name"
field, verbatim, which is normally relative to your image folder. Later
fine-tuning steps let you point an --image-root at that folder. If you would
rather have self-contained absolute paths, pass --absolute-paths (which requires
--image-folder). Downstream tutorial code resolves each filename as: relative to
the CSV's own location, or relative to an image root you specify, or absolute.

One image can appear in more than one row (see --multiple-label-handling).

Run with --help for all options.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict


# Class name assigned to images that have no annotations when
# --unlabeled-image-handling is "include".
UNLABELED_CATEGORY = "unlabeled"


def eprint(*args, **kwargs):
    """Print to stderr (used for warnings and the final summary)."""
    print(*args, file=sys.stderr, **kwargs)


def fail(message):
    """Print an error and exit non-zero, without writing any output."""
    eprint("ERROR: " + message)
    sys.exit(1)


def build_indices(coco):
    """Return (cat_id_to_name, image_records, image_id_to_cat_ids).

    image_records is a list of dicts with keys: id, file_name, location.
    image_id_to_cat_ids maps image id -> list of category ids (may repeat).
    """
    cat_id_to_name = {}
    for c in coco.get("categories", []):
        cat_id_to_name[c["id"]] = c["name"]

    image_records = []
    missing_location = []
    for im in coco.get("images", []):
        loc = im.get("location")
        if loc is None or (isinstance(loc, str) and loc.strip() == ""):
            missing_location.append(im.get("id", im.get("file_name", "<unknown>")))
        image_records.append(
            {
                "id": im["id"],
                "file_name": im["file_name"],
                "location": loc,
            }
        )

    if missing_location:
        examples = ", ".join(str(x) for x in missing_location[:5])
        fail(
            "{} image(s) have no 'location' field, which this converter requires "
            "(e.g. {}). Every image must record which camera/deployment it came "
            "from.".format(len(missing_location), examples)
        )

    image_id_to_cat_ids = defaultdict(list)
    for ann in coco.get("annotations", []):
        image_id_to_cat_ids[ann["image_id"]].append(ann["category_id"])

    return cat_id_to_name, image_records, image_id_to_cat_ids


def convert(args):
    with open(args.input_json, encoding="utf-8") as f:
        coco = json.load(f)

    cat_id_to_name, image_records, image_id_to_cat_ids = build_indices(coco)

    if args.absolute_paths and not args.image_folder:
        fail("--absolute-paths requires --image-folder")
    if args.check_images and not args.image_folder and not args.absolute_paths:
        fail("--check-images requires --image-folder (to locate the images on disk)")

    # Accumulate output rows; only write after all validation passes.
    rows = []  # each row is (filename_value, category, location)
    per_class = Counter()
    locations_seen = set()

    n_total = len(image_records)
    n_unlabeled_omitted = 0
    n_unlabeled_included = 0
    n_multilabel_omitted = 0
    n_multilabel_expanded = 0  # images that produced >1 row under "all"
    unlabeled_error_examples = []

    for rec in image_records:
        img_id = rec["id"]
        file_name = rec["file_name"]
        location = rec["location"]

        orig_cat_ids = image_id_to_cat_ids.get(img_id, [])
        had_annotations = len(orig_cat_ids) > 0

        names = []
        for cid in orig_cat_ids:
            name = cat_id_to_name.get(cid)
            if name is None:
                fail(
                    "image '{}' references unknown category id {}".format(img_id, cid)
                )
            names.append(name)

        distinct = sorted(set(names))

        if not had_annotations:
            # No labels at all in the source data.
            if args.unlabeled_image_handling == "omit":
                n_unlabeled_omitted += 1
                continue
            elif args.unlabeled_image_handling == "error":
                unlabeled_error_examples.append(img_id)
                continue
            elif args.unlabeled_image_handling == "include":
                distinct = [UNLABELED_CATEGORY]
                n_unlabeled_included += 1

        # Determine the filename value to write.
        if args.absolute_paths:
            filename_value = os.path.abspath(os.path.join(args.image_folder, file_name))
        else:
            filename_value = file_name

        if len(distinct) > 1:
            if args.multiple_label_handling == "omit":
                n_multilabel_omitted += 1
                continue
            # "all": one row per distinct label.
            n_multilabel_expanded += 1
            for cat in distinct:
                rows.append((filename_value, cat, location))
                per_class[cat] += 1
        else:
            cat = distinct[0]
            rows.append((filename_value, cat, location))
            per_class[cat] += 1

        locations_seen.add(location)

    # Honor --unlabeled-image-handling error.
    if args.unlabeled_image_handling == "error" and unlabeled_error_examples:
        examples = ", ".join(str(x) for x in unlabeled_error_examples[:5])
        fail(
            "{} image(s) have no annotations and --unlabeled-image-handling is "
            "'error' (e.g. {}).".format(len(unlabeled_error_examples), examples)
        )

    # Optionally verify every referenced image exists on disk.
    n_checked_missing = 0
    if args.check_images:
        missing = []
        for filename_value, _, _ in rows:
            if args.absolute_paths:
                src = filename_value
            else:
                src = os.path.join(args.image_folder, filename_value)
            if not os.path.isfile(src):
                missing.append(src)
        n_checked_missing = len(missing)
        if missing:
            for m in missing[:10]:
                eprint("MISSING: " + m)
            if len(missing) > 10:
                eprint("... and {} more".format(len(missing) - 10))
            fail(
                "{} referenced image(s) were not found on disk; CSV not "
                "written.".format(len(missing))
            )

    # Write the CSV.
    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "category", "location"])
        writer.writerows(rows)

    print_summary(
        args=args,
        n_total=n_total,
        n_rows=len(rows),
        per_class=per_class,
        n_locations=len(locations_seen),
        n_unlabeled_omitted=n_unlabeled_omitted,
        n_unlabeled_included=n_unlabeled_included,
        n_multilabel_omitted=n_multilabel_omitted,
        n_multilabel_expanded=n_multilabel_expanded,
        n_checked_missing=n_checked_missing,
    )


def print_summary(
    args,
    n_total,
    n_rows,
    per_class,
    n_locations,
    n_unlabeled_omitted,
    n_unlabeled_included,
    n_multilabel_omitted,
    n_multilabel_expanded,
    n_checked_missing,
):
    eprint("")
    eprint("=" * 64)
    eprint("Conversion summary")
    eprint("=" * 64)
    eprint("Input JSON              : {}".format(args.input_json))
    eprint("Output CSV              : {}".format(args.output_csv))
    eprint("Source images           : {}".format(n_total))
    eprint("Rows written            : {}".format(n_rows))
    eprint("Distinct classes        : {}".format(len(per_class)))
    eprint("Distinct locations      : {}".format(n_locations))
    eprint("")
    eprint("Multi-label images expanded to rows ('all') : {}".format(n_multilabel_expanded))
    eprint("Multi-label images omitted                  : {}".format(n_multilabel_omitted))
    eprint("Unlabeled images omitted                    : {}".format(n_unlabeled_omitted))
    eprint("Unlabeled images included as '{}'  : {}".format(
        UNLABELED_CATEGORY, n_unlabeled_included))
    if args.check_images:
        eprint("Referenced images missing on disk           : {}".format(n_checked_missing))
    eprint("")
    eprint("Rows per class (descending):")
    for name, count in per_class.most_common():
        eprint("  {:>7d}  {}".format(count, name))
    eprint("=" * 64)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input_json", help="path to a COCO Camera Traps .json file")
    p.add_argument("output_csv", help="path to the CSV file to create")
    p.add_argument(
        "--image-folder",
        help="base folder the images live in; required for --check-images or "
        "--absolute-paths",
    )
    p.add_argument(
        "--multiple-label-handling",
        choices=["omit", "all"],
        default="omit",
        help="what to do with images that have more than one distinct label: "
        "'omit' (default) drops them, 'all' writes one row per label",
    )
    p.add_argument(
        "--unlabeled-image-handling",
        choices=["omit", "error", "include"],
        default="omit",
        help="what to do with images that have no annotations at all: 'omit' "
        "(default) drops them, 'error' aborts, 'include' writes them with the "
        "category '{}'".format(UNLABELED_CATEGORY),
    )
    p.add_argument(
        "--absolute-paths",
        action="store_true",
        help="write absolute image paths instead of the verbatim COCO file_name "
        "(requires --image-folder)",
    )
    p.add_argument(
        "--check-images",
        action="store_true",
        help="verify every referenced image exists on disk before writing the CSV "
        "(requires --image-folder)",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    convert(args)


if __name__ == "__main__":
    main()
