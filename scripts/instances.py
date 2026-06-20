#%% Header

"""
instances.py

Turn the data CSV plus a MegaDetector results file into the list of training
"instances" (crops) the classifier actually learns from.

Each instance is one MegaDetector animal box, cropped from its image, labeled
with that image's category. Because an image can contain several animals, one
image can produce several instances; conversely an image with no animal boxes
above threshold produces none (so, for example, most "blank" images contribute
nothing, which is expected for a crop-based classifier).

The pipeline, in order:
  1. apply the optional category mapping (rename / merge / drop);
  2. keep only MegaDetector detections in the "animal" category, with
     confidence >= threshold, taking at most max_boxes of them per image
     (highest confidence first);
  3. drop categories with fewer than min_instances instances (after mapping).

Validation is lenient: images missing from disk, labeled images absent from the
MD file, and MD/disk mismatches become warnings (recorded in the report). The
only hard error is if no instances survive at all.
"""

#%% Imports and constants

import csv
import json
import os
from collections import Counter
from dataclasses import dataclass

from mapping import load_mapping, apply_mapping, mapping_warnings

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


#%% Data classes

@dataclass
class Instance:
    filename: str          # path as written in the CSV (relative to image folder, or absolute)
    bbox: tuple            # MegaDetector bbox [x, y, w, h], normalized to [0, 1]
    conf: float            # detection confidence
    category: str          # mapped class label
    location: str          # camera / deployment


#%% Support functions

def load_csv_rows(csv_path):
    """
    Read the data CSV (columns filename, category, location).
    """

    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = {(n or "").strip().lower(): n for n in (reader.fieldnames or [])}
        for req in ("filename", "category", "location"):
            if req not in fields:
                raise ValueError(
                    "data CSV '%s' must have columns filename, category, location "
                    "(found: %s)" % (csv_path, reader.fieldnames))
        for row in reader:
            rows.append({
                "filename": (row.get(fields["filename"]) or "").strip(),
                "category": (row.get(fields["category"]) or "").strip(),
                "location": (row.get(fields["location"]) or "").strip(),
            })
    return rows


def load_md_animal_boxes(md_path, conf_threshold, max_boxes):
    """
    Read a MegaDetector results file and select animal boxes per image.

    Returns (selected, md_files, stats) where selected maps file -> list of
    (bbox, conf) for the kept animal boxes. Per-detection classifications in the
    MD file (if any) are ignored. Raises ValueError if there is no detection
    category named 'animal'.
    """

    with open(md_path, encoding="utf-8") as f:
        md = json.load(f)
    det_cats = md.get("detection_categories")
    if not det_cats:
        raise ValueError("MD file '%s' has no 'detection_categories'." % md_path)
    animal_ids = [k for k, v in det_cats.items() if v == "animal"]
    if not animal_ids:
        raise ValueError(
            "MD file '%s' has no detection category named 'animal' (found: %s). "
            "This script only trains on animal boxes." % (md_path, det_cats))
    animal_id = animal_ids[0]

    selected = {}
    md_files = set()
    n_animal_dets = 0
    n_selected = 0

    for im in md.get("images", []):

        fname = im["file"]
        md_files.add(fname)
        animal = [d for d in (im.get("detections") or []) if d.get("category") == animal_id]
        n_animal_dets += len(animal)
        kept = [d for d in animal if (d.get("conf") or 0.0) >= conf_threshold]
        kept.sort(key=lambda d: d.get("conf") or 0.0, reverse=True)
        kept = kept[:max_boxes]
        selected[fname] = [(tuple(d["bbox"]), float(d.get("conf") or 0.0)) for d in kept]
        n_selected += len(kept)

    # ...for each image

    stats = {
        "md_n_images": len(md_files),
        "md_n_animal_dets": n_animal_dets,
        "md_n_selected": n_selected,
        "animal_category_id": animal_id,
    }

    return selected, md_files, stats

# ...def load_md_animal_boxes(...)


def _resolve(filename, image_folder):
    return filename if os.path.isabs(filename) else os.path.join(image_folder, filename)


def md_vs_disk(md_files, image_folder, scan_disk=True, example_cap=10):
    """
    Compare the MD file's images against what's on disk (informational).
    """

    md_not_on_disk = [f for f in md_files if not os.path.isfile(_resolve(f, image_folder))]
    disk_not_in_md = []
    n_disk = 0
    if scan_disk and os.path.isdir(image_folder):
        for root, _dirs, files in os.walk(image_folder):
            for fn in files:
                if os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
                    n_disk += 1
                    rel = os.path.relpath(os.path.join(root, fn), image_folder).replace("\\", "/")
                    if rel not in md_files and fn not in md_files:
                        disk_not_in_md.append(rel)
    return {
        "md_not_on_disk_count": len(md_not_on_disk),
        "md_not_on_disk_examples": md_not_on_disk[:example_cap],
        "disk_scanned": scan_disk,
        "disk_image_count": n_disk,
        "disk_not_in_md_count": len(disk_not_in_md),
        "disk_not_in_md_examples": disk_not_in_md[:example_cap],
    }


def prepare_instance_list(csv_path,
                          md_path,
                          image_folder,
                          mapping_path=None,
                          conf_threshold=0.3,
                          max_boxes=5,
                          min_instances=100,
                          scan_disk=True):
    """
    Build the list of training instances (crops) from a data CSV and a
    MegaDetector results file.

    For each CSV row the category is remapped (if a mapping is given), the image's
    animal boxes are selected (confidence >= [conf_threshold], at most [max_boxes]
    per image, highest confidence first), and one Instance is produced per
    selected box. Categories with fewer than [min_instances] instances, counted
    after mapping, are then dropped. Images missing from disk, absent from the MD
    file, or with no surviving box are skipped and counted in the report rather
    than raising; the only hard error is if no instances survive at all.

    Args:
        csv_path (str): path to the data CSV, with columns filename, category, and
            location
        md_path (str): path to the MegaDetector results .json file
        image_folder (str): base folder the CSV filenames are relative to, used to
            check that images exist on disk
        mapping_path (str, optional): path to a category mapping CSV (input,output);
            if None, categories are used as-is
        conf_threshold (float, optional): minimum MegaDetector confidence for an
            animal box to become an instance (default 0.3)
        max_boxes (int, optional): maximum number of animal boxes to keep per
            image, highest confidence first (default 5)
        min_instances (int, optional): drop any category with fewer than this many
            instances, counted after mapping (default 100)
        scan_disk (bool, optional): if True, walk [image_folder] to count images on
            disk that are absent from the MD file, for the report (default True)

    Returns:
        tuple: a 3-tuple (instances, classes, report). instances (list of Instance)
        has one entry per kept animal box, each with filename, bbox, conf, mapped
        category, and location; classes (list of str) is the sorted list of
        surviving category names, which become the model's classes; report (dict)
        summarizes the preparation (row and instance counts, per-category counts,
        dropped categories, non-fatal warnings, and MD-versus-disk statistics) and
        is used to write the run summary
    """

    warnings = []

    mapping = load_mapping(mapping_path) if mapping_path else {}
    rows = load_csv_rows(csv_path)
    selected, md_files, md_stats = load_md_animal_boxes(md_path, conf_threshold, max_boxes)

    if mapping:
        warnings.extend(mapping_warnings(mapping, {r["category"] for r in rows}))

    instances = []
    n_dropped_by_remove = 0
    csv_missing_on_disk = []
    n_csv_not_in_md = 0
    n_csv_no_boxes = 0

    for r in rows:

        fname = r["filename"]
        cat = apply_mapping(r["category"], mapping)
        if cat is None:
            n_dropped_by_remove += 1
            continue
        if not os.path.isfile(_resolve(fname, image_folder)):
            csv_missing_on_disk.append(fname)
            continue
        if fname not in selected:
            n_csv_not_in_md += 1
            continue
        boxes = selected[fname]
        if not boxes:
            n_csv_no_boxes += 1
            continue
        for bbox, conf in boxes:
            instances.append(Instance(fname, bbox, conf, cat, r["location"]))

    # ...for each row

    counts_before = Counter(i.category for i in instances)
    dropped_by_mincount = sorted(
        [(c, n) for c, n in counts_before.items() if n < min_instances],
        key=lambda x: -x[1])
    kept = {c for c, n in counts_before.items() if n >= min_instances}
    instances = [i for i in instances if i.category in kept]
    counts_final = Counter(i.category for i in instances)
    classes = sorted(counts_final)

    report = {
        "params": {"conf_threshold": conf_threshold, "max_boxes": max_boxes,
                   "min_instances": min_instances, "mapping_path": mapping_path},
        "n_csv_rows": len(rows),
        "n_dropped_by_remove": n_dropped_by_remove,
        "csv_missing_on_disk_count": len(csv_missing_on_disk),
        "csv_missing_on_disk_examples": csv_missing_on_disk[:10],
        "n_csv_not_in_md": n_csv_not_in_md,
        "n_csv_no_boxes": n_csv_no_boxes,
        "n_instances_before_mincount": sum(counts_before.values()),
        "counts_before_mincount": dict(counts_before),
        "dropped_by_mincount": dropped_by_mincount,
        "n_instances_final": len(instances),
        "counts_final": dict(counts_final),
        "classes": classes,
        "n_classes": len(classes),
        "warnings": warnings,
    }

    report.update(md_stats)
    report.update(md_vs_disk(md_files, image_folder, scan_disk=scan_disk))

    if not instances:
        raise ValueError(
            "No training instances survived preparation. Check the MD confidence "
            "threshold, --min-instances, the category mapping, and that the CSV "
            "filenames match the MD 'file' fields.")

    return instances, classes, report

# ...def prepare_instance_list(...)
