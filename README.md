# SpeciesNet fine-tuning tutorial

## Overview

### Goals of this tutorial

### Notes on what to expect from species classification in generally

* TODO most of the benefit comes from getting you through your common classes quickly, so prioritize high precision and adequate recall on common classes, rather than getting hung up on rare classes (even if the rare clases are the ones you care about)

### What might I try before fine-tuning?

* TODO See whether there's already a model available (https://agentmorris.github.io/camera-trap-ml-survey/#publicly-available-ml-models-for-camera-traps)
* TODO Try SpeciesNet off the shelf, with a custom taxonomy list instead of the standard geofence

### What are situations where this is probably a bad idea?

* TODO Fine-tuning SpeciesNet still relies on MegaDetector to work, so cases where MD isn't very good - small animal cameras, cameras with a lot of semi-aquatic mammals in the water, etc. - can't be "rescued" by fine-tuning SpeciesNet
* TODO Fine-tuning still generally can't help AI see anything humans can't see in individual images

## Setting up your environment

* TODO instructions on installing miniforge, checking out this git repo, pip installing stuff, conda activate

## Preparing your data

### The format this tutorial expects

This tutorial does not expect you to reorganize your images on disk. Your images can stay wherever they already are (one big folder, or a tree of subfolders — it doesn't matter). Instead, you will describe your dataset with a .csv file with one row per labeled image and the following three columns:

| Column | What it contains |
|---|---|
| `filename` | The path to one image (see "How filenames are resolved" below). |
| `category` | The label for that image... this usually a taxon (`zebra`, `impala`, `rodent`) or the label `blank`, but it can also include any other class you want the model to learn. |
| `location` | The camera (a.k.a. "deployment" or "site") the image came from.  This is not latitude and longitude, just a unique name for each camera. |

A minimal CSV looks like this:

```csv
filename,category,location
2018_NB47_000508.JPG,baboon,NB47
2018_NB46_000062.JPG,impala,NB46
2018_NB44_000435.JPG,blank,NB44
```

Two things to know about this format:

* **One image can appear in more than one row.** If a single photo contains both a zebra and an impala, it can have a `zebra` row and an `impala` row. (The converter below decides whether to do this — see `--multiple-label-handling`.)
* **The class names are entirely up to you.** They become the classes your fine-tuned model predicts. Whatever string you put in `category` is what the model learns; `blank` is the conventional name for "no animal", and we recommend keeping it.

### Why `location` matters

A camera-trap dataset of 1,000,000 images might come from only ~200 cameras, and the images from a single camera are highly repetitive (same background, same lighting, often the same individual animals passing repeatedly). If you let images from one camera land in both your training and validation sets, your validation accuracy will look great but will be a lie: the model is partly being tested on scenes it already memorized.

So the right thing to do is to split the data by camera: every image from a given camera goes entirely into training *or* entirely into validation, never both.  Code that you will use later in this tutorial does that splitting for you, but to do it, it needs to know which images share a camera.  That is the only reason the `location` column exists.  You don't have to think about the splitting itself; you just have to tell us the camera for each image.

If you genuinely have no camera/location information, you can put the same value (e.g. `unknown`) in every row, but be aware that your validation numbers will then be optimistic, and treat them with suspicion.

### How filenames are resolved

The `filename` value can be any of three things, and later steps in the tutorial will handle all three:

* **Relative to the CSV file's own location** (e.g. the CSV sits in your image folder and `filename` is just `2018_NB47_000508.JPG`).
* **Relative to an image root** that you pass to the later fine-tuning steps (e.g. `--image-root /data/maasai-mara`).
* **An absolute path** (e.g. `/data/maasai-mara/2018_NB47_000508.JPG`).

You don't have to choose now; just be consistent, and remember where your images actually are.

### If your data is already in COCO Camera Traps format

[COCO Camera Traps](https://github.com/agentmorris/MegaDetector/blob/main/megadetector/data_management/README.md#coco-camera-traps-format) (CCT) is a common JSON format for camera-trap labels. If your labels are in a CCT `.json` file, the script `scripts/coco_to_csv.py` produces the CSV for you. Your CCT file must have a `location` field on every image (the script will stop with a clear error if any image is missing one).

You can run the script like this:

```bash
python scripts/coco_to_csv.py path/to/labels.json path/to/output.csv
```

The options:

| Option | Default | What it does |
|---|---|---|
| `--multiple-label-handling` | `omit` | What to do with images that have more than one distinct category. `omit` drops them; `all` writes one row per category. |
| `--unlabeled-image-handling` | `omit` | What to do with images that have *no* label at all. `omit` drops them; `error` stops so you can investigate; `include` keeps them, labeled as `unlabeled`. |
| `--image-folder` | (none) | The folder your images live in. Only needed for `--check-images` or `--absolute-paths`. |
| `--absolute-paths` | off | Write absolute image paths instead of the paths as they appear in the JSON. Requires `--image-folder`. |
| `--check-images` | off | Before writing, confirm every referenced image actually exists on disk. Requires `--image-folder`. |

#### A note on `--multiple-label-handling` and rare classes

The default, `omit`, throws away any image that contains more than one category.  This is because we have no way to determine which animal in the image goes with which category.

### If your data is in some other format

For now, the only converter provided is COCO Camera Traps → CSV. If your labels live somewhere else (a different JSON schema, a database export, a set of per-camera spreadsheets, or images already sorted into folders by species), you can produce the three-column CSV yourself with whatever tool you're comfortable in — R, Excel, or a few lines of Python. As long as the result has `filename`, `category`, and `location` columns, the rest of the tutorial doesn't care how you made it. (We expect to add more converters over time, including one for the common case of images already organized into per-species and/or per-location folders.)

## Running MegaDetector

### Running SpeciesNet while you're running MD

* TODO You might want to do this to compare a fine-tuned model to a custom taxonomy mapping later

## Fine-tuning

#### Preparing a mapping file

TODO 

#### Visualizing your training data before training

TODO

#### Renaming, merging, and removing classes with `--category-remap`

TODO: this section is left over from an earlier state when remapping was part of .csv preparation

Real datasets often need cleanup: the same animal labeled two ways, sex/age splits you don't want (`lion`, `lion_male`, `lion_female`), or difficult/rare categories that you don't expect AI to be able to separate (e.g. you may want to merge individual species that are rare into categories like `rodent` or `bird`). You can fix these without editing your source data by passing `--category-remap remap.csv`, where `remap.csv` has two columns, `input` and `output`:

```csv
input,output
lion_male,lion
lion_female,lion
animal,remove
animal,remove
```

* A normal `output` *renames* the category. If several inputs map to the same output, they are *merged*.
* The special output value `remove` drops that class entirely.
* Leaving `output` blank is an error.  If you meant to drop the class, write `remove` explicitly so it's clear you intended to.
* Remapping happens *before* multi-species detection, so merging `lion_male` + `lion_female` into `lion` correctly turns a photo labeled with both into a single, unambiguous `lion` image rather than a multi-species one.

## Evaluation

### Comparing to taxonomically-mapped SpeciesNet

## Running your fine-tuned model

## Other approaches


