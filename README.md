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

* TODO: talk about ways to run MD, AddaxAI, Python, etc.

## Visualizing your training data before training

TODO

## Preparing a mapping file

A mapping file is an optional CSV that renames, merges, or drops your categories at training time.  It is kept separate from your data CSV on purpose: the data CSV stays a literal record of your labels, while the mapping captures modeling decisions, so you can try different groupings without ever editing your data.

### Why you might remap

Common reasons:

* **The same animal is labeled two ways.** For example `wildebeest` and `blue_wildebeest`, or a misspelling sitting alongside the correct spelling.  Map them to one name so they count as a single class.
* **You have splits you do not want the model to make.** Sex or age labels like `lion`, `lion_male`, and `lion_female` are usually better merged into `lion`, unless you specifically need to tell them apart and have enough examples of each.
* **Some classes are too rare or too hard to tell apart.** A long tail of bird species with a handful of images each will not train well individually.  Merging them into a coarser class like `other_bird` (or `rodent`, `reptile`, and so on) usually gives a more useful model than dozens of classes the model can barely learn.
* **Some labels are not real classes.** A vague catch-all like `animal`, or a junk label, can be dropped entirely.

You do not have to remap anything: if you skip the mapping file, every category (above the minimum count) is trained as its own class.

### The mapping CSV format

The mapping CSV has two columns that matter, `input` and `output` (any other columns, such as the `count` column described below, are ignored):

```csv
input,output
hyena_spotted,hyena
hyena_striped,hyena
lion_male,lion
lion_female,lion
crane,other_bird
eagle,other_bird
animal,remove
zebra,
```

The `output` column decides what happens to each `input` category:

* **A name** renames the category; several inputs sharing one output are merged (so `hyena_spotted` and `hyena_striped` above both become `hyena`).
* **The exact word `remove`** drops the category entirely; its images contribute nothing to training.
* **Left blank** leaves the category unchanged, which is identical to not listing it at all (the `zebra` row above is just documentation; you could delete it).

Two rules: each `input` may appear only once, and the mapping is applied in a single pass, so if you map `A` to `B` and also `B` to `C`, an `A` becomes `B`, not `C`.

### Generating a template

You can write the mapping CSV by hand, but if your labels are in COCO Camera Traps format it is easier to start from a generated template.  `scripts/coco_to_mapping_file.py` lists every category for you, sorted from most to least common, with the `output` column left blank to fill in:

```bash
python scripts/coco_to_mapping_file.py path/to/labels.json mapping.csv
```

The result has one row per category (plus a row for `unlabeled`, the images with no annotations, which is always included) and a `count` column giving the number of images that contain that category:

```csv
input,output,count
wildebeest,,6870
gazelle_thomsons,,6227
...
sparrow,,1
unlabeled,,0
```

Use `count` to decide what to merge or drop; the long tail at the bottom is where merging into coarser classes usually helps.  Then fill in the `output` column and pass the file to training with `--mapping` (see "Fine-tuning").

One caveat about `count`: it is the number of *images* per category, which is a good planning guide, but the model actually trains on MegaDetector crops, so the number of training examples per class will differ (an image of a herd yields many crops; a `blank` image usually yields none).  The exact per-class crop counts that training used, after your mapping and the minimum-count filter, are reported in the run's `summary.md`.

## Fine-tuning

Fine-tuning takes the SpeciesNet classifier and continues training it on your own labeled crops, so it learns the species in your ecosystem (and your label names) instead of SpeciesNet's full global taxonomy.  The training script does three things for you: it turns your data into training crops, it splits your cameras into training and validation sets, and it trains the model while recording everything in a single run folder.

By default each labeled image becomes one crop per MegaDetector animal box (up to five boxes per image, at confidence 0.3 or higher; see `--max-boxes` and `--conf-threshold`), and each crop inherits its image's label.  This is the same "classify the animal box" approach SpeciesNet itself uses.  One consequence is worth understanding: an image with no animal box above the threshold produces no training crops, so most `blank` images contribute nothing.  A crop-based classifier only learns `blank` from MegaDetector's false-positive boxes, which is expected.  Finally, the classifier head is replaced with a fresh head over your classes, and part of the backbone is unfrozen so it can adapt (see "Choosing how much to fine-tune" below).

### What you need

Before fine-tuning you should already have:

* **A data CSV** with `filename`, `category`, and `location` columns (see "Preparing your data").
* **A MegaDetector results file** covering those images (see "Running MegaDetector").
* **The SpeciesNet starting weights** (see the next subsection).
* **Optionally, a mapping file** to rename, merge, or drop classes (see "Preparing a mapping file").

### Getting the SpeciesNet starting weights

Fine-tuning starts from a PyTorch copy of SpeciesNet's EfficientNetV2-M backbone.  Download the pre-converted checkpoint, `speciesnet_timm_m.pt`, from [TODO: add release link], and pass it with `--backbone-checkpoint`.  If you leave that option off, the script falls back to generic ImageNet weights, which is only useful for checking that your setup runs at all; for real results you want the SpeciesNet weights.

That checkpoint is produced by converting SpeciesNet's original Keras weights into a `timm` model.  You do not need to run that conversion yourself (it needs a separate environment; see `requirements-conversion.txt` and "Other approaches"), because the download above already did it once.

How faithful is the converted model?  We compared it against the officially released PyTorch SpeciesNet on 4,837 random animal crops, looking at each model's top prediction.  When the original model is confident, the two agree almost perfectly: 97% agreement at confidence above 0.5, 99.3% above 0.7, and 99.8% above 0.9 (and 92% across all crops, including very low-confidence ones).  The few disagreements are low-confidence, visually ambiguous cases (for example one gazelle species versus another).  In short, the converted weights are a faithful starting point.

### Running a training run

A minimal run looks like this (it is shown on several lines for readability; put it on one line, or use your shell's line-continuation character):

```bash
python scripts/train.py \
    --data-csv data.csv \
    --image-root /path/to/images \
    --md-results md_results.json \
    --backbone-checkpoint speciesnet_timm_m.pt \
    --mapping mapping.csv \
    --run-folder runs/my-first-run
```

`--run-folder` is required and must not already exist; everything from the run is written there.  The script prints a running summary, and when it finishes it writes the fine-tuned model to `runs/my-first-run/model_best.pt`.

### Options

Run `python scripts/train.py --help` for the complete list; these are the ones most worth knowing.

| Option | Default | What it does |
|---|---|---|
| `--backbone-checkpoint` | (ImageNet) | The converted SpeciesNet weights to start from.  Strongly recommended. |
| `--mapping` | (none) | Mapping CSV to rename, merge, or drop classes (see "Preparing a mapping file"). |
| `--unfreeze-blocks` | `2` | How much of the backbone to train: `0` = the new head only, `N` = the head plus the last N of the backbone's 7 stages, `-1` = the whole network. |
| `--min-instances` | `100` | Drop any class with fewer than this many training crops (counted after mapping). |
| `--val-fraction` | `0.15` | Target fraction of crops to hold out for validation (chosen by camera). |
| `--conf-threshold` | `0.3` | Minimum MegaDetector confidence for a box to become a training crop. |
| `--max-boxes` | `5` | Maximum animal boxes to use per image (highest confidence first). |
| `--weighted-loss` | off | Weight the loss by inverse class frequency, to help rare classes. |
| `--epochs` | `20` | Number of passes over the training data. |
| `--batch-size` | `32` | Crops per step, per GPU. |
| `--lr` | `1e-4` | Learning rate. |
| `--workers` | `8` | Data-loading worker processes, per GPU. |
| `--checkpoint-every-n-epochs` | `1` | How often to save a checkpoint (every epoch by default). |
| `--patience` | `0` | If greater than 0, stop early when validation macro-accuracy has not improved for this many epochs. |
| `--devices` | `auto` | `auto` uses all GPUs; or give an integer count. |
| `--seed` | `0` | Random seed (this also determines the train/val split). |

### What a run produces

Everything for one run lives in its run folder:

* **`summary.md`**: a human-readable report of the run, and the first thing to read.  It lists your final classes and how many crops each has, the train/val split balance (per camera and per class), every data warning (images missing from disk, images with no animal box, classes dropped below the minimum, and so on), and the final metrics.
* **`model_best.pt`**: a compact, self-describing checkpoint of the best epoch, ready for "Running your fine-tuned model".  It records the class list and the exact preprocessing, so you never have to remember them.
* **`checkpoints/`**: one checkpoint per epoch (by default) plus `last.ckpt`, which is what resuming uses.
* **`metrics.csv`**: per-epoch training and validation metrics.
* **`config.json`**: the full configuration, which is what makes resuming possible.
* **`split.csv`**: which camera (location) was assigned to the training set and which to validation.
* **`hparams.yaml`**: the model's hyperparameters, as recorded by the training engine (PyTorch Lightning).

### Resuming an interrupted run

If a run stops partway (a crash, a reboot, or you stopping it), continue it using only the folder name:

```bash
python scripts/train.py --resume runs/my-first-run
```

It reads `config.json`, finds the most recent checkpoint, and picks up where it left off, restoring the optimizer, the learning-rate schedule, and the epoch count.  You do not need to remember any of the other settings; they were saved for you.

### Using both GPUs, and Windows versus WSL

If your machine has more than one GPU, the script uses all of them by default (pass `--devices 1` to force a single GPU).  On a two-GPU machine each epoch is roughly twice as fast.

This works on both native Windows and Linux/WSL.  In our testing the throughput was about the same on both (around 190 to 200 images per second per RTX 4090, reading images from a Windows drive in either case), so for this model you do not need to bother with WSL.  Unlike large transformer models, EfficientNetV2 does not benefit from the Linux-only acceleration features, so the usual reason to prefer WSL does not apply here.

### Choosing how much to fine-tune

The most important knob is `--unfreeze-blocks`.  Freezing more of the backbone (a smaller number) trains fewer parameters: it is faster and less prone to overfitting on small datasets, but it adapts less.  Unfreezing more (a larger number, or `-1` for the whole network) can fit your data better, but it needs more data and overfits more easily.  The default of `2` is a reasonable middle ground that still leaves most of the network trainable.  If your dataset is small, or you watch validation accuracy peak early and then decline, try a smaller number; if your dataset is large and diverse, try a larger one.

A few other practical notes:

* **Watch the per-class counts in `summary.md`.** Classes below `--min-instances` are dropped, and the "omit multi-label images" default during data preparation can quietly shrink rare classes.  If a class you care about is missing or tiny, that report is where you will notice.
* **For imbalanced data, try `--weighted-loss`.** Camera-trap datasets are very long-tailed, and weighting the loss pushes the model to pay more attention to rarer classes, at some cost to accuracy on the common ones.
* **Your validation numbers are only as honest as your locations.** Because the split is by camera, validation accuracy reflects how well the model will do on cameras it has never seen.  If you put every image under one location, those numbers will be optimistic.
* **Rare, visually similar classes are hard.** If two species are nearly indistinguishable in your images, consider merging them in the mapping file rather than expecting the model to separate them.

## Running your fine-tuned model

Once you have a fine-tuned model (`model_best.pt` from your run folder), you run it on new images the same way you prepared your training data: run MegaDetector on the new images, then classify each animal box.  The `scripts/predict.py` script does the classification step.

`predict.py` does not run MegaDetector for you, so you need a MegaDetector results file for the new images first (see "Running MegaDetector").  It also does not need the SpeciesNet starting weights or anything from training besides `model_best.pt`: that file already records your class list and the exact preprocessing, so predictions match training automatically.

A minimal run:

```bash
python scripts/predict.py \
    runs/my-first-run/model_best.pt \
    md_results.json \
    --image-root /path/to/new/images \
    --output predictions.json
```

By default the output is a MegaDetector-format results file: a copy of your input MD file with the model's classifications added to each animal detection at or above the confidence threshold.  Every original detection is kept (person and vehicle boxes, and animal boxes below the threshold, are preserved with no classification added), so the file drops straight into MegaDetector's own postprocessing and evaluation tools (see "Evaluation").  Pass `--csv-output` to instead write a simple per-box CSV (described below), which is handier if you only want to skim results in a spreadsheet.

### Options

| Option | Default | What it does |
|---|---|---|
| `--image-root` | (required) | The folder the MegaDetector filenames are relative to. |
| `--output` | (required) | Where to write the results (a MegaDetector-format `.json` by default). |
| `--csv-output` | off | Write a flat per-box CSV to `--output` instead of MegaDetector format. |
| `--conf-threshold` | `0.1` | Only classify animal boxes at or above this MegaDetector confidence. |
| `--topk` | `1` | How many top predictions (with scores) to record per box. |
| `--batch-size` | `32` | Crops classified per batch. |
| `--device` | `auto` | `auto` picks a GPU if one is available, otherwise the CPU. |

### The MegaDetector-format output

In the default output, each classified animal detection gains a `classifications` list, ordered most likely first, where each entry pairs a class id (an index into the top-level `classification_categories`) with a score.  With `--topk 5` you get the top five classes per box.  Because the whole detection file is preserved, this output is exactly what MegaDetector's classification postprocessing expects, which is how the "Evaluation" step below works.

### The CSV output

With `--csv-output` you get one row per classified animal box, not per image, because an image can contain several animals.  Each row gives the filename, the box (as normalized `x, y, w, h`), the MegaDetector detection confidence, and the model's prediction(s) and score(s):

```csv
filename,x,y,w,h,detection_conf,pred1_class,pred1_score
2019_AB01_000123.JPG,0.21,0.34,0.08,0.15,0.94,impala,0.972
2019_AB01_000123.JPG,0.55,0.41,0.10,0.18,0.88,zebra,0.910
2019_AB07_000044.JPG,0.30,0.29,0.40,0.55,0.97,elephant,0.995
```

With `--topk 3`, each row also carries `pred2_class`, `pred2_score`, `pred3_class`, and `pred3_score`, which is useful for seeing the model's second guess on hard crops.

A few things to keep in mind:

* **Only animal boxes are classified.** Person and vehicle detections from MegaDetector are never classified, and animal boxes below `--conf-threshold` are skipped.  In the MegaDetector-format output those detections are still present, just without a `classifications` entry; in the CSV output they do not appear at all.
* **`blank` is a prediction, not an absence of a box.** If you trained a `blank` class, the model can label an animal box as `blank` when it thinks the box is actually empty (a MegaDetector false positive).  An image where MegaDetector found no animal at all is "blank" in a different sense: it has no animal box to classify.
* **The model only knows your classes.** It predicts from the label set you trained on, using your names, and has no knowledge of SpeciesNet's broader taxonomy or its geofence.  Anything outside your classes will be forced into the closest class you did train.
* **Run it in your training environment.** Use the same environment you fine-tuned in; `predict.py` rebuilds the model from `model_best.pt` and needs the same packages.

## Evaluation

* TODO: describe analyze_classification_results.py (for labeled data)
* TODO: describe postprocess_batch_results.py(for unlabeled data)

### Comparing to taxonomically-mapped SpeciesNet

* TODO: describe restrict_to_taxa list, comparison script

## Working with your results

* TODO: talk about things people do next, especially Timelapse

## Future work

* TODO: Conversion and mapping file scripts for other input formats