#%% Header

"""
train.py

Fine-tune SpeciesNet (a timm EfficientNetV2-M) on your own camera-trap data.

Inputs are the data CSV (filename, category, location), the image folder, and a
MegaDetector results file; the script crops to animal boxes, splits cameras into
train/val, optionally remaps/merges/drops classes, freezes part of the backbone,
and trains. Everything for one run is written to a required run folder.

Fresh run:
  python scripts/train.py --data-csv data.csv --image-root IMAGES \\
      --md-results md.json --run-folder runs/myrun [options]

Resume an interrupted run (needs only the folder):
  python scripts/train.py --resume runs/myrun

See the README "Fine-tuning" section for the full walkthrough.
"""

#%% Imports and constants

import argparse
import csv
import json
import os
import re
import sys
import platform
from pathlib import Path

import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
import timm
from torchmetrics.classification import MulticlassAccuracy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instances import prepare                       # noqa: E402
from split import make_split                        # noqa: E402
from dataset import CropDataset                     # noqa: E402
from model import (build_model, freeze_backbone,    # noqa: E402
                   DEFAULT_TIMM_MODEL, IMG_SIZE, NORM_MEAN, NORM_STD, SPECIESNET_TIMM_URL)

INFERENCE_FORMAT = "speciesnet-finetune-classifier-v1"

# Arguments that define the data/model and must be identical when resuming.
CONFIG_KEYS = [
    "data_csv", "image_root", "md_results", "mapping", "val_fraction",
    "min_instances", "conf_threshold", "max_boxes", "seed", "epochs",
    "batch_size", "workers", "lr", "weight_decay", "unfreeze_blocks",
    "timm_model", "backbone_checkpoint", "weighted_loss",
    "checkpoint_every_n_epochs", "patience",
]


#%% Support functions

def is_ddp_child():
    """
    True if this process is a Lightning-spawned DDP worker (not the launcher).
    """

    return "LOCAL_RANK" in os.environ


# --------------------------------------------------------------------------- #
# Lightning module
# --------------------------------------------------------------------------- #
class LitClassifier(L.LightningModule):
    def __init__(self, num_classes, classes, timm_model, unfreeze_blocks, lr,
                 weight_decay, epochs, class_weights=None, backbone_checkpoint=None):
        super().__init__()
        # class_weights is data, not a hyperparameter we want re-instantiated on load.
        self.save_hyperparameters(ignore=["class_weights"])
        self.model = build_model(
            num_classes, timm_model=timm_model,
            speciesnet_checkpoint=backbone_checkpoint)
        self.n_trainable, self.n_total = freeze_backbone(self.model, unfreeze_blocks)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_macro = MulticlassAccuracy(num_classes=num_classes, average="macro")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.val_acc(logits, y)
        self.val_macro(logits, y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_macro_acc", self.val_macro, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs)
        return {"optimizer": opt, "lr_scheduler": sched}


def compute_class_weights(instances, classes):
    counts = {c: 0 for c in classes}
    for inst in instances:
        counts[inst.category] += 1
    total = sum(counts.values())
    n = len(classes)
    return torch.tensor([total / (n * counts[c]) for c in classes], dtype=torch.float32)


def read_best(run_folder, best_path, trainer):
    """
    Return (best_epoch, best_val_metrics) for the monitored best checkpoint.

    The best checkpoint's epoch is parsed from its filename; its validation
    metrics are looked up in metrics.csv. Falls back to the final-epoch metrics
    if the lookup fails.
    """
    best_epoch = None
    if best_path:
        m = re.search(r"epoch0*([0-9]+)", os.path.basename(best_path))
        if m:
            best_epoch = int(m.group(1))
    metrics = {}
    mpath = os.path.join(run_folder, "metrics.csv")
    if best_epoch is not None and os.path.exists(mpath):
        with open(mpath, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("epoch") and row.get("val_macro_acc") and int(float(row["epoch"])) == best_epoch:
                    for k in ("val_loss", "val_acc", "val_macro_acc"):
                        if row.get(k):
                            metrics[k] = round(float(row[k]), 4)
    if not metrics:
        metrics = {k: round(float(v), 4) for k, v in trainer.callback_metrics.items()
                   if k.startswith("val")}
    return best_epoch, metrics


def find_latest_checkpoint(run_folder):
    ckpt_dir = Path(run_folder) / "checkpoints"
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return str(last)
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    return str(ckpts[-1]) if ckpts else None


def export_inference_checkpoint(best_ckpt_path, out_path, timm_model, num_classes,
                                classes, epoch, metrics):
    """
    Write a compact, self-describing checkpoint for predict.py.
    """

    ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    model_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    model = timm.create_model(timm_model, pretrained=False, num_classes=num_classes)
    model.load_state_dict(model_sd)
    torch.save({
        "format": INFERENCE_FORMAT,
        "timm_model": timm_model,
        "num_classes": num_classes,
        "classes": classes,
        "img_size": IMG_SIZE,
        "norm_mean": list(NORM_MEAN),
        "norm_std": list(NORM_STD),
        "preprocessing": ("crop to MegaDetector animal box, resize to %dx%d, "
                          "scale to [0,1] (no mean/std normalization)" % (IMG_SIZE, IMG_SIZE)),
        "source_checkpoint": os.path.basename(best_ckpt_path),
        "epoch": epoch,
        "metrics": metrics,
        "state_dict": model.state_dict(),
    }, out_path)


def write_summary(run_folder, config, prep_report, split_report, classes,
                  model_info=None, final=None):
    lines = []
    a = lines.append
    a("# SpeciesNet fine-tuning run\n")
    a("Run folder: `%s`\n" % run_folder)
    a("## Configuration\n")
    a("| setting | value |")
    a("|---|---|")
    for k in CONFIG_KEYS:
        a("| %s | %s |" % (k, config.get(k)))
    a("")
    a("## Data\n")
    p = prep_report
    a("- CSV rows: %d" % p["n_csv_rows"])
    a("- Instances after box selection + mapping: %d (before min-count)" % p["n_instances_before_mincount"])
    a("- Instances used for training: **%d** across **%d** classes" % (p["n_instances_final"], p["n_classes"]))
    a("- MD animal detections: %d; selected (conf>=%.2f, max %d/image): %d"
      % (p["md_n_animal_dets"], p["params"]["conf_threshold"], p["params"]["max_boxes"], p["md_n_selected"]))
    a("")
    a("### Warnings (not fatal)\n")
    a("- Labeled images dropped because all labels were `remove`: %d" % p["n_dropped_by_remove"])
    a("- Labeled images missing from disk: %d" % p["csv_missing_on_disk_count"])
    a("- Labeled images absent from the MD file: %d" % p["n_csv_not_in_md"])
    a("- Labeled images with no animal box above threshold: %d" % p["n_csv_no_boxes"])
    a("- MD entries not found on disk: %d" % p["md_not_on_disk_count"])
    a("- Images on disk not in the MD file: %d" % p["disk_not_in_md_count"])
    for w in p["warnings"]:
        a("- %s" % w)
    if p["dropped_by_mincount"]:
        a("- Classes dropped below min-count (%d): %s"
          % (p["params"]["min_instances"],
             ", ".join("%s(%d)" % (c, n) for c, n in p["dropped_by_mincount"])))
    a("")
    a("## Train/val split (by camera)\n")
    s = split_report
    a("- Locations: %d total, %d train, %d val (%.1f%% of locations in val)"
      % (s["n_locations"], s["n_train_locations"], s["n_val_locations"], 100 * s["frac_val_locations"]))
    a("- Instances: %d train, %d val (%.1f%% of instances in val; target %.0f%%)"
      % (s["n_instances_train"], s["n_instances_val"], 100 * s["frac_val_instances"],
         100 * s["val_fraction_requested"]))
    if s["classes_missing_train"]:
        a("- WARNING: classes with no training instances: %s" % ", ".join(s["classes_missing_train"]))
    if s["classes_missing_val"]:
        a("- WARNING: classes with no validation instances: %s" % ", ".join(s["classes_missing_val"]))
    a("")
    a("| class | total | train | val | val %% |")
    a("|---|---|---|---|---|")
    for c in sorted(classes, key=lambda c: -s["per_category"][c]["total"]):
        d = s["per_category"][c]
        a("| %s | %d | %d | %d | %.1f |" % (c, d["total"], d["train"], d["val"], 100 * d["val_fraction"]))
    a("")
    if model_info:
        a("## Model\n")
        a("- timm model: `%s`, classes: %d" % (model_info["timm_model"], model_info["num_classes"]))
        a("- Backbone init: %s" % model_info["init"])
        a("- Trainable parameter tensors: %d of %d (unfreeze_blocks=%s)"
          % (model_info["n_trainable"], model_info["n_total"], config.get("unfreeze_blocks")))
        a("")
    if final:
        a("## Result\n")
        a("- Best epoch: %s" % final.get("epoch"))
        a("- Best val metrics: %s" % final.get("metrics"))
        a("- Inference checkpoint: `%s`" % final.get("inference_checkpoint"))
        a("")
    with open(os.path.join(run_folder, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_split(run_folder, split):
    """
    Write the camera-to-split assignment to split.csv (location, split).
    """

    with open(os.path.join(run_folder, "split.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["location", "split"])
        for loc in sorted(split):
            w.writerow([loc, split[loc]])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resume", metavar="RUN_FOLDER",
                   help="resume a previous run from its folder (ignores other data/model args)")
    p.add_argument("--data-csv", help="data CSV with columns filename, category, location")
    p.add_argument("--image-root", help="folder the image filenames are relative to")
    p.add_argument("--md-results", help="MegaDetector results .json file")
    p.add_argument("--run-folder", help="output folder for this run (must not already exist)")
    p.add_argument("--mapping", help="optional category mapping CSV (input,output)")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--min-instances", type=int, default=100)
    p.add_argument("--conf-threshold", type=float, default=0.3)
    p.add_argument("--max-boxes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--unfreeze-blocks", type=int, default=2,
                   help="0=head only, N=last N backbone stages, -1=all")
    p.add_argument("--timm-model", default=DEFAULT_TIMM_MODEL)
    p.add_argument("--backbone-checkpoint", default=SPECIESNET_TIMM_URL,
                   help="converted SpeciesNet timm checkpoint (URL or local path) to start "
                   "from; defaults to the released checkpoint. Pass 'imagenet' to start from "
                   "ImageNet weights instead (only for checking that a setup runs).")
    p.add_argument("--weighted-loss", action="store_true",
                   help="weight the loss by inverse class frequency")
    p.add_argument("--checkpoint-every-n-epochs", type=int, default=1)
    p.add_argument("--patience", type=int, default=0,
                   help="early-stopping patience on val_macro_acc (0 = disabled)")
    p.add_argument("--devices", default="auto", help="'auto' (all GPUs) or an integer count")
    p.add_argument("--limit-batches", type=int, default=0,
                   help="debug: cap train/val batches per epoch (0 = no cap)")
    return p.parse_args(argv)


def resolve_config(args):
    """
    Return (config dict, run_folder, resuming).
    """

    if args.resume:
        run_folder = args.resume
        with open(os.path.join(run_folder, "config.json"), encoding="utf-8") as f:
            config = json.load(f)
        return config, run_folder, True
    for req in ("data_csv", "image_root", "md_results", "run_folder"):
        if getattr(args, req) is None:
            raise SystemExit("ERROR: --%s is required for a fresh run" % req.replace("_", "-"))
    config = {k: getattr(args, k) for k in CONFIG_KEYS}
    return config, args.run_folder, False


#%% Core training function


#%% Command-line driver

def main(argv=None):
    args = parse_args(argv)
    config, run_folder, resuming = resolve_config(args)
    L.seed_everything(config["seed"], workers=True)

    # Fresh-run folder management happens only in the launcher process.
    if not resuming and not is_ddp_child():
        if os.path.exists(run_folder):
            raise SystemExit(
                "ERROR: run folder '%s' already exists. Use --resume to continue it, "
                "or choose a new folder." % run_folder)
        os.makedirs(os.path.join(run_folder, "checkpoints"))
        with open(os.path.join(run_folder, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    # Data preparation (deterministic, so every DDP rank agrees).
    instances, classes, prep_report = prepare(
        config["data_csv"], config["md_results"], config["image_root"],
        mapping_path=config["mapping"], conf_threshold=config["conf_threshold"],
        max_boxes=config["max_boxes"], min_instances=config["min_instances"],
        scan_disk=not is_ddp_child())
    split, split_report = make_split(instances, val_fraction=config["val_fraction"],
                                     seed=config["seed"])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    train_inst = [i for i in instances if split[i.location] == "train"]
    val_inst = [i for i in instances if split[i.location] == "val"]

    train_ds = CropDataset(train_inst, class_to_idx, config["image_root"], train=True)
    val_ds = CropDataset(val_inst, class_to_idx, config["image_root"], train=False)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["workers"], pin_memory=True, drop_last=True,
        persistent_workers=config["workers"] > 0)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["workers"], pin_memory=True,
        persistent_workers=config["workers"] > 0)

    class_weights = compute_class_weights(train_inst, classes) if config["weighted_loss"] else None
    model = LitClassifier(
        num_classes=len(classes), classes=classes, timm_model=config["timm_model"],
        unfreeze_blocks=config["unfreeze_blocks"], lr=config["lr"],
        weight_decay=config["weight_decay"], epochs=config["epochs"],
        class_weights=class_weights, backbone_checkpoint=config["backbone_checkpoint"])
    bc = config["backbone_checkpoint"]
    init_desc = "ImageNet (stand-in)" if (not bc or bc == "imagenet") else ("SpeciesNet: " + bc)
    model_info = {"timm_model": config["timm_model"], "num_classes": len(classes),
                  "init": init_desc,
                  "n_trainable": model.n_trainable, "n_total": model.n_total}

    # Write the split assignment and an initial summary (so the data/split report
    # exists even if training fails).
    if not is_ddp_child():
        write_split(run_folder, split)
        write_summary(run_folder, config, prep_report, split_report, classes, model_info)

    # Devices / strategy (gloo on native Windows, NCCL elsewhere).
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.devices == "auto":
        devices, n_proc = ("auto", max(1, n_gpus))
    else:
        devices = int(args.devices)
        n_proc = devices
    accelerator = "gpu" if n_gpus > 0 else "cpu"
    if accelerator == "gpu" and n_proc > 1:
        if platform.system() == "Windows":
            # Windows has no NCCL, so use gloo. Pin the rendezvous address so the
            # process group doesn't try to resolve a bogus host (e.g. Docker
            # Desktop's kubernetes.docker.internal entry).
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            from lightning.pytorch.strategies import DDPStrategy
            strategy = DDPStrategy(process_group_backend="gloo")
        else:
            strategy = "ddp"
    else:
        strategy = "auto"
    precision = "bf16-mixed" if accelerator == "gpu" else "32-true"

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(run_folder, "checkpoints"),
            filename="epoch{epoch:03d}-valmacro{val_macro_acc:.3f}",
            auto_insert_metric_name=False, monitor="val_macro_acc", mode="max",
            save_top_k=-1, save_last=True,
            every_n_epochs=config["checkpoint_every_n_epochs"]),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if config["patience"] and config["patience"] > 0:
        callbacks.append(EarlyStopping(monitor="val_macro_acc", mode="max",
                                       patience=config["patience"]))

    logger = CSVLogger(save_dir=run_folder, name="", version="")
    limit = args.limit_batches or 1.0
    trainer = L.Trainer(
        max_epochs=config["epochs"], accelerator=accelerator, devices=devices,
        strategy=strategy, precision=precision, default_root_dir=run_folder,
        logger=logger, callbacks=callbacks, log_every_n_steps=10,
        limit_train_batches=limit, limit_val_batches=limit)

    ckpt_path = find_latest_checkpoint(run_folder) if resuming else None
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    # Export the best model + final summary (rank 0 only).
    if trainer.is_global_zero:
        cb = trainer.checkpoint_callback
        best = cb.best_model_path or find_latest_checkpoint(run_folder)
        best_epoch, best_metrics = read_best(run_folder, best, trainer)
        inf_path = os.path.join(run_folder, "model_best.pt")
        final = {"epoch": best_epoch, "metrics": best_metrics,
                 "inference_checkpoint": "model_best.pt"}
        if best and os.path.exists(best):
            export_inference_checkpoint(best, inf_path, config["timm_model"],
                                        len(classes), classes, best_epoch, best_metrics)
        write_summary(run_folder, config, prep_report, split_report, classes,
                      model_info, final=final)
        print("Done. Best checkpoint: %s" % best)
        print("Inference model: %s" % inf_path)
        print("Summary: %s" % os.path.join(run_folder, "summary.md"))


if __name__ == "__main__":
    main()
