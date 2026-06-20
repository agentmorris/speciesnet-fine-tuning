#%% Header

"""
model.py

Build the timm EfficientNetV2-M classifier we fine-tune, and provide backbone
freezing by stage.

SpeciesNet is an EfficientNetV2-M trained at 480x480 with pixels scaled to
[0, 1] and NO mean/std normalization (the official inference path feeds img/255,
which pjbull's conversion preserves and verifies). timm's tf_efficientnetv2_m
defaults to 384px and to mean/std=0.5, but EfficientNetV2 handles other input
sizes via adaptive pooling, so we feed 480px crops in [0, 1]. The normalization
below is therefore the identity.

By default the model starts from the released, converted SpeciesNet weights
(downloaded and cached on first use). Passing "imagenet" instead starts from
timm's ImageNet weights (which strictly want mean/std=0.5); that is only for
checking that a setup runs, not for real results.
"""

#%% Imports and constants

import timm
import torch

DEFAULT_TIMM_MODEL = "tf_efficientnetv2_m"
IMG_SIZE = 480
NORM_MEAN = (0.0, 0.0, 0.0)
NORM_STD = (1.0, 1.0, 1.0)

# Default starting weights: the converted SpeciesNet EfficientNetV2-M checkpoint,
# downloaded and cached by torch.hub on first use.
SPECIESNET_TIMM_URL = "https://lila.science/speciesnet-timm"

# Pass this instead of a URL/path to start from timm's ImageNet weights.
IMAGENET_SENTINEL = "imagenet"


#%% Support functions

def _load_checkpoint(src):
    """
    Load a checkpoint from a local path or an http(s) URL (URLs are cached).
    """

    if str(src).startswith(("http://", "https://")):
        return torch.hub.load_state_dict_from_url(src, map_location="cpu", weights_only=False)
    return torch.load(src, map_location="cpu", weights_only=False)


def build_model(num_classes, timm_model=DEFAULT_TIMM_MODEL,
                speciesnet_checkpoint=SPECIESNET_TIMM_URL):
    """
    Create the classifier.

    speciesnet_checkpoint may be a URL or a local path to converted SpeciesNet
    weights (pjbull "speciesnet-convert" format: a dict containing a 'state_dict'),
    in which case the backbone is loaded and the classifier head is reset to
    num_classes. It defaults to the released SpeciesNet timm checkpoint
    (SPECIESNET_TIMM_URL), downloaded and cached on first use. Pass
    IMAGENET_SENTINEL ("imagenet") to start from timm's ImageNet weights instead,
    which is only useful for checking that a setup runs.
    """

    if not speciesnet_checkpoint or speciesnet_checkpoint == IMAGENET_SENTINEL:
        return timm.create_model(timm_model, pretrained=True, num_classes=num_classes)

    model = timm.create_model(timm_model, pretrained=False, num_classes=num_classes)
    ckpt = _load_checkpoint(speciesnet_checkpoint)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    # Drop the original (full-taxonomy) classifier so our fresh head survives
    state_dict = {k: v for k, v in state_dict.items()
                  if not (k.startswith("classifier.") or k.startswith("head."))}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    bad_missing = [m for m in missing
                   if not (m.startswith("classifier.") or m.startswith("head."))]
    if bad_missing:
        raise RuntimeError(
            "Missing non-classifier keys when loading SpeciesNet weights: %s"
            % bad_missing[:10])
    if unexpected:
        raise RuntimeError(
            "Unexpected keys when loading SpeciesNet weights: %s" % unexpected[:10])
    return model


def freeze_backbone(model, unfreeze_blocks):
    """
    Freeze the backbone, then unfreeze the head and the last N block-stages.

    unfreeze_blocks ==  0 -> train only the classifier head (+ final conv);
    unfreeze_blocks ==  N -> additionally unfreeze the last N stages of model.blocks;
    unfreeze_blocks == -1 -> train the whole network (no freezing).

    Returns (n_trainable, n_total) parameter-tensor counts for reporting.
    """

    if unfreeze_blocks == -1:
        for p in model.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False
        # Always train the classifier head and the final conv/bn
        for attr in ("classifier", "conv_head", "bn2"):
            mod = getattr(model, attr, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = True
        n_stages = len(model.blocks)
        for i in range(max(0, n_stages - unfreeze_blocks), n_stages):
            for p in model.blocks[i].parameters():
                p.requires_grad = True

    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    n_total = sum(1 for _ in model.parameters())
    return n_trainable, n_total
