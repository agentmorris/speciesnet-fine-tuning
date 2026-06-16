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

The model can start either from converted SpeciesNet weights (the real goal) or,
as a stand-in during development, from timm's ImageNet weights (which strictly
want mean/std=0.5, but that mismatch only affects throwaway dev smoke tests).
"""

import timm
import torch

DEFAULT_TIMM_MODEL = "tf_efficientnetv2_m"
IMG_SIZE = 480
NORM_MEAN = (0.0, 0.0, 0.0)
NORM_STD = (1.0, 1.0, 1.0)


def build_model(num_classes, timm_model=DEFAULT_TIMM_MODEL,
                speciesnet_checkpoint=None, imagenet_pretrained=True):
    """Create the classifier.

    If speciesnet_checkpoint is given, load the converted SpeciesNet backbone
    (pjbull "speciesnet-convert" format: a dict containing a 'state_dict') and
    reset the classifier head to num_classes. Otherwise start from timm's
    ImageNet weights, which is a reasonable stand-in while the converted weights
    are being produced.
    """
    if speciesnet_checkpoint:
        model = timm.create_model(timm_model, pretrained=False, num_classes=num_classes)
        ckpt = torch.load(speciesnet_checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        # Drop the original (full-taxonomy) classifier so our fresh head survives.
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
    else:
        model = timm.create_model(timm_model, pretrained=imagenet_pretrained,
                                  num_classes=num_classes)
    return model


def freeze_backbone(model, unfreeze_blocks):
    """Freeze the backbone, then unfreeze the head and the last N block-stages.

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
        # Always train the classifier head and the final conv/bn.
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
