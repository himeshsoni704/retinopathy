"""Backbones and heads. Checkpoints saved by this project always contain full weights."""
import argparse
import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models.vision_transformer import interpolate_embeddings


class Heads(nn.Module):
    def __init__(self, encoder, features, task, frozen=False):
        super().__init__()
        self.encoder, self.task, self.frozen = encoder, task, frozen
        if frozen:
            encoder.requires_grad_(False)
        self.primary = nn.Linear(features, {"grade":5,"quality":3,"damage":8}[task])
        self.secondary = nn.Linear(features, {"grade":1,"damage":8}[task]) if task != "quality" else None

    def forward(self, x):
        if self.frozen:
            self.encoder.eval()
        features = self.encoder(x).flatten(1)
        first = self.primary(features)
        return first, self.secondary(features) if self.secondary is not None else None


def retfound(size, weights):
    from timm.models.vision_transformer import VisionTransformer
    encoder = VisionTransformer(img_size=size, patch_size=16, embed_dim=1024, depth=24,
                                num_heads=16, mlp_ratio=4, qkv_bias=True, global_pool="avg",
                                num_classes=0, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    if weights:
        with torch.serialization.safe_globals([argparse.Namespace]):
            raw = torch.load(weights, map_location="cpu", weights_only=True)
        source = raw.get("model", raw.get("state_dict", raw))
        source = {k.removeprefix("module.").removeprefix("backbone."): v for k,v in source.items()}
        expected = set(encoder.state_dict())
        state = {k:v for k,v in source.items() if k in expected}
        if "pos_embed" not in state:
            raise ValueError("Expected a RETFound-MAE colour-fundus encoder with pos_embed")
        pos = state["pos_embed"]
        if pos.shape != encoder.pos_embed.shape:
            side = math.isqrt(pos.size(1)-1)
            grid = pos[:,1:].reshape(1,side,side,1024).permute(0,3,1,2)
            grid = F.interpolate(grid, (size//16,size//16), mode="bicubic", align_corners=False)
            state["pos_embed"] = torch.cat((pos[:,:1],grid.flatten(2).transpose(1,2)),dim=1)
        # MAE pretraining can supply only the pretraining norm. Global-pool LayerNorm
        # starts with identity affine parameters when fc_norm is absent.
        missing, _ = encoder.load_state_dict(state, strict=False)
        if set(missing) - {"fc_norm.weight","fc_norm.bias"}:
            raise ValueError(f"Incompatible RETFound-MAE encoder: missing keys {missing[:8]}")
    return encoder


def build(task, arch, size, pretrained=False, retinal_weights=None):
    if arch == "retfound":
        if task != "grade" or (pretrained and not retinal_weights):
            raise ValueError("RETFound grade training needs a local colour-fundus MAE checkpoint")
        return Heads(retfound(size, retinal_weights), 1024, task, frozen=True)
    if arch == "vit":
        encoder = models.vit_b_16(image_size=size)
        if pretrained:
            state = models.ViT_B_16_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
            encoder.load_state_dict(interpolate_embeddings(size,16,state))
        encoder.heads = nn.Identity()
        return Heads(encoder, 768, task)
    factory, weights = {"r50":(models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
                        "r18":(models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1)}[arch]
    encoder = factory(weights=weights if pretrained else None)
    features = encoder.fc.in_features
    encoder.fc = nn.Identity()
    return Heads(encoder, features, task)


def restore(path, device):
    data = torch.load(path, map_location="cpu", weights_only=True)
    net = build(data["task"], data["arch"], data["size"])
    net.load_state_dict(data.pop("state"))
    return net.eval().to(device), data
