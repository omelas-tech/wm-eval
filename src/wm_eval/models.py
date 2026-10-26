"""Load released world-model checkpoints without their training stacks.

Released LeWM checkpoints ship a ``config.json`` whose encoder targets
``stable_pretraining.backbone.utils.vit_hf``. Installing stable-pretraining
pulls in Lightning, wandb, timm and friends, none of which inference needs, so
we re-point that target at :func:`vit_hf` below, which builds the identical
``transformers.ViTModel``. ``strict=True`` weight loading guarantees the two
architectures match parameter for parameter.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HF_BASE_URL = "https://huggingface.co"

_SPT_VIT = "stable_pretraining.backbone.utils.vit_hf"
_VIT_SIZES = {
    "tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
    "small": {"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6},
    "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
}


def vit_hf(
    size: str = "tiny",
    patch_size: int = 16,
    image_size: int = 224,
    pretrained: bool = False,
    use_mask_token: bool = True,
    **kwargs,
):
    """Drop-in for ``stable_pretraining.backbone.utils.vit_hf`` (from scratch)."""
    from transformers import ViTConfig, ViTModel

    if pretrained:
        raise ValueError("wm-eval only rebuilds architectures; weights come from the checkpoint")
    params = dict(_VIT_SIZES[size])
    params["intermediate_size"] = params["hidden_size"] * 4
    params.update(image_size=image_size, patch_size=patch_size, **kwargs)
    model = ViTModel(ViTConfig(**params), add_pooling_layer=False, use_mask_token=use_mask_token)
    model.config.interpolate_pos_encoding = True
    return model


@dataclass(frozen=True)
class LoadedModel:
    model: object
    ref: str
    weights_sha256: str
    config: dict


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    tmp.rename(dest)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# A downloaded config.json is untrusted input: hydra will import and call any
# ``_target_`` it names. Only architecture builders may appear there.
ALLOWED_TARGET_PREFIXES = (
    "stable_worldmodel.wm.",
    "torch.nn.",
    f"{__name__}.vit_hf",
)


class UntrustedConfigError(ValueError):
    pass


def _retarget(node):
    if isinstance(node, dict):
        out = {k: _retarget(v) for k, v in node.items()}
        target = out.get("_target_")
        if target == _SPT_VIT:
            out["_target_"] = target = f"{__name__}.vit_hf"
        if target is not None and not str(target).startswith(ALLOWED_TARGET_PREFIXES):
            raise UntrustedConfigError(f"config target not allowed: {target!r}")
        return out
    if isinstance(node, list):
        return [_retarget(v) for v in node]
    return node


def resolve_checkpoint(ref: str, cache_dir: Path) -> tuple[Path, Path]:
    """Return local ``(weights.pt, config.json)`` for ``hf:<user>/<repo>`` or a folder."""
    if ref.startswith("hf:"):
        repo = ref[3:]
        local = cache_dir / repo.replace("/", "--")
        for name in ("config.json", "weights.pt"):
            if not (local / name).exists():
                _fetch(f"{HF_BASE_URL}/{repo}/resolve/main/{name}", local / name)
        return local / "weights.pt", local / "config.json"
    folder = Path(ref)
    return folder / "weights.pt", folder / "config.json"


def load_world_model(ref: str, cache_dir: Path, device: str = "cpu") -> LoadedModel:
    import torch
    from hydra.utils import instantiate

    weights, config_path = resolve_checkpoint(ref, cache_dir)
    config = json.loads(config_path.read_text())
    model = instantiate(_retarget(config))
    # weights_only: a checkpoint is data, never code (no pickle execution).
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    return LoadedModel(model=model, ref=ref, weights_sha256=sha256_file(weights), config=config)
