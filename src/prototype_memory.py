from pathlib import Path

import torch


def load_prototype_bank(path, map_location="cpu"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prototype bank not found: {path}")

    bank = torch.load(path, map_location=map_location)
    if "pos_prototypes" not in bank or "neg_prototypes" not in bank:
        raise ValueError(f"Prototype bank is missing required tensors: {path}")

    pos_prototypes = bank["pos_prototypes"].float()
    neg_prototypes = bank["neg_prototypes"].float()

    if pos_prototypes.ndim != 2 or neg_prototypes.ndim != 2:
        raise ValueError("Prototype tensors must be rank-2 [N, C]")

    return {
        "pos_prototypes": pos_prototypes,
        "neg_prototypes": neg_prototypes,
        "meta": bank.get("meta", {}),
    }
