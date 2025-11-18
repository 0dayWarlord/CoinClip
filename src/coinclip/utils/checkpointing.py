#checkpoint save/load utilities

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Optional[Dict[str, float]] = None,
    filepath: Path = Path("checkpoints/checkpoint.pt"),
    is_best: bool = False,
) -> None:
    """
    save model checkpoint

    Args:
        model: model to save
        optimizer: optimizer state
        epoch: current epoch
        loss: current loss value
        metrics: optional metrics dictionary
        filepath: path to save checkpoint
        is_best: if True, also save as best checkpoint
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "metrics": metrics or {},
    }

    torch.save(checkpoint, filepath)

    if is_best:
        best_path = filepath.parent / "best_model.pt"
        torch.save(checkpoint, best_path)


def load_checkpoint(
    filepath: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    load model checkpoint

    Args:
        filepath: path to checkpoint file
        model: model to load state into
        optimizer: optional optimizer to load state into
        device: device to load checkpoint on

    Returns:
        dictionary containing epoch, loss, and metrics
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "loss": checkpoint.get("loss", float("inf")),
        "metrics": checkpoint.get("metrics", {}),
    }

