#training loop implementation

import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from coinclip.models.coinclip_model import CoinCLIPModel
from coinclip.training.losses import CombinedLoss
from coinclip.training.metrics import compute_classification_metrics
from coinclip.utils.checkpointing import load_checkpoint, save_checkpoint
from coinclip.utils.logging import setup_logger


def train_model(
    model: CoinCLIPModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    num_epochs: int,
    device: str = "cuda",
    checkpoint_dir: Path = Path("checkpoints"),
    log_dir: Path = Path("logs"),
    save_best_metric: str = "val_loss",
    logger=None,
) -> Dict[str, list]:
    """
    train the model

    Args:
        model: model to train
        train_loader: training data loader
        val_loader: optional validation data loader
        optimizer: optimizer
        scheduler: optional learning rate scheduler
        num_epochs: number of training epochs
        device: device to train on
        checkpoint_dir: directory to save checkpoints
        log_dir: directory to save logs
        save_best_metric: metric to use for saving best model
        logger: optional logger

    Returns:
        dictionary with training history
    """
    if logger is None:
        logger = setup_logger()

    model = model.to(device)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    #initialize loss function
    loss_function = CombinedLoss(
        classification_weight=1.0,
        contrastive_weight=0.1 if model.use_contrastive else 0.0,
        contrastive_temperature=model.contrastive_temperature if model.use_contrastive else 0.07,
    )

    #training history
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_validation_metric = float("inf") if "loss" in save_best_metric else 0.0

    for epoch in range(num_epochs):
        #training phase
        model.train()
        train_loss = 0.0
        train_metrics = {"accuracy": 0.0}
        number_of_train_batches = 0

        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in train_progress_bar:
            #move to device
            text = {k: v.to(device) for k, v in batch["text"].items()}
            image = batch["image"].to(device)
            numeric = batch["numeric"].to(device)
            labels = batch["label"].to(device)

            #forward pass
            optimizer.zero_grad()
            outputs = model(text=text, image=image, numeric=numeric)

            #compute loss
            embeddings = outputs.get("contrastive_emb") if model.use_contrastive else None
            losses = loss_function(
                logits=outputs["logits"],
                labels=labels,
                embeddings=embeddings,
            )

            #backward pass
            losses["total"].backward()
            optimizer.step()

            #accumulate metrics
            train_loss += losses["total"].item()
            batch_metrics = compute_classification_metrics(
                outputs["logits"], labels, num_classes=model.classifier[-1].out_features
            )
            train_metrics["accuracy"] += batch_metrics["accuracy"]
            number_of_train_batches += 1

            #update progress bar
            train_progress_bar.set_postfix({"loss": losses["total"].item()})

        #average training metrics
        train_loss /= number_of_train_batches
        train_metrics["accuracy"] /= number_of_train_batches
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_metrics["accuracy"])

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.4f}"
        )

        #validation phase
        if val_loader is not None:
            model.eval()
            validation_loss = 0.0
            validation_metrics = {"accuracy": 0.0}
            number_of_validation_batches = 0

            all_logits = []
            all_labels = []

            with torch.no_grad():
                validation_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
                for batch in validation_progress_bar:
                    #move to device
                    text = {k: v.to(device) for k, v in batch["text"].items()}
                    image = batch["image"].to(device)
                    numeric = batch["numeric"].to(device)
                    labels = batch["label"].to(device)

                    #forward pass
                    outputs = model(text=text, image=image, numeric=numeric)

                    #compute loss
                    embeddings = outputs.get("contrastive_emb") if model.use_contrastive else None
                    losses = loss_function(
                        logits=outputs["logits"],
                        labels=labels,
                        embeddings=embeddings,
                    )

                    #accumulate
                    validation_loss += losses["total"].item()
                    all_logits.append(outputs["logits"])
                    all_labels.append(labels)
                    number_of_validation_batches += 1

            #compute validation metrics
            validation_loss /= number_of_validation_batches
            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            validation_metrics = compute_classification_metrics(
                all_logits, all_labels, num_classes=model.classifier[-1].out_features
            )

            history["val_loss"].append(validation_loss)
            history["val_accuracy"].append(validation_metrics["accuracy"])

            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - Val Loss: {validation_loss:.4f}, "
                f"Val Acc: {validation_metrics['accuracy']:.4f}, "
                f"Val F1: {validation_metrics.get('f1', 0.0):.4f}"
            )

            #save best model
            current_metric = validation_loss if "loss" in save_best_metric else validation_metrics.get(
                save_best_metric.replace("val_", ""), 0.0
            )
            is_best = (
                current_metric < best_validation_metric
                if "loss" in save_best_metric
                else current_metric > best_validation_metric
            )

            if is_best:
                best_validation_metric = current_metric
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=validation_loss,
                    metrics=validation_metrics,
                    filepath=checkpoint_dir / "checkpoint.pt",
                    is_best=True,
                )
                logger.info(f"Saved best model (metric: {save_best_metric} = {best_validation_metric:.4f})")

        #learning rate scheduling
        if scheduler is not None:
            scheduler.step()

        #save regular checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=train_loss,
            metrics=train_metrics,
            filepath=checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt",
            is_best=False,
        )

        #save training history
        with open(log_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    logger.info("Training completed!")
    return history

