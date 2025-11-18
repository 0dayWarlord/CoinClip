#!/usr/bin/env python3

#training script for CoinCLIP model

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

#add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coinclip.data import CoinVibeDataset, collate_fn
from coinclip.models import CoinCLIPModel
from coinclip.training.train_loop import train_model
from coinclip.utils.config import load_config, merge_configs, parse_cli_overrides
from coinclip.utils.logging import setup_logger
from coinclip.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train CoinCLIP model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config values (key=value format)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (overrides config)",
    )

    arguments = parser.parse_args()

    #load configuration
    configuration = load_config(arguments.config)

    #parse CLI overrides
    if arguments.override:
        overrides = parse_cli_overrides(arguments.override)
        from omegaconf import OmegaConf

        override_configuration = OmegaConf.create(overrides)
        configuration = merge_configs(configuration, override_configuration)

    #override seed and device if provided
    if arguments.seed is not None:
        configuration.training.seed = arguments.seed
    if arguments.device is not None:
        configuration.training.device = arguments.device

    #set seed
    seed = configuration.training.get("seed", 42)
    set_seed(seed)

    #setup logger
    log_directory = Path(configuration.training.get("log_dir", "logs"))
    logger = setup_logger(log_file=log_directory / "training.log")

    logger.info(f"Starting training with config: {arguments.config}")
    logger.info(f"Seed: {seed}")

    #device
    device = configuration.training.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    #data directories
    data_directory = Path(configuration.data.data_dir)
    processed_directory = Path(configuration.data.processed_dir)

    #create datasets
    train_dataset = CoinVibeDataset(
        data_dir=data_directory,
        split="train",
        metadata_file=processed_directory / "train_metadata.csv",
        max_text_length=configuration.data.get("max_text_length", 512),
        image_size=tuple(configuration.data.get("image_size", [224, 224])),
        augment=configuration.data.get("augment", True),
    )

    validation_dataset = CoinVibeDataset(
        data_dir=data_directory,
        split="val",
        metadata_file=processed_directory / "val_metadata.csv",
        max_text_length=configuration.data.get("max_text_length", 512),
        image_size=tuple(configuration.data.get("image_size", [224, 224])),
        augment=False,
    )

    #create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=configuration.training.batch_size,
        shuffle=True,
        num_workers=configuration.training.get("num_workers", 4),
        collate_fn=collate_fn,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=configuration.training.batch_size,
        shuffle=False,
        num_workers=configuration.training.get("num_workers", 4),
        collate_fn=collate_fn,
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(validation_dataset)}")

    #create model
    model = CoinCLIPModel(
        text_encoder_name=configuration.model.text_encoder_name,
        text_embedding_dim=configuration.model.text_embedding_dim,
        image_encoder_name=configuration.model.image_encoder_name,
        image_embedding_dim=configuration.model.image_embedding_dim,
        numeric_input_dim=configuration.model.numeric_input_dim,
        numeric_hidden_dims=configuration.model.numeric_hidden_dims,
        numeric_embedding_dim=configuration.model.numeric_embedding_dim,
        fusion_type=configuration.model.fusion_type,
        fusion_hidden_dim=configuration.model.fusion_hidden_dim,
        fusion_output_dim=configuration.model.fusion_output_dim,
        num_classes=configuration.model.num_classes,
        use_contrastive=configuration.model.get("use_contrastive", True),
        contrastive_temperature=configuration.model.get("contrastive_temperature", 0.07),
        dropout=configuration.model.get("dropout", 0.1),
        freeze_text_encoder=configuration.model.get("freeze_text_encoder", False),
        freeze_image_encoder=configuration.model.get("freeze_image_encoder", False),
    )

    logger.info(f"Model created with {sum(parameter.numel() for parameter in model.parameters())} parameters")

    #create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=configuration.training.learning_rate,
        weight_decay=configuration.training.get("weight_decay", 0.01),
    )

    #create scheduler
    scheduler = None
    if configuration.training.get("scheduler", None) == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=configuration.training.num_epochs
        )

    #training
    checkpoint_directory = Path(configuration.training.get("checkpoint_dir", "checkpoints"))
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=validation_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=configuration.training.num_epochs,
        device=device,
        checkpoint_dir=checkpoint_directory,
        log_dir=log_directory,
        save_best_metric=configuration.training.get("save_best_metric", "val_loss"),
        logger=logger,
    )

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()

