#!/usr/bin/env python3

#evaluation script for CoinCLIP model

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

#add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coinclip.data import CoinVibeDataset, collate_fn
from coinclip.models import CoinCLIPModel
from coinclip.evaluation.eval_pipeline import evaluate_model, save_evaluation_results
from coinclip.utils.config import load_config
from coinclip.utils.checkpointing import load_checkpoint
from coinclip.utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Evaluate CoinCLIP model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to evaluate on",
    )

    arguments = parser.parse_args()

    #load configuration
    configuration = load_config(arguments.config)

    #setup logger
    logger = setup_logger()

    #device
    device = arguments.device or configuration.training.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    #data directories
    data_directory = Path(configuration.data.data_dir)
    processed_directory = Path(configuration.data.processed_dir)

    #create dataset
    dataset = CoinVibeDataset(
        data_dir=data_directory,
        split=arguments.split,
        metadata_file=processed_directory / f"{arguments.split}_metadata.csv",
        max_text_length=configuration.data.get("max_text_length", 512),
        image_size=tuple(configuration.data.get("image_size", [224, 224])),
        augment=False,
    )

    #create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=configuration.training.get("eval_batch_size", configuration.training.batch_size),
        shuffle=False,
        num_workers=configuration.training.get("num_workers", 4),
        collate_fn=collate_fn,
    )

    logger.info(f"Evaluating on {arguments.split} split: {len(dataset)} samples")

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

    #load checkpoint
    checkpoint_path = Path(arguments.checkpoint)
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    load_checkpoint(checkpoint_path, model, device=device)

    #evaluate
    results = evaluate_model(
        model=model,
        data_loader=data_loader,
        device=device,
        logger=logger,
        return_predictions=True,
    )

    #save results
    if arguments.output:
        output_path = Path(arguments.output)
    else:
        output_path = Path("evaluation_results") / f"{arguments.split}_results.json"

    save_evaluation_results(results, output_path)
    logger.info(f"Results saved to {output_path}")

    #print summary
    print("\n" + "=" * 50)
    print(f"Evaluation Results ({arguments.split} split)")
    print("=" * 50)
    for key, value in results["metrics"].items():
        print(f"{key}: {value:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()

