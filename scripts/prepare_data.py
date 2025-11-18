#!/usr/bin/env python3

#data preparation script for CoinVibe dataset

import argparse
import json
import sys
from pathlib import Path

#add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(
    raw_directory: Path,
    output_directory: Path,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_seed: int = 42,
):
    """
    prepare CoinVibe dataset from raw data

    Args:
        raw_directory: directory containing raw data
        output_directory: directory to save processed data
        test_size: proportion of data for test set
        validation_size: proportion of data for validation set (from remaining after test)
        random_seed: random seed for splitting
    """
    raw_directory = Path(raw_directory)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    #look for metadata file
    metadata_files = list(raw_directory.glob("*.csv")) + list(raw_directory.glob("*.json"))
    if not metadata_files:
        #create a sample metadata file if none exists
        print(f"No metadata file found in {raw_directory}. Creating sample metadata...")
        create_sample_metadata(raw_directory, output_directory)
        return

    metadata_file = metadata_files[0]
    print(f"Loading metadata from {metadata_file}")

    #load metadata
    if metadata_file.suffix == ".json":
        with open(metadata_file, "r") as file:
            metadata = pd.DataFrame(json.load(file))
    else:
        metadata = pd.read_csv(metadata_file)

    #ensure required columns exist
    required_columns = ["label"]  #at minimum, we need labels
    for column in required_columns:
        if column not in metadata.columns:
            if column == "label":
                #create dummy labels if missing
                metadata["label"] = 0
                print(f"Warning: 'label' column not found. Created dummy labels.")

    #split data
    #first split: train+validation vs test
    train_validation, test = train_test_split(
        metadata, test_size=test_size, random_state=random_seed, stratify=metadata.get("label")
    )

    #second split: train vs validation
    validation_size_adjusted = validation_size / (1 - test_size)  #adjust for remaining data
    train, validation = train_test_split(
        train_validation,
        test_size=validation_size_adjusted,
        random_state=random_seed,
        stratify=train_validation.get("label"),
    )

    #save splits
    train_path = output_directory / "train_metadata.csv"
    validation_path = output_directory / "val_metadata.csv"
    test_path = output_directory / "test_metadata.csv"

    train.to_csv(train_path, index=False)
    validation.to_csv(validation_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"Data split completed:")
    print(f"  Train: {len(train)} samples -> {train_path}")
    print(f"  Validation: {len(validation)} samples -> {validation_path}")
    print(f"  Test: {len(test)} samples -> {test_path}")

    #save split info
    split_info = {
        "train_size": len(train),
        "val_size": len(validation),
        "test_size": len(test),
        "total_size": len(metadata),
    }
    with open(output_directory / "split_info.json", "w") as file:
        json.dump(split_info, file, indent=2)


def create_sample_metadata(raw_directory: Path, output_directory: Path, number_of_samples: int = 100):
    """create sample metadata for testing"""
    import numpy as np

    metadata = []
    for index in range(number_of_samples):
        sample = {
            "id": index,
            "description": f"Sample memecoin {index} description",
            "social_posts": f"Social media content for coin {index}",
            "whitepaper": f"Whitepaper content for coin {index}",
            "image_path": f"images/coin_{index}.png",
            "market_cap": np.random.uniform(1000, 1000000),
            "volume_24h": np.random.uniform(100, 100000),
            "price_change_24h": np.random.uniform(-50, 50),
            "holders_count": np.random.randint(10, 10000),
            "transactions_24h": np.random.randint(10, 1000),
            "liquidity": np.random.uniform(1000, 100000),
            "label": np.random.randint(0, 2),
        }
        metadata.append(sample)

    dataframe = pd.DataFrame(metadata)
    metadata_path = raw_directory / "metadata.csv"
    dataframe.to_csv(metadata_path, index=False)
    print(f"Created sample metadata at {metadata_path}")

    #now prepare the data
    prepare_data(raw_directory, output_directory)


def main():
    parser = argparse.ArgumentParser(description="Prepare CoinVibe dataset")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed data",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for test set",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Proportion of data for validation set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting",
    )

    arguments = parser.parse_args()

    prepare_data(
        raw_directory=Path(arguments.raw_dir),
        output_directory=Path(arguments.output_dir),
        test_size=arguments.test_size,
        validation_size=arguments.val_size,
        random_seed=arguments.seed,
    )


if __name__ == "__main__":
    main()

