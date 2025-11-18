#configuration loading and management

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from omegaconf import DictConfig, OmegaConf


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """
    load configuration from YAML file

    Args:
        config_path: path to YAML config file

    Returns:
        omegaConf DictConfig object
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    configuration = OmegaConf.load(config_path)
    return configuration


def merge_configs(
    base_config: DictConfig, override_config: Optional[DictConfig] = None, **kwargs
) -> DictConfig:
    """
    merge configurations with overrides

    Args:
        base_config: base configuration
        override_config: optional override configuration
        **kwargs: additional key-value overrides

    Returns:
        merged configuration
    """
    configuration = OmegaConf.create(base_config)

    if override_config:
        configuration = OmegaConf.merge(configuration, override_config)

    if kwargs:
        override_dictionary = OmegaConf.create(kwargs)
        configuration = OmegaConf.merge(configuration, override_dictionary)

    return configuration


def save_config(config: DictConfig, output_path: Union[str, Path]) -> None:
    """
    save configuration to YAML file

    Args:
        config: configuration to save
        output_path: output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_path)


def parse_cli_overrides(overrides: list[str]) -> Dict[str, Any]:
    """
    parse command-line override strings (key=value format)

    Args:
        overrides: list of override strings

    Returns:
        dictionary of parsed overrides
    """
    result = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: {override}. Expected key=value")
        key, value = override.split("=", 1)
        #try to parse as number or boolean
        try:
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif "." in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass  #keep as string
        result[key] = value
    return result

