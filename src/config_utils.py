"""
Configuration Utilities Module

This module provides utility functions for loading configuration data from YAML files.
"""
import yaml


def load_config(path='../configs/config_baseline.yaml'):
    """
    Load a YAML configuration file and return its contents as a Python dictionary.

    Parameters:
        path (str): File system path to the YAML configuration file. Defaults to '../configs/config_baseline.yaml'.

    Returns:
        dict: Parsed configuration data.

    Raises:
        FileNotFoundError: If the specified configuration file is not found.
        yaml.YAMLError: If the YAML content cannot be parsed.
    """
    # Open the YAML file in read mode and parse its contents safely.
    with open(path, 'r') as f:
        return yaml.safe_load(f)
