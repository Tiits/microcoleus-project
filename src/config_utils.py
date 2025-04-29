# src/config_utils.py
import yaml

def load_config(path='configs/config_baseline.yaml'):
    """
    Charge et retourne la configuration YAML.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)
