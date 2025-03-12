import yaml
import os

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "settings.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

config = load_config()