import os
from pathlib import Path

import yaml


def load_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def check_configs(model_dir, suffix='yaml'):
    model_dir = Path(model_dir)
    vocab_file = model_dir / f"vocab.{suffix}"
    assert os.path.exists(vocab_file), f"{vocab_file} does not exist"
    config_file = model_dir / f"config.{suffix}"
    assert os.path.exists(config_file), f"{config_file} does not exist"

    vocab = load_yaml(vocab_file)
    dictionaries = vocab.get("dictionaries", [])
    for language, dictionary in dictionaries.items():
        if dictionary is not None:
            dictionary_path = model_dir / dictionary
            assert dictionary_path.exists(), f"{dictionary_path.absolute()} does not exist"
