import copy
import os
import re
from pathlib import Path

import yaml


def load_config(path, **overrides):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config = copy.deepcopy(config)
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    for key in ("data", "output", "encoder_repo", "encoder_weights"):
        value = os.path.expandvars(config.get(key) or "")
        if key == "encoder_repo" and (not value or re.search(r"\$\{[^}]+\}", value)):
            config[key] = ""
            continue
        if not value or re.search(r"\$\{[^}]+\}", value):
            raise ValueError(f"Set {key} through the configuration, environment or command line")
        config[key] = str(Path(value).expanduser().resolve())
    output = Path(config["output"])
    config.update(
        base_checkpoint=str(output / "dsr" / "best.pt"),
        base_complete=str(output / "dsr" / "COMPLETE.json"),
        base_training_names=str(output / "dsr" / "training_names.json"),
    )
    return config
