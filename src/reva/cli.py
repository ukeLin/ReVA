import argparse
from pathlib import Path

from filelock import FileLock
from PIL import Image
import numpy as np
import torch

from .config import load_config
from .protocol import PHASES


def main():
    parser = argparse.ArgumentParser(prog="reva")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["train", "evaluate", "check"]:
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--dataset")
        command.add_argument("--data-root", dest="data")
        command.add_argument("--output-dir", dest="output")
        command.add_argument("--encoder-weights")
        command.add_argument("--encoder-repo")
        command.add_argument("--seed", type=int)
        command.add_argument("--device")
        if name == "train":
            command.add_argument(
                "--stage",
                choices=["all", "dsr", "cache", "ser", "critic", "ovp", "freeze"],
                default="all",
            )
        if name == "evaluate":
            command.add_argument("--split", choices=["val", "test"], default="test")
    predict = commands.add_parser("predict")
    predict.add_argument("--run", required=True)
    predict.add_argument("--encoder-weights", required=True)
    predict.add_argument("--encoder-repo", default="")
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "predict":
        from .predictor import ReVA

        model = ReVA(args.run, args.encoder_weights, args.encoder_repo, args.device)
        with Image.open(args.input) as image:
            result = model.predict_image(image)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        mask = (result["logits"][0, 0] >= 0).cpu().numpy().astype(np.uint8) * 255
        Image.fromarray(mask).save(output)
        return
    overrides = {
        k: getattr(args, k, None)
        for k in ["data", "output", "dataset", "seed", "device", "encoder_weights", "encoder_repo"]
    }
    config = load_config(args.config, **overrides)
    from .engine import Run

    run = Run(config, enable_gpu=config.get("device", "cuda").startswith("cuda"))
    if args.command == "check":
        print(run.base_check()[1])
        return
    with FileLock(str(run.root / "run.lock"), timeout=0):
        if args.command == "evaluate":
            run.execute("evaluate-" + args.split)
            return
        if args.stage in ["all", "dsr"]:
            from .train_dsr import train_dsr

            train_dsr(config)
        phases = (
            PHASES[:5]
            if args.stage == "all"
            else (
                ()
                if args.stage == "dsr"
                else ({"ser": "executor", "ovp": "policy"}.get(args.stage, args.stage),)
            )
        )
        for phase in phases:
            run.execute(phase)
