import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import SegmentationDataset
from .engine import cpu_load, save_pt, seed_all
from .evidence import loss_each
from .models.dsr import MixedTokenSegmenter, component_loss
from .protocol import atomic


@torch.no_grad()
def validate(encoder, decoder, loader, device):
    decoder.eval()
    scores = []
    for image, target, _ in loader:
        image, target = image.to(device), target.to(device)
        features, padded = encoder(image)
        logits = decoder(features, padded, image.shape[-2:], image)["logits"].float()
        prediction = (
            F.interpolate(logits, target.shape[-2:], mode="bilinear", align_corners=False) >= 0
        )
        target = target.bool()
        scores.append(
            float(
                (2 * (prediction & target).sum() + 1e-8) / (prediction.sum() + target.sum() + 1e-8)
            )
        )
    return float(np.mean(scores))


def train_dsr(config):
    seed_all(config["seed"])
    device = torch.device(config.get("device", "cuda"))
    settings = config["training"]["dsr"]
    output = Path(config["output"]) / "dsr"
    if (output / "COMPLETE.json").exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    model = MixedTokenSegmenter(
        config["encoder_repo"], config["encoder_weights"], mode="single"
    ).to(device)
    train = SegmentationDataset(config["data"], "train")
    val = SegmentationDataset(config["data"], "val")
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(
        train,
        batch_size=settings["batch_size"],
        shuffle=True,
        num_workers=config.get("workers", 2),
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    validation = DataLoader(val, batch_size=1, num_workers=config.get("workers", 2))
    optimizer = torch.optim.AdamW(
        model.decoder.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    ema_decay = settings.get("ema_decay", 0)
    ema = copy.deepcopy(model.decoder).requires_grad_(False).eval() if ema_decay else None
    evaluated = ema if ema is not None else model.decoder
    first, best = 1, -1.0
    resume = output / "last.pt"
    if resume.exists():
        checkpoint = cpu_load(resume)
        model.decoder.load_state_dict(checkpoint["live_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if ema is not None:
            ema.load_state_dict(checkpoint["decoder"])
        generator.set_state(checkpoint["loader_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        import random

        random.setstate(checkpoint["python_rng"])
        first, best = checkpoint["epoch"] + 1, checkpoint["best_score"]
    atomic(
        output / "config.json", settings | {"dataset": config["dataset"], "seed": config["seed"]}
    )
    atomic(output / "training_names.json", [p.name for p, _ in train.pairs])
    for epoch in range(first, settings["epochs"] + 1):
        warmup = settings.get("warmup_epochs", 0)
        if warmup and epoch <= warmup:
            rate = settings["lr"] * epoch / warmup
        elif warmup:
            rate = (
                settings["lr"]
                * 0.5
                * (1 + math.cos(math.pi * (epoch - warmup) / (settings["epochs"] - warmup)))
            )
        else:
            rate = settings["lr"] * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / settings["epochs"]))
        for group in optimizer.param_groups:
            group["lr"] = rate
        model.train()
        total, count = 0.0, 0
        for images, masks, _ in loader:
            images, masks = images.to(device), masks.to(device)
            for side in settings["scales"]:
                image = (
                    images
                    if side == 352
                    else F.interpolate(images, (side, side), mode="bilinear", align_corners=True)
                )
                target = (
                    masks
                    if side == 352
                    else F.interpolate(masks, (side, side), mode="bilinear", align_corners=True)
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    result = model(image)
                    auxiliary = component_loss(result, target)
                    loss = loss_each(result["logits"].float(), target).mean()
                    loss = loss + sum(
                        weight * term
                        for weight, term in zip(settings["auxiliary_weights"], auxiliary)
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite DSR loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
                optimizer.step()
                if ema is not None:
                    with torch.no_grad():
                        for a, b in zip(ema.parameters(), model.decoder.parameters()):
                            a.mul_(ema_decay).add_(b, alpha=1 - ema_decay)
                        for a, b in zip(ema.buffers(), model.decoder.buffers()):
                            a.copy_(b)
                total += float(loss.detach()) * len(images)
                count += len(images)
        score = validate(model.encoder, evaluated, validation, device)
        metadata = dict(
            method="own_base",
            mode="single",
            width=128,
            epochs=settings["epochs"],
            dataset=config["dataset"],
            seed=config["seed"],
        )
        pack = dict(
            decoder=evaluated.state_dict(), epoch=epoch, val={"dice": score}, config=metadata
        )
        if score > best:
            best = score
            save_pt(output / "best.pt", pack)
        import random

        save_pt(
            resume,
            pack
            | dict(
                live_state=model.decoder.state_dict(),
                optimizer=optimizer.state_dict(),
                best_score=best,
                loader_rng=generator.get_state(),
                torch_rng=torch.get_rng_state(),
                python_rng=random.getstate(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            ),
        )
        record = dict(epoch=epoch, loss=total / count, validation_dice=score, lr=rate)
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    atomic(
        output / "COMPLETE.json",
        dict(
            completed=True,
            full_training_completed=True,
            epochs=settings["epochs"],
            best_validation_dice=best,
        ),
    )
