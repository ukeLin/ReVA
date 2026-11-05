import csv
import json
import os
from pathlib import Path
import numpy as np

VERSION = "reva-v1"
GROUPS = ("executor_fit", "executor_selection", "planner_fit", "planner_calibration")
WEIGHTS = (75, 5, 16, 4)
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PHASES = ("cache", "executor", "critic", "policy", "freeze", "evaluate-val", "evaluate-test")
DATASETS = (
    "Kvasir",
    "ClinicDB",
    "ColonDB",
    "ETIS",
    "BKAI",
    "BUSI",
    "ISIC2017_Official",
    "ISIC2018_Official",
)
DEPENDENCIES = {
    "cache": (),
    "executor": ("cache",),
    "critic": ("cache",),
    "policy": ("executor", "critic"),
    "freeze": ("policy",),
    "evaluate-val": ("freeze",),
    "evaluate-test": ("freeze",),
}


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def allocation(n):
    counts = [n * w // 100 for w in WEIGHTS]
    order = sorted(range(4), key=lambda k: (-(n * WEIGHTS[k] % 100), k))
    for k in order[: n - sum(counts)]:
        counts[k] += 1
    return dict(zip(GROUPS, counts))


def pairs(data, split):
    folder = Path(data) / split
    images = sorted((p for p in (folder / "images").iterdir() if p.suffix.lower() in EXTENSIONS))
    mask_paths = [p for p in (folder / "masks").iterdir() if p.suffix.lower() in EXTENSIONS]
    masks = {p.stem: p for p in mask_paths}
    if not images or len({p.stem for p in images}) != len(images):
        raise ValueError(f"Empty or duplicate image stems: {folder}")
    if len(masks) != len(mask_paths) or {p.stem for p in images} != set(masks):
        raise ValueError(f"Missing/duplicate masks: {folder}")
    return [(p, masks[p.stem]) for p in images]


def validate_config(config):
    for key in (
        "dataset",
        "seed",
        "split_seed",
        "data",
        "output",
        "base_checkpoint",
        "base_complete",
        "encoder_weights",
    ):
        if key not in config:
            raise ValueError("Missing configuration key: " + key)
    if not isinstance(config["seed"], int):
        raise ValueError("seed must be an integer")


def build_manifest(config):
    validate_config(config)
    train = pairs(config["data"], "train")
    names = [p.name for p, _ in train]
    counts = allocation(len(names))
    if counts["executor_fit"] < 4 or any((counts[k] < 2 for k in GROUPS[1:])):
        raise ValueError(f"Too few examples to fit/select/calibrate: {counts}")
    order = np.random.default_rng(config["split_seed"]).permutation(len(names))
    shuffled = [names[int(i)] for i in order]
    groups = {}
    cursor = 0
    for key in GROUPS:
        groups[key] = shuffled[cursor : cursor + counts[key]]
        cursor += counts[key]
    flat = sum([groups[k] for k in GROUPS], [])
    if len(flat) != len(names) or len(set(flat)) != len(names):
        raise ValueError("Training-role partition is not disjoint/exhaustive")
    splits = {s: [p.name for p, _ in pairs(config["data"], s)] for s in ("val", "test")}
    audit = read(config["data_audit"]) if config.get("data_audit") else None
    preparation = audit.get("preparation", {}).get(config["dataset"]) if audit else None
    result = dict(
        algorithm=VERSION,
        dataset=config["dataset"],
        seed=config["seed"],
        split_seed=config["split_seed"],
        groups=groups,
        counts={k: len(v) for k, v in groups.items()},
        training_images=len(names),
        validation=splits["val"],
        test=splits["test"],
        base_checkpoint=config["base_checkpoint"],
        base_saw_full_training_partition=True,
        group_rounding="integer largest remainder; fixed group-order tie break",
        test_labels_read=False,
        val_used_for_base_checkpoint_selection=True,
        downstream_selection="training-internal executor_selection only",
        fixed_visual_cases={s: splits[s][:3] for s in splits},
    )
    if preparation:
        result["visual_cases_match_baseline_audit"] = (
            preparation.get("fixed_visual_cases") == result["fixed_visual_cases"]
        )
    return result


def load_or_create(config):
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    locked = output / "config.json"
    canonical = dict(config, algorithm=VERSION)
    if locked.exists() and read(locked) != canonical:
        raise ValueError("Run configuration changed; choose a new output directory")
    atomic(locked, canonical)
    proposed = build_manifest(config)
    path = output / "manifest.json"
    if path.exists() and read(path) != proposed:
        raise ValueError("Dataset membership or partition changed after manifest freeze")
    atomic(path, proposed)
    return proposed
