import functools
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from .protocol import (
    VERSION,
    GROUPS,
    PHASES,
    DEPENDENCIES,
    atomic,
    read,
    write_csv,
    pairs,
    load_or_create,
    validate_config,
)

STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def save_pt(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def cpu_load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state(sampler=None):
    state = dict(
        python=random.getstate(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()
    )
    if sampler:
        state["sampler"] = sampler.getstate()
    return state


def restore_rng(state, sampler=None):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    if sampler:
        sampler.setstate(state["sampler"])


class Run:
    def __init__(self, config, enable_gpu=False):
        validate_config(config)
        self.config = config
        self.root = Path(config["output"])
        self.enable_gpu = enable_gpu
        self.device = torch.device(config.get("device", "cuda"))
        self.seed = config["seed"]
        self.manifest = load_or_create(config)
        self.groups = self.manifest["groups"]
        self.core = None

    def imports(self):
        from . import evidence

        return evidence

    def interrupted(self):
        if STOP or (self.root / "STOP").exists():
            raise InterruptedError("Runner stop requested; latest complete epoch is resumable")

    def gpu_guard(self):
        self.interrupted()
        if self.device.type == "cuda" and (not torch.cuda.is_available()):
            raise RuntimeError("CUDA is unavailable; select a compatible PyTorch installation")

    def status(self, phase, **values):
        row = dict(
            phase=phase,
            dataset=self.config["dataset"],
            seed=self.seed,
            pid=os.getpid(),
            unix=time.time(),
            **values,
        )
        atomic(self.root / phase / "status.json", row)
        atomic(self.root / "status.json", row)
        print(json.dumps(row), flush=True)

    def stage_config(self, phase, **values):
        folder = self.root / phase
        folder.mkdir(parents=True, exist_ok=True)
        recipe = dict(
            algorithm=VERSION,
            dataset=self.config["dataset"],
            seed=self.seed,
            dependencies=list(DEPENDENCIES[phase]),
            base=self.config["base_checkpoint"],
            global_side=352,
            crop_side=224,
            regions=9,
            scales=[1.0, 1.5],
            max_additional_encoder_calls=1,
            **values,
        )
        atomic(folder / "config.json", recipe)
        return folder

    def complete(self, phase, **values):
        folder = self.root / phase
        if not (folder / "history.jsonl").exists():
            (folder / "history.jsonl").write_text("", encoding="utf-8")
        atomic(
            folder / "COMPLETE.json",
            dict(
                completed=True,
                phase=phase,
                dataset=self.config["dataset"],
                seed=self.seed,
                unix=time.time(),
                **values,
            ),
        )
        self.status(phase, completed=True, **values)

    def dependencies(self, phase):
        for dependency in DEPENDENCIES[phase]:
            path = self.root / dependency / "COMPLETE.json"
            if not path.exists() or not read(path).get("completed"):
                raise RuntimeError(f"Dependency incomplete: {dependency}")

    def base_check(self):
        pack = cpu_load(self.config["base_checkpoint"])
        state = pack.get("decoder")
        if state is None:
            state = {
                k.removeprefix("decoder."): v
                for k, v in pack["state"].items()
                if k.startswith("decoder.")
            }
        from .models.dsr import MixedTokenDecoder

        decoder = MixedTokenDecoder(384, 128, "single", 16)
        decoder.load_state_dict(state, strict=True)
        expected = set(sum(self.groups.values(), []))
        names = read(self.config["base_training_names"])
        if set(names) != expected:
            raise ValueError("DSR checkpoint and training split differ")
        return (state, dict(selected_epoch=pack["epoch"], mode="single", width=128))

    def base(self):
        self.gpu_guard()
        state, _ = self.base_check()
        from .models.dsr import MixedTokenSegmenter

        model = MixedTokenSegmenter(
            self.config["encoder_repo"],
            self.config["encoder_weights"],
            "dinov3_vits16",
            "single",
            128,
        )
        model.decoder.load_state_dict(state, strict=True)
        return model.requires_grad_(False).to(self.device).eval()

    def source(self, split):
        return {p.name: (p, m) for p, m in pairs(self.config["data"], split)}

    @staticmethod
    def rgb(path):
        with Image.open(path) as image:
            return image.convert("RGB")

    @staticmethod
    def mask(path):
        with Image.open(path) as image:
            return torch.from_numpy(
                (np.array(image.convert("RGB")).max(-1) > 127).astype(np.uint8)
            )[None]

    @functools.lru_cache(maxsize=12)
    def packet(self, name):
        return cpu_load(self.root / "cache" / "train" / (name + ".pt"))

    def load_head(self):
        path = self.component_paths()["executor"]
        head = self.imports().CropRefiner().to(self.device)
        head.load_state_dict(cpu_load(path)["state"], strict=True)
        return head.eval()

    def load_critic(self):
        pack = cpu_load(self.component_paths()["critic"])
        model = self.imports().SupportCritic(pack["class_weights"]).to(self.device)
        model.load_state_dict(pack["state"], strict=True)
        return model.eval()

    def component_paths(self):
        return {k: self.root / k / "best.pt" for k in ("executor", "critic", "policy")} | {
            "base": Path(self.config["base_checkpoint"])
        }

    @torch.no_grad()
    def cache(self):
        core = self.imports()
        folder = self.stage_config(
            "cache",
            split="train",
            preserves_original_source=True,
            candidate_encoding_is_training_only=True,
            view_batch=18,
        )
        model = self.base()
        source = self.source("train")
        for index, name in enumerate(sum([self.groups[k] for k in GROUPS], [])):
            self.interrupted()
            output = folder / "train" / (name + ".pt")
            if output.exists():
                continue
            if index % 20 == 0:
                self.gpu_guard()
                self.status("cache", completed_images=index, total=len(source))
            image_path, target_path = source[name]
            rgb = self.rgb(image_path)
            target = self.mask(target_path)
            if target.shape[-2:] != (rgb.height, rgb.width):
                raise ValueError("Native image and mask dimensions differ: " + name)
            small = rgb.resize((352, 352), Image.Resampling.BILINEAR)
            boxes, views = core.geometry(*rgb.size)
            crops = []
            for view in views.tolist():
                bounds = tuple((round(a * s) for a, s in zip(view, (rgb.width, rgb.height) * 2)))
                crops.append(rgb.crop(bounds).resize((224, 224), Image.Resampling.BILINEAR))
            initial = model(core.image_tensor(small)[None].to(self.device))
            encoded, _ = model.encoder(
                torch.stack([core.image_tensor(im) for im in crops]).to(self.device)
            )
            observed = model.decoder.fusion(
                torch.cat([p(f) for p, f in zip(model.decoder.projections, encoded)], 1)
            )
            p = dict(
                name=name,
                source=str(image_path),
                native_size=(rgb.height, rgb.width),
                mask=target,
                boxes=boxes,
                views=views,
                small_rgb=torch.from_numpy(np.array(small).transpose(2, 0, 1).copy()),
                state=initial["dense_state"][0].float().cpu(),
                logits=initial["logits"][0].float().cpu(),
                view_state=observed.half().cpu(),
                view_rgb=torch.stack(
                    [torch.from_numpy(np.array(im).transpose(2, 0, 1).copy()) for im in crops]
                ),
            )
            d, entropy = core.policy_features(p, self.device)
            p.update(descriptors=d.cpu(), entropy=entropy.cpu())
            save_pt(output, p)
        self.complete("cache", images=len(source), split="train", test_labels_read=False)

    def resume(self, folder, model, optimizer, schedule=None, sampler=None):
        path = folder / "last.pt"
        if not path.exists():
            return (1, -float("inf"))
        pack = cpu_load(path)
        model.load_state_dict(pack["state"])
        optimizer.load_state_dict(pack["optimizer"])
        if schedule:
            schedule.load_state_dict(pack["scheduler"])
        restore_rng(pack["rng"], sampler)
        return (pack["epoch"] + 1, pack["best_score"])

    def epoch_end(
        self,
        folder,
        model,
        optimizer,
        epoch,
        score,
        best,
        schedule=None,
        sampler=None,
        extra=None,
        record=None,
    ):
        improved = score > best
        best = max(best, score)
        pack = dict(
            state=model.state_dict(),
            epoch=epoch,
            seed=self.seed,
            selection_score=score,
            **extra or {},
        )
        if improved:
            save_pt(folder / "best.pt", pack)
        if schedule:
            schedule.step()
        save_pt(
            folder / "last.pt",
            dict(
                pack,
                optimizer=optimizer.state_dict(),
                scheduler=schedule.state_dict() if schedule else None,
                rng=rng_state(sampler),
                best_score=best,
            ),
        )
        with (folder / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record or dict(epoch=epoch, score=score)) + "\n")
        return best

    @torch.no_grad()
    def executor_selection(self, head):
        core = self.imports()
        head.eval()
        scores = []
        for index, name in enumerate(self.groups["executor_selection"]):
            self.interrupted()
            p = self.packet(name)
            action = 2 * int(p["entropy"].argmax()) + index % 2
            inputs, boxes = core.prepare_inputs(p, [action], self.device)
            logits = core.corrected(p, head(*inputs), boxes, self.device)
            scores.append(float(core.dice_each(logits, p["mask"][None].to(self.device))))
        return float(np.mean(scores))

    def executor(self):
        self.gpu_guard()
        settings = self.config["training"]["executor"]
        core = self.imports()
        folder = self.stage_config(
            "executor",
            epochs=settings["epochs"],
            lr=settings["lr"],
            weight_decay=settings["weight_decay"],
            batch=settings["batch_size"],
            augmentation="none",
            selection="internal fixed entropy-region Dice",
            loss="native weighted BCE + weighted IoU",
        )
        seed_all(self.seed)
        sampler = random.Random(self.seed)
        head = core.CropRefiner().to(self.device)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
        )
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, settings["epochs"])
        first, best = self.resume(folder, head, optimizer, schedule, sampler)
        for epoch in range(first, settings["epochs"] + 1):
            self.gpu_guard()
            head.train()
            names = list(self.groups["executor_fit"])
            sampler.shuffle(names)
            total = 0.0
            for start in range(0, len(names), settings["batch_size"]):
                self.interrupted()
                arguments, boxes_list, packets = ([], [], [])
                for name in names[start : start + settings["batch_size"]]:
                    p = self.packet(name)
                    r = (
                        sampler.choice(p["entropy"].topk(3).indices.tolist())
                        if sampler.random() < 0.75
                        else sampler.randrange(9)
                    )
                    inputs, boxes = core.prepare_inputs(
                        p, [2 * r + sampler.randrange(2)], self.device
                    )
                    arguments.append(inputs)
                    boxes_list.append(boxes)
                    packets.append(p)
                joined = [torch.cat([a[j] for a in arguments]) for j in range(8)]
                optimizer.zero_grad(set_to_none=True)
                delta = head(*joined)
                terms = [
                    core.loss_each(
                        core.corrected(p, delta[i : i + 1], boxes_list[i], self.device),
                        p["mask"][None].float().to(self.device),
                    ).mean()
                    for i, p in enumerate(packets)
                ]
                loss = torch.stack(terms).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite SER loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(packets)
            score = self.executor_selection(head)
            best = self.epoch_end(
                folder,
                head,
                optimizer,
                epoch,
                score,
                best,
                schedule,
                sampler,
                record=dict(epoch=epoch, fit_loss=total / len(names), selection_dice=score),
            )
            self.status("executor", epoch=epoch, selection_dice=score)
        self.complete(
            "executor",
            epochs=settings["epochs"],
            selected_epoch=cpu_load(folder / "best.pt")["epoch"],
        )

    def critic(self):
        self.gpu_guard()
        settings = self.config["training"]["critic"]
        core = self.imports()
        folder = self.stage_config(
            "critic",
            epochs=settings["epochs"],
            lr=settings["lr"],
            weight_decay=settings["weight_decay"],
            batch=settings["batch_size"],
            loss="weighted conditional Bernoulli",
            selection="internal conditional loss",
        )
        seed_all(self.seed)
        fit = self.groups["executor_fit"]
        selected = self.groups["executor_selection"]
        data = {}
        with torch.no_grad():
            for index, name in enumerate(fit + selected):
                if index % 100 == 0:
                    self.gpu_guard()
                p = self.packet(name)
                data[name] = (
                    core.support_input(p, self.device),
                    core.defect_target(p, self.device),
                )
        freq = torch.cat([data[n][1] for n in fit]).mean((0, 2, 3))
        weights = freq.clamp_min(1e-05).rsqrt()
        weights /= weights.mean()
        exposure = torch.cat([data[n][0][:, -1:] for n in fit]).mean()
        if not 0 < float(exposure) < 1:
            raise ValueError(
                "All fitting predictions have one support class; original conditional-rate initialization is undefined"
            )
        rates = (freq[1:] / torch.stack((exposure, 1 - exposure))).clamp(1e-06, 1 - 1e-06)
        model = core.SupportCritic(weights).to(self.device)
        with torch.no_grad():
            model.readout.bias.copy_(torch.logit(rates) + (weights[1:] / weights[0]).log())
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
        )
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, settings["epochs"])
        first, best = self.resume(folder, model, optimizer, schedule)
        for epoch in range(first, settings["epochs"] + 1):
            self.gpu_guard()
            model.train()
            total = 0.0
            for ids in torch.randperm(len(fit)).split(settings["batch_size"]):
                self.interrupted()
                items = [data[fit[int(i)]] for i in ids]
                x = torch.cat([item[0] for item in items])
                y = torch.cat([item[1] for item in items])
                optimizer.zero_grad(set_to_none=True)
                loss = core.conditional_loss(model(x), y, x[:, -1:], weights)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite critic loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(items)
            model.eval()
            with torch.no_grad():
                score = -float(
                    np.mean(
                        [
                            float(
                                core.conditional_loss(
                                    model(data[n][0]), data[n][1], data[n][0][:, -1:], weights
                                )
                            )
                            for n in selected
                        ]
                    )
                )
            best = self.epoch_end(
                folder,
                model,
                optimizer,
                epoch,
                score,
                best,
                schedule,
                extra=dict(class_weights=weights.cpu(), class_frequency=freq.cpu()),
                record=dict(epoch=epoch, fit_loss=total / len(fit), selection_loss=-score),
            )
            self.status("critic", epoch=epoch, selection_loss=-score)
        self.complete(
            "critic",
            epochs=settings["epochs"],
            selected_epoch=cpu_load(folder / "best.pt")["epoch"],
        )

    @torch.no_grad()
    def value_rows(self, names, head, critic):
        core = self.imports()
        folder = self.root / "policy" / "outcomes"
        rows = []
        for index, name in enumerate(names):
            output = folder / (name + ".pt")
            if output.exists():
                rows.append(cpu_load(output))
                continue
            if index % 20 == 0:
                self.gpu_guard()
                self.status("policy", phase_detail="outcome collection", completed_images=index)
            p = self.packet(name)
            descriptors = core.features(p, self.device, critic)
            target = p["mask"][None].to(self.device)
            initial = F.interpolate(
                p["logits"][None].float().to(self.device),
                target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            before = float(core.dice_each(initial, target))
            gains = []
            for start in range(0, 18, 3):
                self.interrupted()
                args, boxes = core.prepare_inputs(p, list(range(start, start + 3)), self.device)
                gains.extend(
                    (
                        core.dice_each(core.corrected(p, head(*args), boxes, self.device), target)
                        - before
                    )
                    .cpu()
                    .tolist()
                )
            row = dict(
                name=name, features=descriptors.cpu(), dice_gain=torch.tensor(gains).reshape(9, 2)
            )
            save_pt(output, row)
            rows.append(row)
        return rows

    def policy(self):
        self.gpu_guard()
        settings = self.config["training"]["policy"]
        core = self.imports()
        folder = self.stage_config(
            "policy",
            epochs=settings["epochs"],
            lr=settings["lr"],
            weight_decay=settings["weight_decay"],
            batch=settings["batch_size"],
            scheduler=None,
            target="direct native binary Dice gain",
            calibration_images=len(self.groups["planner_calibration"]),
        )
        head, critic = (self.load_head(), self.load_critic())
        fit = self.value_rows(self.groups["planner_fit"], head, critic)
        cal = self.value_rows(self.groups["planner_calibration"], head, critic)
        x = torch.cat([r["features"] for r in fit]).to(self.device)
        y = torch.cat([r["dice_gain"] for r in fit]).to(self.device)
        xc = torch.cat([r["features"] for r in cal]).to(self.device)
        yc = torch.cat([r["dice_gain"] for r in cal]).to(self.device)
        mean, std = (x.mean(0), x.std(0).clamp_min(0.001))
        x = (x - mean) / std
        seed_all(self.seed)
        model = core.GainPlanner(446).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
        )
        first, _ = self.resume(folder, model, optimizer)
        for epoch in range(first, settings["epochs"] + 1):
            self.interrupted()
            if epoch % 20 == 1:
                self.gpu_guard()
            model.train()
            total = 0.0
            for ids in torch.randperm(len(x), device=self.device).split(settings["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(model(x[ids]) * 100, y[ids] * 100)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite planner loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(ids)
            save_pt(
                folder / "last.pt",
                dict(
                    state=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=None,
                    epoch=epoch,
                    rng=rng_state(),
                    best_score=0.0,
                ),
            )
            with (folder / "history.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(epoch=epoch, fit_loss=total / len(x))) + "\n")
            if epoch % 20 == 0:
                self.status("policy", epoch=epoch, fit_loss=total / len(x))
        model.eval()
        with torch.no_grad():
            prediction = model((xc - mean) / std)
            bias = (yc - prediction).mean(0)
            threshold = max(
                1e-06, float((prediction + bias).reshape(len(cal), 18).max(1).values.median())
            )
        save_pt(
            folder / "best.pt",
            dict(
                state=model.state_dict(),
                epoch=settings["epochs"],
                seed=self.seed,
                mean=mean.cpu(),
                std=std.cpu(),
                bias=bias.cpu(),
                threshold=threshold,
                dimensions=446,
                kind="support",
                mode="grid_local",
            ),
        )
        self.complete(
            "policy", epochs=settings["epochs"], threshold=threshold, bias=bias.cpu().tolist()
        )

    def freeze(self):
        folder = self.stage_config("freeze", official_test_used_for_selection=False)
        _, base_metadata = self.base_check()
        sources = {}
        for key, path in self.component_paths().items():
            stat = path.stat()
            pack = cpu_load(path)
            sources[key] = dict(
                path=str(path),
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                selected_epoch=pack.get("epoch", 200 if key == "policy" else None),
            )
        atomic(
            folder / "FROZEN.json",
            dict(
                algorithm=VERSION,
                sources=sources,
                base=base_metadata,
                selection="base val-selected; downstream training-internal selection/calibration",
                test_labels_read=False,
                no_new_model_choice_after_test=True,
            ),
        )
        self.complete("freeze", test_labels_read=False)

    def frozen_check(self):
        frozen = read(self.root / "freeze" / "FROZEN.json")
        for value in frozen["sources"].values():
            stat = Path(value["path"]).stat()
            if stat.st_size != value["size"] or stat.st_mtime_ns != value["mtime_ns"]:
                raise RuntimeError("Frozen component changed before evaluation: " + value["path"])

    @torch.no_grad()
    def infer(self, rgb, base, head, critic, planner, pack, force_one=False):
        core = self.imports()
        small = rgb.resize((352, 352), Image.Resampling.BILINEAR)
        image = core.image_tensor(small)[None].to(self.device)
        initial = base(image)
        boxes, views = core.geometry(*rgb.size)
        p = dict(
            state=initial["dense_state"][0],
            logits=initial["logits"][0],
            boxes=boxes,
            views=views,
            native_size=(rgb.height, rgb.width),
            small_rgb=torch.from_numpy(np.array(small).transpose(2, 0, 1).copy()),
        )
        p["descriptors"], _ = core.policy_features(p, self.device)
        features = core.features(p, self.device, critic)
        gains = planner(
            (features - pack["mean"].to(self.device)) / pack["std"].to(self.device)
        ) + pack["bias"].to(self.device)
        value, action = gains.flatten().max(0)
        initial_native = F.interpolate(
            initial["logits"].float(), (rgb.height, rgb.width), mode="bilinear", align_corners=False
        )
        observe = float(value) > pack["threshold"]
        output = dict(
            base_logits=initial_native,
            logits=initial_native,
            action=-1,
            chosen_action=int(action),
            observe=observe,
            predicted_gain=float(value),
            extra_encoder_calls=0,
        )
        if not (force_one or observe):
            return output
        action = int(action)
        viewbox = views[action].tolist()
        bounds = tuple((round(v * s) for v, s in zip(viewbox, (rgb.width, rgb.height) * 2)))
        crop = core.image_tensor(rgb.crop(bounds).resize((224, 224), Image.Resampling.BILINEAR))[
            None
        ].to(self.device)
        encoded, _ = base.encoder(crop)
        view = (
            base.decoder.fusion(
                torch.cat([p(f) for p, f in zip(base.decoder.projections, encoded)], 1)
            )
            .half()
            .float()
        )
        core_box, view_box = (
            boxes[action // 2 : action // 2 + 1].to(self.device),
            views[action : action + 1].to(self.device),
        )
        rel = core.relative(core_box, view_box)
        args = (
            view,
            core.roi_sample(initial["logits"].sigmoid(), view_box, 14),
            core.roi_sample(initial["dense_state"], core_box, 14),
            core.roi_sample(initial["logits"], core_box, 224),
            core.roi_sample(crop, rel, 224),
            rel,
            core.roi_sample(initial["dense_state"], view_box, 14),
            core.roi_sample(core.roi_sample(image, view_box, 224), rel, 224),
        )
        delta = head(*args)
        output.update(
            logits=initial_native
            + core.paste_roi(delta, core_box, (rgb.height, rgb.width), (1.0, 1.0)),
            action=action,
            extra_encoder_calls=1,
        )
        return output

    @torch.no_grad()
    def evaluate(self, split):
        phase = "evaluate-" + split
        self.gpu_guard()
        self.frozen_check()
        folder = self.stage_config(
            phase,
            split=split,
            checkpoint_selection=False,
            latency_measured=False,
            metric_space="native pixels",
        )
        from .metrics import metrics

        base, head, critic = (self.base(), self.load_head(), self.load_critic())
        pack = cpu_load(self.component_paths()["policy"])
        planner = self.imports().GainPlanner(446).to(self.device)
        planner.load_state_dict(pack["state"])
        planner.eval()
        source = self.source(split)
        fixed = set(self.manifest["fixed_visual_cases"][split])
        audit = read(self.config["data_audit"]) if self.config.get("data_audit") else None
        signatures = (
            {r["name"]: r for r in audit["datasets"][self.config["dataset"]][split]}
            if audit
            else {}
        )
        rows = []
        for index, (name, (image_path, target_path)) in enumerate(source.items()):
            self.interrupted()
            if index % 20 == 0:
                self.gpu_guard()
                self.status(phase, completed_images=index, total=len(source))
            item = folder / "per-image" / (name + ".json")
            if item.exists():
                rows.extend(read(item))
                continue
            rgb = self.rgb(image_path)
            result = self.infer(rgb, base, head, critic, planner, pack, force_one=True)
            target = self.mask(target_path)[0].numpy().astype(bool)
            base_mask = (result["base_logits"][0, 0] >= 0).cpu().numpy()
            one_mask = (result["logits"][0, 0] >= 0).cpu().numpy()
            masks = dict(
                base=base_mask, one=one_mask, stop=one_mask if result["observe"] else base_mask
            )
            image_rows = []
            for policy, prediction in masks.items():
                observe = policy == "one" or (policy == "stop" and result["observe"])
                meta = signatures.get(name, {})
                row = dict(
                    dataset=self.config["dataset"],
                    seed=self.seed,
                    split=split,
                    name=name,
                    policy=policy,
                    action=result["chosen_action"] if observe else -1,
                    extra_encoder_calls=int(observe),
                    cluster=meta.get("cluster"),
                    exclude_clean_test=bool(meta.get("exclude_clean_test", False)),
                    **metrics(prediction, target),
                )
                image_rows.append(row)
                dest = folder / "predictions" / policy
                dest.mkdir(parents=True, exist_ok=True)
                Image.fromarray(prediction.astype(np.uint8) * 255).save(
                    dest / (Path(name).stem + ".png")
                )
            atomic(item, image_rows)
            rows.extend(image_rows)
        write_csv(folder / "per-image.csv", rows)
        summary = []
        for policy in ("base", "one", "stop"):
            for scope in ("raw", "clean"):
                values = [
                    r
                    for r in rows
                    if r["policy"] == policy
                    and (not (scope == "clean" and split == "test" and r["exclude_clean_test"]))
                ]
                summary.append(
                    dict(
                        dataset=self.config["dataset"],
                        seed=self.seed,
                        split=split,
                        policy=policy,
                        scope=scope,
                        images=len(values),
                        dice_percent=100 * float(np.mean([r["dice"] for r in values]))
                        if values
                        else None,
                        hd95_pixels=float(np.mean([r["hd95_pixels"] for r in values]))
                        if values
                        else None,
                        call_fraction=float(np.mean([r["extra_encoder_calls"] for r in values]))
                        if values
                        else None,
                    )
                )
        write_csv(folder / "summary.csv", summary)
        self.complete(
            phase,
            images=len(source),
            official_test_labels_read=split == "test",
            test_used_for_selection=False,
            latency_measured=False,
            fixed_visual_cases=sorted(fixed),
            all_predictions_saved=True,
        )

    def execute(self, phase):
        if (self.root / phase / "COMPLETE.json").exists():
            print(json.dumps(dict(phase=phase, already_complete=True)), flush=True)
            return
        self.dependencies(phase)
        if phase.startswith("evaluate-"):
            self.evaluate(phase.split("-", 1)[1])
        else:
            getattr(self, phase)()
        self.packet.cache_clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def overall_complete(self):
        missing = [phase for phase in PHASES if not (self.root / phase / "COMPLETE.json").exists()]
        if not missing:
            atomic(
                self.root / "COMPLETE.json",
                dict(
                    completed=True,
                    algorithm=VERSION,
                    dataset=self.config["dataset"],
                    seed=self.seed,
                    phases=list(PHASES),
                    official_test_used_for_selection=False,
                ),
            )
