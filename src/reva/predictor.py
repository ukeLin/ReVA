from pathlib import Path

import torch
from torch import nn

from . import evidence
from .engine import Run, cpu_load
from .models.dsr import MixedTokenSegmenter


class ReVA(nn.Module):
    infer = Run.infer

    def __init__(self, run, encoder_weights, encoder_repo="", device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        run = Path(run)
        self.segmenter = MixedTokenSegmenter(encoder_repo, encoder_weights, mode="single")
        checkpoint = cpu_load(run / "dsr" / "best.pt")
        state = checkpoint.get("decoder")
        if state is None:
            state = {k.removeprefix("decoder."): v for k, v in checkpoint["state"].items()}
        self.segmenter.decoder.load_state_dict(state, strict=True)
        self.refiner = evidence.CropRefiner()
        self.refiner.load_state_dict(cpu_load(run / "executor" / "best.pt")["state"])
        critic = cpu_load(run / "critic" / "best.pt")
        self.critic = evidence.SupportCritic(critic["class_weights"])
        self.critic.load_state_dict(critic["state"])
        self.policy_pack = cpu_load(run / "policy" / "best.pt")
        self.planner = evidence.GainPlanner(446)
        self.planner.load_state_dict(self.policy_pack["state"])
        self.to(self.device).requires_grad_(False).eval()

    def imports(self):
        return evidence

    @torch.no_grad()
    def predict_image(self, image):
        return self.infer(
            image.convert("RGB"),
            self.segmenter,
            self.refiner,
            self.critic,
            self.planner,
            self.policy_pack,
        )
