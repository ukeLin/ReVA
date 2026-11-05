import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from reva.config import load_config
from reva.geometry import geometry, relative, roi_sample, paste_roi
from reva.metrics import metrics
from reva.models.dsr import MixedTokenDecoder
from reva.models.ovp import SupportCritic
from reva.models.ser import CropRefiner
from reva.protocol import allocation


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_geometry_covers_image_and_keeps_context_inside(self):
        boxes, views = geometry(157, 113)
        self.assertEqual(tuple(boxes.shape), (9, 4))
        self.assertEqual(tuple(views.shape), (18, 4))
        self.assertTrue(bool(((views >= 0) & (views <= 1)).all()))
        area = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).sum()
        self.assertAlmostEqual(float(area), 1.0)
        self.assertTrue(torch.allclose(views[::2], boxes))
        self.assertTrue(bool((relative(boxes, views[1::2]) >= 0).all()))

    def test_local_paste_changes_only_core(self):
        box = torch.tensor([[0.25, 0.25, 0.75, 0.75]])
        correction = paste_roi(torch.ones(1, 1, 8, 8), box, (32, 32), (1.0, 1.0))
        self.assertEqual(float(correction[:, :, :8].abs().sum()), 0.0)
        self.assertEqual(float(correction[:, :, :, 24:].abs().sum()), 0.0)
        self.assertGreater(float(correction[:, :, 10:22, 10:22].mean()), 0.99)
        self.assertEqual(tuple(roi_sample(correction, box, 8).shape), (1, 1, 8, 8))

    def test_readout_shapes_and_gradient(self):
        decoder = MixedTokenDecoder(384, mode="single")
        features = [torch.randn(1, 384, 2, 2) for _ in range(4)]
        output = decoder(features, (32, 32), (32, 32), torch.randn(1, 3, 32, 32))
        self.assertEqual(tuple(output["logits"].shape), (1, 1, 32, 32))
        self.assertTrue(torch.equal(output["component_fg"], output["component_bg"]))
        output["logits"].square().mean().backward()
        self.assertGreater(float(decoder.query.weight.grad.abs().sum()), 0.0)
        self.assertEqual(sum(p.numel() for p in decoder.parameters()), 1021026)

    def test_shared_response_zero_for_identical_evidence(self):
        head = CropRefiner()
        torch.nn.init.normal_(head.fine[-1].weight)
        view = torch.randn(1, 128, 14, 14)
        prior = torch.rand(1, 1, 14, 14)
        core = torch.randn(1, 128, 14, 14)
        base = torch.randn(1, 1, 224, 224)
        rgb = torch.randn(1, 3, 224, 224)
        box = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        result = head(view, prior, core, base, rgb, box, view, rgb)
        self.assertEqual(float(result.detach().abs().max()), 0.0)

    def test_critic_maps_respect_support(self):
        critic = SupportCritic(torch.ones(3))
        state = torch.randn(1, 134, 4, 4)
        state[:, -1] = 0
        maps = critic.probabilities(state)
        self.assertEqual(float(maps[:, 1].detach().abs().sum()), 0.0)
        self.assertTrue(torch.allclose(maps.sum(1), torch.ones(1, 4, 4)))

    def test_engine_evidence_interface(self):
        from reva import evidence
        from reva.engine import Run

        for name in [
            "geometry",
            "CropRefiner",
            "SupportCritic",
            "GainPlanner",
            "prepare_inputs",
            "features",
            "image_tensor",
            "relative",
            "roi_sample",
            "paste_roi",
        ]:
            self.assertTrue(hasattr(evidence, name), name)
        self.assertTrue(callable(Run.infer))

    def test_native_metrics_empty_and_identical_masks(self):
        mask = np.zeros((16, 32), dtype=bool)
        self.assertEqual(metrics(mask, mask)["hd95_pixels"], 0.0)
        self.assertEqual(metrics(mask, mask)["dice"], 1.0)
        other = mask.copy()
        other[4:8, 10:20] = True
        self.assertEqual(metrics(other, other)["hd95_pixels"], 0.0)
        self.assertAlmostEqual(metrics(mask, other)["hd95_pixels"], np.hypot(16, 32))

    def test_training_role_allocation(self):
        self.assertEqual(
            allocation(800),
            dict(executor_fit=600, executor_selection=40, planner_fit=128, planner_calibration=32),
        )
        self.assertEqual(sum(allocation(113).values()), 113)

    def test_paths_are_configurable(self):
        config_file = Path(__file__).resolve().parents[1] / "configs/default.yaml"
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(
                os.environ,
                {
                    "REVA_DATA_ROOT": folder,
                    "REVA_OUTPUT_DIR": folder,
                    "DINOV3_WEIGHTS": "weights.pth",
                },
            ),
        ):
            config = load_config(config_file, encoder_repo="")
            self.assertEqual(config["data"], str(Path(folder).resolve()))
            self.assertEqual(config["training"]["dsr"]["epochs"], 100)


if __name__ == "__main__":
    unittest.main()
