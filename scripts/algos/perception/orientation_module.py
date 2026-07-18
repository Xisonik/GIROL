"""Orientation feature modules.

Boundary schema:
    OrientationFeature wraps an optional OrientationModule.
        mode='gt'   : state.orientation -> [B, 1]
        mode='pred' : OrientationModule(img, graph_emb) -> predicted angle [B, 1]

Only this file knows orientation modes, angle bins, losses, and optional metrics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

GRAPH_EMB_DIM = 128
NUM_ORIENT_BINS = 36

_eval_gt_angles: list[torch.Tensor] = []
_eval_pred_angles: list[torch.Tensor] = []
_eval_step_counter = 0


def _as_column(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(-1)
    if x.dim() == 2 and x.shape[-1] == 1:
        return x
    raise ValueError(f"Expected orientation tensor [B] or [B,1], got {tuple(x.shape)}")


def collect_orientation_data(gt: torch.Tensor, pred: torch.Tensor) -> None:
    global _eval_step_counter
    _eval_gt_angles.append(_as_column(gt).detach().cpu())
    _eval_pred_angles.append(_as_column(pred).detach().cpu())
    _eval_step_counter += 1


def print_orientation_accuracy(peep: bool = False):
    global _eval_step_counter
    if not _eval_gt_angles:
        return None

    gt = torch.cat(_eval_gt_angles, dim=0)
    pred = torch.cat(_eval_pred_angles, dim=0)
    err = torch.atan2(torch.sin(gt - pred), torch.cos(gt - pred)).abs()
    metrics = (
        (err < 10.0 * torch.pi / 180.0).float().mean().item(),
        (err < 20.0 * torch.pi / 180.0).float().mean().item(),
        (err < 30.0 * torch.pi / 180.0).float().mean().item(),
    )
    if not peep:
        _eval_gt_angles.clear()
        _eval_pred_angles.clear()
        _eval_step_counter = 0
    return metrics


class OrientationModule(nn.Module):
    """Learned predictor: img + graph_emb -> pred_angle, probs, logits."""

    def __init__(self, img_dim: int, graph_emb_dim: int = GRAPH_EMB_DIM, num_bins: int = NUM_ORIENT_BINS):
        super().__init__()
        self.num_bins = int(num_bins)
        self.net = nn.Sequential(
            nn.Linear(int(img_dim) + int(graph_emb_dim), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_bins),
        )

        bin_size = 2 * torch.pi / self.num_bins
        centers = torch.linspace(-torch.pi, torch.pi, self.num_bins + 1)[:-1] + bin_size / 2
        self.register_buffer("bin_centers", centers)

    def forward(self, img: torch.Tensor, graph_emb: torch.Tensor):
        logits = self.net(torch.cat([img, graph_emb], dim=-1))
        probs = F.softmax(logits, dim=-1)
        pred_angle = self.bin_centers[probs.argmax(-1)].unsqueeze(-1)
        return pred_angle, probs, logits

    def compute_loss(self, logits: torch.Tensor, probs: torch.Tensor, gt_yaw: torch.Tensor):
        gt_yaw = _as_column(gt_yaw).squeeze(-1)
        gt_norm = torch.atan2(torch.sin(gt_yaw), torch.cos(gt_yaw))
        bin_size = 2 * torch.pi / self.num_bins
        labels = ((gt_norm + torch.pi) / bin_size).long().clamp(0, self.num_bins - 1)

        loss = F.cross_entropy(logits, labels, label_smoothing=0.05)

        with torch.no_grad():
            pred_bins = logits.argmax(-1)
            bin_dist = torch.abs(pred_bins - labels)
            bin_dist = torch.minimum(bin_dist, self.num_bins - bin_dist)
            pred_angles = self.bin_centers[pred_bins]
            ang_err = torch.atan2(torch.sin(gt_norm - pred_angles), torch.cos(gt_norm - pred_angles))
            metrics = {
                "orient/loss": loss.item(),
                "orient/acc_relaxed": (bin_dist <= 1).float().mean().item(),
                "orient/acc_strict": (pred_bins == labels).float().mean().item(),
                "orient/mean_error_deg": (ang_err.abs().mean() * 180 / torch.pi).item(),
                "orient/confidence": probs.max(-1)[0].mean().item(),
            }
        return loss, metrics


class OrientationFeature(nn.Module):
    """Wrapper for GT or predicted yaw features."""

    def __init__(
        self,
        mode: str = "gt",
        img_dim: int | None = None,
        graph_emb_dim: int = GRAPH_EMB_DIM,
        num_bins: int = NUM_ORIENT_BINS,
        predictor: OrientationModule | None = None,
        force_predictor: bool = False,
        log_metrics: bool = False,
    ):
        super().__init__()
        if mode not in {"gt", "pred"}:
            raise ValueError(f"OrientationFeature mode must be 'gt' or 'pred', got {mode!r}")
        self.mode = mode
        self.log_metrics = bool(log_metrics)

        if predictor is None and (mode == "pred" or force_predictor):
            if img_dim is None:
                raise ValueError("img_dim is required when OrientationFeature creates OrientationModule")
            predictor = OrientationModule(img_dim=int(img_dim), graph_emb_dim=int(graph_emb_dim), num_bins=int(num_bins))
        self.predictor = predictor

    def forward(self, gt_yaw: torch.Tensor, img: torch.Tensor | None = None, graph_emb: torch.Tensor | None = None) -> torch.Tensor:
        gt_yaw = _as_column(gt_yaw)
        if self.mode == "gt":
            return gt_yaw

        if self.predictor is None:
            raise RuntimeError("OrientationFeature(mode='pred') has no predictor")
        if img is None or graph_emb is None:
            raise RuntimeError("OrientationFeature(mode='pred') expects gt_yaw, img, graph_emb")

        pred_angle, _, _ = self.predictor(img, graph_emb)
        if self.log_metrics:
            collect_orientation_data(gt_yaw, pred_angle)
        return pred_angle

    def predict(self, img: torch.Tensor, graph_emb: torch.Tensor):
        if self.predictor is None:
            raise RuntimeError("OrientationFeature has no predictor")
        return self.predictor(img, graph_emb)

    def compute_loss(self, logits: torch.Tensor, probs: torch.Tensor, gt_yaw: torch.Tensor):
        if self.predictor is None:
            raise RuntimeError("OrientationFeature has no predictor")
        return self.predictor.compute_loss(logits, probs, gt_yaw)
