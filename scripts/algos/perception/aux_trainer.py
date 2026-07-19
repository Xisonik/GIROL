"""Auxiliary orientation trainer.

This trainer is intentionally specific:

    rollout states -> img, graph, orientation
    graph_encoder(graph) -> graph_emb
    orientation_module.predict(img, graph_emb) -> logits
    supervised yaw loss -> update graph_encoder + orientation_module

Actor/value networks are not touched.

Saved aux checkpoint contract:
    {
        "timestep": int,
        "graph_encoder": state_dict,
        "orientation_module": state_dict,
        "graph_optimizer": state_dict, optional,
        "orient_optimizer": state_dict, optional,
    }

TensorBoard tags:
    aux/*
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from skrl.utils.spaces.torch import unflatten_tensorized_space


class OrientationAuxTrainer:
    """Supervised orientation training from rollout memory."""

    def __init__(
        self,
        modules: dict,
        agent,
        obs_space,
        device,
        graph_ref: str = "graph_encoder",
        orientation_ref: str = "orientation_module",
        lr_graph: float = 3e-5,
        lr_orient: float = 3e-5,
        batch_size: int = 256,
        train_steps_per_call: int = 1,
        log_interval: int = 2000,
        grad_norm_clip: float = 1.0,
        save_interval: int = 1000,
        checkpoint_dir: Optional[str] = None,
        resume_from: Optional[str] = None,
        save_optimizer: bool = True,
        tensorboard_dir: Optional[str] = None,
    ):
        self.graph_encoder = modules[graph_ref]
        self.orientation_module = modules[orientation_ref]
        self.agent = agent
        self.obs_space = obs_space
        self.device = device

        self.batch_size = int(batch_size)
        self.train_steps = int(train_steps_per_call)
        self.log_interval = int(log_interval)
        self.grad_norm_clip = float(grad_norm_clip)

        self.save_interval = int(save_interval)
        self.save_optimizer = bool(save_optimizer)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.writer = None
        if tensorboard_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(str(tensorboard_dir))
            except Exception as exc:
                print(f"[AuxTrain] TensorBoard writer disabled: {exc}", flush=True)

        # Strict by design: aux sees the same normalized states as policy/value.
        # If you disable state preprocessing, use a separate aux trainer class.
        self.state_preprocessor = agent._state_preprocessor

        self.graph_optimizer = torch.optim.AdamW(
            self.graph_encoder.parameters(),
            lr=float(lr_graph),
            weight_decay=1e-4,
        )
        self.orient_optimizer = torch.optim.AdamW(
            self.orientation_module.parameters(),
            lr=float(lr_orient),
            weight_decay=1e-4,
        )

        self._metric_sum: dict[str, float] = {}
        self._metric_count = 0
        self.last_metrics: dict[str, float] = {}

        if resume_from:
            self.load(resume_from, load_optimizer=self.save_optimizer)

    def step(self, timestep: int) -> None:
        mem = self.agent.memory
        if not mem.filled and mem.memory_index < self.batch_size:
            return

        self.graph_encoder.train()
        self.orientation_module.train()

        for _ in range(self.train_steps):
            states = self._sample_states()
            _, metrics = self._train_batch(states)
            self._accumulate_metrics(metrics)

        self.graph_encoder.eval()
        self.orientation_module.eval()

        if timestep % self.log_interval == 0:
            self._print_metrics(timestep)

        if self.save_interval > 0 and timestep % self.save_interval == 0:
            self.save(timestep)

    def _sample_states(self) -> dict[str, torch.Tensor]:
        sample = self.agent.memory.sample(names=["states"], batch_size=self.batch_size)[0]
        raw_states = sample[0]

        with torch.no_grad():
            states = self.state_preprocessor(raw_states, train=False)

        return unflatten_tensorized_space(self.obs_space, states)

    def _train_batch(self, states: dict[str, torch.Tensor]):
        img = states["img"]
        graph = states["graph"]
        gt_yaw = states["orientation"]

        graph_emb = self.graph_encoder(graph)
        _, probs, logits = self.orientation_module.predict(img, graph_emb)
        loss, metrics = self.orientation_module.compute_loss(logits, probs, gt_yaw)

        self.graph_optimizer.zero_grad(set_to_none=True)
        self.orient_optimizer.zero_grad(set_to_none=True)

        loss.backward()

        graph_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.graph_encoder.parameters(),
            self.grad_norm_clip,
        )
        orient_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.orientation_module.parameters(),
            self.grad_norm_clip,
        )

        self.graph_optimizer.step()
        self.orient_optimizer.step()

        metrics = dict(metrics)
        metrics["loss"] = float(loss.detach().item())
        metrics["graph_grad_norm"] = float(graph_grad_norm)
        metrics["orient_grad_norm"] = float(orient_grad_norm)

        return loss, metrics

    def _accumulate_metrics(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self._metric_sum[key] = self._metric_sum.get(key, 0.0) + float(value)
        self._metric_count += 1

    def _print_metrics(self, timestep: int) -> None:
        if self._metric_count == 0:
            return

        n = self._metric_count
        averaged = {
            key: value / n
            for key, value in sorted(self._metric_sum.items())
        }
        self.last_metrics = averaged

        line = " | ".join(
            f"{key}: {value:.4f}"
            for key, value in averaged.items()
        )
        print(f"[AuxTrain {timestep}] {line}", flush=True)

        if self.writer is not None:
            for key, value in averaged.items():
                self.writer.add_scalar(f"aux/{key}", float(value), timestep)
            self.writer.flush()

        self._metric_sum.clear()
        self._metric_count = 0

    def state_dict(self, timestep: int = 0) -> dict:
        payload = {
            "timestep": int(timestep),
            "graph_encoder": self.graph_encoder.state_dict(),
            "orientation_module": self.orientation_module.state_dict(),
        }
        if self.save_optimizer:
            payload["graph_optimizer"] = self.graph_optimizer.state_dict()
            payload["orient_optimizer"] = self.orient_optimizer.state_dict()
        return payload

    def save(self, timestep: int) -> None:
        if self.checkpoint_dir is None:
            return

        payload = self.state_dict(timestep=timestep)

        path = self.checkpoint_dir / f"aux_{int(timestep)}.pt"
        latest_path = self.checkpoint_dir / "aux_latest.pt"

        torch.save(payload, path)
        torch.save(payload, latest_path)

        print(f"[AuxTrain] saved checkpoint: {path}", flush=True)

    def load(self, path: str, load_optimizer: bool = True) -> None:
        payload = torch.load(path, map_location=self.device)

        self.graph_encoder.load_state_dict(payload["graph_encoder"])
        self.orientation_module.load_state_dict(payload["orientation_module"])

        if load_optimizer and self.save_optimizer:
            if "graph_optimizer" in payload:
                self.graph_optimizer.load_state_dict(payload["graph_optimizer"])
            if "orient_optimizer" in payload:
                self.orient_optimizer.load_state_dict(payload["orient_optimizer"])

        self.graph_encoder.eval()
        self.orientation_module.eval()

        timestep = int(payload.get("timestep", 0))
        print(f"[AuxTrain] loaded checkpoint: {path} (timestep={timestep})", flush=True)
