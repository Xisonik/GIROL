"""Explicit frozen navigation feature extraction.

Default feature path:

    flat states -> selected state features + selected module outputs -> concat

This extractor does not control internal modes of perception modules. It only
selects which state fields and module outputs are appended to actor/value input.

External perception modules are attached as inference providers: they are not
registered as actor/value submodules and their forward pass is wrapped in
`torch.no_grad()`. To train a perception module through RL, create a separate
actor/critic class that explicitly owns it as a trainable submodule.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from skrl.utils.spaces.torch import unflatten_tensorized_space
from configs.clock import get_global_step

def space_dim(observation_space, key: str) -> int:
    return int(observation_space.spaces[key].shape[0])


def attach_inference_module(owner: nn.Module, public_name: str, modules: dict[str, nn.Module], ref: str) -> None:
    """Attach module without registering its parameters on owner.

    This keeps external perception modules out of actor/value optimizer ownership.
    """
    if ref not in modules:
        raise KeyError(f"Required module {ref!r} is not in model.modules")
    owner.__dict__[public_name] = modules[ref]


class NavFeatureExtractor(nn.Module):
    """Build one feature vector for actor/value heads.
    """

    def __init__(
        self,
        observation_space,
        modules: dict[str, nn.Module],
        features_cfg: dict,
    ):
        super().__init__()
        self.observation_space = observation_space

        self.use_img = bool(features_cfg.get("use_img", False))
        self.gt_orientation_step_end = int(features_cfg.get("gt_orientation_step_end", 0))
        self.use_memory = bool(features_cfg.get("use_memory", False))
        self.use_goal = bool(features_cfg.get("use_goal", False))
        self.use_gt_orientation = bool(features_cfg.get("use_gt_orientation", False))

        self.use_graph = bool(features_cfg.get("use_graph", False))
        self.graph_ref = str(features_cfg.get("graph_ref", "graph_encoder"))
        self.graph_dim = int(features_cfg.get("graph_dim", 128))

        self.use_orientation_module = bool(features_cfg.get("use_orientation_module", False))
        self.orientation_ref = str(features_cfg.get("orientation_ref", "orientation_module"))
        self.orientation_dim = int(features_cfg.get("orientation_dim", 1))
        self.use_gt_orientation_only = bool(features_cfg.get("use_gt_orientation_only", False))

        # if self.use_graph:
        attach_inference_module(self, "graph_encoder", modules, self.graph_ref)

        # if self.use_orientation_module:
        attach_inference_module(self, "orientation_module", modules, self.orientation_ref)

        self._output_dim = self._build_output_dim()

    @property
    def output_dim(self) -> int:
        return int(self._output_dim)


    def _build_output_dim(self) -> int:
        dim = 0

        if self.use_img:
            dim += space_dim(self.observation_space, "img")
        if self.use_memory:
            dim += space_dim(self.observation_space, "memory")
        if self.use_goal:
            dim += space_dim(self.observation_space, "goal")
        if self.use_gt_orientation:
            dim += space_dim(self.observation_space, "orientation")
        if self.use_graph:
            dim += self.graph_dim
        if self.use_orientation_module:
            dim += self.orientation_dim

        if dim <= 0:
            raise ValueError("Feature extractor has zero output dim; enable at least one feature")
        return dim

    def forward(self, flat_states: torch.Tensor) -> torch.Tensor:
        states = unflatten_tensorized_space(self.observation_space, flat_states)
        feats: list[torch.Tensor] = []

        img = None
        img = states["img"]
        if self.use_img and not self.use_gt_orientation_only:
            feats.append(img)

        if self.use_memory:
            feats.append(states["memory"])

        if self.use_goal:
            feats.append(states["goal"])
        
        if not self.use_gt_orientation_only:
            graph_emb = None
            with torch.no_grad():
                graph_emb = self.graph_encoder(states["graph"])

        if self.use_graph and not self.use_gt_orientation_only:
            # FROZEN INFERENCE: graph_encoder is intentionally not trained by actor/value.
            # To train it through RL, make a separate actor/critic class, register the
            # module normally as self.graph_encoder, and remove this no_grad block.
            feats.append(graph_emb)

        if self.use_gt_orientation:
            feats.append(states["orientation"])
        if self.use_orientation_module and not self.use_gt_orientation_only:
            step = get_global_step()
            # FROZEN INFERENCE: orientation_module is intentionally not trained by actor/value.
            # It receives only the inputs assembled here. The module owns its internal mode.
            with torch.no_grad():
                orient = self.orientation_module(
                    states["orientation"],
                    img=img,
                    graph_emb=graph_emb,
                )
            if step > self.gt_orientation_step_end:
                feats.append(orient)
            else:
                feats.append(states["orientation"])

        x = torch.cat(feats, dim=-1)
        got = int(x.shape[-1])
        if not torch.isfinite(x).all():
            print("no no no ")
        if got != self.output_dim:
            raise RuntimeError(f"Feature dim mismatch: got {got}, expected {self.output_dim}")
        return x

    def describe(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "output_dim": self.output_dim,
            "use_img": self.use_img,
            "use_memory": self.use_memory,
            "use_goal": self.use_goal,
            "use_gt_orientation": self.use_gt_orientation,
            "use_graph": self.use_graph,
            "graph_ref": self.graph_ref,
            "graph_dim": self.graph_dim,
            "use_orientation_module": self.use_orientation_module,
            "orientation_ref": self.orientation_ref,
            "orientation_dim": self.orientation_dim,
        }

def check_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        bad = ~torch.isfinite(tensor)
        raise RuntimeError(
            f"[NaN DEBUG] {name} contains NaN/Inf: "
            f"shape={tuple(tensor.shape)}, "
            f"bad_count={int(bad.sum().item())}, "
            f"min={torch.nan_to_num(tensor).min().item()}, "
            f"max={torch.nan_to_num(tensor).max().item()}"
        )
