"""Tiny bridge from runner config to IsaacLab env code.

In the runner:
    os.environ["ALOHA_NAV_ENV_CFG"] = json.dumps(cfg["env"])

In aloha_env_base.py / aloha_env_hab_wr.py:
    from aloha_nav.env_bridge.env_config import load_env_config
    self.runtime_cfg = load_env_config()
"""

from __future__ import annotations

import json
import os


ENV_CONFIG_VAR = "ALOHA_NAV_ENV_CFG"


DEFAULT_ENV_CONFIG = {
    "task": "nav",
    "turn_task": False,
    "camera": True,
    "memory": True,
    "controller": False,
    "curriculum": True,
    "action_angle_deg": 30,
}


def load_env_config() -> dict:
    raw = os.environ.get(ENV_CONFIG_VAR)
    cfg = dict(DEFAULT_ENV_CONFIG)
    if raw:
        cfg.update(json.loads(raw))
    return cfg
