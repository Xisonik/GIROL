from __future__ import annotations

import json
import os

from isaaclab.assets import RigidObject, RigidObjectCfg
import isaaclab.sim as sim_utils

from .graveyard_layout import graveyard_capacity, graveyard_position
from .scene_item_config import read_object_transform


class AssetManager:
    """Spawn exactly the physical instances declared in scene_items.json.

    This class owns USD spawning concerns: USD path, rigid/collision properties,
    static asset rotation and asset scale. Runtime scene placement remains the
    responsibility of SceneManager.
    """

    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as file:
            cfg = json.load(file)

        self.items = cfg["objects"]
        self.prim_paths: dict[str, list[str]] = {}
        self.counts: dict[str, int] = {}

    def _abs_paths(self, usd_paths: list[str]) -> list[str]:
        root = os.getcwd()
        return [
            os.path.join(
                root,
                "source/isaaclab_assets/data/aloha_assets",
                path,
            )
            for path in usd_paths
        ]

    @staticmethod
    def _rigid_props_for(types: list[str]):
        if "static_obstacle" in types or "movable_obstacle" in types:
            return sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            )
        return sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=True,
        )

    def spawn_assets_in_env0(self):
        self.prim_paths.clear()
        self.counts.clear()

        total_object_count = sum(int(obj["count"]) for obj in self.items)
        if total_object_count > graveyard_capacity():
            raise RuntimeError(
                "Not enough graveyard cells: "
                f"objects={total_object_count}, "
                f"capacity={graveyard_capacity()}"
            )

        # One index across all object types. Per-type instance_id must not be
        # used for parking because every type starts from instance_id=0.
        global_object_index = 0

        for obj in self.items:
            name = obj["name"]
            types = obj["type"]
            count = int(obj["count"])
            if count < 0:
                raise ValueError(f"Negative count for object {name!r}: {count}")

            if "info" in types:
                global_object_index += count
                continue

            usd_paths = self._abs_paths(obj["usd_paths"])
            if not usd_paths:
                raise ValueError(f"Object {name!r} has no usd_paths")

            transform = read_object_transform(obj)
            self.prim_paths[name] = []
            self.counts[name] = count

            base_size = obj["size"]
            if not isinstance(base_size, list) or len(base_size) != 3:
                raise ValueError(
                    f"Object {name!r}: size must be [x, y, z], "
                    f"got {base_size!r}"
                )
            scaled_height = float(base_size[2]) * float(transform.scale[2])

            is_collision_object = (
                "movable_obstacle" in types
                or "static_obstacle" in types
            )

            for instance_id in range(count):
                if is_collision_object:
                    prim_path = (
                        f"/World/envs/env_0/obstacles/{name}_{instance_id}"
                    )
                else:
                    prim_path = f"/World/envs/env_0/{name}_{instance_id}"

                parked_position = graveyard_position(
                    global_object_index,
                    scaled_height=scaled_height,
                    offset=transform.offset,
                )

                spawn_kwargs = {
                    "usd_path": usd_paths[0],
                    "scale": transform.scale,
                    "rigid_props": self._rigid_props_for(types),
                    "activate_contact_sensors": False,
                }
                if is_collision_object:
                    spawn_kwargs["collision_props"] = (
                        sim_utils.CollisionPropertiesCfg(
                            collision_enabled=True
                        )
                    )

                spawn_cfg = sim_utils.UsdFileCfg(**spawn_kwargs)
                RigidObject(
                    RigidObjectCfg(
                        prim_path=prim_path,
                        spawn=spawn_cfg,
                        init_state=RigidObjectCfg.InitialStateCfg(
                            pos=parked_position,
                            rot=transform.rotation_quat_wxyz,
                        ),
                    )
                )
                self.prim_paths[name].append(prim_path)
                global_object_index += 1

        if global_object_index != total_object_count:
            raise RuntimeError(
                "Asset graveyard indexing mismatch: "
                f"indexed={global_object_index}, expected={total_object_count}"
            )

        return self.prim_paths, self.counts
