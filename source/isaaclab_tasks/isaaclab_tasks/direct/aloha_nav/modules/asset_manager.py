import json
import os

from isaaclab.assets import RigidObject, RigidObjectCfg
import isaaclab.sim as sim_utils


class AssetManager:
    """Spawn exactly the physical instances declared in scene_items.json.

    Room assignment is a per-reset logical property. Assets are therefore not
    multiplied by the number of physical rooms.
    """

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        self.items = cfg["objects"]
        self.prim_paths = {}
        self.counts = {}

    def _abs_paths(self, usd_paths):
        root = os.getcwd()
        return [
            os.path.join(
                root,
                "source/isaaclab_assets/data/aloha_assets",
                path,
            )
            for path in usd_paths
        ]

    def _rigid_props_for(self, types: list[str]):
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

        for obj in self.items:
            name = obj["name"]
            types = obj["type"]
            if "info" in types:
                continue

            count = int(obj["count"])
            if count < 0:
                raise ValueError(f"Negative count for object {name!r}: {count}")
            usd_paths = self._abs_paths(obj["usd_paths"])
            if not usd_paths:
                raise ValueError(f"Object {name!r} has no usd_paths")

            self.prim_paths[name] = []
            self.counts[name] = count

            default_rot = (1.0, 0.0, 0.0, 0.0)
            if name == "bowl":
                default_rot = (0.0, 0.7071, 0.0, 0.7071)

            for i in range(count):
                is_collision_object = (
                    "movable_obstacle" in types
                    or "static_obstacle" in types
                )
                if is_collision_object:
                    prim_path = f"/World/envs/env_0/obstacles/{name}_{i}"
                else:
                    prim_path = f"/World/envs/env_0/{name}_{i}"

                spawn_kwargs = {
                    "usd_path": usd_paths[0],
                    "rigid_props": self._rigid_props_for(types),
                    "activate_contact_sensors": False,
                }
                if is_collision_object:
                    spawn_kwargs["collision_props"] = (
                        sim_utils.CollisionPropertiesCfg(collision_enabled=True)
                    )
                spawn_cfg = sim_utils.UsdFileCfg(**spawn_kwargs)

                RigidObject(
                    RigidObjectCfg(
                        prim_path=prim_path,
                        spawn=spawn_cfg,
                        init_state=RigidObjectCfg.InitialStateCfg(
                            pos=(
                                (i % 16 - 8) * 0.5,
                                (i // 16 - 2) * 0.5,
                                -20.0,
                            ),
                            rot=default_rot,
                        ),
                    )
                )
                self.prim_paths[name].append(prim_path)

        return self.prim_paths, self.counts
