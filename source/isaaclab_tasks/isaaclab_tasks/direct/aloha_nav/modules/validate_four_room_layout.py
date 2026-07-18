"""Run from the repository root after installing the patched module files."""

import os
import torch

from source.isaaclab_tasks.isaaclab_tasks.direct.aloha_nav.modules.scene_manager import SceneManager


def main() -> None:
    config_path = os.path.join(
        os.getcwd(),
        "source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/configs/scene_items.json",
    )
    manager = SceneManager(num_envs=8, config_path=config_path, device="cpu")
    env_ids = torch.arange(8)
    manager.randomize_scene(env_ids)

    assert manager.num_rooms == 4
    assert manager.num_total_objects == 84

    goal_indices = manager.type_map["possible_goal"]
    goal_counts = manager.active[:, goal_indices].sum(dim=1)
    assert torch.all(goal_counts == 1), goal_counts

    for env_id in env_ids.tolist():
        floor_mask = manager.active[env_id] & (manager.on_surface_idx[env_id] < 0)
        floor_positions = manager.positions[env_id, floor_mask]
        forbidden = manager.room_mapper.forbidden_inner_wall_mask(floor_positions)
        assert not forbidden.any(), floor_positions[forbidden]

    print("Four-room layout validation passed.")
    print("Object count:", manager.num_total_objects)
    print("Active objects per env:", manager.active.sum(dim=1).tolist())
    print("Goal room ids:", manager.active_goal_room_ids.tolist())


if __name__ == "__main__":
    main()
