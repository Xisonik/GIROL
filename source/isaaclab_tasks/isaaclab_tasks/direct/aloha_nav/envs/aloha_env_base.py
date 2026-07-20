# env.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
 
import gymnasium as gym
import torch
import math
import numpy as np
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCamera, TiledCameraCfg, ContactSensorCfg
from ..modules.scene_manager import SceneManager
from ..modules.evaluation_manager import EvaluationManager
from ..modules.control_manager import VectorizedPurePursuit
from ..modules.path_manager import Path_manager
from ..modules.memory_manager import MemoryManager
from ..modules.asset_manager import AssetManager
from ..modules.room_geometry import RoomGeometryConfig
from ..modules.Noise_modules import RelativeYawObservationNoise
from configs.clock import tick_global_step, get_global_step
import omni.kit.commands
import datetime
import torch.nn.functional as F
import random
from ..modules.env_config import load_env_config
from isaaclab_assets.robots.aloha import ALOHA_CFG
from transformers import CLIPProcessor, CLIPModel
import json


def _expanded_scene_object_count() -> int:
    """Return the exact number of physical object instances."""
    aloha_nav_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    scene_path = os.path.join(
        aloha_nav_dir, "configs", "scene_items.json"
    )
    with open(scene_path, "r") as f:
        scene_cfg = json.load(f)
    return sum(int(obj["count"]) for obj in scene_cfg["objects"])


num_total_objects = _expanded_scene_object_count()

_ENV_CFG = json.loads(os.environ.get("ALOHA_NAV_ENV_CFG", "{}"))

def env_get(name: str, default=None):
    return _ENV_CFG.get(name, default)

noise_orientation_steps_start_step = int(env_get("noise_orientation_steps_start_step", 0))
CL_ON = int(env_get("CL_ON", True))

class WheeledRobotEnvWindow(BaseEnvWindow):
    def __init__(self, env: 'BaseWheeledRobotEnv', window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)

def quat_conjugate(quat):
    """
    Сопряжённый кватернион.
    Предполагаем формат (w, x, y, z)
    """
    # Проверь формат!
    # print(f"quat example: {quat[0]}")  # Посмотри первые 4 значения
    
    # Если (w, x, y, z):
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.stack([w, -x, -y, -z], dim=-1)

@configclass
class BaseWheeledRobotEnvCfg(DirectRLEnvCfg):
    episode_length_s = 512.0
    decimation = 8
    action_space = gym.spaces.Box(
        low=np.array([-1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
        shape=(2,)
    )
    # Observation space is now the ResNet18 embedding size (512)
    m = 1  # Например, 3 эмбеддинга и действия
    # TODO automat compute num_total_objects
    num_total_objects = num_total_objects #36 12 num_total_objects * 5

    observation_space = gym.spaces.Dict({
        "img": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(512,), dtype=np.float32),  #518 512*4+4+2
        "memory": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(2056,)), 
        "goal": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(2,), dtype=np.float32),
        "orientation": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(1,), dtype=np.float32),
        "graph": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(6*num_total_objects,), dtype=np.float32)
    })
    state_space = 0
    debug_vis = False
 
    ui_window_class_type = WheeledRobotEnvWindow

    sim: SimulationCfg = SimulationCfg(
        dt=1/60,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="min",
            restitution_combine_mode="min",
            static_friction=0.2,
            dynamic_friction=0.15,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        debug_vis=False,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=24, replicate_physics=True)
    robot: ArticulationCfg = ALOHA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    wheel_radius = 0.068
    wheel_distance = 0.34
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/box2_Link/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.35, 0, 1.1), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=35.0, focus_distance=2.0, horizontal_aperture=36, clipping_range=(0.2, 10.0)
        ),
        width=224,
        height=224,
    )
    current_dir = os.getcwd()
    room = sim_utils.UsdFileCfg(
        usd_path=os.path.join(current_dir, "source/isaaclab_assets/data/aloha_assets", "scenes/scenes_sber_kitchen_for_BBQ/room_big_wall.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            kinematic_enabled=False,
            rigid_body_enabled=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
    )
    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        update_period=0.1,
        history_length=3,
        debug_vis=False,
        filter_prim_paths_expr=["/World/envs/env_.*"], #/obstacles/.*
    )

class BaseWheeledRobotEnv(DirectRLEnv):
    cfg: BaseWheeledRobotEnvCfg

    # ---- Variant hooks -------------------------------------------------
    # Subclasses override these small hooks instead of copying __init__.
    def _default_use_staff(self) -> bool:
        return True

    def _default_use_obstacles(self) -> bool:
        return True

    def _init_actuator_handles(self) -> None:
        self._left_wheel_id = self._robot.find_joints("left_wheel")[0]
        self._right_wheel_id = self._robot.find_joints("right_wheel")[0]

    def _init_variant_after_actuators(self) -> None:
        pass

    def _reset_variant_state(self, env_ids: torch.Tensor) -> None:
        """Reset extra state owned by a concrete environment variant.

        This hook is called after the common actor reset and before goal placement.
        Variants should use it for controller state, predicted-collision flags, etc.
        """
        pass

    def _reset_actor_on_episode_start(self, env_ids: torch.Tensor) -> None:
        """Common per-episode actor reset that is safe for all variants.

        The base environment owns the ordering. Concrete variants only add/clear
        their own buffers via ``_reset_variant_state``.
        """
        self._robot.reset(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.zeros_like(self.episode_length_buf)
        self._actions[env_ids] = 0.0
        self._reset_variant_state(env_ids)

    def _write_actor_state_to_sim(
        self,
        env_ids: torch.Tensor,
        robot_pos: torch.Tensor,
        robot_quats: torch.Tensor,
    ) -> None:
        """Write the placed actor state to simulation.

        Default implementation is for the wheeled ALOHA Articulation.
        RigidObject-based variants must override this method instead of
        overriding the whole ``_reset_idx``.
        """
        self._write_actor_state_to_sim(env_ids=env_ids, robot_pos=robot_pos, robot_quats=robot_quats)

    def __init__(self, cfg: BaseWheeledRobotEnvCfg, render_mode: str | None = None, **kwargs):
        self._super_init = True
        self.current_dir = os.getcwd()
        self.config_path=os.path.join(self.current_dir, "source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/configs/scene_items.json")
        self.scene_objects = {}
        self.CAMERA = True
        self.memory_on = True
        super().__init__(cfg, render_mode, **kwargs)
        runtime_cfg = load_env_config()
        if self.memory_on:
            self.memory_manager = MemoryManager(
                num_envs=self.num_envs,
                embedding_size=512,  # Размер эмбеддинга ResNet18
                action_size=2,      # Размер действия (линейная и угловая скорость)
                device=self.device
            )
        self._super_init = False
        self.EVAL = False
        self.random_actions = False
        self.scene_manager = SceneManager(self.num_envs, self.config_path, self.device)

        self.CL_ON = CL_ON
        self.stage = 0
        self.use_staff = self._default_use_staff()
        self.use_obstacles = self._default_use_obstacles()
        self.use_controller = True #kwargs.get('expert', False)
        self.imitation = False #kwargs.get('imitation', False)
        self.cur_angle_error = 0
        self.mean_radius = 0
        self.warm_len = 2000
        self.my_episode_lenght = 256

        self.turn_on_obstacles = False
        self.turn_on_obstacles_always = False

        if self.imitation:
            self.use_controller = True
        if self.use_controller:
            self.path_manager = Path_manager(scene_manager=self.scene_manager, ratio=4.0, shift=[10, 10], device=self.device)
            self.control_module = VectorizedPurePursuit(num_envs=self.num_envs, device=self.device)
        self.scene_embeddings = torch.zeros(self.num_envs, 6*num_total_objects, device=self.device)

        self._actions = torch.zeros((self.num_envs, 2), device=self.device)
        self._actions[:, 1] = 0.0
        self._left_wheel_vel = torch.zeros(self.num_envs, device=self.device)
        self._right_wheel_vel = torch.zeros(self.num_envs, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        self._init_actuator_handles()
        self._init_variant_after_actuators()

        self.set_debug_vis(self.cfg.debug_vis)

        self.turn_on_controller_step = 0
        
        if self.turn_on_obstacles_always:
            self.use_obstacles = True
        self.previous_distance_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.previous_angle_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.previous_lin_vel = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.previous_ang_vel = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.angular_speed = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.success_rate = 0
        self.sr_stack_capacity = 0

        self._step_update_counter = 0
        self.max_angle_error = 1 * torch.pi

        self.warm = True
        self.without_imitation = self.warm_len / 2
        self.without_imitation_log = False
        self.success_ep_num = 0
        self.first_ep = [True, True] # TODO: to dictionaty

        self.episode_lengths = torch.zeros(self.num_envs, device=self.device)
        self.episode_count = 0
        self.total_episode_length = 0.0
        self.tensorboard_step = 0
        self.cur_step = 0
        self.velocities = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        if self.CAMERA:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.eval()  # Установить в режим оценки
        self.foult_ep_num = 0
        # Инициализация стеков для хранения успехов (1 - успех, 0 - неуспех)
        self.success_stacks = [[] for _ in range(self.num_envs)]  # policy-only stacks
        self.controller_success_stacks = [[] for _ in range(self.num_envs)]
        self.all_success_stacks = [[] for _ in range(self.num_envs)]
        self.controller_success_rate = 0.0
        self.all_success_rate = 0.0
        self.max_stack_size = 10  # Максимальный размер стека
        self.sr_stack_full = False
        self.start_mean_radius = 0
        self.min_level_radius = 0
        self.sr_treshhold = 85
        self.LOG = False
        self.text_embeddings = torch.zeros((self.num_envs, 512), device=self.device)
        if self.LOG:
            from comet_ml import start
            from comet_ml.integration.pytorch import log_model
            self.experiment = start(
                api_key=os.environ.get("COMET_API_KEY"),
                project_name="general",
                workspace="xisonik"
            )
        self.print_config_info()

        self.choose_speed_step = 0
        self.choosen_linear_speed = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.choosen_angular_speed = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        from ..modules.history_tracker import SceneHistoryTracker, NaNDetector
        self.history_tracker = SceneHistoryTracker(self.num_envs, num_total_objects, 100, self.device)
        self.nan_detector = NaNDetector(self.history_tracker, "./nan_debug", raise_on_nan=True)
        self.setup_omni_warning_handler()
        self.first_nan = True
        self.controlled_env_ids = set()
        self.control_percentage = 0
        self.assistance_ratio = 0
        self.assistance_num_envs = 0

        # --- Средняя награда за эпизод (скользящее окно) ---
        self._reward_window_size = 200
        self._episode_reward_window: list[float] = []   # суммы наград за последние N эпизодов
        self._current_episode_reward = torch.zeros(self.num_envs, device=self.device)
        self.mean_episode_reward = 0.0
        self.relative_yaw_noise = RelativeYawObservationNoise(
            accuracy=runtime_cfg.get("relative_yaw_accuracy", 100.0),
            variance=runtime_cfg.get("relative_yaw_noise_variance", 0.0),
            enabled=runtime_cfg.get("relative_yaw_noise", False),
        )

        self.TURN_TASK = True   # turn-in-place task: robot rotates to face the goal
        if self.TURN_TASK:
            self.stage = 4          # fixed placement: spawn ±90° off the goal
            self.CL_ON = False      # no curriculum progression (stage stays 4)
            self.use_controller = False
        self.DEF_TURN = False
        self._update_controlled_envs()

    def eval_mode(self, ON, eval_stage: int = 0):
        if ON:
            """Сохраняет train-состояние и переключает env в eval."""
            self._saved_train_state = {
                'stage': self.stage,
                'mean_radius': self.mean_radius,
                'cur_angle_error': self.cur_angle_error,
                'success_stacks': [s.copy() for s in self.success_stacks],
                'sr_stack_full': self.sr_stack_full,
                'success_rate': self.success_rate,
                'CL_ON': self.CL_ON,
                'use_controller': self.use_controller,
            }
            self.stage = eval_stage
            self.mean_radius = 0.0
            self.cur_angle_error = 0.0
            self.CL_ON = False
            self.EVAL = True
            # Очищаем SR стеки — eval считается с нуля
            self.success_stacks = [[] for _ in range(self.num_envs)]
            self.sr_stack_full = False
            self.success_rate = 0.0
            # Форсируем reset всех envs на следующем шаге
            self.episode_length_buf[:] = self.my_episode_lenght
            print(f"[EVAL] Entered eval mode (stage={eval_stage})")
        else:
            """Восстанавливает train-состояние и логирует eval SR."""
            eval_sr = self.success_rate
            s = self._saved_train_state
            self.stage = s['stage']
            self.mean_radius = s['mean_radius']
            self.cur_angle_error = s['cur_angle_error']
            self.success_stacks = s['success_stacks']
            self.sr_stack_full = s['sr_stack_full']
            self.success_rate = s['success_rate']
            self.CL_ON = s['CL_ON']
            self.use_controller = s['use_controller']
            self.EVAL = False
            # Форсируем reset всех envsм
            self.episode_length_buf[:] = self.my_episode_lenght
            print(f"[EVAL] Exited eval mode. Eval SR={eval_sr:.1f}%, restored train SR={self.success_rate:.1f}%")
            return eval_sr

    def _update_controlled_envs(self, env_ids = None):
        """
        Быстрое заполнение управляемых сред.
        
        Если available >= нужно добавить → выбираем ровно столько сразу.
        Если available < нужно → берем все что есть.
        """
        # Вычисляем целевой процент (верхняя граница 0.6)
        self.control_percentage = max(0.10, min(0.60, 0.9 - 0.8 * (self.success_rate / 100.0)))
        target_count = max(1, int(self.num_envs * self.control_percentage))
        
        if env_ids is None:
            while len(self.controlled_env_ids) > target_count:
                self.controlled_env_ids.discard(random.choice(list(self.controlled_env_ids)))
            return self.control_percentage
        
        # Кандидаты берем ТОЛЬКО из текущих env_ids
        env_ids_set = set(int(e.item()) for e in env_ids)
        available = env_ids_set - self.controlled_env_ids
        
        # Сколько нужно добавить
        need_to_add = target_count - len(self.controlled_env_ids)
        
        # Быстрое заполнение: выбираем ровно столько сколько нужно
        if need_to_add > 0 and available:
            to_add_count = min(need_to_add, len(available))  # берем минимум из нужного и доступного
            to_add = random.sample(list(available), to_add_count)
            self.controlled_env_ids.update(to_add)
        
        # Удаляем лишние (если somehow перешли предел)
        while len(self.controlled_env_ids) > target_count:
            self.controlled_env_ids.discard(random.choice(list(self.controlled_env_ids)))
        
        # Специальные режимы
        if self.imitation:
            self.controlled_env_ids = set(range(self.num_envs))
        elif not self.use_controller:
            self.controlled_env_ids.clear()
        
        self.assistance_ratio = self.control_percentage
        self.assistance_num_envs = len(self.controlled_env_ids)
        
        # if self.cur_step % 256 == 0:
        #     print(f"[CONTROL] SR={self.success_rate:.1f}% → {self.control_percentage:.1%} "
        #         f"({len(self.controlled_env_ids)}/{self.num_envs} envs controlled)")
        
        return self.control_percentage


    def print_config_info(self):
        print("__________[ CONGIFG INFO ]__________")
        print(f"|")
        print(f"| Start mean radius is: {self.mean_radius}")
        print(f"|")
        print(f"| Start amx angle is: {self.max_angle_error}")
        print(f"|")
        print(f"| Use controller: {self.use_controller}")
        print(f"|")
        print(f"| Full imitation: {self.imitation}")
        print(f"|")
        print(f"| Use memory: {self.memory_on}")
        print(f"|")
        print(f"| Use obstacles: {self.use_obstacles}")
        print(f"|")
        print(f"| Start radius: {self.start_mean_radius}, min: {self.min_level_radius}")
        print(f"|")
        print(f"| Warm len: {self.warm_len}")
        print(f"|")
        print(f"| stack size: {self.max_stack_size}")
        print(f"|")
        print(f"| Turn on obstacles always: {self.turn_on_obstacles_always}")
        print(f"|")
        print(f"| Turn on curriculum learnong: {self.CL_ON}")
        print(f"|")
        print(f"| Turn on random actions: {self.random_actions}")
        print(f"|")
        print(f"_______[ CONGIFG INFO CLOSE ]_______")
        if self.EVAL:
            print(f"|")
            print(f"|")
            print(f"______!!ATTENTION!!_____")
            print(f"|")
            print(f"|")
            print("!!! IT IS EVAL NOW in ALOHA_ENV !!!")


    def get_metrics(self) -> dict:
        return {
            "success_rate": self.success_rate,
            "controller_success_rate": self.controller_success_rate,
            "all_success_rate": self.all_success_rate,
            "mean_radius": self.mean_radius,
            "mean_episode_reward": self.mean_episode_reward,
            "assistance_ratio": self.assistance_ratio,
            "assistance_num_envs": self.assistance_num_envs,
            "max_angle_error": float(self.max_angle_error),
            "cur_angle_error": float(self.cur_angle_error),
            "episode_count": self.episode_count,
            "stage": self.stage,
            "avg_episode_length": (
                self.total_episode_length / self.episode_count 
                if self.episode_count > 0 else 0
            ),
        }

    def _setup_scene(self):
        from isaaclab.sensors import ContactSensor
        from omni.usd import get_context
        from pxr import UsdGeom
        from isaaclab.sim.spawners.from_files import spawn_from_usd

        self._robot = Articulation(self.cfg.robot)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        if self.CAMERA:
            self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
            self.scene.sensors["tiled_camera"] = self._tiled_camera

        stage = get_context().get_stage()
        UsdGeom.Xform.Define(stage, "/World/envs/env_0/obstacles")

        spawn_from_usd(
            prim_path="/World/envs/env_0/obstacles/room",
            cfg=self.cfg.room,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )

        self.asset_manager = AssetManager(config_path=self.config_path)
        prim_paths_env0, counts = self.asset_manager.spawn_assets_in_env0()

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self._robot
        self.scene_objects = {}
        for name, count in counts.items():
            for i in range(count):
                if "/obstacles/" in prim_paths_env0[name][i]:
                    prim_path_view = f"/World/envs/env_.*/obstacles/{name}_{i}"
                else:
                    prim_path_view = f"/World/envs/env_.*/{name}_{i}"

                ro_view = RigidObject(RigidObjectCfg(prim_path=prim_path_view, spawn=None))
                self.scene.rigid_objects[f"{name}_{i}"] = ro_view
                self.scene_objects.setdefault(name, []).append(ro_view)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=300.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_observations(self) -> dict:
        
        self.tensorboard_step += 1
        self.cur_step += 1
        self.episode_lengths += 1
        # Получение RGB изображений с камеры
        if self.CAMERA:
            camera_data = self._tiled_camera.data.output["rgb"].clone()  # (num_envs, 224, 224, 3)

            # Переводим в float и нормализуем для CLIP
            imgs = camera_data.to(device=self.device, dtype=torch.float32, non_blocking=True) / 255.0
            imgs = imgs.permute(0, 3, 1, 2)  # (N, 3, H, W)
            first_img_2 = imgs[0]
            inputs = self.clip_processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                image_embeddings = self.clip_model.get_image_features(**inputs)
                image_embeddings = image_embeddings / (image_embeddings.norm(dim=1, keepdim=True) + 1e-9)

        # Получение скоростей робота
        root_lin_vel_w = torch.norm(self._robot.data.root_lin_vel_w[:, :2], dim=1).unsqueeze(-1)
        root_ang_vel_w = self._robot.data.root_ang_vel_w[:, 2].unsqueeze(-1)
        root_pos_w =  self.to_local(self._robot.data.root_pos_w)
        angle = self._robot.data.root_quat_w

        root_quat_w = self._robot.data.root_quat_w  # shape [N, 4]
        base_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=root_quat_w.device)

        # mask = torch.isnan(root_quat_w).any(dim=-1)
        # root_quat_w[mask] = base_quat

        if torch.isnan(root_quat_w).any():
            print("Oh Nooooooo")
        # Локальный вектор взгляда робота (вперёд по оси X)
        local_forward = torch.tensor([1.0, 0.0, 0.0], device=root_quat_w.device, dtype=root_quat_w.dtype)
        local_forward = local_forward.unsqueeze(0).repeat(root_quat_w.shape[0], 1)  # [N, 3]

        # Вектор взгляда в мировых координатах
        forward_w = self.quat_rotate(root_quat_w, local_forward)  # [N, 3]

        # Вектор от робота к цели
        root_pos_w = self._robot.data.root_pos_w  # [N, 3]
        to_goal = self._desired_pos_w - root_pos_w  # [N, 3]

        # Нормализуем векторы
        forward_w_norm = torch.nn.functional.normalize(forward_w[:, :2] , dim=1)
        to_goal_norm = torch.nn.functional.normalize(to_goal[:, :2] , dim=1)

        # Косинус угла между векторами взгляда и направления на цель
        cos_angle = torch.sum(forward_w_norm * to_goal_norm, dim=1)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)  # для безопасности

        # Вычисляем угол между векторами
        angle = torch.acos(cos_angle)
        angle = angle
        self.memory_manager.update(image_embeddings, self.velocities)
        embedding = self.memory_manager.get_observations()
        robot_quat = self._robot.data.root_quat_w # [num_envs, 4]
        mask = torch.isnan(robot_quat).any(dim=-1)
        robot_quat[mask] = base_quat
        
        if torch.isnan(robot_quat).any():
            print("Oh Nooooooo robot_quat")
        # Конвертируем quaternion → yaw
        # ВНИМАНИЕ: Isaac Lab использует (w,x,y,z) или (x,y,z,w) - ПРОВЕРЬТЕ!
        w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
        robot_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
        # print("robot angle:", robot_yaw)
        # print(f"robot angle: {torch.rad2deg(robot_yaw).item():.1f}°")
        obs_img = torch.cat([embedding, root_lin_vel_w*0.1, root_ang_vel_w*0.1, self.to_local(self._desired_pos_w)], dim=-1)

        # Вектор взгляда в мировых координатах
        # Знаковый угол [-π, π]
        root_quat_w = self._robot.data.root_quat_w  # [N, 4]
        root_pos_w = self._robot.data.root_pos_w    # [N, 3]
        mask = torch.isnan(root_quat_w).any(dim=-1)
        root_quat_w[mask] = base_quat
        
        if torch.isnan(root_quat_w).any():
            print("Oh Nooooooo root_quat_w")

        base_pos = torch.tensor([0.0, 0.0, 0.0], device=root_pos_w.device)

        mask = torch.isnan(root_pos_w).any(dim=-1)
        root_pos_w[mask] = base_pos

        if torch.isnan(root_pos_w).any():
            print("Oh Nooooooo root_pos_w")
        # 1. Вектор от робота к цели в мировых координатах
        to_goal_world = self._desired_pos_w - root_pos_w  # [N, 3]

        # 2. Конвертируем в ЛОКАЛЬНЫЕ координаты робота
        # Нужна обратная ротация (ротация мира в систему координат робота)
        quat_inv = quat_conjugate(root_quat_w)  # Инвертируем кватернион
        to_goal_local = self.quat_rotate(quat_inv, to_goal_world)  # [N, 3]

        # 3. Берём только XY компоненты (игнорируем высоту Z)
        to_goal_local_xy = to_goal_local[:, :2]  # [N, 2]

        # 4. Вычисляем угол через atan2 (в плоскости XY)
        # В системе координат робота:
        #   X - вперёд (куда смотрит робот)
        #   Y - влево
        # Тогда:
        #   atan2(y, x) даёт угол от оси X (вперёд) до вектора цели
        relative_yaw = torch.atan2(to_goal_local_xy[:, 1], to_goal_local_xy[:, 0])  # [N]
        # obs_img = torch.cat([embedding, ], dim=-1)
        # if self.EVAL:
        # print(f"Relative yaw 2: {torch.rad2deg(relative_yaw[0]):.1f}°")
        # print(self.to_local(self._desired_pos_w).shape)
        step = get_global_step()
        if step > noise_orientation_steps_start_step:
            relative_yaw = self.relative_yaw_noise(relative_yaw.unsqueeze(1))
        else:
            relative_yaw = relative_yaw.unsqueeze(1)
        obs = {
            "img": image_embeddings.unsqueeze(1),          # нормализуем
            "memory": embedding.unsqueeze(1),
            "goal": self.to_local(self._desired_pos_w).unsqueeze(1),
            "orientation": relative_yaw,
            "graph": self.scene_embeddings # НЕ нормализуем
        }
        self.previous_ang_vel = self.angular_speed

        observations = {"policy": obs}
        return observations

    def _pre_physics_step(self, actions: torch.Tensor):
        if not self.TURN_TASK:
            env_ids = self._robot._ALL_INDICES.clone()
            self._actions = actions.clone().clamp(-1.0, 1.0)

            nan_mask = torch.isnan(self._actions) | torch.isinf(self._actions)
            nan_indices = torch.nonzero(nan_mask.any(dim=1), as_tuple=False).squeeze()
            if nan_indices.numel() > 0:
                if self.first_nan:
                    self.first_nan = False
                    print(f"[MY WARNING] NaN/Inf in actions for envs: {nan_indices.tolist()}")
                self._actions[nan_mask] = 0.0
                actions[nan_mask] = 0.0

            r = self.cfg.wheel_radius
            L = self.cfg.wheel_distance
            self._step_update_counter += 1
            
            # МАСКА управляемых сред
            controlled_mask = torch.tensor(
                [int(e.item()) in self.controlled_env_ids for e in env_ids],
                dtype=torch.bool, device=self.device
            )
            
            # ШАГ 1: считаем скорости из actions для ВСЕх сред (базовая RL)
            linear_speed = 0.6 * (self._actions[:, 0] + 1.0)
            angular_speed = 2 * self._actions[:, 1]
            
            # ШАГ 2: если есть управляемые → пересчитываем их через контроллер
            if controlled_mask.any() or self.imitation:
                self.turn_on_controller_step += 1
                
                quat = self._robot.data.root_quat_w
                siny_cosp = 2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2])
                cosy_cosp = 1 - 2 * (quat[:, 2] * quat[:, 2] + quat[:, 3] * quat[:, 3])
                yaw = torch.atan2(siny_cosp, cosy_cosp)
                
                # Получаем скорости от контроллера для ВСЕх сред
                lin_sp_all, ang_sp_all = self.control_module.compute_controls(
                    self.to_local(self._robot.data.root_pos_w[:, :2], env_ids),
                    yaw
                )
                
                # Пересчитываем ТОЛЬКО управляемые среды
                controlled_indices = torch.where(controlled_mask)[0]
                linear_speed[controlled_indices] = lin_sp_all[controlled_indices]
                angular_speed[controlled_indices] = ang_sp_all[controlled_indices]
                
                # Обновляем actions ТОЛЬКО для управляемых
                self._actions[controlled_indices, 0] = (linear_speed[controlled_indices] / 0.6) - 1
                self._actions[controlled_indices, 1] = angular_speed[controlled_indices] / 2
                actions.copy_(self._actions.clamp(-1.0, 1.0))
            
            # ШАГ 3: переводим скорости в управления моторами
            self.angular_speed = angular_speed
            self.velocities = torch.stack([linear_speed, angular_speed], dim=1)
            self._left_wheel_vel = (linear_speed - (angular_speed * L / 2)) / r
            self._right_wheel_vel = (linear_speed + (angular_speed * L / 2)) / r
        else:
            r = self.cfg.wheel_radius
            L = self.cfg.wheel_distance
            self._actions = actions.clone().clamp(-1.0, 1.0)
            linear_speed = 0.0*(self._actions[:, 0] + 1.0) # [num_envs], всегда > 0
            angular_speed = 2*self._actions[:, 1]  # [num_envs], оставляем как есть от RL

            if self.DEF_TURN:
                linear_speed = torch.zeros_like(self._actions[:, 0])
                angular_speed = torch.full_like(self._actions[:, 1], -2.0)
            self.angular_speed = angular_speed
            self.velocities = torch.stack([linear_speed, angular_speed], dim=1)
            self._left_wheel_vel = (linear_speed - (angular_speed * L / 2)) / r
            self._right_wheel_vel = (linear_speed + (angular_speed * L / 2)) / r

    def _apply_action(self):
        wheel_velocities = torch.stack([self._left_wheel_vel, self._right_wheel_vel], dim=1).unsqueeze(-1).to(dtype=torch.float32)
        self.last_actions = wheel_velocities
        self._robot.set_joint_velocity_target(wheel_velocities, joint_ids=[self._left_wheel_id, self._right_wheel_id])

    def _get_rewards(self) -> torch.Tensor:
        goal_reached, num_subs, r_error, a_error = self.goal_reached(get_num_subs=True)
        gamma = 0.5
        if self.TURN_TASK:
            F_s = -self.previous_angle_error 
            F_s_next = -a_error
            turnes = (F_s_next - F_s)
            self.previous_angle_error = a_error
        else:
            progress = self.previous_distance_error - r_error  # >0 если ближе к цели
            turnes = gamma * progress

        has_contact = torch.logical_or(self.get_contact(), self.out_of_bounds())

        collision_penalty = -3.0 * has_contact.float()
        goal_bonus = 5.0 * goal_reached.float()
        reward = -0.05 + turnes + collision_penalty + goal_bonus # TODO: turnes + !Add this back

        died, time_out = self._get_dones(inner=True)
        if torch.any(died | time_out):
            sr = self.update_success_rate(goal_reached)

        self.previous_distance_error = r_error
        self._current_episode_reward += reward.detach()
        return reward
    
    def quat_rotate(self, quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """
        Вращение вектора vec кватернионом quat.
        quat: [N, 4] (w, x, y, z)
        vec: [N, 3]
        Возвращает: [N, 3] - вектор vec, повернутый кватернионом quat
        """
        w, x, y, z = quat.unbind(dim=1)
        vx, vy, vz = vec.unbind(dim=1)

        # Кватернионное умножение q * v
        qw = -x*vx - y*vy - z*vz
        qx = w*vx + y*vz - z*vy
        qy = w*vy + z*vx - x*vz
        qz = w*vz + x*vy - y*vx

        # Обратный кватернион q*
        rw = w
        rx = -x
        ry = -y
        rz = -z

        # Результат (q * v) * q*
        rx_new = qw*rx + qx*rw + qy*rz - qz*ry
        ry_new = qw*ry - qx*rz + qy*rw + qz*rx
        rz_new = qw*rz + qx*ry - qy*rx + qz*rw

        return torch.stack([rx_new, ry_new, rz_new], dim=1)


    def goal_reached(self, angle_threshold: float = 20, radius_threshold: float = 1.3, get_num_subs=False):
        """
        Проверяет достижение цели с учётом расстояния и направления взгляда робота.
        distance_to_goal: [N] расстояния до цели
        angle_threshold: максимально допустимый угол в радианах между направлением взгляда и вектором на цель
        Возвращает: [N] булев тензор, True если цель достигнута
        """
        root_pos_w = self._robot.data.root_pos_w[:, :2]

        distance_to_goal = torch.linalg.norm(self._desired_pos_w[:, :2] - root_pos_w, dim=1)
        close_enough = distance_to_goal <= radius_threshold
        root_quat_w = self._robot.data.root_quat_w  # shape [N, 4]

        # Локальный вектор взгляда робота (вперёд по оси X)
        local_forward = torch.tensor([1.0, 0.0, 0.0], device=root_quat_w.device, dtype=root_quat_w.dtype)
        local_forward = local_forward.unsqueeze(0).repeat(root_quat_w.shape[0], 1)  # [N, 3]

        # Вектор взгляда в мировых координатах
        forward_w = self.quat_rotate(root_quat_w, local_forward)  # [N, 3]

        root_pos_w = self._robot.data.root_pos_w  # [N, 3]
        to_goal = self._desired_pos_w - root_pos_w  # [N, 3]
        forward_w_norm = torch.nn.functional.normalize(forward_w[:, :2] , dim=1)
        to_goal_norm = torch.nn.functional.normalize(to_goal[:, :2] , dim=1)

        cos_angle = torch.sum(forward_w_norm * to_goal_norm, dim=1)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)  # для безопасности

        angle = torch.acos(cos_angle)
        angle_degrees = torch.abs(angle) * 180.0 / 3.141592653589793

        facing_goal = angle_degrees < angle_threshold

        conditions = torch.stack([close_enough, facing_goal], dim=1)  # shape [N, M]
        num_conditions_met = conditions.sum(dim=1)  # shape [N], количество True в каждой строк

        returns = torch.logical_and(close_enough, facing_goal)
        if self.TURN_TASK: #TODO: WRONG DOING
            returns = facing_goal
        if get_num_subs == False:
            return returns
        return returns, num_conditions_met, distance_to_goal+0.1-radius_threshold, angle_degrees

    def get_contact(self):
        force_matrix = self.scene["contact_sensor"].data.net_forces_w
        force_matrix[..., 2] = 0
        # вычисляем модуль силы для каждого контакта
        if force_matrix is not None and force_matrix.numel() > 0:
            contact_forces = torch.norm(force_matrix, dim=-1)
            num_contacts_per_env = torch.sum(contact_forces > 0.05, dim=1)
            high_contact_envs = num_contacts_per_env >= 1
        else:
            print("force_matrix_w is None or empty")
            high_contact_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return high_contact_envs

    def update_success_rate(self, goal_reached):
        """Update policy-only, controller-only and aggregate success rates.

        ``success_rate`` remains policy-only because curriculum learning must not
        be improved by expert-controlled episodes. The previous implementation
        returned immediately when every env was controlled, which made the only
        visible SR stay at 0 during full imitation.
        """
        died, time_out = self._get_dones(inner=True)
        completed = died | time_out

        if torch.any(completed):
            completed_ids = torch.where(completed)[0]
            for env_id_tensor in completed_ids:
                env_id = int(env_id_tensor.item())
                value = int(bool(goal_reached[env_id].item()))

                self.all_success_stacks[env_id].append(value)
                if len(self.all_success_stacks[env_id]) > self.max_stack_size:
                    self.all_success_stacks[env_id].pop(0)

                if env_id in self.controlled_env_ids:
                    self.controller_success_stacks[env_id].append(value)
                    if len(self.controller_success_stacks[env_id]) > self.max_stack_size:
                        self.controller_success_stacks[env_id].pop(0)
                else:
                    self.success_stacks[env_id].append(value)
                    if len(self.success_stacks[env_id]) > self.max_stack_size:
                        self.success_stacks[env_id].pop(0)

        def _rate(stacks, include_env):
            values = [
                result
                for env_id, stack in enumerate(stacks)
                if include_env(env_id)
                for result in stack
            ]
            return (
                100.0 * float(sum(values)) / float(len(values))
                if values else 0.0,
                len(values),
            )

        self.success_rate, policy_count = _rate(
            self.success_stacks,
            lambda env_id: env_id not in self.controlled_env_ids,
        )
        self.controller_success_rate, _ = _rate(
            self.controller_success_stacks,
            lambda env_id: env_id in self.controlled_env_ids,
        )
        self.all_success_rate, _ = _rate(
            self.all_success_stacks,
            lambda _env_id: True,
        )

        self.sr_stack_capacity = policy_count
        uncontrolled_count = self.num_envs - len(self.controlled_env_ids)
        if (
            uncontrolled_count > 0
            and policy_count >= 2 * uncontrolled_count * 0.9
        ):
            self.sr_stack_full = True

        return self.success_rate

    def out_of_bounds(self):
        if not self.TURN_TASK:
            poses = self.to_local(self._robot.data.root_pos_w)
            x, y = poses[..., 0], poses[..., 1]

            bounds = self.scene_manager.room_bounds
            xmin = bounds['x_min'] + 1.5
            xmax = bounds['x_max'] - 1.5
            ymin = bounds['y_min'] + 1.5
            ymax = bounds['y_max'] - 1.5
            inside_outer = (
                (x >= xmin) & (x <= xmax)
                & (y >= ymin) & (y <= ymax)
            )
            inside_active = (
                self.scene_manager.positions_in_active_navigation_area(poses)
            )
            return ~(inside_outer & inside_active)
        else:
            root_quat_w = self._robot.data.root_quat_w
            root_pos_w = self._robot.data.root_pos_w
            to_goal = self._desired_pos_w - root_pos_w
            local_forward = torch.tensor(
                [1.0, 0.0, 0.0],
                device=root_quat_w.device,
                dtype=root_quat_w.dtype,
            ).unsqueeze(0).repeat(root_quat_w.shape[0], 1)
            forward_w = self.quat_rotate(root_quat_w, local_forward)
            forward_w_norm = torch.nn.functional.normalize(
                forward_w[:, :2], dim=1
            )
            to_goal_norm = torch.nn.functional.normalize(
                to_goal[:, :2], dim=1
            )
            cos_angle = torch.sum(
                forward_w_norm * to_goal_norm, dim=1
            ).clamp(-1.0, 1.0)
            angle_degrees = torch.abs(torch.acos(cos_angle)) * 180.0 / math.pi
            return angle_degrees > 150

    def update_sr_stack(self):
        self.success_stacks = [[] for _ in range(self.num_envs)]  # Список списков для каждой среды
        self.sr_stack_full = False

    def _get_dones(self, inner=False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        inner flag - not changes in buffers
        """
        time_out = self.is_time_out(self.my_episode_lenght - 1)
        died = self.goal_reached(get_num_subs=False) | self.get_contact() | self.out_of_bounds() #| time_out

        if not inner:
            self.episode_length_buf[died] = 0
        return died, time_out
    
    def is_time_out(self, max_episode_length=256):
        if self.first_ep[1]:
            self.first_ep[1] = False
            max_episode_length = 2
        time_out = self.episode_length_buf >= max_episode_length
        return time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if self.first_ep[0] or env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES.clone()

        super()._reset_idx(env_ids)
        extras = dict()
        extras["Episode/success_rate"] = float(self.success_rate)
        extras["Episode/controller_success_rate"] = float(
            self.controller_success_rate
        )
        extras["Episode/all_success_rate"] = float(self.all_success_rate)
        extras["Episode/angle"] = float(self.cur_angle_error)
        extras["Episode/radius"] = float(self.mean_radius)
        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        possible_orientations = torch.tensor([0.0], device=self.device)

        E = len(env_ids)
        random_indices = torch.randint(0, len(possible_orientations), (E,), device=self.device)
        random_orientations = possible_orientations[random_indices]
        config = {"orientation": random_orientations}
        self.scene_manager.randomize_scene(env_ids=env_ids)
        self.scene_manager.get_graph_embedding(self.to_local(self._robot.data.root_pos_w), self._robot._ALL_INDICES.clone())
        goal_pos_local  = self.scene_manager.get_active_goal_state(env_ids)

        self._desired_pos_w[env_ids, :3] = goal_pos_local 
        self._desired_pos_w[env_ids, :2] = self.to_global(goal_pos_local , env_ids)

        if self.CL_ON:
            self.curriculum_learning_module(env_ids) 

        if (self.sr_stack_full and
            self.use_controller and
            not self.first_ep[0] or
            self.imitation):

            self.turn_on_controller_step = 0
            self._update_controlled_envs(env_ids)

        if (((self.turn_on_obstacles_always or self.warm and (self.mean_radius >= 3.5 or self.mean_radius <= 1.5))) and not self.first_ep[0]) and self.use_obstacles: # 
            if self.turn_on_obstacles_always and self.cur_step % 300:
                print("[ WARNING ] ostacles allways turn on")

            self.turn_on_obstacles = True
        else:
            self.turn_on_obstacles = False
        env_ids = env_ids.to(dtype=torch.long)

        self._reset_actor_on_episode_start(env_ids)

        method_name = f"place_robot_for_goal_stage_{self.stage}"
        if hasattr(self.scene_manager, method_name):
            method = getattr(self.scene_manager, method_name)
            robot_pos_local, robot_quats = method(
                config=config,
                env_ids=env_ids,
                mean_dist=self.mean_radius,
                min_dist=1.2,
                max_dist=8.0,
                angle_error=self.cur_angle_error,
            )
        else:
            print("WRANG STAGE")
        
        robot_pos  = robot_pos_local
        
        if self.use_controller or self.imitation:
            # Находим пути для всех ресетящихся (env_ids)
            env_ids_for_control = env_ids  # только те что сейчас ресетятся
            robot_pos_for_control = robot_pos
            goal_pos_for_control = goal_pos_local[:, :2]
            
            paths = None
            possible_try_steps = 3
            obstacle_positions_list = self.scene_manager.get_active_obstacle_positions_for_path_planning(
                env_ids_for_control
            )
            
            for i in range(possible_try_steps):
                paths = self.path_manager.get_paths(
                    env_ids=env_ids_for_control,
                    active_obstacles_by_type_list=obstacle_positions_list,
                    start_positions=robot_pos_for_control,
                    target_positions=goal_pos_for_control[:, :2]
                )
                if paths is None:
                    print(f"[ ERROR ] GET NONE PATH {i + 1} times")
                    self.scene_manager.randomize_scene(env_ids=env_ids_for_control)
                    goal_pos_local_retry = self.scene_manager.get_active_goal_state(env_ids_for_control)
                    self._desired_pos_w[env_ids_for_control, :3] = goal_pos_local_retry
                    self._desired_pos_w[env_ids_for_control, :2] = self.to_global(goal_pos_local_retry, env_ids_for_control)
                    goal_pos_for_control = goal_pos_local_retry[:, :2]
                else:
                    break
            self.control_module.update_paths(env_ids_for_control, paths, goal_pos_for_control)
        if self.memory_on:
            self.memory_manager.reset(env_ids)
        self._write_actor_state_to_sim(
            env_ids=env_ids,
            actor_pos_local=robot_pos,
            actor_quats=robot_quats,
        )
        self._update_scene_objects(env_ids) #self._robot._ALL_INDICES.clone())

        # Логируем длину эпизодов для сброшенных сред
        self.total_episode_length += torch.sum(self.episode_lengths[env_ids]).item()
        self.episode_count += len(env_ids)
        # Сбрасываем счетчик длины для сброшенных сред
        self.episode_lengths[env_ids] = 0
        _, _, r_error, a_error = self.goal_reached(get_num_subs=True)
        self.previous_distance_error[env_ids] = r_error[env_ids]
        self.previous_angle_error[env_ids] = a_error[env_ids]
        self.first_ep[0] = False

        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        reference_yaws = None
        prompts = self.scene_manager.get_navigation_prompts(
            env_ids=env_ids_t,
            radius=4.0,
            use_local_frame=True,
            reference_yaws=reference_yaws
        )
        if self.CAMERA:
            text_inputs = self.clip_processor(
                text=prompts, return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                text_embeddings = self.clip_model.get_text_features(**text_inputs)
                text_embeddings = text_embeddings / (text_embeddings.norm(dim=1, keepdim=True) + 1e-9)
            self.text_embeddings[env_ids] = text_embeddings
        self.scene_embeddings[env_ids] = self.scene_manager.encode_scene_graph(env_ids)

        # Логируем суммарные награды завершённых эпизодов в скользящее окно
        for env_idx in env_ids:
            self._episode_reward_window.append(
                float(self._current_episode_reward[env_idx].item())
            )
        # Обрезаем окно до последних N
        if len(self._episode_reward_window) > self._reward_window_size:
            self._episode_reward_window = self._episode_reward_window[-self._reward_window_size:]

        if self._episode_reward_window:
            self.mean_episode_reward = sum(self._episode_reward_window) / len(self._episode_reward_window)

        # Сбрасываем накопленную награду для ресетящихся сред
        self._current_episode_reward[env_ids] = 0.0

        if self.LOG and self.sr_stack_full:
            self.experiment.log_metric("success_rate", self.success_rate, step=self.tensorboard_step)
            self.experiment.log_metric("mean_radius", self.mean_radius, step=self.tensorboard_step)
            self.experiment.log_metric("max_angle", self.max_angle_error, step=self.tensorboard_step)
            # self.experiment.log_metric("use obstacles", self.turn_on_obstacles.float(), step=self.tensorboard_step)

    def _write_actor_state_to_sim(
        self,
        env_ids: torch.Tensor,
        actor_pos_local: torch.Tensor,
        actor_quats: torch.Tensor,
    ) -> None:
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :2] = self.to_global(actor_pos_local, env_ids)
        default_root_state[:, 3:7] = actor_quats
        default_root_state[:, 7:] = 0.0

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def to_local(self, pos, env_ids=None, env_origins=None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES.clone()
        if env_origins is None:
            env_origins = self._terrain.env_origins
        return pos[:, :2] - env_origins[env_ids, :2]
    
    def to_global(self, pos, env_ids=None, env_origins=None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES.clone()
        if env_origins is None:
            env_origins = self._terrain.env_origins
        return pos[:, :2] + env_origins[env_ids, :2]

    def curriculum_learning_module(self, env_ids: torch.Tensor):
        """
        Stage-based curriculum learning:
        
        Stage 0 (Warm):   0 → 1024 steps
                        Learning basics, all envs guided
        
        Stage 1 (Main):   1024+ steps → radius >= 6.0
                        Increase difficulty (angle → radius)
        
        Stage 2 (Final):  radius >= 6.0+
                        Maximum difficulty
        """
        
        # ============ ПЕРЕХОДЫ МЕЖДУ СТЕЙДЖАМИ ============
        
        # Stage 0 → 1: по времени
        if self.stage == 0 and self.cur_step >= self.warm_len:
            self.stage = 1
            self.mean_radius = self.start_mean_radius
            self.cur_angle_error = 0
            print(f"✓ [STAGE 0→1] Warm complete. Starting main training.")
        
        # Stage 1 → 2: по достижению радиуса
        elif self.stage == 1 and self.mean_radius >= 6.0:
            self.stage = 2
            self.cur_angle_error = 0
            print(f"✓ [STAGE 1→2] Final stage reached (radius={self.mean_radius:.1f}m)")
        
        # ============ ЛОГИКА СЛОЖНОСТИ (Stage 1+) ============
        
        if self.stage >= 1 and self.sr_stack_full:
            if self.success_rate >= self.sr_treshhold:
                self.success_ep_num += 1
                self.foult_ep_num = 0
                if self.success_ep_num > 512:
                    self.success_ep_num = 0
                    self._increase_difficulty()
            elif self.success_rate <= 70:
                self.foult_ep_num += 1
                if self.foult_ep_num > 15120:
                    self.success_ep_num = 0
                    self.foult_ep_num = 0
                    self._decrease_difficulty()

    def _increase_difficulty(self):
        """Увеличить сложность при высоком успехе"""
        # Сначала увеличиваем углы
        self.cur_angle_error += self.max_angle_error / 7
        
        # Когда углы исчерпаны → переходим на радиус
        if self.cur_angle_error > self.max_angle_error:
            self.cur_angle_error = 0
            increment = 0.5 if self.mean_radius == 0 else 1.0
            self.mean_radius = min(8.0, self.mean_radius + increment)
        
        print(f"[UP ↑] SR={self.success_rate:.0f}% → r={self.mean_radius:.1f}m, a={self.cur_angle_error:.2f}rad")
        self._step_update_counter = 0
        self.update_sr_stack()

    def _decrease_difficulty(self):
        """Уменьшить сложность при низком успехе"""
        # Уменьшаем радиус
        if self.cur_angle_error == 0:
            if self.mean_radius <= 0.5:
                self.mean_radius = 0
            elif self.mean_radius <= 1:
                self.mean_radius = 0.5
            else:
                self.mean_radius -= 0.5
            
            self.mean_radius = max(self.min_level_radius, self.mean_radius)
        
        self.cur_angle_error = 0
        self._step_update_counter = 0
        
        print(f"[DOWN ↓] SR={self.success_rate:.0f}% → r={self.mean_radius:.1f}m")
        self.update_sr_stack()

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass

    def close(self):
        super().close()

    def _update_scene_objects(self, env_ids: torch.Tensor):
        """Векторизованное обновление позиций всех объектов в симуляторе."""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES.clone()
        # Получаем все локальные позиции из scene_manager'а
        all_local_positions = self.scene_manager.positions
        
        # Конвертируем в глобальные координаты
        env_origins_expanded = self._terrain.env_origins.unsqueeze(1).expand_as(all_local_positions)
        all_global_positions = all_local_positions + env_origins_expanded
        
        # Создаем тензор для ориентации (по умолчанию Y-up: w=1)
        all_quats = torch.zeros(self.num_envs, self.scene_manager.num_total_objects, 4, device=self.device)
        all_quats[..., 0] = 1.0
        
        # Собираем полные состояния (поза + ориентация)
        all_root_states = torch.cat([all_global_positions, all_quats], dim=-1)
        
        # Итерируемся по объектам, управляемым симулятором
        for name, object_instances in self.scene_objects.items():
            if name not in self.scene_manager.object_map:
                continue
            
            # Получаем индексы для данного типа объектов
            indices = self.scene_manager.object_map[name]['indices']
            
            # Собираем состояния только для этих объектов
            object_root_states = all_root_states[:, indices, :]
            
            # Обновляем каждый экземпляр этого типа
            for i, instance in enumerate(object_instances):
                # Выбираем срез для i-го экземпляра по всем окружениям
                instance_states = object_root_states[:, i, :]
                # Применяем маску: неактивные объекты берём из default_positions
                active_mask = self.scene_manager.active[:, indices[i]]
                # Используем дефолтные позиции из SceneManager
                inactive_pos = self.scene_manager.default_positions[0, indices[i]]  # (3,)
                inactive_pos = inactive_pos.expand(self.num_envs, -1)  # (num_envs, 3)
                # Конвертируем в глобальные координаты
                inactive_pos_global = inactive_pos + env_origins_expanded[:, indices[i], :]
                # Векторизованное обновление позиций
                final_positions = torch.where(
                    active_mask.unsqueeze(-1),
                    instance_states[:, :3],
                    inactive_pos_global
                )
                instance_states[:, :3] = final_positions
                if name == "bowl":
                    rot = torch.tensor([0.0, 0.0, 0.7071, 0.7071], device=self.device).expand(self.num_envs, -1)
                    instance_states[:, 3:7] = rot
                if name == "cabinet":
                    # --- параметры и данные сцены ---
                    bounds = self.scene_manager.room_bounds  # {'x_min','x_max','y_min','y_max'}
                    margin = 0.03  # небольшой отступ от стены (м)
                    # размеры этого экземпляра во всех env (Bx3)
                    inst_size = self.scene_manager.sizes.expand(self.num_envs, -1, -1)[:, indices[i]]  # [N, 3]
                    half_x = inst_size[:, 0] * 0.5
                    half_y = inst_size[:, 1] * 0.5

                    # текущие (мировые) позиции для активных/неактивных уже собраны в instance_states[:, :3]
                    states = self.to_local(instance_states)
                    px = states[:, 0]
                    py = states[:, 1]

                    # расстояния до 4 стен (без учёта размера/отступа — для выбора ближайшей)
                    d_left   = (px - bounds['x_min']).abs()      # стена x_min
                    d_right  = (bounds['x_max'] - px).abs()      # стена x_max
                    d_bottom = (py - bounds['y_min']).abs()      # стена y_min
                    d_top    = (bounds['y_max'] - py).abs()      # стена y_max

                    # индекс ближайшей стены: 0=x_min, 1=x_max, 2=y_min, 3=y_max
                    dists = torch.stack([d_left, d_right, d_bottom, d_top], dim=1)  # [N, 4]
                    wall_idx = dists.argmin(dim=1)  # [N]
                    mask_x_walls = (wall_idx == 0) | (wall_idx == 1)  # стены "вдоль Y" (x фикс)
                    mask_y_walls = (wall_idx == 2) | (wall_idx == 3)  # стены "вдоль X" (y фикс)

                    # кватернионы в (w, x, y, z)
                    q_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=instance_states.dtype)
                    q_rot90z  = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=self.device, dtype=instance_states.dtype)

                    # если по умолчанию шкаф ориентирован вдоль X,
                    # то у "x-стен" (вдоль Y) — поверни на 90°; у "y-стен" — оставь identity
                    if mask_x_walls.any():
                        instance_states[mask_x_walls, 3:7] = q_identity.expand(mask_x_walls.sum(), 4)
                    if mask_y_walls.any():
                        instance_states[mask_y_walls, 3:7] = q_rot90z.expand(mask_y_walls.sum(), 4)

                # Записываем состояния в симулятор
                zero_vel = torch.zeros((env_ids.numel(), 6), device=self.device, dtype=instance_states.dtype)
                instance.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
                instance.write_root_pose_to_sim(instance_states[env_ids], env_ids=env_ids)

    def step(self, action: torch.Tensor):
        obs_buf, reward_buf, reset_terminated, reset_time_outs, extras = super().step(action)
        
        # ========== НАШЕ ОТСЛЕЖИВАНИЕ ==========
        # Записываем историю позиций объектов
        self.history_tracker.record_step(
            positions=self.scene_manager.positions,
            names=self.scene_manager.names,
            active=self.scene_manager.active,
            step_number=int(self.common_step_counter)
        )
       
        # ========== RETURN ==========
        tick_global_step(1)
        return obs_buf, reward_buf, reset_terminated, reset_time_outs, extras

    def setup_omni_warning_handler(self):
        import logging
        
        omni_logger = logging.getLogger("omni.usd")
        
        class BreakpointWarningHandler(logging.Handler):
            def __init__(self, env):
                super().__init__()
                self.env = env
                self.count = 0
            
            def emit(self, record):
                if "OrthogonalizeBasis did not converge" in record.getMessage():
                    self.count += 1
                    print(f"\n[ORTHONORMALIZE WARNING #{self.count}]")
                    print(f"Step: {self.env.common_step_counter}")
                    
                    # ✅ ОСТАНОВИТЬ В ДЕБАГЕ
                    import pdb
                    pdb.set_trace()
        
        handler = BreakpointWarningHandler(self)
        omni_logger.addHandler(handler)
        omni_logger.setLevel(logging.WARNING)

    def get_environment_which_is_closest_to_camera_lookat(self):
        env_positions = self._terrain.env_origins
        camera_lookat = torch.tensor(self.cfg.viewer.lookat).to(env_positions.device)

        # Small substraction to prefer environments which are closer from
        # positive side
        distances = torch.linalg.norm(env_positions - camera_lookat - 0.001, dim=-1)

        closest_env_idx = distances.argmin().item()
        return closest_env_idx

    def render_fpv(self):
        assert self.CAMERA, "Render is only available when CAMERA mode is enabled."
        # Choose an environment which is closest to the origin. A small number (0.001)
        # is substracted to prefer environments with origins
        camera_data = self._tiled_camera.data.output["rgb"].clone().cpu().numpy()  # Shape: (num_envs, 224, 224, 3)
        return camera_data


# Backward-compatible aliases for code that imports this module directly.
WheeledRobotEnvCfg = BaseWheeledRobotEnvCfg
WheeledRobotEnv = BaseWheeledRobotEnv
