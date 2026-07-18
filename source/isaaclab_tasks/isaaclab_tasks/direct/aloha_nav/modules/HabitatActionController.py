import torch
import math

class HabitatActionController:
    """
    Habitat-style macro-action controller.

    Actions:
        0 -> turn_left_30
        1 -> turn_right_30
        2 -> move_forward_25cm

    Один RL action запускает один macro-action.
    Внутри decimation контроллер каждый physics step задаёт скорости.
    Если action завершился раньше конца decimation, env стоит на месте.
    В конце decimation незавершённые action считаются timeout и перезаписываются
    на следующем _pre_physics_step.
    """

    ACTION_TURN_LEFT = 0
    ACTION_TURN_RIGHT = 1
    ACTION_FORWARD = 2

    def __init__(
        self,
        env,
        robot,
        num_envs: int,
        device,
        decimation: int,
        turn_angle_deg: float = 30.0,
        forward_distance: float = 0.25,
        yaw_tolerance_deg: float = 2.5,
        distance_tolerance: float = 0.02,
        kp_yaw: float = 5.0,
        kp_distance: float = 6.0,
        kp_heading: float = 4.0,
        max_linear_speed: float = 1.0,
        max_angular_speed: float = 2.0,
        print_stats: bool = True,
    ):
        self.env = env
        self.robot = robot
        self.num_envs = num_envs
        self.device = device
        self.decimation = int(decimation)

        self.turn_angle = math.radians(turn_angle_deg)
        self.forward_distance = float(forward_distance)
        self.yaw_tolerance = math.radians(yaw_tolerance_deg)
        self.distance_tolerance = float(distance_tolerance)

        self.kp_yaw = float(kp_yaw)
        self.kp_distance = float(kp_distance)
        self.kp_heading = float(kp_heading)
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.print_stats = bool(print_stats)

        # --- one-step history / per-env state ---
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.success = torch.zeros(num_envs, dtype=torch.bool, device=device)

        self.action_id = torch.zeros(num_envs, dtype=torch.long, device=device)

        self.start_pos_l = torch.zeros(num_envs, 2, dtype=torch.float32, device=device)
        self.start_yaw = torch.zeros(num_envs, dtype=torch.float32, device=device)

        self.target_yaw = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.forward_dir_l = torch.zeros(num_envs, 2, dtype=torch.float32, device=device)

        # Сколько physics/sim steps реально было отправлено для текущего macro-action.
        self.elapsed_sim_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

        # На каком sim step action завершился. -1 значит ещё не завершился.
        self.finished_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)

        self.active[env_ids] = False
        self.done[env_ids] = False
        self.success[env_ids] = False
        self.action_id[env_ids] = 0
        self.start_pos_l[env_ids] = 0.0
        self.start_yaw[env_ids] = 0.0
        self.target_yaw[env_ids] = 0.0
        self.forward_dir_l[env_ids] = 0.0
        self.elapsed_sim_steps[env_ids] = 0
        self.finished_step[env_ids] = -1

    @staticmethod
    def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _get_robot_pose_local(self):
        """
        Возвращает:
            pos_l: [num_envs, 2] позиция робота в локальных координатах env
            yaw:   [num_envs]    yaw робота в радианах
        """
        env_ids = self.robot._ALL_INDICES.clone()

        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w

        base_pos = torch.tensor([0.0, 0.0, 0.0], device=self.device, dtype=root_pos_w.dtype)
        base_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=root_quat_w.dtype)

        pos_nan = torch.isnan(root_pos_w).any(dim=-1) | torch.isinf(root_pos_w).any(dim=-1)
        quat_nan = torch.isnan(root_quat_w).any(dim=-1) | torch.isinf(root_quat_w).any(dim=-1)

        if pos_nan.any():
            print("[ ERROR ]: some pos is nan")
            root_pos_w = root_pos_w.clone()
            root_pos_w[pos_nan] = base_pos

        if quat_nan.any():
            print("[ ERROR ]: some quat is nan")
            root_quat_w = root_quat_w.clone()
            root_quat_w[quat_nan] = base_quat

        # Используем существующую функцию env: world -> env-local.
        # Она вычитает env_origins для каждой среды.
        pos_l = self.env.to_local(root_pos_w[:, :2], env_ids)

        # Isaac Lab quaternion здесь используется как [w, x, y, z],
        # как уже сделано в твоём aloha_env.py.
        w, x, y, z = root_quat_w[:, 0], root_quat_w[:, 1], root_quat_w[:, 2], root_quat_w[:, 3]
        yaw = torch.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

        return pos_l[:, :2], yaw

    def start(self, actions: torch.Tensor):
        """
        Вызывается один раз в _pre_physics_step.
        Полностью перезаписывает текущий macro-action для всех envs.
        """
        actions = actions.to(device=self.device)

        if actions.dim() > 1:
            actions = actions.squeeze(-1)

        actions = actions.long().clamp(0, 2)

        pos_l, yaw = self._get_robot_pose_local()

        self.action_id[:] = actions

        self.start_pos_l[:] = pos_l
        self.start_yaw[:] = yaw

        self.active[:] = True
        self.done[:] = False
        self.success[:] = False
        self.elapsed_sim_steps[:] = 0
        self.finished_step[:] = -1

        # Цели для поворотов.
        turn_left = actions == self.ACTION_TURN_LEFT
        turn_right = actions == self.ACTION_TURN_RIGHT

        self.target_yaw[:] = yaw
        self.target_yaw[turn_left] = self.wrap_to_pi(yaw[turn_left] + self.turn_angle)
        self.target_yaw[turn_right] = self.wrap_to_pi(yaw[turn_right] - self.turn_angle)

        # Направление движения вперёд в env-local XY.
        self.forward_dir_l[:, 0] = torch.cos(yaw)
        self.forward_dir_l[:, 1] = torch.sin(yaw)

    def step(self):
        """
        Вызывается каждый physics step из _apply_action.

        Returns:
            linear_speed:  [num_envs]
            angular_speed: [num_envs]
        """
        pos_l, yaw = self._get_robot_pose_local()

        linear_speed = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        angular_speed = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        running = self.active & (~self.done)
        if not running.any():
            return linear_speed, angular_speed

        is_turn = running & (
            (self.action_id == self.ACTION_TURN_LEFT) |
            (self.action_id == self.ACTION_TURN_RIGHT)
        )
        is_forward = running & (self.action_id == self.ACTION_FORWARD)

        # ============================================================
        # 1. Проверяем завершение поворотов
        # ============================================================
        if is_turn.any():
            yaw_error = self.wrap_to_pi(self.target_yaw - yaw)
            turn_reached = is_turn & (torch.abs(yaw_error) <= self.yaw_tolerance)

            if turn_reached.any():
                self.done[turn_reached] = True
                self.success[turn_reached] = True
                self.active[turn_reached] = False
                self.finished_step[turn_reached] = self.elapsed_sim_steps[turn_reached]

            still_turn = is_turn & (~self.done)

            if still_turn.any():
                yaw_error = self.wrap_to_pi(self.target_yaw - yaw)
                # angular_speed[still_turn] = torch.clamp(
                #     self.kp_yaw * yaw_error[still_turn],
                #     -self.max_angular_speed,
                #     self.max_angular_speed,
                # )
                turn_sign = torch.sign(yaw_error[still_turn])
                turn_sign[turn_sign == 0] = 1.0

                angular_speed[still_turn] = torch.clamp(
                    turn_sign * self.max_angular_speed,
                    -self.max_angular_speed,
                    self.max_angular_speed,
                )
        # ============================================================
        # 2. Проверяем завершение движения вперёд
        # ============================================================
        if is_forward.any():
            delta = pos_l - self.start_pos_l
            progress = torch.sum(delta * self.forward_dir_l, dim=-1)

            forward_reached = is_forward & (
                progress >= self.forward_distance - self.distance_tolerance
            )

            if forward_reached.any():
                self.done[forward_reached] = True
                self.success[forward_reached] = True
                self.active[forward_reached] = False
                self.finished_step[forward_reached] = self.elapsed_sim_steps[forward_reached]

            still_forward = is_forward & (~self.done)

            if still_forward.any():
                distance_left = self.forward_distance - progress
                lin = self.kp_distance * distance_left
                lin = torch.clamp(lin, 0.0, self.max_linear_speed)

                # Держим yaw, который был на старте move_forward.
                heading_error = self.wrap_to_pi(self.start_yaw - yaw)
                ang = torch.clamp(
                    self.kp_heading * heading_error,
                    -self.max_angular_speed,
                    self.max_angular_speed,
                )

                linear_speed[still_forward] = lin[still_forward]
                angular_speed[still_forward] = ang[still_forward]

        # Считаем только те envs, которым реально отправили команду на этот physics step.
        commanded = self.active & (~self.done)
        self.elapsed_sim_steps[commanded] += 1

        return linear_speed, angular_speed

    def finish_decimation_and_print(self, rl_step: int | None = None):
        """
        Вызывается один раз на substep == decimation - 1.

        Все незавершённые macro-actions считаются timeout.
        После этого новый _pre_physics_step перезапишет задачу.
        """
        timeout = self.active & (~self.done)

        if timeout.any():
            self.done[timeout] = True
            self.success[timeout] = False
            self.active[timeout] = False
            self.finished_step[timeout] = self.decimation

        if self.print_stats:
            self._print_stats(rl_step=rl_step)

    def _format_group_stats(self, name: str, mask: torch.Tensor) -> str:
        n = int(mask.sum().item())
        if n == 0:
            return f"{name}: n=0"

        steps = self.finished_step[mask].to(torch.float32)
        success_count = int(self.success[mask].sum().item())
        fail_count = n - success_count

        mean = steps.mean().item()
        var = steps.var(unbiased=False).item()
        min_v = int(steps.min().item())
        max_v = int(steps.max().item())

        return (
            f"{name}: "
            f"n={n}, "
            f"success={success_count}, "
            f"fail={fail_count}, "
            f"steps_mean={mean:.2f}, "
            f"steps_var={var:.2f}, "
            f"steps_min={min_v}, "
            f"steps_max={max_v}"
        )

    def _print_stats(self, rl_step: int | None = None):
        is_turn = (
            (self.action_id == self.ACTION_TURN_LEFT) |
            (self.action_id == self.ACTION_TURN_RIGHT)
        )
        is_forward = self.action_id == self.ACTION_FORWARD

        prefix = "[HabitatAction]"
        if rl_step is not None:
            prefix += f" rl_step={rl_step}"

        all_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        all_stats = self._format_group_stats("all", all_mask)
        turn_stats = self._format_group_stats("turn", is_turn)
        forward_stats = self._format_group_stats("forward", is_forward)

        print(f"all_stats | {prefix} {all_stats}]")
        print(f"turn_stats | {turn_stats}")
        print(f"forward_stats | {forward_stats}")