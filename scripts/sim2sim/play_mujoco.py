# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

import time
import threading
import socket
import struct

import numpy as np
import torch
import mujoco
import mujoco.viewer
from cc.udp import UDP

from berkeley_humanoid_lite_lowlevel.policy.policy_runner import parse_arguments, PolicyRunner
from berkeley_humanoid_lite_lowlevel.policy.udp_joystick import UdpJoystick

class Ema:
    def __init__(self, cutoff_freq: float, dt: float, init_value: torch.Tensor):
        self.alpha = dt / (dt + 1 / (2 * np.pi * cutoff_freq))
        self.value = init_value

    def filter(self, x: torch.Tensor):
        self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value



cfg = parse_arguments()

if not cfg:
    raise ValueError(f"Failed to load config.")


def quat_rotate_inverse(q, v):
    q_w = q[0]
    q_vec = q[1:4]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (torch.dot(q_vec, v)) * 2.0
    return a - b + c



class Cfg:

    # Physics configurations
    policy_dt: float = cfg.policy_dt

    physics_dt: float = cfg.physics_dt

    cutoff_freq: float = cfg.cutoff_freq

    # Articulation configurations
    num_joints = cfg.num_joints
    joint_kp = cfg.joint_kp
    joint_kd = cfg.joint_kd
    
    default_joint_positions = cfg.default_joint_positions
    
    num_actions = cfg.num_actions
    action_indices = cfg.action_indices
    default_base_position = cfg.default_base_position
    effort_limits = cfg.effort_limits


class MujocoEnv:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg

        self.physics_substeps = int(np.round(self.cfg.policy_dt / self.cfg.physics_dt))

        if cfg.num_joints == 22:
            self.mj_model = mujoco.MjModel.from_xml_path("source/berkeley_humanoid_lite_assets/data/mjcf/bhl_scene.xml")
        else:
            self.mj_model = mujoco.MjModel.from_xml_path("source/berkeley_humanoid_lite_assets/data/mjcf/bhl_biped_scene.xml")
        
        self.mj_data = mujoco.MjData(self.mj_model)
        
        self.mj_model.opt.timestep = self.cfg.physics_dt

        self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)


        self.sensordata_dof_size = 3 * self.mj_model.nu

        self.gravity_vector = torch.tensor([0.0, 0.0, -1.0])

        self.joint_kp = torch.zeros((self.cfg.num_joints,), dtype=torch.float32)
        self.joint_kd = torch.zeros((self.cfg.num_joints,), dtype=torch.float32)
        self.effort_limits = torch.zeros((self.cfg.num_joints,), dtype=torch.float32)

        self.joint_kp[:] = torch.tensor(self.cfg.joint_kp)
        self.joint_kd[:] = torch.tensor(self.cfg.joint_kd)
        self.effort_limits[:] = torch.tensor(self.cfg.effort_limits)

        self.n_steps = 0

        print("Policy frequency: ", 1 / self.cfg.policy_dt)
        print("Physics frequency: ", 1 / self.cfg.physics_dt)
        print("Physics substeps: ", self.physics_substeps)

        self.is_killed = threading.Event()

        # default to rl control mode
        self.mode = 3.0
        self.command_velocity_x = 0.0
        self.command_velocity_y = 0.0
        self.command_velocity_yaw = 0.0

        self.joystick_thread = threading.Thread(target=self.joystick_receive_thread, daemon=True)
        self.joystick_thread.start()

        self.ema_filter = Ema(cutoff_freq=self.cfg.cutoff_freq, dt=self.cfg.physics_dt, init_value=torch.zeros((self.cfg.num_joints,)))
    
    def joystick_receive_thread(self):
        # Create UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Bind to empty string to listen on all interfaces
        server_address = ("0.0.0.0", 10011)  # Empty string means listen on all available interfaces
        sock.bind(server_address)
        
        print(f"Listening for UDP broadcast messages on port {server_address[1]}")
        
        while not self.is_killed.is_set():
            # Receive data (buffer size of 1024 bytes should be plenty for joystick data)
            data, address = sock.recvfrom(1024)
            

            command_mode_switch, command_velocity_x, command_velocity_y, command_velocity_yaw = struct.unpack("<Bfff", data)

            if command_mode_switch != 0:
                self.mode = command_mode_switch
            self.command_velocity_x = command_velocity_x
            self.command_velocity_y = command_velocity_y
            self.command_velocity_yaw = command_velocity_yaw

        sock.close()


    def reset(self):
        self.mj_data.qpos[0:3] = self.cfg.default_base_position
        self.mj_data.qpos[3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.mj_data.qpos[7:] = self.cfg.default_joint_positions
        self.mj_data.qvel[:] = 0

        observations = self._get_observations()

        return observations
    
    def step(self, actions: torch.Tensor) -> torch.Tensor:
        step_start_time = time.perf_counter()

        for _ in range(self.physics_substeps):
            self._apply_actions(actions)
            mujoco.mj_step(self.mj_model, self.mj_data)

        self.mj_viewer.sync()

        observations = self._get_observations()

        time_until_next_step = self.cfg.policy_dt - (time.perf_counter() - step_start_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        self.n_steps += 1

        return observations

    def _apply_actions(self, actions: torch.Tensor):
        target_positions = torch.zeros((self.cfg.num_joints,))
        target_positions[self.cfg.action_indices] = actions

        output_torques = self.joint_kp * (target_positions - self._get_joint_pos()) + self.joint_kd * (-self._get_joint_vel())
        
        output_torques_filtered = self.ema_filter.filter(output_torques)
        
        output_torques_clipped = torch.clip(output_torques_filtered, -self.effort_limits, self.effort_limits)

        self.mj_data.ctrl[:] = output_torques_clipped.numpy()

    def _get_base_pos(self):
        return torch.tensor(self.mj_data.qpos[:3], dtype=torch.float32)
    
    def _get_base_quat(self):
        # return torch.tensor(self.mj_data.qpos[3:7], dtype=torch.float32)
        return torch.tensor(self.mj_data.sensordata[self.sensordata_dof_size+0:self.sensordata_dof_size+4], dtype=torch.float32)
    
    def _get_base_ang_vel(self):
        # return torch.tensor(self.mj_data.qvel[3:6], dtype=torch.float32)
        return torch.tensor(self.mj_data.sensordata[self.sensordata_dof_size+4:self.sensordata_dof_size+7], dtype=torch.float32)
    
    def _get_projected_gravity(self):
        base_quat = self._get_base_quat()
        projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vector)
        return projected_gravity
    
    def _get_joint_pos(self):
        # return torch.tensor(self.mj_data.qpos[7:7+self.cfg.num_joints], dtype=torch.float32)
        return torch.tensor(self.mj_data.sensordata[0:self.cfg.num_joints], dtype=torch.float32)

    def _get_joint_vel(self):
        # return torch.tensor(self.mj_data.qvel[6:6+self.cfg.num_joints], dtype=torch.float32)
        return torch.tensor(self.mj_data.sensordata[self.cfg.num_joints:2*self.cfg.num_joints], dtype=torch.float32)

    def _get_observations(self) -> torch.Tensor:
        return torch.cat([
            self._get_base_quat(),
            self._get_base_ang_vel(),
            self._get_joint_pos()[self.cfg.action_indices],
            self._get_joint_vel()[self.cfg.action_indices],
            torch.tensor([self.mode, self.command_velocity_x, self.command_velocity_y, self.command_velocity_yaw], dtype=torch.float32),
        ], dim=-1)


env = MujocoEnv(Cfg())

obs = env.reset()

udp = UDP((cfg.ip_robot_addr, cfg.ip_policy_acs_port), (cfg.ip_host_addr, cfg.ip_policy_obs_port))
# udp = UDP(("0.0.0.0", cfg.ip_policy_acs_port), ("127.0.0.1", cfg.ip_policy_obs_port))

runner = PolicyRunner(cfg)
runner_thread = threading.Thread(target=runner.run, daemon=True)
runner_thread.start()

stick = UdpJoystick(publish_address="127.0.0.1", publish_port=10011)
stick_thread = threading.Thread(target=stick.run, daemon=True)
stick_thread.start()


default_actions = np.array(cfg.default_joint_positions, dtype=np.float32)[env.cfg.action_indices]

while True:
    udp.send_numpy(obs.numpy())

    actions = udp.recv_numpy(dtype=np.float32, timeout=0.2)
    if actions is None:
        actions = default_actions
    
    actions = torch.tensor(actions)
    obs = env.step(actions)


