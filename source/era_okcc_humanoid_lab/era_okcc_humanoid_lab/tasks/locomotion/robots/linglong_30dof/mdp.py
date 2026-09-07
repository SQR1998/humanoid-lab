"""LingLong-specific locomotion command and reward helpers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from ...mdp.commands import CustomUniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class LingLongUniformVelocityCommand(CustomUniformVelocityCommand):
    """L7 gait generator adapted to LingLong 2.0 leg geometry."""

    thigh_length = 0.3823
    shank_length = 0.4200
    nominal_hip_to_ankle = (thigh_length + shank_length) * math.cos(math.pi / 6.0)

    _neutral_knee = math.acos(
        (nominal_hip_to_ankle**2 - thigh_length**2 - shank_length**2)
        / (2.0 * thigh_length * shank_length)
    )
    _neutral_hip = -math.atan2(
        shank_length * math.sin(_neutral_knee),
        thigh_length + shank_length * math.cos(_neutral_knee),
    )

    def _calculate_ik_xz(self, x: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map desired sagittal foot offsets to hip/knee offsets.

        The zero-foot-height solution is offset to zero joint position so the
        reference remains compatible with the neutral URDF pose.
        """
        x_target = x
        z_target = self.nominal_hip_to_ankle - z

        distance = torch.sqrt(x_target.square() + z_target.square())
        min_reach = abs(self.thigh_length - self.shank_length) + 1.0e-4
        max_reach = self.thigh_length + self.shank_length - 1.0e-4
        clipped_distance = torch.clamp(distance, min=min_reach, max=max_reach)
        scale = clipped_distance / torch.clamp(distance, min=1.0e-6)
        x_target = x_target * scale
        z_target = z_target * scale

        cos_knee = (
            x_target.square() + z_target.square() - self.thigh_length**2 - self.shank_length**2
        ) / (2.0 * self.thigh_length * self.shank_length)
        knee = torch.acos(torch.clamp(cos_knee, -1.0, 1.0))
        hip = torch.atan2(-x_target, z_target) - torch.atan2(
            self.shank_length * torch.sin(knee),
            self.thigh_length + self.shank_length * torch.cos(knee),
        )

        return hip - self._neutral_hip, knee - self._neutral_knee

    def _update_command(self) -> None:
        """Update walking references and replace standing references with a static stance."""
        super()._update_command()

        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        if standing_env_ids.numel() == 0:
            return

        self.ref_action[standing_env_ids] = 0.0
        self.stance_mask[standing_env_ids] = True
        self.swing_phase[standing_env_ids] = 0.0
        self.stance_phase[standing_env_ids] = 0.0
        self.contact_number_des[standing_env_ids] = 2
        self.feet_desired_x[standing_env_ids] = 0.0
        self.feet_desired_z[standing_env_ids] = 0.0


def _standing_mask(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return a float mask that is one only for explicit standing samples."""
    command: LingLongUniformVelocityCommand = env.command_manager.get_term(command_name)
    return command.is_standing_env.float()


def standing_leg_joint_pos_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize leg pose deviation from the default pose only while standing."""
    asset = env.scene[asset_cfg.name]
    joint_error = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.mean(joint_error.square(), dim=1) * _standing_mask(env, command_name)


def standing_leg_joint_vel_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize moving leg joints only while standing."""
    asset = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(joint_vel.square(), dim=1) * _standing_mask(env, command_name)


def standing_feet_lin_vel_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize foot motion only while standing."""
    asset = env.scene[asset_cfg.name]
    feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    return torch.sum(feet_vel.square(), dim=(1, 2)) * _standing_mask(env, command_name)


def standing_feet_airborne(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize either foot losing ground contact only while standing."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > threshold
    airborne_ratio = torch.mean((~feet_contact).float(), dim=1)
    return airborne_ratio * _standing_mask(env, command_name)


def feet_contact_number(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward left/right foot contact matching the desired stance mask."""
    command: LingLongUniformVelocityCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > threshold
    return torch.mean(torch.where(feet_contact == command.stance_mask, 1.0, -0.2), dim=1)


def feet_contact_number_sum(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward the desired number of contacting feet."""
    command: LingLongUniformVelocityCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > threshold
    actual_contact_num = torch.sum(feet_contact.float(), dim=1)
    return torch.where(actual_contact_num == command.contact_number_des, 1.0, -0.2)


def bad_contacts_task(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate when any configured LingLong trunk body exceeds the contact threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_history = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    force_magnitude = torch.linalg.vector_norm(force_history, dim=-1)
    return torch.amax(force_magnitude, dim=(1, 2)) > threshold
