"""Load LingLong 2.0 in Isaac Sim and hold its neutral standing pose."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Visual and physics check for the LingLong 2.0 URDF.")
parser.add_argument("--steps", type=int, default=0, help="Number of physics steps; 0 runs until the window closes.")
parser.add_argument("--print_interval", type=int, default=100, help="Status print interval in physics steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from era_okcc_humanoid_lab.robots.linglong_30dof import LINGLONG_30DOF_CFG


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    sim.set_camera_view(eye=[2.5, 2.5, 1.8], target=[0.0, 0.0, 0.9])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = LINGLONG_30DOF_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(cfg=robot_cfg)

    sim.reset()
    if robot.num_joints != 30:
        raise RuntimeError(f"Expected 30 joints, imported {robot.num_joints}: {robot.joint_names}")
    if robot.num_bodies != 31:
        raise RuntimeError(f"Expected 31 bodies, imported {robot.num_bodies}: {robot.body_names}")

    print(f"[INFO] Imported {robot.num_bodies} bodies and {robot.num_joints} joints.")
    print(f"[INFO] Joint names: {robot.joint_names}")

    sim_dt = sim.get_physics_dt()
    step = 0
    while simulation_app.is_running() and (args_cli.steps <= 0 or step < args_cli.steps):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        step += 1

        if step % args_cli.print_interval == 0:
            root_z = robot.data.root_pos_w[0, 2].item()
            max_joint_error = torch.max(
                torch.abs(robot.data.joint_pos - robot.data.default_joint_pos)
            ).item()
            finite = bool(
                torch.isfinite(robot.data.root_state_w).all().item()
                and torch.isfinite(robot.data.joint_pos).all().item()
            )
            print(
                f"[CHECK] step={step:6d} root_z={root_z:.4f} "
                f"max_joint_error={max_joint_error:.4f} finite={finite}"
            )
            if not finite:
                raise RuntimeError("Non-finite LingLong simulation state detected.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
