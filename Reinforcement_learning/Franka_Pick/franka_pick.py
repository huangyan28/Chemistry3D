from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.franka import Franka
from omniisaacgymenvs.robots.articulations.views.franka_view import FrankaView

from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.prims import RigidPrim, RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import GeometryPrim

from omni.isaac.cloner import Cloner

import numpy as np
import torch
import math

from pxr import Usd, UsdGeom, UsdPhysics

class FrankaPickTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # Number of environments and spacing between them
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        # Maximum length of each episode
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        # Scale for action and noise parameters
        self.action_scale = self._task_cfg["env"]["actionScale"]
        self.start_position_noise = self._task_cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self._task_cfg["env"]["startRotationNoise"]
        self.num_props = self._task_cfg["env"]["numProps"]

        # Reward scales
        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.target_lift_height = 0.3
        self.grasp_reward_scale = self._task_cfg["env"]["graspRewardScale"]
        self.lift_reward_scale = self._task_cfg["env"]["liftRewardScale"]
        self.around_handle_reward_scale = self._task_cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self._task_cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self._task_cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.finger_close_reward_scale = self._task_cfg["env"]["fingerCloseRewardScale"]
        self.grasp_penalty_scale = self._task_cfg["env"]["grasp_penalty_scale"]

        self.distX_offset = 0.04
        self.dt = 1/60.

        self._num_observations = 21
        self._num_actions = 9

        # Counters for steps, episodes, and epochs


        # Initialize RLTask
        RLTask.__init__(self, name, env)
        return

    def set_up_scene(self, scene) -> None:
        """
        Set up the scene by adding the Franka robot and the beaker.
        """
        self.get_franka()
        self.get_beaker()

        super().set_up_scene(scene)

        # Create views for Franka and beaker
        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")
        self._beaker = RigidPrimView(prim_paths_expr="/World/envs/.*/beaker", name="beaker_view")

        # Add views to the scene
        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)
        scene.add(self._beaker)

        self.init_data()
        return

    def get_franka(self):
        """
        Add Franka robot to the scene.
        """
        franka = Franka(prim_path=self.default_zero_env_path + "/franka", name="franka", usd_path="/home/huangyan/.local/share/ov/pkg/isaac_sim-2023.1.1/standalone_examples/Chem_lab/Franka/franka_instanceable.usd")
        self._sim_config.apply_articulation_settings("franka", get_prim_at_path(franka.prim_path), self._sim_config.parse_actor_config("franka"))

    def get_beaker(self):
        """
        Add a beaker to the scene.
        """
        beaker_path = '/home/huangyan/.local/share/ov/pkg/isaac_sim-2023.1.1/standalone_examples/Chem_lab/Object_Usd/250_ml_beaker.usd'
        add_reference_to_stage(usd_path=beaker_path, prim_path=self.default_zero_env_path + "/beaker")
        beaker = GeometryPrim(
            self.default_zero_env_path + "/beaker",
            name="beaker",
            position=torch.tensor([0.6, 0.25, 0.0], dtype=torch.float32),
            orientation=torch.tensor([1, 0.0, 0.0, 0.0], dtype=torch.float32),
            scale=torch.tensor([0.6, 0.6, 0.6], dtype=torch.float32),
        )
        self._sim_config.apply_articulation_settings("beaker", get_prim_at_path(beaker.prim_path), self._sim_config.parse_actor_config("beaker"))

    def init_data(self) -> None:
        """
        Initialize data for the task.
        """
        def get_env_local_pose(env_pos, xformable, device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device, dtype=torch.float)

        stage = get_current_stage()
        hand_pose = get_env_local_pose(self._env_pos[0], UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_link7")), self._device)
        lfinger_pose = get_env_local_pose(self._env_pos[0], UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_leftfinger")), self._device)
        rfinger_pose = get_env_local_pose(self._env_pos[0], UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_rightfinger")), self._device)

        finger_pose = torch.zeros(7, device=self._device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = (tf_inverse(hand_pose[3:7], hand_pose[0:3]))

        grasp_pose_axis = 1
        franka_local_grasp_pose_rot, franka_local_pose_pos = tf_combine(hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3])
        self.franka_local_grasp_pos = franka_local_pose_pos.repeat((self._num_envs, 1))
        self.franka_local_grasp_rot = franka_local_grasp_pose_rot.repeat((self._num_envs, 1))

        beaker_local_grasp_pose = torch.tensor([0.0, 0.0, 0.03, 1.0, 0.0, 0.0, 0.0], device=self._device)
        self.beaker_local_grasp_pos = beaker_local_grasp_pose[0:3].repeat((self._num_envs, 1))
        self.beaker_local_grasp_rot = beaker_local_grasp_pose[3:7].repeat((self._num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))
        self.forward_axis = torch.tensor([-1, 0, 0], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))
        self.beaker_up_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))

        self.franka_default_dof_pos = torch.tensor([1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035], device=self._device)

        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)

    def get_observations(self) -> dict:
        """
        Get observations from the environment.
        """
        hand_pos, hand_rot = self._frankas._hands.get_world_poses(clone=False)
        beaker_pos, beaker_rot = self._beaker.get_world_poses(clone=False)
        self.beaker_height = beaker_pos[:, 2]

        franka_dof_pos = self._frankas.get_joint_positions(clone=False)
        franka_dof_vel = self._frankas.get_joint_velocities(clone=False)
        self.franka_dof_pos = franka_dof_pos

        (
            self.franka_grasp_rot,
            self.franka_grasp_pos,
            self.beaker_grasp_rot,
            self.beaker_grasp_pos
        ) = self.compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.franka_local_grasp_rot,
            self.franka_local_grasp_pos,
            beaker_rot,
            beaker_pos,
            self.beaker_local_grasp_rot,
            self.beaker_local_grasp_pos,
        )

        self.franka_lfinger_pos, self.franka_lfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)
        self.franka_rfinger_pos, self.franka_rfinger_rot = self._frankas._rfingers.get_world_poses(clone=False)

        dof_pos_scaled = (
            2.0 * (franka_dof_pos - self.franka_dof_lower_limits) / (self.franka_dof_upper_limits - self.franka_dof_lower_limits) - 1.0
        )
        to_target = self.beaker_grasp_pos - self.franka_grasp_pos

        self.obs_buf = torch.cat((dof_pos_scaled, franka_dof_vel * self.dof_vel_scale, to_target), dim=-1)

        observations = {
            self._frankas.name: {
                "obs_buf": self.obs_buf
            }
        }

        return observations

    def pre_physics_step(self, actions) -> None:
        """
        Perform actions before each physics step.
        """
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        targets = self.franka_dof_targets + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.franka_dof_targets[:] = tensor_clamp(targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        env_ids_int32 = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)

        self._frankas.set_joint_position_targets(self.franka_dof_targets, indices=env_ids_int32)

    def reset_idx(self, env_ids):
        """
        Reset environments by indices.
        """
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # Reset Franka
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self._device) - 0.5),
            self.franka_dof_lower_limits,
            self.franka_dof_upper_limits,
        )
        dof_pos = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_pos[:, :] = pos
        self.franka_dof_targets[env_ids, :] = pos
        self.franka_dof_pos[env_ids, :] = pos

        # Reset beaker
        self._beaker.set_world_poses(self.default_beaker_pos[env_ids], self.default_beaker_rot[env_ids], env_ids.to(torch.int32))

        self._frankas.set_joint_position_targets(self.franka_dof_targets[env_ids], indices=indices)
        self._frankas.set_joint_positions(dof_pos, indices=indices)
        self._frankas.set_joint_velocities(dof_vel, indices=indices)

        # Bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        """
        Perform operations after resetting the environment.
        """
        self.num_franka_dofs = self._frankas.num_dof
        self.franka_dof_pos = torch.zeros((self._num_envs, self.num_franka_dofs), device=self._device)
        dof_limits = self._frankas.get_dof_limits()
        self.franka_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.franka_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[self._frankas.gripper_indices] = 0.1
        self.franka_dof_targets = torch.zeros((self._num_envs, self.num_franka_dofs), dtype=torch.float, device=self._device)

        self.default_beaker_pos, self.default_beaker_rot = self._beaker.get_world_poses()

        # Randomize all environments
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        """
        Calculate reward metrics for the task.
        """
        self.rew_buf[:] = self.compute_pick_and_lift_reward(
            self.reset_buf, self.progress_buf, self.actions,
            self.franka_grasp_pos, self.beaker_grasp_pos, self.franka_grasp_rot, self.beaker_grasp_rot,
            self.franka_lfinger_pos, self.franka_rfinger_pos,
            self.gripper_forward_axis, self.gripper_up_axis, self.beaker_up_axis,
            self._num_envs, self.dist_reward_scale, self.rot_reward_scale, self.grasp_reward_scale, self.lift_reward_scale,
            self.around_handle_reward_scale, self.grasp_penalty_scale, self.finger_dist_reward_scale, self.action_penalty_scale, self.target_lift_height, self._max_episode_length, self.franka_dof_pos,
            self.finger_close_reward_scale,
        )

    def is_done(self) -> None:
        """
        Determine if the task is done.
        """
        # Reset if beaker is lifted to the target height or max length reached
        height_diff = torch.abs(self.beaker_height - self.target_lift_height)
        close_to_target_height = height_diff < 0.02

        self.reset_buf = torch.where(close_to_target_height, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

    def compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        beaker_rot,
        beaker_pos,
        beaker_local_grasp_rot,
        beaker_local_grasp_pos,
    ):
        """
        Compute grasp transforms for the Franka robot and the beaker.
        """
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )

        global_beaker_rot, global_beaker_pos = tf_combine(
            beaker_rot, beaker_pos, beaker_local_grasp_rot, beaker_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos, global_beaker_rot, global_beaker_pos

    def compute_pick_and_lift_reward(
        self, reset_buf, progress_buf, actions,
        franka_grasp_pos, beaker_grasp_pos, franka_grasp_rot, beaker_grasp_rot,
        franka_lfinger_pos, franka_rfinger_pos,
        gripper_forward_axis, gripper_up_axis, beaker_up_axis,
        num_envs, dist_reward_scale, rot_reward_scale, grasp_reward_scale, lift_reward_scale,
        around_handle_reward_scale, grasp_penalty_scale, finger_dist_reward_scale, action_penalty_scale, target_lift_height, max_episode_length, joint_positions,
        finger_close_reward_scale
    ):
        """
        Compute the reward for picking and lifting the beaker.
        """
        # Distance reward: distance between the gripper and the beaker
        d = torch.norm(franka_grasp_pos - beaker_grasp_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d ** 2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.05, dist_reward * 2, dist_reward)  # Increase reward if very close

        # Alignment reward: gripper orientation alignment
        axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
        axis4 = torch.tensor([1, 0, 0], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))
        dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        rot_reward = torch.sign(dot2) * dot2 ** 2

        around_handle_reward = torch.zeros_like(rot_reward)
        around_handle_reward = torch.where(
            franka_lfinger_pos[:, 0] > beaker_grasp_pos[:, 0],
            torch.where(
                franka_rfinger_pos[:, 0] < beaker_grasp_pos[:, 0], around_handle_reward + 0.5, around_handle_reward
            ),
            around_handle_reward,
        )

        # Reward for the distance of each finger from the beaker
        finger_dist_reward = torch.zeros_like(rot_reward)
        lfinger_dist = torch.abs(franka_lfinger_pos[:, 0] - beaker_grasp_pos[:, 0])
        rfinger_dist = torch.abs(franka_rfinger_pos[:, 0] - beaker_grasp_pos[:, 0])
        finger_dist_reward = torch.where(
            franka_lfinger_pos[:, 0] > beaker_grasp_pos[:, 0],
            torch.where(
                franka_rfinger_pos[:, 0] < beaker_grasp_pos[:, 0],
                (0.04 - lfinger_dist) + (0.04 - rfinger_dist),
                finger_dist_reward,
            ),
            finger_dist_reward,
        )

        grasp_reward = torch.zeros_like(rot_reward)
        grasp_reward = torch.where(d <= 0.05, (0.04 - joint_positions[:, 7]) + (0.04 - joint_positions[:, 8]), grasp_reward)

        keep_height = torch.where(franka_grasp_pos[:, 2] < 0.2, 1.0 / (1.0 + franka_grasp_pos[:, 2]), 0)

        # Compute the deviation of the beaker's orientation from the vertical direction
        beaker_up_vector = tf_vector(beaker_grasp_rot, beaker_up_axis)
        vertical_vector = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat((num_envs, 1))
        dot_vertical = torch.bmm(beaker_up_vector.view(num_envs, 1, 3), vertical_vector.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)

        grasp_penalty = 1.0 - torch.sign(dot_vertical) * dot_vertical ** 2

        height_diff = torch.abs(target_lift_height - beaker_grasp_pos[:, 2])
        lift_reward = 1.0 / (5 * height_diff + 0.025)

        # Action penalty: reduce unnecessary actions
        action_penalty = torch.sum(actions ** 2, dim=-1)

        # Compute the total reward
        rewards = (
            dist_reward_scale * dist_reward
            + rot_reward_scale * rot_reward
            + lift_reward_scale * lift_reward
            - 3 * grasp_penalty
        )

        rewards = torch.where(height_diff < 0.269, rewards + 2.0 * lift_reward, rewards)
        rewards = torch.where(height_diff < 0.2, rewards + 4.0 * lift_reward, rewards)
        rewards = torch.where(height_diff < 0.1, rewards + 6.0 * lift_reward, rewards)

        return rewards
