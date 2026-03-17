# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.sensors import RayCasterCfg, patterns, CameraCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from mbrl.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from .rough_env_cfg import UnitreeA1RoughEnvCfg
import mbrl.tasks.manager_based.locomotion.velocity.mdp as mdp
from mbrl.tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg, MySceneCfg
from .flat_env_cfg import UnitreeA1FlatEnvCfg_AMP

from mbrl.assets.unitree import UNITREE_A1_CFG  # isort: skip
from mbrl.terrains.config.rough import ROUGH_TERRAINS_CFG


@configclass
class SceneCfg(MySceneCfg):
    # sensors
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # depth scanner
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
        update_period=0.1,
        height=64,
        width=64,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    # foot height
    foot_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )  
    # forward height
    forward_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(1.2, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[2.4, 2.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

@configclass
class ObservationsCfg_WMP(ObservationsCfg):
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # -- domain randomization parameters 50dim
        friction = ObsTerm(
            func=mdp.body_material_friction,
            params={"asset_cfg": SceneEntityCfg("robot"), "friction_type": "dynamic"},
        )
        restitution = ObsTerm(
            func=mdp.body_material_restitution,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        added_base_mass = ObsTerm(
            func=mdp.body_mass,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )
        base_com_pos = ObsTerm(
            func=mdp.body_com_pos,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )
        joint_stiffness_ratio = ObsTerm(
            func=mdp.joint_stiffness_ratio,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        joint_damping_ratio = ObsTerm(
            func=mdp.joint_damping_ratio,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        # -- contact forces and flags (all foot/thigh/base contacts)
        contact_forces = ObsTerm(
            func=mdp.contact_forces_flat,  # TODO: check mdp.body_contact differences
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )
        contact_flags = ObsTerm(
            func=mdp.contact_flag,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"), 
                "threshold": 0.1,
                "body_names": [".*thigh", ".*calf"]
            },
        )
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        # height scans
        foot_height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("foot_height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        """Observations for critic group."""
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CameraCfg(ObsGroup):
        """Observations for camera group."""

        # observation terms (order preserved)
        # depth
        depth_scan = ObsTerm(
            func=mdp.processed_image,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "data_type": "distance_to_image_plane",
                "normalize": True,
                "far_clip": 2.0,
                "near_clip": 0.0,
            },
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ForwardHeightCfg(ObsGroup):
        """Observations for forward height group."""

        # -- forward height scans 525 dim
        forward_height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("forward_height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    camera: CameraCfg = CameraCfg()
    forward_height: ForwardHeightCfg = ForwardHeightCfg()


@configclass
class UnitreeA1RoughEnvCfg_WMP(LocomotionVelocityRoughEnvCfg):

    # override scene
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    # override observation terms
    observations: ObservationsCfg_WMP = ObservationsCfg_WMP()

    base_link_name = "base"
    foot_link_name = ".*_foot"
    # fmt: off
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    # fmt: on

    # action
    num_actions = 12
    # amp
    num_amp_observations = 2
    # joint_pos(12)、foot_pos_local(12)、base_lin_vel(3)、base_ang_vel(3)、joint_vel(12)、z_pos(1)
    amp_observation_space = 43
    reference_body = "base"
    reset_strategy = "default"
    amp_motion_files = 'datasets/mocap_motions_a1/a1_motions.npz'

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        # self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.robot = UNITREE_A1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel.scale = 1.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),       # no height jitter — robot always spawns at terrain surface height
                "roll": (0.0, 0.0),    # WMP never randomizes roll (always upright)
                "pitch": (0.0, 0.0),   # WMP never randomizes pitch (always upright)
                "yaw": (-3.14, 3.14),  # yaw randomisation is safe and adds orientation diversity
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -1.0  # multify 
        self.rewards.ang_vel_xy_l2.weight = 0.0  # multify
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0.35
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -1e-4  # multify 
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -2.5e-7  # multify
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_l1", -0.04, [".*_joint"]) # multify 
        self.rewards.joint_pos_limits.weight = 0
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_power.weight = -2e-5
        self.rewards.stand_still.weight = -2.0
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.03 # multify

        # Contact sensor
        self.rewards.undesired_contacts.weight = 0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = 0  
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.5  # multify 
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.15)  # multify 
        self.rewards.track_ang_vel_z_exp.weight = 0.5  # multify 
        self.rewards.track_ang_vel_z_exp.params["std"] = math.sqrt(0.15)  # multify 

        # Others
        self.rewards.feet_air_time.weight = 0.5  # multify 
        self.rewards.feet_air_time.params["threshold"] = 0.5  # multify 
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = -0.1 # multify
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        # Stronger stumble penalty gated to gap terrain (mirrors WMP feet_edge for gap+pit).
        # Starts at -0.1 (matching WMP's initial curriculum coef of 0.1 × -1.0 = -0.1) and
        # ramps to -1.0 between runner iterations 4000-10000 via the curriculum term below.
        self.rewards.feet_stumble_on_gap.weight = -0.1
        self.rewards.feet_stumble_on_gap.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        self.rewards.upward.weight = 0
        self.rewards.collision.weight = -1.0 # multify
        self.rewards.collision.params["sensor_cfg"].body_names = [".*thigh", ".*calf"]  # multify
        self.rewards.stuck.weight = -1.0 # multify
        self.rewards.cheat.weight = -1.0 # multify
        # Exclude rough-flat terrain from cheat penalty (matches WMP: applied only to non-flat terrains)
        self.rewards.cheat.params["excluded_terrain"] = "random_rough"
        # ------------------------------Terminations------------------------------
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        self.terminations.illegal_contact = None

        # ------------------------------Curriculums------------------------------
        # self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.2, 1.0)
        # self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
        # Ramp feet_stumble_on_gap weight from -0.1 to -1.0 between runner iterations
        # 4000 and 10000, mirroring WMP's feet_edge reward curriculum.
        self.curriculum.feet_stumble_on_gap_ramp = CurrTerm(
            func=mdp.reward_weight_linear_ramp,
            params={
                "reward_term_name": "feet_stumble_on_gap",
                "initial_weight": -0.1,
                "final_weight": -1.0,
                "start_iter": 4000,
                "end_iter": 10000,
                "num_steps_per_env": 24,
            },
        )

        # no height scan
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        
        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeA1RoughEnvCfg_WMP":
            self.disable_zero_weight_rewards()

@configclass
class UnitreeA1RoughEnvCfg_WMP_PLAY(UnitreeA1RoughEnvCfg_WMP):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # command
        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, -0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeA1RoughEnvCfg_WMP_PLAY":
            self.disable_zero_weight_rewards()