# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.sensors import RayCasterCfg, patterns, CameraCfg
import isaaclab.sim as sim_utils

from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.rough_env_cfg import AnymalDRoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg, RewardsCfg, MySceneCfg

from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

from mbrl.mbrl.envs.mdp.commands import UniformVelocityCommand_Visualize, SampleUniformVelocityCommand
import mbrl.tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class RewardsCfg_TRAIN(RewardsCfg):
    stand_still = RewTerm(
        func=mdp.joint_pos_stand_still, weight=-1.0, params={"command_name": "base_velocity", "threshold": 0.05}
        )


@configclass
class AnymalDFlatEnvCfg(AnymalDRoughEnvCfg):
    
    rewards: RewardsCfg_TRAIN = RewardsCfg_TRAIN()
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.feet_air_time.weight = 0.5
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


@configclass
class SceneCfg(MySceneCfg):
    # sensors
    # depth scanner
    camera = CameraCfg(
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
class ObservationsCfg_PRETRAIN(ObservationsCfg):

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
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT")},
        )
        contact_flags = ObsTerm(
            func=mdp.contact_flag,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"), 
                "threshold": 0.1,
                "body_names": [".*THIGH", ".*SHANK"]
            },
        )

        # -- observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
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
    class CameraCfg(ObsGroup):
        """Observations for camera group."""

        # observation terms (order preserved)
        # depth
        depth_scan = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "data_type": "distance_to_image_plane",
                "normalize": True,
            },
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


    @configclass
    class CriticCfg(ObsGroup):  # 285dim
        """Privileged observations for critic (not available to actor at test time)."""

        # -- policy obs (no noise, clean version) 48dim
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SystemStateCfg(ObsGroup):

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        joint_torque = ObsTerm(func=mdp.joint_effort)
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


    @configclass
    class SystemActionCfg(ObsGroup):

        # observation terms (order preserved)
        pred_actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


    @configclass
    class SystemExtensionCfg(ObsGroup):
        # -- height scans 187 dim
        # foot_height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("foot_height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )
        # -- forward height scans 525 dim
        forward_height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("forward_height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        # # -- domain randomization parameters 50dim
        # friction = ObsTerm(
        #     func=mdp.body_material_friction,
        #     params={"asset_cfg": SceneEntityCfg("robot"), "friction_type": "dynamic"},
        # )
        # restitution = ObsTerm(
        #     func=mdp.body_material_restitution,
        #     params={"asset_cfg": SceneEntityCfg("robot")},
        # )
        # added_base_mass = ObsTerm(
        #     func=mdp.body_mass,
        #     params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        # )
        # base_com_pos = ObsTerm(
        #     func=mdp.body_com_pos,
        #     params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        # )
        # joint_stiffness_ratio = ObsTerm(
        #     func=mdp.joint_stiffness_ratio,
        #     params={"asset_cfg": SceneEntityCfg("robot")},
        # )
        # joint_damping_ratio = ObsTerm(
        #     func=mdp.joint_damping_ratio,
        #     params={"asset_cfg": SceneEntityCfg("robot")},
        # )
        # # -- contact forces and flags (all foot/thigh/base contacts)
        # contact_forces = ObsTerm(
        #     func=mdp.contact_forces_flat,  # TODO: check mdp.body_contact differences
        #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT")},
        # )
        # contact_flags = ObsTerm(
        #     func=mdp.contact_flag,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"), 
        #         "threshold": 0.1,
        #         "body_names": [".*THIGH", ".*SHANK"]
        #     },
        # )

        # def __post_init__(self):
        #     self.enable_corruption = False
        #     self.concatenate_terms = True


    @configclass
    class SystemContactCfg(ObsGroup):

        # observation terms (order preserved)
        thigh_contact = ObsTerm(func=mdp.body_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0})
        foot_contact = ObsTerm(func=mdp.body_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"), "threshold": 1.0})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


    @configclass
    class SystemTerminationCfg(ObsGroup):

        # observation terms (order preserved)
        base_contact = ObsTerm(func=mdp.body_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PolicyCfg = PolicyCfg()
    camera: CameraCfg = CameraCfg()
    system_state: SystemStateCfg = SystemStateCfg()
    system_action: SystemActionCfg = SystemActionCfg()
    system_extension: SystemExtensionCfg = SystemExtensionCfg()
    system_contact: SystemContactCfg = SystemContactCfg()
    system_termination: SystemTerminationCfg = SystemTerminationCfg()


@configclass
class WMAnymalDFlatEnvCfg_PRETRAIN(AnymalDFlatEnvCfg):
    
    # override scene
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)

    # override observation terms
    observations: ObservationsCfg_PRETRAIN = ObservationsCfg_PRETRAIN()
    

@configclass
class WMAnymalDFlatEnvCfg_FINETUNE(WMAnymalDFlatEnvCfg_PRETRAIN):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # override commands
        self.commands.base_velocity.class_type = SampleUniformVelocityCommand


@configclass
class WMAnymalDFlatEnvCfg_VISUALIZE(WMAnymalDFlatEnvCfg_PRETRAIN):
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for visualize
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        # disable randomization for visualize
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # override commands
        self.commands.base_velocity.class_type = UniformVelocityCommand_Visualize
        self.commands.base_velocity.resampling_time_range = (2.0, 2.0)
        # override randomization
        self.events.reset_base.func = mdp.reset_root_state_uniform_visualize
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (1.57, 1.57)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            }
        }
        self.events.reset_robot_joints.func = mdp.reset_joints_by_scale_visualize
