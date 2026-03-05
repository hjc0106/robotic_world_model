from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, ContactSensor, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_body_names,
    record_dtype,
    record_joint_names,
    record_joint_pos_offsets,
    record_joint_vel_offsets,
    record_shape,
)


@generic_io_descriptor(
    observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def over_orientation(env: ManagerBasedEnv, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Bad orientation.
    
    Note: This function is typically used as a termination condition.
    
    Args:
        env: The environment.
        limit_angle: The limit angle in radians.
        asset_cfg: The RigidObject associated with this observation.

    Returns:
        A boolean tensor indicating whether the orientation is bad.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.acos(-asset.data.projected_gravity_b[:, 2]).abs().unsqueeze(-1) > limit_angle


@generic_io_descriptor(observation_type="BodyState", on_inspect=[record_shape, record_dtype, record_body_names])
def body_height_w(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The height of bodies of an Articulation.

    Note: Only the bodies configured in :attr:`asset_cfg.body_ids` will have their poses returned.

    Args:
        env: The environment.
        asset_cfg: The Articulation associated with this observation.

    Returns:
        The height of the bodies of the Articulation. Output is stacked horizontally per body.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_pos_w[:, asset_cfg.body_ids, 2]


@generic_io_descriptor(observation_type="BodyState", on_inspect=[record_shape, record_dtype, record_body_names])
def body_lin_vel_w_norm(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The linear velocity of bodies of an Articulation.

    Note: Only the bodies configured in :attr:`asset_cfg.body_ids` will have their poses returned.

    Args:
        env: The environment.
        asset_cfg: The Articulation associated with this observation.

    Returns:
        The linear velocity of bodies of an Articulation. Output is stacked horizontally per body.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)


# ==============================================================================
# Privileged observation functions
# ==============================================================================

def body_mass(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
) -> torch.Tensor:
    """Added mass on specified bodies relative to default mass.

    Returns the difference between current and default mass for the specified
    bodies (i.e. the randomized added mass). Shape: (num_envs, num_bodies).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    current_mass = asset.root_physx_view.get_masses().to(env.device)   # (num_envs, num_bodies)
    default_mass = asset.data.default_mass.to(env.device)              # (num_envs, num_bodies)
    added_mass = current_mass[:, asset_cfg.body_ids] - default_mass[:, asset_cfg.body_ids]
    return added_mass


def body_com_pos(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
) -> torch.Tensor:
    """Center-of-mass position of specified bodies in the body local frame.

    Shape: (num_envs, num_bodies * 3).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # get_coms returns (num_envs, num_bodies, 7): [px, py, pz, qw, qx, qy, qz]
    coms = asset.root_physx_view.get_coms().to(env.device)
    com_pos = coms[:, asset_cfg.body_ids, :3]   # (num_envs, num_bodies, 3)
    return com_pos.view(env.num_envs, -1)


def joint_stiffness_ratio(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ratio of current joint stiffness to default joint stiffness minus 1.

    Reflects relative change introduced by domain randomization.
    Shape: (num_envs, num_joints).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    current = asset.data.joint_stiffness[:, asset_cfg.joint_ids]
    default = asset.data.default_joint_stiffness[:, asset_cfg.joint_ids]
    return current / (default + 1e-6) - 1.0


def joint_damping_ratio(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ratio of current joint damping to default joint damping minus 1.

    Reflects relative change introduced by domain randomization.
    Shape: (num_envs, num_joints).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    current = asset.data.joint_damping[:, asset_cfg.joint_ids]
    default = asset.data.default_joint_damping[:, asset_cfg.joint_ids]
    return current / (default + 1e-6) - 1.0


def body_material_friction(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    friction_type: str = "static",
) -> torch.Tensor:
    """Static or dynamic friction coefficient of the asset's geometry shapes.

    Reads material properties set by ``randomize_rigid_body_material``. Since each body
    can have multiple collision shapes, the mean over all shapes is returned.

    Args:
        asset_cfg: The articulation/rigid-body asset to query.
        friction_type: ``"static"`` (col 0) or ``"dynamic"`` (col 1).

    Returns:
        Tensor of shape (num_envs, 1) with the mean friction coefficient.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # shape: (num_envs, max_shapes, 3)  cols: [static_friction, dynamic_friction, restitution]
    materials = asset.root_physx_view.get_material_properties().to(env.device)
    col = 0 if friction_type == "static" else 1
    return materials[:, :, col].mean(dim=1, keepdim=True)   # (num_envs, 1)


def body_material_restitution(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Restitution coefficient of the asset's geometry shapes.

    Reads material properties set by ``randomize_rigid_body_material``.
    Returns the mean restitution over all shapes.

    Returns:
        Tensor of shape (num_envs, 1).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    materials = asset.root_physx_view.get_material_properties().to(env.device)
    return materials[:, :, 2].mean(dim=1, keepdim=True)    # (num_envs, 1)


def contact_forces_flat(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Flatten net contact forces of specified bodies.

    Shape: (num_envs, num_bodies * 3).
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]   # (num_envs, num_bodies, 3)
    return forces.view(env.num_envs, -1)


def contact_flag(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 0.1,
    body_names: str | list[str] | None = None,
) -> torch.Tensor:
    """Binary contact flag: 1 if contact force norm > threshold, else 0.

    Args:
        sensor_cfg: Contact sensor config. body_names in sensor_cfg determines the base body set.
        threshold: Force norm threshold for contact detection.
        body_names: Optional regex string or list of strings to further filter bodies within
            sensor_cfg. E.g. ``".*THIGH"`` or ``[".*THIGH", ".*SHANK"]``.
            If None, all bodies in sensor_cfg are used.

    Returns:
        Boolean tensor of shape (num_envs, num_selected_bodies).
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (num_envs, num_bodies, 3)
    if body_names is not None:
        # find_bodies returns (indices_in_sensor, matched_names)
        local_ids, _ = sensor.find_bodies(body_names)
        # convert sensor-global ids to positions within sensor_cfg.body_ids
        body_id_set = list(sensor_cfg.body_ids)
        local_ids = [body_id_set.index(i) for i in local_ids if i in body_id_set]
        forces = forces[:, local_ids, :]
    return (torch.norm(forces, dim=-1) > threshold).float()

def processed_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
    far_clip: float = 2.0,
    near_clip: float = 0.0,
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.

    Returns:
        The images produced at the last time-step
    """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type]

    # depth image conversion
    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    # rgb/depth/normals image normalization
    if normalize:
        if data_type == "rgb":
            images = images.float() / 255.0
            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
            images -= mean_tensor
        elif "distance_to" in data_type or "depth" in data_type:
            images = torch.clamp(images, max=-near_clip, min=-far_clip)
            images = images * -1
            images = (images - near_clip) / (far_clip - near_clip) - 0.5
        elif "normals" in data_type:
            images = (images + 1.0) * 0.5

    return images.clone()
