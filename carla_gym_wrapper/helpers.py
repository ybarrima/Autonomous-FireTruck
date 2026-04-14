"""
Utility / helper functions for the CARLA-Gym wrapper.

These functions handle coordinate transforms, distance calculations,
image conversions, and vehicle geometry operations needed by
CarlaEnv and BirdeyeRender.
"""

import math
import numpy as np
import carla
import pygame
from skimage.transform import resize


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def set_carla_transform(pose):
    """Create a carla.Transform from [x, y, yaw] list."""
    transform = carla.Transform()
    transform.location.x = pose[0]
    transform.location.y = pose[1]
    transform.rotation.yaw = pose[2]
    return transform


def get_pos(vehicle):
    """Return (x, y) position of a CARLA vehicle."""
    trans = vehicle.get_transform()
    return trans.location.x, trans.location.y


def get_info(vehicle):
    """Return (x, y, yaw, length, width) of a CARLA vehicle."""
    trans = vehicle.get_transform()
    x = trans.location.x
    y = trans.location.y
    yaw = trans.rotation.yaw / 180 * np.pi
    bb = vehicle.bounding_box
    l = bb.extent.x
    w = bb.extent.y
    return x, y, yaw, l, w


def get_local_pose(global_pose, ego_pose):
    """
    Transform a global (x, y, yaw) pose into the ego vehicle's
    local coordinate frame.
    """
    x, y, yaw = global_pose
    ego_x, ego_y, ego_yaw = ego_pose
    R = np.array([
        [np.cos(ego_yaw), np.sin(ego_yaw)],
        [-np.sin(ego_yaw), np.cos(ego_yaw)]
    ])
    vec_local = R @ np.array([x - ego_x, y - ego_y])
    yaw_local = yaw - ego_yaw
    return vec_local[0], vec_local[1], yaw_local


def get_pixel_info(local_info, d_behind, obs_range, image_size):
    """
    Convert local-frame vehicle info to pixel coordinates for
    the PIXOR representation.
    """
    x, y, yaw, l, w = local_info
    x_pixel = (x + d_behind) / obs_range * image_size
    y_pixel = (y + obs_range / 2) / obs_range * image_size
    yaw_pixel = yaw
    l_pixel = l / obs_range * image_size
    w_pixel = w / obs_range * image_size
    return x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel


def get_pixels_inside_vehicle(pixel_info, pixel_grid):
    """
    Get all pixel coordinates that fall inside a vehicle's
    bounding box in PIXOR representation.
    """
    x, y, yaw, l, w = pixel_info
    # Rotation matrix
    R = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])
    # Translate grid to vehicle center
    shifted = pixel_grid - np.array([x, y])
    # Rotate to vehicle frame
    rotated = (np.linalg.inv(R) @ shifted.T).T
    # Check which pixels are inside the box
    inside = (np.abs(rotated[:, 0]) <= l) & (np.abs(rotated[:, 1]) <= w)
    return pixel_grid[inside].astype(int)


# ---------------------------------------------------------------------------
# Distance / geometry helpers
# ---------------------------------------------------------------------------

def distance_vehicle(waypoint, vehicle_transform):
    """Euclidean distance from a waypoint to a vehicle transform."""
    loc = vehicle_transform.location
    wp_loc = waypoint.transform.location
    dx = loc.x - wp_loc.x
    dy = loc.y - wp_loc.y
    return math.sqrt(dx * dx + dy * dy)


def is_within_distance_ahead(target_location, current_location, orientation, max_distance):
    """
    Check whether a target location is within max_distance directly
    ahead of the current location (within a ±90° cone).
    """
    target_vec = np.array([
        target_location.x - current_location.x,
        target_location.y - current_location.y
    ])
    norm = np.linalg.norm(target_vec)
    if norm < 0.001:
        return True
    if norm > max_distance:
        return False

    forward_vec = np.array([
        math.cos(math.radians(orientation)),
        math.sin(math.radians(orientation))
    ])
    d_angle = math.degrees(
        math.acos(np.clip(np.dot(forward_vec, target_vec) / norm, -1.0, 1.0))
    )
    return d_angle < 90.0


def compute_magnitude_angle(target_location, current_location, orientation):
    """
    Compute the magnitude (distance) and angle between a target
    location and the current vehicle heading.
    """
    target_vec = np.array([
        target_location.x - current_location.x,
        target_location.y - current_location.y
    ])
    norm = np.linalg.norm(target_vec)
    forward_vec = np.array([
        math.cos(math.radians(orientation)),
        math.sin(math.radians(orientation))
    ])
    if norm < 0.001:
        return 0.0, 0.0
    d_angle = math.degrees(
        math.acos(np.clip(np.dot(forward_vec, target_vec) / norm, -1.0, 1.0))
    )
    return norm, d_angle


# ---------------------------------------------------------------------------
# Lane / waypoint distance helpers
# ---------------------------------------------------------------------------

def get_lane_dis(waypoints, x, y):
    """
    Calculate the lateral distance from a point (x, y) to the
    nearest segment of the waypoint path, and return the unit
    direction vector of that segment.
    """
    min_dis = float('inf')
    min_w = np.array([0.0, 0.0])

    for i in range(len(waypoints) - 1):
        x1, y1 = waypoints[i][0], waypoints[i][1]
        x2, y2 = waypoints[i + 1][0], waypoints[i + 1][1]

        # Direction vector of this segment
        dx = x2 - x1
        dy = y2 - y1
        seg_len = math.sqrt(dx * dx + dy * dy)
        if seg_len < 1e-6:
            continue

        # Unit direction
        w = np.array([dx, dy]) / seg_len

        # Project point onto segment
        t = ((x - x1) * dx + (y - y1) * dy) / (seg_len * seg_len)
        t = np.clip(t, 0.0, 1.0)

        # Closest point on segment
        px = x1 + t * dx
        py = y1 + t * dy

        # Signed lateral distance (cross product)
        dis = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        dis /= seg_len

        if abs(dis) < abs(min_dis):
            min_dis = dis
            min_w = w

    return min_dis, min_w


def get_preview_lane_dis(waypoints, x, y, idx=2):
    """
    Like get_lane_dis but uses a waypoint slightly ahead (index idx)
    for a preview of upcoming curvature.
    """
    # Use a forward-looking segment
    if len(waypoints) > idx + 1:
        subset = waypoints[idx:]
    else:
        subset = waypoints
    return get_lane_dis(subset, x, y)


# ---------------------------------------------------------------------------
# Image conversion helpers
# ---------------------------------------------------------------------------

def display_to_rgb(display, obs_size):
    """Resize a pygame display array to (obs_size, obs_size, 3)."""
    rgb = resize(display, (obs_size, obs_size)) * 255
    return rgb.astype(np.uint8)


def rgb_to_display_surface(rgb, display_size):
    """
    Convert an RGB numpy array to a pygame Surface of
    (display_size, display_size).
    """
    surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    surface = pygame.transform.scale(surface, (display_size, display_size))
    return surface
