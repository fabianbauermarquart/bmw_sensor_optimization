import numpy as np
from typing import List, Tuple

from utils.models import FieldOfView, Sensor


def create_cubic_data(start_point: np.array, end_point: np.array) -> np.ndarray:
    """
    Create cubic data from two corner points of the cube.
    :param start_point: start point.
    :param end_point: end point.
    :return: cubic data.
    """
    base_voxel = np.array([[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
                           [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
                           [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
                           [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
                           [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
                           [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]], dtype=float)

    return start_point + base_voxel * (end_point - start_point)


def rotate_point(point: np.array, z_angle: float = 0.0, y_angle: float = 0.0, x_angle: float = 0.0,
                 extrinsic: bool = False) -> np.array:
    """
    Rotate a point.
    :param point: original point.
    :param z_angle: z axis rotation angle (yaw).
    :param y_angle: y axis rotation angle (pitch).
    :param x_angle: x axis rotation angle (roll).
    :return: Rotated point.
    """
    if extrinsic:
        x_angle, y_angle, z_angle = z_angle, y_angle, x_angle

    z_rot = np.array([[np.cos(z_angle), -np.sin(z_angle), 0],
                      [np.sin(z_angle), np.cos(z_angle), 0],
                      [0, 0, 1]])

    y_rot = np.array([[np.cos(y_angle), 0, np.sin(y_angle)],
                      [0, 1, 0],
                      [-np.sin(y_angle), 0, np.cos(y_angle)]])

    x_rot = np.array([[1, 0, 0],
                      [0, np.cos(x_angle), -np.sin(x_angle)],
                      [0, np.sin(x_angle), np.cos(x_angle)]])

    rot_matrix = np.matmul(np.matmul(z_rot, y_rot), x_rot)
    rotated_point = np.matmul(rot_matrix, np.array([point]).T)

    return rotated_point.T[0]


def get_vector_angles(vector: np.array) -> Tuple[float, float]:
    """
    Get angles from vector.
    :param vector: (x, y, z) coordinates.
    :return: two angles.
    """
    x, y, z = vector

    vertical_angle = np.arctan2(np.sqrt(z ** 2 + x ** 2), y)
    horizontal_angle = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)

    return (vertical_angle, horizontal_angle)


def calculate_plane_normal(corners: List) -> np.array:
    """
    Calculate the plane normal from the plane's corner points.
    :param corners: 4 corner points (x, y, z)
    :return:
    """
    A = np.array(corners[0])
    B = np.array(corners[1])
    C = np.array(corners[2])

    normal = np.cross((B - A), (C - A))

    return normal / np.linalg.norm(normal)


def vector_intersects_plane(vector_start: np.array, vector_end: np.array, plane_point: np.array, plane_normal: np.array,
                            epsilon=1e-6):
    """
    Returns true if vector intersects plane.
    :param vector_start: start coordinate.
    :param vector_end: end coordinate.
    :param plane_point: point on the plane.
    :param plane_normal: plane's normal vector.
    :param epsilon: error tolerance.
    :return: True if vector intersects plane.
    """
    support_vector = vector_end - vector_start
    dot_product = np.dot(plane_normal, support_vector)

    if abs(dot_product) > epsilon:
        # If 'factor' is between (0 - 1) the point intersects with the segment.
        w = vector_start - plane_point
        factor = -np.dot(plane_normal, w) / dot_product
        return factor >= 0 and factor <= 1
    else:
        # The segment is parallel to plane.
        return False


def sample_parallelogram_points(corners: List, n: int = 2, m: int = 2, distance: float = None) -> List[np.array]:
    """
    Grid sampling of points on a 3D parallelogram.
    :param corners: 4 corner points (x, y, z).
    :param n: number of points for the width (AB side)
    :param m: number of points for the length (AC side)
    :param distance: approximate distance between points (optional, overrides n and m)
    :return: sampled points.
    """
    sampled_points = []

    A, B, _, D = corners

    A = np.array(A)
    B = np.array(B)
    D = np.array(D)

    # override n, m when a distance value is given
    if distance is not None:
        norm_ab = np.linalg.norm(B - A)
        norm_ad = np.linalg.norm(D - A)

        # minimum sample points per side is 2
        n = max(2, int(norm_ab / distance))
        m = max(2, int(norm_ad / distance))

    for i in range(n + 1):
        for j in range(m + 1):
            sampled_points.append(A
                                  + i * (B - A) / float(n)
                                  + j * (D - A) / float(m))

    return sampled_points


def is_point_visible_by_sensor(args: Tuple[Sensor, np.array, int, int]) -> Tuple[bool, int, int]:
    """
    Returns true if a given point is in the visible range of a given sensor.
    :param args: point, sensor, point index and sensor index needed for multiprocessing.
    :return: truth value.
    """
    (sensor, point, point_idx, sensor_idx) = args

    return (is_inside_elliptic_cone(point,
                                    sensor.position,
                                    sensor.characteristic.field_of_view,
                                    sensor.orientation),
            sensor_idx,
            point_idx)


def is_inside_elliptic_cone(point: np.array, apex: np.array, field_of_view: FieldOfView, orientation: np.array) -> bool:
    """
    Returns true if a given point is inside the elliptic cone.

    :param point: point which should be tested if it lies inside or outside the cone.
    :param apex: position of the cone's tip (i.e. the sensor placement).
    :param field_of_view: field of view.
    :param orientation: cone's orientation, given by two angles.
    :return: truth value.
    """
    fov_h_angle = field_of_view.horizontal_angle
    fov_v_angle = field_of_view.vertical_angle
    fov_range = field_of_view.fov_range

    or_h_angle = orientation[0]
    or_v_angle = orientation[1]

    # The mid point of the base
    base_midpoint = apex + rotate_point(np.array([fov_range, 0, 0]), or_v_angle, 0, or_h_angle)

    # Project point onto axis vector
    projected_point = apex \
                      + (np.dot(point - apex, base_midpoint - apex) / np.linalg.norm(base_midpoint - apex) ** 2) \
                      * (base_midpoint - apex)

    # Check if projected point is within fov range
    projected_point_distance = np.linalg.norm(projected_point - apex)

    if projected_point_distance < 0 or projected_point_distance > fov_range:
        return False

    # Calculate angle from projected point to point in question
    reference_vector = rotate_point(np.array([0, 1, 0]), or_v_angle, 0, or_h_angle)
    angle = np.dot(point - projected_point, reference_vector)

    # Get elliptical radius at the line between the point and its projection
    horizontal_radius = np.linalg.norm(projected_point - apex) * np.tan(fov_h_angle)
    vertical_radius = np.linalg.norm(projected_point - apex) * np.tan(fov_v_angle)

    radius = horizontal_radius * vertical_radius \
             / np.sqrt(horizontal_radius ** 2 * np.sin(angle) ** 2 + vertical_radius ** 2 * np.cos(angle) ** 2)

    # True if the distance between the projection and the point is smaller or equal to the radius, else False.
    return np.linalg.norm(point - projected_point) <= radius
