import csv
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from seaborn import diverging_palette

from utils.geometry import rotate_point, create_cubic_data, sample_parallelogram_points, is_visible_by_sensor
from utils.models import Sensor, SensorType, Characteristic, FieldOfView
from utils.types import is_number, flatten_list, group_list

allowed_sensor_positions = {}
occlusion_geometry = {}
criticality_grid = []
regions_of_interest_cubic = {}
regions_of_interest_sector = {}

# Read allowed sensor positions
with open('Sensor_position/allowed_sensor_positions.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    column_names = None

    row_count = 0
    for row in csv_reader:
        if row_count == 0:
            column_names = row
            row_count += 1
        else:
            allowed_sensors = [SensorType[str] for str in row[13].lower().split(', ')]

            allowed_sensor_positions[row[0]] = {'coordinates': group_list(row[1:13], 3),
                                                'allowed_sensors': allowed_sensors}
            row_count += 1

# Read occlusion geometry
with open('Sensor_position/occlusion_geometry.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    column_names = None

    row_count = 0
    for row in csv_reader:
        if row_count == 0:
            column_names = row
            row_count += 1
        else:
            row_entries = [v for v in row if is_number(v)]
            occlusion_geometry[row[0]] = {'coordinates': group_list(row_entries, 3)}
            row_count += 1

# Read criticality grid
scaling_factor = 1000

with open('Sensor_position/criticallity_grid_0_5.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    row_count = 0

    for row in csv_reader:
        if row_count == 0:
            row_count += 1
        else:
            point = np.array(row, dtype=float)
            point[0:3] = point[0:3] * scaling_factor

            criticality_grid.append(point)
            row_count += 1

criticality_grid = np.array(criticality_grid, dtype=float)
criticality_grid_display = np.array(criticality_grid, dtype=float)

# Subsample for easy viewing
for _ in range(5):
    criticality_grid_display = np.delete(criticality_grid_display,
                                         list(range(0, criticality_grid_display.shape[0], 2)), axis=0)

# Read regions of interest (cubic)
with open('Sensor_position/regions_of_interest_cubic.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    row_count = 0

    for row in csv_reader:
        if row_count == 0:
            row_count += 1
        else:
            start_point = np.array(row[1:4], dtype=float) * scaling_factor
            end_point = np.array(row[4:7], dtype=float) * scaling_factor

            regions_of_interest_cubic[row[0]] = {'start_point': start_point,
                                                 'end_point': end_point,
                                                 'rel_speed': float(row[7]),
                                                 'distance': float(row[8]),
                                                 'critical_index': float(row[9]),
                                                 'required_parameters': [v for v in row[10:13] if v != ''],
                                                 'speed': float(row[13]),
                                                 'env_types': [v for v in row[14:17] if v != '']}
            row_count += 1

# Read regions of interest (spherical sector)
with open('Sensor_position/regions_of_interest_sector.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    row_count = 0

    for row in csv_reader:
        if row_count == 0:
            row_count += 1
        else:
            center_point = np.array(row[1:4], dtype=float)
            radius_max = float(row[4])
            radius_min = float(row[5])
            yaw_angle = float(row[6])
            height = float(row[7])
            h_angle = float(row[8])

            regions_of_interest_sector[row[0]] = {'center_point': center_point,
                                                  'radius_max': radius_max,
                                                  'radius_min': radius_min,
                                                  'yaw_angle': yaw_angle,
                                                  'height': height,
                                                  'h_angle': h_angle,
                                                  'rel_speed': float(row[9]),
                                                  'distance': float(row[10]),
                                                  'critical_index': float(row[11]),
                                                  'required_parameters': [v for v in row[12:15] if v != ''],
                                                  'speed': float(row[15]),
                                                  'env_types': [v for v in row[16:19] if v != '']}

            regions_of_interest_sector[row[0]]['radius_max_point'] \
                = (center_point + rotate_point(np.array([radius_max, 0, 0]), yaw_angle, 0, h_angle)) * scaling_factor

            regions_of_interest_sector[row[0]]['radius_min_point'] \
                = (center_point + rotate_point(np.array([radius_min, 0, 0]), yaw_angle, 0, h_angle)) * scaling_factor

            row_count += 1

# Print outs for plausability check
# print(allowed_sensor_positions)
# print(occlusion_geometry)
# print(criticality_grid[0])
# print(regions_of_interest_cubic)
# print(regions_of_interest_sector)

# Set up plot
fig = plt.figure()
ax = Axes3D(fig)

# Car only
ax.set_xlim3d(-10000, 5000)
ax.set_ylim3d(-4000, 4000)
ax.set_zlim3d(0, 4000)

# Regions of interest
ax.set_xlim3d(-75000, 125000)
ax.set_ylim3d(-100000, 100000)
ax.set_zlim3d(0, 100000)

sensor_pos_collection = Poly3DCollection(
    [v['coordinates'] for v in allowed_sensor_positions.values()],
    linewidths=0.5, alpha=0.8, edgecolors='black', facecolors='blue')

occlusion_geometry_collection = Poly3DCollection(
    [v['coordinates'] for v in occlusion_geometry.values()],
    linewidths=0.5, alpha=0.5, edgecolors='black', facecolors='green')

regions_of_interest_cubic_collection = Poly3DCollection(
    flatten_list([create_cubic_data(v['start_point'], v['end_point']) for v in regions_of_interest_cubic.values()]),
    linewidths=0.5, edgecolors='black', alpha=0.3, facecolors='red')

# ax.add_collection3d(occlusion_geometry_collection)
ax.add_collection3d(sensor_pos_collection)
ax.add_collection3d(regions_of_interest_cubic_collection)

for value in regions_of_interest_sector.values():
    xs = [value['radius_min_point'][0], value['radius_max_point'][0]]
    ys = [value['radius_min_point'][1], value['radius_max_point'][1]]
    zs = [value['radius_min_point'][2], value['radius_max_point'][2]]

    ax.plot(xs, ys, zs)

palette = diverging_palette(0, 255, sep=8, n=256)
criticality_palette = [palette[int(v * 255)] for v in criticality_grid_display[:, 3]]

# ax.scatter(criticality_grid_display[:, 0], criticality_grid_display[:, 1], criticality_grid_display[:, 2],
#            s=1,
#            c=criticality_palette, alpha=0.5)


# Basic sensors

# Livox Mid-100
lidar_characteristic = Characteristic(SensorType.lidar,
                                      FieldOfView(vertical_angle=38.4 * np.pi / 180,
                                                  horizontal_angle=98.4 * np.pi / 180,
                                                  fov_range=90 * 1000),
                                      price=1500.00)

# Unspecified short range radar
short_range_radar_characteristic = Characteristic(SensorType.radar,
                                                  FieldOfView(vertical_angle=80 * np.pi / 180,
                                                              horizontal_angle=80 * np.pi / 180,
                                                              fov_range=30 * 1000),
                                                  price=200.00)

# Bosch front radar
long_range_radar_characteristic = Characteristic(SensorType.radar,
                                                 FieldOfView(vertical_angle=6 * np.pi / 180,
                                                             horizontal_angle=3 * np.pi / 180,
                                                             fov_range=210 * 1000),
                                                 price=200.00)

# Bosch multi purpose camera
camera_characteristic = Characteristic(SensorType.camera,
                                       FieldOfView(vertical_angle=58 * np.pi / 180,
                                                   horizontal_angle=50 * np.pi / 180,
                                                   fov_range=500 * 1000),
                                       price=3000.00)

# Bosch ultrasonic sensor
ultrasonic_characteristic = Characteristic(SensorType.ultrasound,
                                           FieldOfView(vertical_angle=35 * np.pi / 180,
                                                       horizontal_angle=70 * np.pi / 180,
                                                       fov_range=5.5 * 1000),
                                           price=60.0)

# Place sensors on an even grid on all surfaces, considering all cardinal orientations except down
sensor_position_points = []
sensor_candidates = []

for position in allowed_sensor_positions.values():
    # Generate points on surface
    corners = position['coordinates']
    sampled_points = sample_parallelogram_points(corners, distance=150)
    sensor_position_points.extend(sampled_points)

    # Generate sensor candidates
    allowed_sensor_types = position['allowed_sensors']

    for sampled_position in sampled_points:
        for type in allowed_sensor_types:
            orientations = [np.array([0.0, 0.0]),  # front
                            np.array([np.pi / 2, 0.0]),  # right
                            np.array([-np.pi / 2, 0.0]),  # left
                            np.array([np.pi, 0.0]),  # back
                            np.array([0.0, np.pi / 2])]  # up

            # TODO: choose orientations more intelligently
            for orientation in orientations:
                if type == SensorType.lidar:
                    sensor_candidates.append(Sensor(lidar_characteristic, sampled_position, orientation))
                elif type == SensorType.radar:
                    sensor_candidates.append(Sensor(short_range_radar_characteristic, sampled_position, orientation))
                    sensor_candidates.append(Sensor(long_range_radar_characteristic, sampled_position, orientation))
                elif type == SensorType.camera:
                    sensor_candidates.append(Sensor(camera_characteristic, sampled_position, orientation))
                elif type == SensorType.ultrasound:
                    sensor_candidates.append(Sensor(ultrasonic_characteristic, sampled_position, orientation))

sensor_position_points = np.array(sensor_position_points)

ax.scatter(sensor_position_points[:, 0], sensor_position_points[:, 1], sensor_position_points[:, 2],
           s=2.5, linewidths=0.5, edgecolors='black', c='white')

plt.show()

# Calculate coverage booleans
print(len(sensor_candidates))

coverage_grid: List[bool] = []

for point in criticality_grid:
    for sensor in sensor_candidates:
        coverage_grid.append(is_visible_by_sensor(point[0:3], sensor))


def evaluate_coverage(region_of_interest_grid: np.ndarray, coverage_booleans: np.ndarray) -> float:
    """
    Evaluate the covered regions of interest: Ratio of the RoI grid covered by the sensor setup.

    Additionally, all regions of interest with a critical index above c_i ≥ 0.7 need to be covered by
    at least two sensors of a different type.

    :param region_of_interest_grid: region of interest grid point weights.
    :param coverage_booleans: 1 - grid point is covered by sensor setup, 0 - otherwise.
    :return: Coverage volume.
    """
    return sum(np.array(region_of_interest_grid) * np.array(coverage_booleans)) \
           / sum(np.array(region_of_interest_grid))


print(evaluate_coverage(criticality_grid, coverage_grid))
