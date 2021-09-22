from enum import Enum
from typing import Dict

import numpy as np


class SensorType(Enum):
    """
    Sensors type enum: Lidar, radar, camera and ultrasound.
    """
    lidar = 0
    radar = 1
    camera = 2
    ultrasound = 3


class FieldOfView():
    """
    This class represents the field of view, defined by two angles and the range:
    fov_i = {α_{V_i}, α_{H_i}, R}
    """

    def __init__(self, vertical_angle: float, horizontal_angle: float, fov_range: float):
        """
        Field of view constructor.

        :param vertical_angle: radians.
        :param horizontal_angle: radians
        :param fov_range: mm.
        """
        self.vertical_angle = vertical_angle
        self.horizontal_angle = horizontal_angle
        self.fov_range = fov_range


class Characteristic():
    """
    Class defining a sensor characteristic.
    One individual sensor can be defined by three characteristics:
    type, geometry of its field of view, price.
    """

    def __init__(self, type: SensorType, field_of_view: FieldOfView, price: float):
        self.type = type
        self.field_of_view = field_of_view
        self.price = price


class Sensor:
    """
    Class defining a sensor.
    Multiple variables (degrees of freedom) must be set:
    • T: sensor's characteristic (i.e. type, range, field of view, price);
    • P: sensor's position;
    • O: sensor's orientation (given by two angles).
    """

    def __init__(self, characteristic: Characteristic, position: np.array, orientation: np.array, dict: Dict=None):
        """
        Constructor.
        :param characteristic: Characteristic object.
        :param position: cartesian coordinates.
        :param orientation: 2 angular coordinates.
        :param dict: extra information storage.
        """
        self.characteristic = characteristic
        self.position = position
        self.orientation = orientation
        self.dict = dict
        self.coverage = None
