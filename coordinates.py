from __future__ import annotations
import math as m
from dataclasses import dataclass

import torch
from torch import tensor, FloatTensor

# Earth's radius in meters
device: str
earth_radius: tensor
origin: tensor

def cartesian(longitude: float, latitude: float, elevation: float, origin_latitude: float = 39.7881314579999, origin_longitude: float = -86.238784119, origin_altitude: float = 218.9818):
    """Create a cartisian point from long and lat in degrees and elevation in meters."""

    lat = torch.deg2rad(tensor(latitude, dtype=torch.float64, device=device))
    long = torch.deg2rad(tensor(longitude, dtype=torch.float64, device=device))
    alt = tensor(elevation, dtype=torch.float64,  device=device) - tensor(origin_altitude, dtype=torch.float64,  device=device)

    lat0 = torch.deg2rad(torch.tensor(origin_latitude, dtype=torch.float64,  device=device))
    lon0 = torch.deg2rad(torch.tensor(origin_longitude, dtype=torch.float64,  device=device))
    
    X = earth_radius * (long - lat0)
    Y = earth_radius * (lat - lon0)
    
    return X, Y, alt

def setup(device_name: str):
    global device
    global earth_radius
    global origin

    device = device_name

    earth_radius = tensor(6378137, dtype=torch.float64, device=device)

class Location:
    """Represents a location in 3D space using a single tensor (x, y, z)."""
    
    @staticmethod
    def construct(latitude: float, longitude: float, altitude: float) -> FloatTensor:
        """Constructs a Location from latitude, longitude, and altitude."""
        X, Y, Z = cartesian(latitude, longitude, altitude)        
        return torch.tensor([X, Y, Z], dtype=torch.float64, device=device)
    
    @staticmethod
    def distance(a: FloatTensor, b: FloatTensor) -> FloatTensor:
        """Returns the Euclidean distance between two locations in the XY plane."""
        return torch.norm(b[:, :2] - a[:, :2], dim=1)

    @staticmethod
    def distance_with_z(a: FloatTensor, b: FloatTensor) -> FloatTensor:
        """Returns the full 3D Euclidean distance between two locations."""
        return torch.norm(b - a)

    @staticmethod
    def slope(a: FloatTensor, b: FloatTensor) -> FloatTensor:
        """Computes the slope angle (radians) to travel from `a` to `b`."""
        distance = Location.distance(a, b)
        height = b[:, 2] - a[:, 2]
        return torch.atan(height / distance)

    @staticmethod
    def bearing(a: FloatTensor, b: FloatTensor) -> FloatTensor:
        """Computes the bearing angle (radians) from `a` to `b`."""
        delta_x = b[:, 0] - a[:, 0]
        delta_y = b[:, 1] - a[:, 1]
        return torch.atan2(delta_y, delta_x)

    @staticmethod
    def interpolated_position_towards_target(
        a: FloatTensor, b: FloatTensor, distance: FloatTensor
    ) -> FloatTensor:
        """Finds a point that is `distance` along the line from `a` to `b`."""
        
        direction = (b - a) / torch.norm(b - a)
        return a + direction * distance


@dataclass(slots=True)
class Checkpoint:
    """A band of points on the track.
    This represents the discretisized version of a line drawn across the track."""

    left: FloatTensor
    right: FloatTensor

    def points(self, n: int) -> list[FloatTensor]:
        """Get a list of points representing valid positions on the checkpoint line.

        Args:
            n (int): Number of points on the line

        Returns:
            list[Location]: points on the checkpoint line
        """
        if n < 0:
            raise ValueError("A checkpoint must contain at least one point")

        distance = Location.distance(self.left.reshape(1, 3), self.right.reshape(1, 3))

        # Track width is 0, just return the left point
        if distance == 0:
            return [self.left]

        step_distance = distance / n

        locations = []
        for i in range(1, n + 1):            
            point = Location.interpolated_position_towards_target(
                self.left, self.right, step_distance * i
            )
            print(f"Interpolated: {point}")
            
            locations.append(point)

        return locations


if __name__ == "__main__":
    print("Testing Checkpoint")

    a = Location.construct(m.radians(20), m.radians(21), 0)
    b = Location.construct(m.radians(25), m.radians(22), 0)

    checkpoint = Checkpoint(a, b)
    print(checkpoint.points(5))
