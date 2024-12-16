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
    alt = tensor(elevation, dtype=torch.float64,  device=device) - tensor(218.9818, dtype=torch.float64,  device=device)

    lat0 = torch.deg2rad(torch.tensor(origin_latitude, dtype=torch.float64,  device=device))
    lon0 = torch.deg2rad(torch.tensor(origin_longitude, dtype=torch.float64,  device=device))
    
    X = earth_radius * (long - lon0) * torch.cos(lat0)
    Y = earth_radius * (lat - lat0)
    
    return X, Y, alt

def setup(device_name: str):
    global device
    global earth_radius
    global origin

    device = device_name

    earth_radius = tensor(6378137, dtype=torch.float64, device=device)

setup("cuda")

@dataclass(slots=True)
class Location:
    """Represents a location in cartesian 3D."""

    x: FloatTensor  # in meters
    y: FloatTensor  # in meters
    z: FloatTensor  # in meters

    def construct(latitude: float, longitude: float, altitude: float) -> Location:
        """Construct a location from latitude and longitude in degrees. Altitude in meters."""
        X, Y, lat = cartesian(latitude, longitude, altitude)        
        return Location(X, Y, lat)

    def distance_with_z(
        a: Location,
        b: Location,
    ) -> FloatTensor:
        return torch.sqrt((b.x-a.x)**2 + (b.y-a.y)**2 + (b.z-a.z)**2)

    def distance(
        a: Location,
        b: Location,
    ) -> FloatTensor:
        return torch.sqrt((b.x-a.x)**2 + (b.y-a.y)**2)

    def slope(
        a: Location,
        b: Location,
    ) -> FloatTensor:
        """Returns the slope in radians to get from a to b"""

        # TODO: Might make more sense to calculate the straight line cartesian distance
        distance = Location.distance(a,b)
        height = b.z - a.z
        angle = torch.atan(height / distance)
        return angle

    def bearing(a: Location, b: Location) -> FloatTensor:
        x_delta = b.x - a.x
        y_delta = b.y - a.y
        
        return torch.atan(y_delta / x_delta)
    
    def interpolated_position_towards_target(
        a: Location, b: Location, distance: FloatTensor
    ) -> Location:
        """Interpolation a position on a sphere of a given radius that is distance towards b from a"""
        
        print(f"ax: {a.x}, bx: {b.x}, ay: {a.y}, by: {b.y}")

        if a.x == b.x and a.y == b.y:
            print("Warning: a and b are identical")

        dx = b.x - a.x
        dy = b.y - a.y

        if dx == 0 and dy == 0:
            print("Warning: dx and dy are 0.")
            return a
        
        if dx == 0:
            if dy > 0:
                return tensor(torch.pi / 2)
            else:
                return tensor(torch.pi * 3 / 2)

        angle = torch.atan(dy/dx)

        print(f"dx: {dx}, dy: {dy}, angle: {angle}")
        
        destination_x = distance * torch.cos(angle)
        destination_y = distance * torch.sin(angle)
        
        total_distance = Location.distance(a, b)
        percentage_of_distance_traveled = distance / total_distance

        destination_z = a.z + percentage_of_distance_traveled * (
            b.z - a.z
        )

        return Location(
            destination_x,
            destination_y,
            destination_z,
        )


@dataclass(slots=True)
class Checkpoint:
    """A band of points on the track.
    This represents the discretisized version of a line drawn across the track."""

    left: Location
    right: Location

    def points(self, n: int) -> list[Location]:
        """Get a list of points representing valid positions on the checkpoint line.

        Args:
            n (int): Number of points on the line

        Returns:
            list[Location]: points on the checkpoint line
        """
        if n < 0:
            raise ValueError("A checkpoint must contain at least one point")

        distance = Location.distance(self.left, self.right)

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
