from __future__ import annotations
import math as m
from dataclasses import dataclass

import torch
from torch import tensor, FloatTensor

# Earth's radius in meters
device = "cuda"
earth_radius = tensor(6378137, device=device)


def setup(device_name: str, radius: float):
    global device
    global earth_radius

    device = device_name
    earth_radius = radius


def great_circle_distance(
    latitude1: FloatTensor,
    longitude1: FloatTensor,
    latitude2: FloatTensor,
    longitude2: FloatTensor,
) -> FloatTensor:
    """Compute the greater circle distance on a sphere using the Haversine formula
    Returns the distance in meters.
    """

    delta_phi = latitude2 - latitude1
    delta_lambda = longitude2 - longitude1

    a = torch.pow(torch.sin(delta_phi / 2.0), 2) + torch.cos(latitude1) * torch.cos(
        latitude2
    ) * torch.pow(torch.sin(delta_lambda / 2.0), 2)

    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    meters = earth_radius * c
    return meters


def distance_with_altitude(
    distance: FloatTensor, altitude1: FloatTensor, altitude2: FloatTensor
) -> FloatTensor:
    return torch.sqrt(torch.pow(distance, 2) + torch.pow((altitude2 - altitude1), 2))


@dataclass(slots=True)
class Location:
    """Represents a location on a sphere."""

    latitude: FloatTensor  # in degrees
    longitude: FloatTensor  # in degrees
    altitude: FloatTensor  # in meters

    def construct(latitude: float, longitude: float, altitude: float) -> Location:
        return Location(
            tensor(latitude, device=device),
            tensor(longitude, device=device),
            tensor(altitude, device=device),
        )

    def distance(
        a: Location,
        b: Location,
    ) -> FloatTensor:
        d = great_circle_distance(a.latitude, a.longitude, b.latitude, b.longitude)
        return distance_with_altitude(d, a.altitude, b.altitude)

    def slope(
        a: Location,
        b: Location,
    ) -> FloatTensor:
        """Returns the slope in radians to get from a to b"""

        # TODO: Might make more sense to calculate the straight line cartesian distance
        distance = great_circle_distance(
            a.latitude, a.longitude, b.latitude, b.longitude
        )
        height = b.altitude - a.altitude
        angle = torch.atan(height / distance)
        return angle

    def bearing(a: Location, b: Location) -> FloatTensor:
        longitude_delta = b.longitude - a.longitude

        y = torch.sin(longitude_delta) * torch.cos(b.latitude)
        x = torch.cos(a.latitude) * torch.sin(b.latitude) - torch.sin(
            a.latitude
        ) * torch.cos(b.latitude) * torch.cos(longitude_delta)
        bearing_radians = torch.atan2(y, x)

        return torch.fmod(bearing_radians + 2 * torch.pi, 2 * torch.pi)

    def interpolated_position_towards_target(
        a: Location, b: Location, distance: FloatTensor
    ) -> Location:
        """Interpolation a position on a sphere of a given radius that is distance towards b from a"""

        distance_radians = distance / earth_radius
        bearing_radians = Location.bearing(a, b)

        initial_latitude_radians = a.latitude
        initial_longitude_radians = a.longitude

        destination_latitude_radians = torch.asin(
            torch.sin(initial_latitude_radians) * torch.cos(distance_radians)
            + torch.cos(initial_latitude_radians)
            * torch.sin(distance_radians)
            * torch.cos(bearing_radians)
        )
        destination_longitude_radians = initial_longitude_radians + torch.atan2(
            torch.sin(bearing_radians)
            * torch.sin(distance_radians)
            * torch.cos(initial_latitude_radians),
            torch.cos(distance_radians)
            - torch.sin(initial_latitude_radians)
            * torch.sin(destination_latitude_radians),
        )

        # Normalize destination longitude to be within -pi and +pi radians
        destination_longitude_radians = (
            destination_longitude_radians + 3 * torch.pi
        ) % (2 * torch.pi) - torch.pi

        total_distance = Location.distance(a, b)
        percentage_of_distance_traveled = (
            1 if total_distance == 0 else distance / total_distance
        )
        destination_altitude = a.altitude + percentage_of_distance_traveled * (
            b.altitude - a.altitude
        )

        return Location(
            destination_latitude_radians,
            destination_longitude_radians,
            destination_altitude,
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

        step_distance = distance / (n + 1)

        locations = []
        for i in range(1, n + 1):
            point = Location.interpolated_position_towards_target(
                self.left, self.right, step_distance * i
            )
            locations.append(point)

        return locations


if __name__ == "__main__":
    print("Testing Checkpoint")

    a = Location.construct(m.radians(20), m.radians(21), 0)
    b = Location.construct(m.radians(25), m.radians(22), 0)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(
        "Distance: ",
        great_circle_distance(a.latitude, a.longitude, b.latitude, b.longitude),
    )

    checkpoint = Checkpoint(a, b)
    print(checkpoint.points(5))
