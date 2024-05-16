from __future__ import annotations
import math as m
from dataclasses import dataclass

# Earth's radius in meters
EARTH_RADIUS = 6378137


def great_circle_distance(
    latitude1: float,
    longitude1: float,
    latitude2: float,
    longitude2: float,
    radius: float,
) -> float:
    """Compute the greater circle distance on a sphere using the Haversine formula
    Returns the distance in meters.
    """

    phi1 = m.radians(latitude1)
    phi2 = m.radians(latitude2)

    lambda1 = m.radians(longitude1)
    lambda2 = m.radians(longitude2)

    delta_phi = phi2 - phi1
    delta_lambda = lambda2 - lambda1

    a = (
        m.sin(delta_phi / 2.0) ** 2
        + m.cos(phi1) * m.cos(phi2) * m.sin(delta_lambda / 2.0) ** 2
    )

    c = 2 * m.atan2(m.sqrt(a), m.sqrt(1 - a))

    meters = radius * c
    return meters


def distance_with_altitude(distance: float, altitude1: float, altitude2: float):
    return m.sqrt(distance**2 + (altitude2 - altitude1) ** 2)


@dataclass(slots=True)
class Location:
    """Represents a location on a sphere."""

    latitude: float  # in degrees
    longitude: float  # in degrees
    altitude: float  # in meters

    def distance(a: Location, b: Location, radius: float = EARTH_RADIUS) -> float:
        d = great_circle_distance(
            a.latitude, a.longitude, b.latitude, b.longitude, radius
        )
        return distance_with_altitude(d, a.altitude, b.altitude)

    def bearing(a: Location, b: Location) -> float:
        latitude_a_radians = m.radians(a.latitude)
        latitude_b_radians = m.radians(b.latitude)
        longitude_delta_radians = m.radians(b.longitude - a.longitude)

        y = m.sin(longitude_delta_radians) * m.cos(latitude_b_radians)
        x = m.cos(latitude_a_radians) * m.sin(latitude_b_radians) - m.sin(
            latitude_a_radians
        ) * m.cos(latitude_b_radians) * m.cos(longitude_delta_radians)
        bearing_radians = m.atan2(y, x)

        return (m.degrees(bearing_radians) + 360) % 360

    def interpolated_position_towards_target(
        a: Location, b: Location, distance: float, radius: float = EARTH_RADIUS
    ) -> Location:
        """Interpolation a position on a sphere of a given radius that is distance towards b from a"""

        distance_radians = distance / radius
        bearing_radians = m.radians(Location.bearing(a, b))

        initial_latitude_radians = m.radians(a.latitude)
        initial_longitude_radians = m.radians(a.longitude)

        destination_latitude_radians = m.asin(
            m.sin(initial_latitude_radians) * m.cos(distance_radians)
            + m.cos(initial_latitude_radians)
            * m.sin(distance_radians)
            * m.cos(bearing_radians)
        )
        destination_longitude_radians = initial_longitude_radians + m.atan2(
            m.sin(bearing_radians)
            * m.sin(distance_radians)
            * m.cos(initial_latitude_radians),
            m.cos(distance_radians)
            - m.sin(initial_latitude_radians) * m.sin(destination_latitude_radians),
        )

        # Normalize destination longitude to be within -pi and +pi radians
        destination_longitude_radians = (destination_longitude_radians + 3 * m.pi) % (
            2 * m.pi
        ) - m.pi

        percentage_of_distance_traveled = distance / Location.distance(a, b)
        destination_altitude = a.altitude + percentage_of_distance_traveled * (
            b.altitude - a.altitude
        )

        return Location(
            m.degrees(destination_latitude_radians),
            m.degrees(destination_longitude_radians),
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
