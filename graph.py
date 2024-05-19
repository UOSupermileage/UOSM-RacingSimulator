from __future__ import annotations
from dataclasses import dataclass
from coordinates import Location, Checkpoint
import math as m
import copy
from collections.abc import Callable


@dataclass(slots=True)
class Transition:
    target: Node
    work_required: float
    time_required: float


@dataclass(slots=True)
class Node:
    """A Node in the Graph"""

    # Ensure uniqueness for identifiability
    id: int

    transitions: list[Transition]

    kinetic_energy: float

    position: Location
    velocity: float

    # Keep track of motor velocity, it will always be a maxium of the current velocity, but can be less when ramping up the motor or coasting
    motor_velocity: float

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Node) and value.id == self.id

    def create_transition(
        self,
        target: Node,
        mass: float,
        coefficient_of_gravity: float,
        coefficient_of_friction: float,
        air_density: float,
        get_coefficent_of_drag: Callable[[float], float],
        get_projected_area: Callable[[float], float],
        wind_velocity: float,
        wind_bearing: float,
    ) -> Transition:

        weight = mass * coefficient_of_gravity

        slope = Location.slope(self.position, target.position)
        bearing = Location.bearing(self.position, target.position)

        # Gravitational Force (Weight)'s parallel component, assists you down a slope, resists you up a slope
        force_gx = weight * m.sin(slope)

        # Frictional Force, it always opposes velocity
        force_friction = coefficient_of_friction * weight * m.cos(slope)

        coefficent_of_drag = get_coefficent_of_drag(bearing)
        projected_area = get_projected_area(bearing)

        delta_bearing = wind_bearing - bearing
        opposing_wind_velocity = wind_velocity * m.cos(m.radians(delta_bearing))
        effective_wind_velocity = opposing_wind_velocity - target.velocity

        force_drag = (
            0.5
            * air_density
            * (effective_wind_velocity**2)
            * coefficent_of_drag
            * projected_area
        )

        # Work needed to reach the target velocity
        delta_work = 0.5 * mass * (target.velocity**2 - self.velocity**2)

        # The x component of gravity always apposes
        affecting_forces = -force_friction

        # If the slope is a negative incline, gravity assists the motion
        affecting_forces += force_gx if slope < 0 else -1 * force_gx

        # If the wind is going faster in the same direction as the car
        affecting_forces += (
            force_drag if effective_wind_velocity > 0 else -1 * force_drag
        )

        work_required = delta_work - affecting_forces

        # Calculate the time required to transition to the given node
        delta_distance = Location.distance(self.position, target.position)
        time_required = (
            0
            if delta_distance == 0
            else (
                float("inf")
                if target.velocity == 0
                else (3.6 / target.velocity) * delta_distance
            )
        )

        transition = Transition(target, work_required, time_required)
        return transition


@dataclass(slots=True)
class Graph:
    start: Node

    def construct(
        checkpoints: list[Checkpoint],
        n_points_per_checkpoint: int,
        max_velocity: float,  # Max velocity the car is allowed to go is 42 km/h
        velocity_step_size: float,
        max_motor_velocity: float,  # Max velocity the motor is allowed to go is 40 km/h
        motor_velocity_step_size: float,
        wind_velocity: float,
        wind_bearing: float,
        mass: float,
        coefficient_of_friction: float,
        get_coefficient_of_drag: Callable[[float], float],
        get_projected_area: Callable[[float], float],
    ) -> Graph:
        """Construct a graph from a list of checkpoints.

        Raises a value error if the list is empty.
        """

        if len(checkpoints) < 2:
            raise ValueError("Cannot create Graph without at least 2 checkpoints")

        node_index = 0

        starting_checkpoint = checkpoints[0]
        starting_point = starting_checkpoint.points(1)[0]
        starting_node = Node(
            id=node_index,
            transitions=[],
            kinetic_energy=0,
            position=starting_point,
            velocity=0,
            motor_velocity=0,
        )
        node_index += 1

        graph = Graph(start=starting_node)

        current_layer: list[Node] = [starting_node]

        # Iterate over all the checkpoints, constructing each checkpoints corresponding layer of nodes
        for i, checkpoint in enumerate(checkpoints[1:]):
            print(f"Processing checkpoint {i}. Layer size: {len(current_layer)}")
            location_points = checkpoint.points(n_points_per_checkpoint)
            targets: list[Node] = []

            for location in location_points:
                velocity = 0
                while velocity < max_velocity:
                    motor_velocity = 0
                    while motor_velocity < max_motor_velocity:
                        target = Node(
                            node_index,
                            transitions=[],
                            kinetic_energy=0,
                            position=location,
                            velocity=velocity,
                            motor_velocity=motor_velocity,
                        )

                        targets.append(target)

                        node_index += 1
                        motor_velocity += motor_velocity_step_size
                    velocity += velocity_step_size

            for node in current_layer:
                for target in targets:
                    transition = node.create_transition(
                        target=target,
                        mass=mass,
                        coefficient_of_gravity=9.8,
                        coefficient_of_friction=coefficient_of_friction,
                        air_density=1.225,
                        get_coefficent_of_drag=get_coefficient_of_drag,
                        get_projected_area=get_projected_area,
                        wind_velocity=wind_velocity,
                        wind_bearing=wind_bearing,
                    )
                    node.transitions.append(transition)

            current_layer = targets

        return graph
