from __future__ import annotations
from dataclasses import dataclass
from coordinates import Location, Checkpoint, device
import math as m
import copy
from collections.abc import Callable

from torch import FloatTensor, IntTensor, tensor
import torch


@dataclass(slots=True)
class Transition:
    target: Node
    work_required: FloatTensor
    time_required: FloatTensor


@dataclass(slots=True)
class Node:
    """A Node in the Graph"""

    # Ensure uniqueness for identifiability
    id: IntTensor

    transitions: list[Transition]

    kinetic_energy: FloatTensor

    position: Location
    velocity: FloatTensor

    # Keep track of motor velocity, it will always be a maxium of the current velocity, but can be less when ramping up the motor or coasting
    motor_velocity: FloatTensor

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Node) and value.id == self.id

    def create_transition(
        self,
        targets: list[Node],
        mass: FloatTensor,
        coefficient_of_gravity: FloatTensor,
        coefficient_of_friction: FloatTensor,
        air_density: FloatTensor,
        get_coefficent_of_drag: Callable[[float], float],
        get_projected_area: Callable[[float], float],
        wind_velocity: FloatTensor,
        wind_bearing: FloatTensor,
    ) -> list[Transition]:

        weight = mass * coefficient_of_gravity

        slopes = torch.tensor(
            [Location.slope(self.position, target.position) for target in targets],
            device=device,
        )
        bearings = torch.tensor(
            [Location.bearing(self.position, target.position) for target in targets],
            device=device,
        )

        target_velocities = torch.tensor(
            [target.velocity for target in targets], device=device
        )

        # Gravitational Force (Weight)'s parallel component, assists you down a slope, resists you up a slope
        forces_gx = weight * torch.sin(slopes)

        # Frictional Force, it always opposes velocity
        forces_friction = coefficient_of_friction * weight * torch.cos(slopes)

        coefficents_of_drag = tensor(
            [get_coefficent_of_drag(bearing) for bearing in bearings], device=device
        )
        projected_areas = tensor(
            [get_projected_area(bearing) for bearing in bearings], device=device
        )

        delta_bearings = wind_bearing - bearings
        opposing_wind_velocities = wind_velocity * torch.cos(delta_bearings)
        effective_wind_velocities = opposing_wind_velocities - target_velocities

        forces_drag = (
            0.5
            * air_density
            * (torch.pow(effective_wind_velocities, 2))
            * coefficents_of_drag
            * projected_areas
        )

        # Work needed to reach the target velocity
        delta_work = (
            0.5 * mass * (torch.pow(target_velocities, 2) - torch.pow(self.velocity, 2))
        )

        # The x component of gravity always apposes
        affecting_forces = -forces_friction

        # If the slope is a negative incline, gravity assists the motion
        affecting_forces += torch.where(slopes < 0, forces_gx, -forces_gx)

        # If the wind is going faster in the same direction as the car
        affecting_forces += torch.where(
            effective_wind_velocities > 0, forces_drag, -forces_drag
        )

        work_required = delta_work - affecting_forces

        # Calculate the time required to transition to the given node
        delta_distances = torch.tensor(
            [Location.distance(self.position, target.position) for target in targets],
            device=device,
        )
        time_required = torch.where(
            delta_distances == 0,
            0,
            torch.where(
                target_velocities == 0,
                float("inf"),
                3.6 * delta_distances / target_velocities,
            ),
        )

        transitions = [
            Transition(target, work_required_item, time_required_item)
            for target, work_required_item, time_required_item in zip(
                targets, work_required, time_required
            )
        ]
        return transitions


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

        max_velocity_tensor = tensor(max_velocity, device=device)
        max_motor_velocity_tensor = tensor(max_motor_velocity, device=device)
        wind_velocity_tensor = tensor(wind_velocity, device=device)
        wind_bearing_tensor = tensor(wind_bearing, device=device)
        mass_tensor = tensor(mass, device=device)
        coefficient_of_friction_tensor = tensor(coefficient_of_friction, device=device)
        coefficient_of_gravity_tensor = tensor(9.8, device=device)
        air_density_tensor = tensor(1.225, device=device)

        node_index = 0

        starting_checkpoint = checkpoints[0]
        starting_point = starting_checkpoint.points(1)[0]
        starting_node = Node(
            id=node_index,
            transitions=[],
            kinetic_energy=tensor(0, device=device),
            position=starting_point,
            velocity=tensor(0, device=device),
            motor_velocity=tensor(0, device=device),
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
                velocity = tensor(0, device=device)
                while velocity < max_velocity_tensor:
                    motor_velocity = tensor(0, device=device)
                    while motor_velocity < max_motor_velocity_tensor:
                        target = Node(
                            node_index,
                            transitions=[],
                            kinetic_energy=tensor(0, device=device),
                            position=location,
                            velocity=velocity,
                            motor_velocity=motor_velocity,
                        )

                        targets.append(target)

                        node_index += 1
                        motor_velocity += motor_velocity_step_size
                    velocity += velocity_step_size

            for node in current_layer:
                transitions = node.create_transition(
                    target=targets,
                    mass=mass_tensor,
                    coefficient_of_gravity=coefficient_of_gravity_tensor,
                    coefficient_of_friction=coefficient_of_friction_tensor,
                    air_density=air_density_tensor,
                    get_coefficent_of_drag=get_coefficient_of_drag,
                    get_projected_area=get_projected_area,
                    wind_velocity=wind_velocity_tensor,
                    wind_bearing=wind_bearing_tensor,
                )
                node.transitions.extend(transitions)
            current_layer = targets

        return graph
