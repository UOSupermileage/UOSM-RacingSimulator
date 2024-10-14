from __future__ import annotations
from dataclasses import dataclass, field
import uuid
from coordinates import Location, Checkpoint, device
import math as m
import copy
from collections.abc import Callable

from torch import FloatTensor, tensor
import torch
import torch.autograd.profiler as profiler
from tqdm import tqdm


class Transition:
    target: Node
    work_required: FloatTensor
    time_required: FloatTensor
    id: int

    def __init__(self, target: Node, work_required: FloatTensor, time_required: FloatTensor, id: int) -> None:
        self.target = target
        self.work_required = work_required
        self.time_required = time_required
        self.id = id

class Node:
    """A Node in the Graph"""

    # Ensure uniqueness for identifiability

    transitions: list[Transition]

    position: Location
    velocity: FloatTensor

    id: int

    def __init__(self, transitions: list[Transition], position: Location, velocity: FloatTensor, id: int) -> None:
        self.transitions = transitions
        self.position = position
        self.velocity = velocity
        self.id = id

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Node) and value.id == self.id

    def __hash__(self):
        return hash(id)
    
    def __str__(self) -> str:
        return f"Node {self.id} [position: {self.position}, velocity: {self.velocity}]"

    def __repr__(self) -> str:
        return self.__str__()

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
        
        weight = mass * coefficient_of_gravity # For forces!!!

        # Gravitational Force (Weight)'s parallel component, assists you down a slope, resists you up a slope
        forces_gx = torch.abs(weight * torch.sin(slopes))

        # Frictional Force, it always opposes velocity
        forces_friction = torch.abs(coefficient_of_friction * weight * torch.cos(slopes))

        coefficents_of_drag = tensor(
            [get_coefficent_of_drag(bearing) for bearing in bearings], device=device
        )
        projected_areas = tensor(
            [get_projected_area(bearing) for bearing in bearings], device=device
        )

        delta_bearings = wind_bearing - bearings
        opposing_wind_velocities = wind_velocity * torch.cos(delta_bearings)
        
        # Consider the average to be more acurate since we don't instantly accelerate
        average_target_velocities = (self.velocity + target_velocities) / 2
        
        # The relative of the wind with respect to the car
        effective_wind_velocities = average_target_velocities - opposing_wind_velocities

        forces_drag = (
            0.5
            * air_density
            * effective_wind_velocities * effective_wind_velocities
            * coefficents_of_drag
            * projected_areas
        )

        # Change in kinetic energy to reach target velocity
        delta_kinetic = (
            0.5 * mass * (torch.pow(target_velocities, 2) - torch.pow(self.velocity, 2))
        )
        
        # The x component of gravity always apposes
        affecting_forces = -forces_friction

        # If the slope is a negative incline, gravity assists the motion
        affecting_forces += torch.where(slopes < 0, forces_gx, -forces_gx)

        # If the wind is going faster in the same direction as the car
        # TODO: This is innacurate if we accelerate faster than the wind speed throughout the step
        affecting_forces += torch.where(
            effective_wind_velocities > 0, forces_drag, -forces_drag
        )
                
        # Calculate the time required to transition to the given node
        delta_distances = torch.tensor(
            [Location.distance(self.position, target.position) for target in targets],
            device=device,
        )

        # Multiply force by distance to get affecting energy
        affecting_energies = affecting_forces * delta_distances
        
        work_required = delta_kinetic + affecting_energies

        delta_velocities = target_velocities - self.velocity

        time_required = torch.abs(2 * delta_distances / delta_velocities)

        transitions = [
            Transition(target, work_required_item, time_required_item, uuid.uuid4().int)
            for target, work_required_item, time_required_item in zip(
                targets, work_required, time_required
            )
        ]
        return transitions


@dataclass(slots=True)
class Graph:
    start: Node
    end: Node = None

    nodes: dict[int, Node] = field(default_factory=lambda: dict())

    def construct(
        checkpoints: list[Checkpoint],
        n_points_per_checkpoint: int,
        max_velocity: float,  # Max velocity the car is allowed to go is 42 km/h
        velocity_step_size: float,
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

        if len(checkpoints) < 1:
            raise ValueError("Cannot create Graph without at least 1 checkpoint")

        max_velocity_tensor = tensor(max_velocity, device=device)
        velocity_step_size_tensor = tensor(velocity_step_size, device=device)
        wind_velocity_tensor = tensor(wind_velocity, device=device)
        wind_bearing_tensor = tensor(wind_bearing, device=device)
        mass_tensor = tensor(mass, device=device)
        coefficient_of_friction_tensor = tensor(coefficient_of_friction, device=device)
        coefficient_of_gravity_tensor = tensor(9.8, device=device)
        air_density_tensor = tensor(1.225, device=device)

        starting_checkpoint = checkpoints[0]
        starting_point = starting_checkpoint.points(1)[0]
        starting_node = Node(
            transitions=[],
            position=starting_point,
            velocity=tensor(0, device=device),
            id=uuid.uuid4().int
        )

        graph = Graph(start=starting_node)
        graph.nodes[starting_node.id] = starting_node

        current_layer: list[Node] = [starting_node]

        # Iterate over all the checkpoints, constructing each checkpoints corresponding layer of nodes
        for i, checkpoint in enumerate(tqdm(checkpoints[1:])):
            location_points = checkpoint.points(n_points_per_checkpoint)
            targets: list[Node] = []

            if i is (len(checkpoints[1:]) - 1):
                max_velocity_tensor = tensor(0, device=device)
                print("Last node must be at 0m/s")
                
            for location in location_points:
                velocity = tensor(0, device=device)
                while velocity <= max_velocity_tensor:                     
                    target = Node(
                        transitions=[],
                        position=location,
                        velocity=velocity,
                        id=uuid.uuid4().int
                    )

                    targets.append(target)
                    graph.nodes[target.id] = target
                    velocity = velocity + velocity_step_size_tensor

            for node in current_layer:
                transitions = node.create_transition(
                    targets=targets,
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
                graph.nodes[node.id] = node

                graph.end = node

            current_layer = targets

        return graph