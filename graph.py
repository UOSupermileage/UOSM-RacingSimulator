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

HALL_RADIUS = 0.2667
HALL_CIRC = HALL_RADIUS * torch.pi * 2
DRIVETRAIN_EFFICIENCY = 0.98
PROJECTED_AREA = (943416 / 1000) / 1000
COEFFICIENT_OF_DRAG = 0.33
GEAR_RATIO = 25 / 7

def convert_velocity_to_motor_rpm(velocity: FloatTensor) -> FloatTensor:
    """Returns the motor's rpm for a given car velocity"""
    return (velocity / HALL_CIRC) * 60


def convert_rpm_to_angular_velocity(rpm: FloatTensor) -> FloatTensor:
    """Returns the motor's angular velocity (rad / s) for a given rpm"""
    return (rpm / 60) * 2 * torch.pi


def motor_efficiency(velocity: FloatTensor) -> FloatTensor:
    """Returns the motor's effeciency at a given velocity from 0 to 1.0"""
    rpm = convert_velocity_to_motor_rpm(velocity)
    return (
        DRIVETRAIN_EFFICIENCY
        * (-0.00000877777777778 * (rpm - 3000) * (rpm - 3000) + 82)
        / 100
    )


def motor_torque(motor_force: FloatTensor, velocity: FloatTensor) -> FloatTensor:
    rpm = convert_velocity_to_motor_rpm(velocity)
    angular_velocity = convert_rpm_to_angular_velocity(rpm)
    return (motor_force / angular_velocity) * motor_efficiency(velocity)


def motor_work(velocity: FloatTensor, wheel_work: FloatTensor):
    """Returns the work required by the motor for a given work at the wheel"""
    motor_e = motor_efficiency(velocity)
    return wheel_work / motor_e


class Transition:
    target: Node
    work_required: FloatTensor
    time_required: FloatTensor
    id: int

    def __init__(
        self,
        target: Node,
        work_required: FloatTensor,
        time_required: FloatTensor,
        id: int,
    ) -> None:
        self.target = target
        self.work_required = work_required
        self.time_required = time_required
        self.id = id


class Node:
    """A Node in the Graph"""

    # Ensure uniqueness for identifiability

    transitions: list[Transition]

    position: FloatTensor
    velocity: FloatTensor

    id: int

    def __init__(
        self,
        transitions: list[Transition],
        position: FloatTensor,
        velocity: FloatTensor,
        id: int,
    ) -> None:
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
        motor_force: FloatTensor,
        brake_power: FloatTensor,
        air_density: FloatTensor,
        wind_velocity: FloatTensor,
        wind_bearing: FloatTensor,
    ) -> list[Transition]:
        current_positions = self.position.repeat(len(targets), 1)
        target_positions = torch.stack([target.position for target in targets], dim=0)
        target_velocities = torch.stack([target.velocity for target in targets], dim=0)

        slopes = Location.slope(current_positions, target_positions)
        bearings = Location.bearing(current_positions, target_positions)

        # Gravitational and frictional forces
        weight = mass * coefficient_of_gravity

        # Gravitational Force (Weight)'s parallel component, assists you down a slope, resists you up a slope
        forces_gx = weight * torch.sin(slopes)

        # Frictional Force, it always opposes velocity
        forces_friction = coefficient_of_friction * weight * torch.cos(slopes)

        coefficents_of_drag = tensor(
            COEFFICIENT_OF_DRAG, dtype=torch.float64, device=device
        )

        projected_areas = tensor(PROJECTED_AREA, dtype=torch.float64, device=device)

        delta_bearings = wind_bearing - bearings
        effective_wind_velocities = wind_velocity * torch.cos(delta_bearings)

        # Consider the average to be more acurate since we don't instantly accelerate
        average_target_velocities = (self.velocity + target_velocities) / 2

        # The relative of the wind with respect to the car
        relative_wind_velocities = average_target_velocities - effective_wind_velocities

        forces_drag = (
            0.5
            * air_density
            * relative_wind_velocities
            * relative_wind_velocities
            * coefficents_of_drag
            * projected_areas
        )

        # Change in kinetic energy to reach target velocity
        delta_kinetic = (
            0.5 * mass * (torch.pow(target_velocities, 2) - torch.pow(self.velocity, 2))
        )

        delta_kinetic_zeroed = torch.maximum(
            delta_kinetic, torch.zeros_like(delta_kinetic, dtype=torch.float64, device=device)
        )

        # The x component of gravity always apposes
        affecting_forces = torch.abs(forces_friction)

        # If the slope is a negative incline, force is negative (gravity assists the motion) else force is positive.
        affecting_forces += forces_gx

        # If the wind is going faster in the same direction as the car
        affecting_forces += torch.where(
            relative_wind_velocities > 0, forces_drag, -forces_drag
        )

        # Calculate the distance required to transition to the given node
        delta_distances = Location.distance_with_z(current_positions, target_positions)

        # Multiply force by distance to get affecting energy
        affecting_energies = affecting_forces * delta_distances

        work_required = delta_kinetic_zeroed + affecting_energies

        wheel_torque_required = (delta_kinetic + affecting_energies) / delta_distances
        motor_torque_required = wheel_torque_required / (
            GEAR_RATIO * DRIVETRAIN_EFFICIENCY
        )
        motor_torque_provided = motor_torque(motor_force, target_velocities)

        work_required_at_motor = work_required / motor_efficiency(target_velocities)

        # Time required
        delta_velocities = target_velocities - self.velocity
        time_required = 2 * delta_distances / torch.abs(delta_velocities)

        # Brake force and power
        # Average required braking force
        brake_force_required = torch.abs(delta_kinetic)
        brake_power_required = brake_force_required * target_velocities

        # Feasibility checks
        motor_feasible = motor_torque_provided >= motor_torque_required
        brake_feasible = brake_power_required >= brake_power

        feasable = torch.logical_or(
            torch.logical_and(delta_velocities >= 0, motor_feasible),
            torch.logical_and(delta_velocities < 0, brake_feasible),
        )

        transitions = []
        for target, work_required_item, time_required_item, feasable_item in zip(
            targets, work_required_at_motor, time_required, feasable
        ):
            # if feasable_item == 1:
            transitions.append(
                Transition(
                    target, work_required_item, time_required_item, uuid.uuid4().int
                )
            )

        return transitions


@dataclass(slots=True)
class Graph:  #
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
        motor_force: float,
        brake_power: float,
        mass: float,
        coefficient_of_friction: float,
    ) -> Graph:
        """Construct a graph from a list of checkpoints.

        Raises a value error if the list is empty.
        """

        if len(checkpoints) < 1:
            raise ValueError("Cannot create Graph without at least 1 checkpoint")

        max_velocity_tensor = tensor(max_velocity, dtype=torch.float64, device=device)
        velocity_step_size_tensor = tensor(
            velocity_step_size, dtype=torch.float64, device=device
        )
        wind_velocity_tensor = tensor(wind_velocity, dtype=torch.float64, device=device)
        wind_bearing_tensor = tensor(wind_bearing, dtype=torch.float64, device=device)
        mass_tensor = tensor(mass, dtype=torch.float64, device=device)
        coefficient_of_friction_tensor = tensor(
            coefficient_of_friction, dtype=torch.float64, device=device
        )
        coefficient_of_gravity_tensor = tensor(9.81, dtype=torch.float64, device=device)
        air_density_tensor = tensor(1.225, dtype=torch.float64, device=device)
        motor_force_tensor = tensor(motor_force, dtype=torch.float64, device=device)
        brake_power_tensor = tensor(brake_power, dtype=torch.float64, device=device)

        starting_checkpoint = checkpoints[0]
        starting_point = starting_checkpoint.points(1)[0]
        starting_node = Node(
            transitions=[],
            position=starting_point,
            velocity=tensor(0, dtype=torch.float64, device=device),
            id=uuid.uuid4().int,
        )

        graph = Graph(start=starting_node)
        graph.nodes[starting_node.id] = starting_node

        current_layer: list[Node] = [starting_node]

        # Iterate over all the checkpoints, constructing each checkpoints corresponding layer of nodes
        for i, checkpoint in enumerate(tqdm(checkpoints[1:])):
            location_points = checkpoint.points(n_points_per_checkpoint)
            targets: list[Node] = []

            if i is (len(checkpoints[1:]) - 1):
                max_velocity_tensor = tensor(0, dtype=torch.float64, device=device)
                print("Last node must be at 0m/s")

            for location in location_points:
                velocity = tensor(0, dtype=torch.float64, device=device)
                while velocity <= max_velocity_tensor:
                    target = Node(
                        transitions=[],
                        position=location,
                        velocity=velocity,
                        id=uuid.uuid4().int,
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
                    motor_force=motor_force_tensor,
                    brake_power=brake_power_tensor,
                    air_density=air_density_tensor,
                    wind_velocity=wind_velocity_tensor,
                    wind_bearing=wind_bearing_tensor,
                )

                node.transitions.extend(transitions)
                # graph.nodes[node.id] = node

                graph.end = node

            current_layer = targets

        return graph

    def get_node(self, id: int) -> Node | None:
        return self.nodes[id] if id in self.nodes else None
