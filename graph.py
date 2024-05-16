from __future__ import annotations
from dataclasses import dataclass
from coordinates import Location, Checkpoint
import math as m
import copy


@dataclass(slots=True)
class Transition:
    target: Node
    delta_work: float
    delta_time: float


@dataclass(slots=True)
class Node:
    """A Node in the Graph"""

    # Ensure uniqueness for identifiability
    id: int

    transitions: list[Transition]

    position: Location
    velocity: float

    # Keep track of motor velocity, it will always be a maxium of the current velocity, but can be less when ramping up the motor or coasting
    motor_velocity: float

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Node) and value.id == self.id

    def create_transition(self, target: Node) -> Transition:

        # TODO: Calculate the work required to transition to the given node
        # TODO: Calculate the time required to transition to the given node

        transition = Transition(target, 0)
        return transition


@dataclass(slots=True)
class Graph:
    start: Node

    def construct(
        checkpoints: list[Checkpoint],
        n_points_per_checkpoint: int = 5,
        max_velocity: float = 42,  # Max velocity the car is allowed to go is 42 km/h
        velocity_step_size: float = 0.5,
        max_motor_velocity: float = 40,  # Max velocity the motor is allowed to go is 40 km/h
        motor_velocity_step_size: float = 0.5,
        wind_velocity: float = 5,  # The wind is blowing at 5 km/h
        wind_direction: float = 0,
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
            position=starting_point,
            velocity=0,
            motor_velocity=0,
        )
        node_index += 1

        graph = Graph(start=starting_node)

        current_layer: list[Node] = [starting_node]

        # Iterate over all the checkpoints, constructing each checkpoints corresponding layer of nodes
        for checkpoint in checkpoints[1:]:
            location_points = checkpoint.points(n_points_per_checkpoint)
            targets: list[Node] = []

            for location in location_points:
                velocity = 0
                while velocity < max_velocity:
                    motor_velocity = 0
                    while motor_velocity < max_motor_velocity:
                        target = Node(
                            id=node_index,
                            transitions=[],
                            position=location,
                            velocity=velocity,
                            motor_velocity=motor_velocity,
                        )

                        targets.append(targets)

                        node_index += 1
                        motor_velocity += motor_velocity_step_size
                    velocity += velocity_step_size

            for node in current_layer:
                for target in targets:
                    transition = node.create_transition(target)
                    node.transitions.append(transition)

            current_layer = targets

        return graph
