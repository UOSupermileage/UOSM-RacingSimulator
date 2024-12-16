import pandas as pd
from coordinates import Checkpoint, Location, device
from torch import tensor
import torch

track_data = pd.read_csv("./sem_2023_us.csv")

track_data = track_data.rename(columns={
    "Metres above sea level": "Altitude"
})

track_data.head(10)

checkpoints: list[Checkpoint] = []
for i, row in track_data.iterrows():
    location = Location(tensor(row["Latitude"], dtype=torch.float64, device=device), tensor(row["Longitude"],dtype=torch.float64,  device=device), tensor(row["Altitude"], dtype=torch.float64, device=device))
    checkpoints.append(Checkpoint(location, location))

from graph import Graph

def get_coefficient_of_drag(bearing: float) -> float:
    return 0.5

def get_projected_area(bearing: float) -> float:
    return 0.5

g = Graph.construct(
    checkpoints=checkpoints[:2],
    n_points_per_checkpoint=1,
    max_velocity=42 * 1000 / 3600,  # Max velocity the car is allowed to go is 42 km/h
    velocity_step_size=1000 / 3600,
    max_motor_velocity=40 * 1000 / 3600,  # Max velocity the motor is allowed to go is 40 km/h
    motor_velocity_step_size=1000 / 3600,
    wind_velocity=5000 / 3600,
    wind_bearing=200,
    mass=1000,
    coefficient_of_friction=0.5, # TODO: Figure this out
    get_coefficient_of_drag=get_coefficient_of_drag,
    get_projected_area=get_projected_area,
)