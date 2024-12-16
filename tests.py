from coordinates import Location
from torch import tensor, FloatTensor
import torch

def test_location_distance():
    """
    Test the flat distance between two points in cartesian space
    """
    a = Location(tensor(2, dtype=torch.float64), tensor(2, dtype=torch.float64), tensor(0, dtype=torch.float64))
    b = Location(tensor(5, dtype=torch.float64), tensor(6, dtype=torch.float64), tensor(0, dtype=torch.float64))

    distance = a.distance(b)

    assert distance == tensor(5, device="cuda")

def test_location_construct():
    """
    Test that the location construct function converts to cartesian space correctly
    """
    a = Location.construct(39.799173325, -86.23799871, 222.3617)
    b = Location.construct(39.799179425, -86.237984708, 222.3676)

    distance = a.distance(b)

    assert distance == tensor(1.375, dtype=torch.float64, device="cuda")