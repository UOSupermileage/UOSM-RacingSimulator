from coordinates import Location
from torch import tensor, FloatTensor

def test_location_distance():
    """
    Test the flat distance between two points in cartesian space
    """
    a = Location(tensor(2), tensor(2), tensor(0))
    b = Location(tensor(5), tensor(6), tensor(0))

    distance = a.distance(b)

    assert distance == tensor(5, device="cuda")

def test_location_construct():
    """
    Test that the location construct function converts to cartesian space correctly
    """
    a = Location.construct(39.799173325, -86.23799871, 222.3617)
    b = Location.construct(39.799179425, -86.237984708, 222.3676)

    distance = a.distance(b)

    assert distance == tensor(1.375,device="cuda")