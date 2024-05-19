import ee


def ee_init(project: str):
    ee.Authenticate()
    ee.Initialize(project=project)


def get_altitude(
    latitude: float,
    longitude: float,
) -> float:
    point = ee.Geometry.Point(longitude, latitude)
    elevation = (
        ee.Image("USGS/SRTMGL1_003").sample(point).first().get("elevation").getInfo()
    )
    return elevation


if __name__ == "__main__":
    print("Testing google_earth.py")

    ee_init("ee-jeremymcote")

    print(f"Altitude: {get_altitude(39.799168013999996 ,-86.23801415599999)}")
    print(f"Altitude: {get_altitude(39.799173325       ,-86.23799871)}")
