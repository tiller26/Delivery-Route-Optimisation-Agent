class Vehicle:
    """
    Represents a delivery vehicle in the VRPTW system.
    Used by the Output Manager when decoding the solution
    into operational route assignments.
    """

    def __init__(self, vehicle_id: int, capacity: float):
        self.vehicle_id    = vehicle_id
        self.capacity      = capacity
        self.current_load  = 0.0
        self.route         = []        # ordered list of Customer objects
        self.departure_time = 0.0
        self.is_available  = True

    def getvehicle_id(self) -> int:
        return self.vehicle_id

    def add_customer(self, customer) -> bool:
        """Adds a customer to this vehicle's route if capacity allows."""
        if self.current_load + customer.get_demand() > self.capacity:
            return False
        self.route.append(customer)
        self.current_load += customer.get_demand()
        return True

    def remove_customer(self, customer) -> None:
        if customer in self.route:
            self.route.remove(customer)
            self.current_load -= customer.get_demand()

    def getroute(self) -> list:
        return list(self.route)

    def get_remainingcapacity(self) -> float:
        return self.capacity - self.current_load

    def getcurrent_load(self) -> float:
        return self.current_load

    def getdeparture_time(self) -> float:
        return self.departure_time

    def setdeparture_time(self, t: float) -> None:
        self.departure_time = t

    def set_available(self, status: bool) -> None:
        self.is_available = status

    def is_available(self) -> bool:
        return self.is_available

    def reset(self) -> None:
        self.route         = []
        self.current_load  = 0.0
        self.departure_time = 0.0
        self.is_available  = True

    def __repr__(self) -> str:
        return (
            f"Vehicle(id={self.vehicle_id}, "
            f"load={self.current_load}/{self.capacity}, "
            f"stops={len(self.route)})"
        )