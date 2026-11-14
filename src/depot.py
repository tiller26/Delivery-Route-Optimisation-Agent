# This class is the depot where the vehicles start and end their routes.
class Depot:
    def __init__(self, depot_id, x_coordinate, y_coordinate, open_time, close_time):
        self.depot_id = depot_id
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate  
        self.open_time = open_time
        self.close_time = close_time

    # This returns the Id of the depot
    def get_depot_id(self):
        return self.depot_id

    # This returns the location of the depot
    def get_location(self):
        return (self.x_coordinate, self.y_coordinate)

    # This returns the operating hours of the depot
    def get_Operating_Hours(self):
        return (self.open_time, self.close_time)  

    # This is a string representation of the depot
    def to_string(self):
        return f"Depot {self.depot_id}: Location ({self.x_coordinate}, {self.y_coordinate}), Operating Hours ({self.open_time}, {self.close_time})"

    

  