# This class is the customer who is waiting for thei deliuvery.
class Customer:
    # Variables 
    served = False # This variable will control whether the customer ghot the parcel or not.

    def __init__(self, customer_Id, x_coordinate, y_coordinate, demand, earliest_time, latest_time, service_time):
        self.customer_Id = customer_Id
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
        self.demand = demand
        self.earliest_time = earliest_time
        self.latest_time = latest_time
        self.service_time = service_time

    # This returns the Id of the customer
    def getcustomer_id(self):
        return self.customer_Id

    # This returns the location opf the customer
    def get_location(self):
        return (self.x_coordinate, self.y_coordinate)

    # This returns the time window of the customeer availability
    def get_time_window(self):
        return (self.earliest_time, self.latest_time)

    # This returns the demand of the customer
    def get_demand(self):
        return self.demand

    # This returns the service time of the customer
    def get_service_time(self):
        return self.service_time

    # This marks the customer as served
    def mark_as_served(self):
        self.served = True

    # This checks if the customer is served
    def is_served(self):
        return self.served

    # This is a string representation of the customer
    def to_string(self):
        return f"Customer {self.customer_Id}: Location ({self.x_coordinate}, {self.y_coordinate}), Demand {self.demand}, Time Window ({self.earliest_time}, {self.latest_time}), Service Time {self.service_time}"
