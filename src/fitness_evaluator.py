class FitnessEvaluator:
    def __init__(self, customers, depot, parameters):
        self.customers = customers
        self.depot = depot
        self.parameters = parameters
        self.evaluation_count = 0

    def evaluate(self, chromosome):
        # This method will calculate the fitness score of the single chromosome.
        # It will decode the chromosome into routes and then calculate the total distance,
        # total time, and number of vehicles used. The fitness score can be a weighted sum of these factors.
        routes = chromosome.decode_routes(self.depot)
        total_distance = 0
        total_violation = 0

        for route in routes:
            route_distance = self.calculate_route_distance(route)
            route_violation = self.calc_time_window_violation(route)
            total_distance += route_distance
            total_violation += route_violation

        num_vehicles = len(routes)

        w1 = self.parameters.w1
        w2 = self.parameters.w2
        w3 = self.parameters.w3

        fitness_score = (w1 * total_distance) + (w2 * total_violation) + (w3 * num_vehicles)    
        
        chromosome.set_fitness(fitness_score)
        self.evaluation_count += 1

        return fitness_score

    def evaluate_population(self, population):
        for chromosome in population:
            if not chromosome.is_evaluated():
                fitness_score = self.evaluate(chromosome)

    def re_evaluate_plan(self, vehicles):
        # This method will re-evaluate the fitness of a given plan (set of routes) after local search or other modifications.
        total_distance = 0.0
        total_violation = 0.0
        vehicles_used = 0

        for vehicle in vehicles:
            route = [c.get_customer_id() for c in vehicle.get_route()]
            if not route:
                continue  # Skip if the route is empty

            route_distance = self.calculate_route_distance(route)
            route_violation = self.calc_time_window_violation(route)
            total_distance += route_distance
            total_violation += route_violation
            vehicles_used += 1

        w1 = self.parameters.w1
        w2 = self.parameters.w2
        w3 = self.parameters.w3

        fitness_score = (w1 * total_distance) + (w2 * total_violation) + (w3 * vehicles_used)    
        
        return fitness_score

    def calc_distance(self, loc1, loc2):
        return ((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2) ** 0.5

    def calc_depot_distance(self, customer_id):
        return self.calc_distance(self.depot.get_location(), self.customers[customer_id].get_location())

    def calculate_route_distance(self, route):
        # This method will calculate the total distance of a given route.
        if not route:
            return 0.0

        total_distance = 0.0

        # depot to first customer
        total_distance += self.calc_depot_distance(route[0])

        # between customers
        for i in range(len(route) - 1):
            from_location = self.get_location(route[i])
            to_location = self.get_location(route[i + 1])
            total_distance += self.calc_distance(from_location, to_location)

        # last customer back to depot
        total_distance += self.calc_depot_distance(route[-1])

        return total_distance

    def total_route_distance(self, routes):
        total_distance = 0
        for route in routes:
            total_distance += self.calculate_route_distance(route)
        return total_distance


    def calc_time_window_violation(self, route):
        timeline = self.simulate_route_timeline(route)
        total_violation = 0.0

        for customer_id, arrival_time, status, deviation in timeline:
            if status == "late (violation)":
                total_violation += deviation  # Add the amount of time by which we are late

        return total_violation

    def simulate_route_timeline(self, route):
        if not route:
            return []

        timeline = []
        depot_location = self.depot.get_location()
        current_time = 0

        # Start at the depot to first customer
        first_customer = self.customers[route[0]]
        current_time += self.calc_distance(depot_location, first_customer.get_location())


        for i, customer_id in enumerate(route):
            customer = self.customers[customer_id]
            earliest, latest = customer.get_time_window()

            arrival_time = current_time
            if arrival_time < earliest:
                # arrived early vehicles until the window opens
                timeline.append((customer_id, arrival_time, 'early (waited)', earliest - arrival_time))
                current_time = earliest
            elif arrival_time > latest:
                # arrived late vehicles after the window closed
                timeline.append((customer_id, arrival_time, 'late (violation)', arrival_time - latest))
            else:
                # arrived on time
                timeline.append((customer_id, arrival_time, 'on time', 0.0))

            # Add service time before travelloing to the next stop
            current_time += customer.get_service_time()

            # Travel to the next customer if there is one
            if i < len(route) - 1:
                next_customer = self.customers[route[i + 1]]
                current_time += self.calc_distance(customer.get_location(), next_customer.get_location())

        return timeline




    def calculate_vehicle_used(self, routes):
        return len(routes)

    def apply_two_opt(self, route):
        if len(route) < 3:
            return route  # No improvement possible for routes with less than 3 nodes

        best_route = list(route)
        best_distance = self.calculate_route_distance(best_route)
        improved = True

        while improved:
            improved = False
            for i in range(len(best_route) - 1):
                for j in range(i + 2, len(best_route)):
                    #reverse the segment between i and j
                    new_route = best_route[:i] + best_route[i:j + 1][::-1] + best_route[j + 1:]
                    new_distance = self.calculate_route_distance(new_route)
                    if new_distance < best_distance - 1e-10:
                        best_route = new_route
                        best_distance = new_distance
                        improved = True
                        break
                if improved:
                    break

        return best_route

    # Counters         

    def get_evaluation_count(self):
        return self.evaluation_count

    def reset_evaluation_count(self):
        self.evaluation_count = 0

    def calculate_route_time(self, route):
        # This method will calculate the total time of a given route, including service times and travel times.
        time = 0
        for i in range(len(route) - 1):
            from_location = self.get_location(route[i])
            to_location = self.get_location(route[i + 1])
            travel_time = self.euclidean_Distance(from_location, to_location) / self.parameters.travel_Speed
            time += travel_time
            if route[i + 1] != self.depot.get_depot_id():  # If it's not the depot, add service time
                customer_id = route[i + 1]
                customer = self.customers[customer_id]
                time += customer.get_service_time()
        return time

    def get_location(self, customer_id):
        if customer_id == self.depot.get_depot_id():
            return self.depot.get_location()
        else:
            return self.customers[customer_id].get_location()


      