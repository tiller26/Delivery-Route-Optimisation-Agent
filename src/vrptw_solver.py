import csv
import os
import math
import time
from src.customer          import Customer
from src.depot             import Depot
from src.ga_parameters     import GAParameters
from src.fitness_evaluator import FitnessEvaluator
from src.island_manager    import IslandManager
from src.chromosome        import Chromosome


class VRPTWSolver:
    """
    Main controller for the VRPTW Island GA system.

    Ties together all components:
        - Loads problem instances from Solomon-format files
        - Configures GA parameters
        - Runs the Island GA optimisation via IslandManager
        - Displays the solution as a route table and metrics summary
        - Exports the solution to CSV

    Usage:
        solver = VRPTWSolver()
        solver.load_problem("data/C101.txt")
        solver.configure_parameters(params)   # optional — defaults are pre-set
        solver.solve()
        solver.displaysolution()
        solver.exportsolution("output/solution.csv")
    """

    def __init__(self):
        self.customers      = {}    # {customer_id: Customer}
        self.depot          = None
        self.vehicle_count  = 0
        self.vehicle_cap    = 0.0
        self.params         = GAParameters()
        self.evaluator      = None
        self.island_manager = None
        self.solution       = None  # best Chromosome after optimisation
        self.schedule       = []    # decoded route schedule (list of dicts)
        self.is_loaded      = False
        self.is_configured  = False
        self.solve_time     = 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # PROBLEM LOADING
    # ══════════════════════════════════════════════════════════════════════════

    def load_problem(self, filepath: str) -> None:
        """
        Loads a VRPTW problem instance from a Solomon-format text file.

        Solomon file format (space-separated columns):
            Section VEHICLE:
                NUMBER   CAPACITY
                <value>  <value>

            Section CUSTOMER:
                CUST_NO  XCOORD  YCOORD  DEMAND  READY_TIME  DUE_DATE  SERVICE_TIME
                Row 0 is the depot (CUST_NO = 0).
                Rows 1+ are customers.

        Args:
            filepath: Path to a Solomon .txt instance file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Problem file not found: {filepath}")

        print(f"\n[VRPTWSolver] Loading: {filepath}")

        with open(filepath, 'r') as f:
            lines = [ln.strip() for ln in f.readlines()]

        # remove blank lines
        lines = [ln for ln in lines if ln]

        vehicle_count    = 0
        vehicle_capacity = 0.0
        raw_rows         = []

        in_vehicle  = False
        in_customer = False

        for line in lines:
            upper = line.upper()

            if upper.startswith('VEHICLE'):
                in_vehicle  = True
                in_customer = False
                continue

            if upper.startswith('CUSTOMER'):
                in_customer = True
                in_vehicle  = False
                continue

            if in_vehicle:
                if upper.startswith('NUMBER'):
                    continue                     # skip header
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        vehicle_count    = int(parts[0])
                        vehicle_capacity = float(parts[1])
                        in_vehicle       = False
                    except ValueError:
                        pass
                continue

            if in_customer:
                if upper.startswith('CUST'):
                    continue                     # skip header
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        raw_rows.append([float(p) for p in parts[:7]])
                    except ValueError:
                        pass

        if not raw_rows:
            raise ValueError(f"No data found in: {filepath}")

        # ── depot is row 0 ────────────────────────────────────────────────
        d = raw_rows[0]
        self.depot = Depot(
            depot_id = int(d[0]),
            x_coordinate = d[1],
            y_coordinate = d[2],
            open_time = d[4],
            close_time = d[5]
        )

        # ── customers are rows 1+ ─────────────────────────────────────────
        self.customers = {}
        for row in raw_rows[1:]:
            cid = int(row[0])
            self.customers[cid] = Customer(
                customer_Id = cid,
                x_coordinate = row[1],
                y_coordinate = row[2],
                demand        = row[3],
                earliest_time = row[4],
                latest_time   = row[5],
                service_time  = row[6]
            )

        self.vehicle_count = vehicle_count
        self.vehicle_cap   = vehicle_capacity
        self.is_loaded     = True

        self._validate_problem_instance()

        print(f"  Customers loaded:  {len(self.customers)}")
        print(f"  Vehicle count:     {self.vehicle_count}")
        print(f"  Vehicle capacity:  {self.vehicle_cap}")
        print(f"  Depot:             {self.depot}")
        print(f"[VRPTWSolver] Problem loaded successfully.\n")

    def _validate_problem_instance(self) -> None:
        """
        Checks that the loaded instance is valid before optimisation.
        Raises ValueError if any issue is found.
        """
        if self.depot is None:
            raise ValueError("Depot is missing from the problem instance.")
        if not self.customers:
            raise ValueError("No customers found in the problem instance.")
        for cid, c in self.customers.items():
            e, l = c.get_time_window()
            if e > l:
                raise ValueError(
                    f"Customer {cid} has invalid time window: "
                    f"earliest={e} > latest={l}"
                )

    # ══════════════════════════════════════════════════════════════════════════
    # PARAMETER CONFIGURATION
    # ══════════════════════════════════════════════════════════════════════════

    def configure_parameters(self, params: GAParameters) -> None:
        """
        Sets the GA parameters to use for the next solve() call.
        If not called, default GAParameters values are used.

        Args:
            params: A configured and validated GAParameters object
        """
        params.validate()
        self.params       = params
        self.is_configured = True
        print(f"[VRPTWSolver] Parameters configured.")
        print(f"  Population size:   {params.population_size}")
        print(f"  Max generations:   {params.max_generations}")
        print(f"  Islands:           {params.num_islands}")
        print(f"  Topology:          {params.island_topology.value}")
        print(f"  Crossover:         {params.crossover_strategy.value} @ {params.crossover_rate}")
        print(f"  Mutation:          {params.mutation_strategy.value} @ {params.mutation_rate}")
        print(f"  Migration:         every {params.migration_interval} gens "
              f"@ rate {params.migration_rate} | strategy {params.migration_strategy.value}")
        print(f"  Fitness weights:   w1={params.w1} w2={params.w2} w3={params.w3}\n")

    # ══════════════════════════════════════════════════════════════════════════
    # SOLVE
    # ══════════════════════════════════════════════════════════════════════════

    def solve(self) -> None:
        """
        Runs the full Island GA optimisation pipeline:
            1. Creates FitnessEvaluator
            2. Creates IslandManager
            3. Initialises all island populations
            4. Runs the evolutionary loop until convergence
            5. Stores the global best solution
            6. Builds the human-readable schedule
        """
        if not self.is_loaded:
            raise RuntimeError("No problem loaded. Call load_problem() first.")

        self.params.validate()

        print(f"[VRPTWSolver] Starting optimisation...\n")
        start = time.time()

        # ── build evaluator ────────────────────────────────────────────────
        self.evaluator = FitnessEvaluator(
            customers  = self.customers,
            depot      = self.depot,
            parameters = self.params
        )

        # ── build island manager ───────────────────────────────────────────
        self.island_manager = IslandManager(
            params    = self.params,
            evaluator = self.evaluator
        )

        # ── initialise populations ─────────────────────────────────────────
        self.island_manager.initialise_all_islands(
            customers = self.customers,
            depot     = self.depot
        )

        # ── run evolution ──────────────────────────────────────────────────
        self.solution = self.island_manager.run_evolution()

        self.solve_time = time.time() - start

        # ── build output schedule ──────────────────────────────────────────
        self.schedule = self.build_outputschedule()

        print(f"\n[VRPTWSolver] Optimisation complete in {self.solve_time:.2f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # SCHEDULE BUILDER
    # ══════════════════════════════════════════════════════════════════════════

    def build_outputschedule(self) -> list:
        """
        Decodes the best chromosome into a human-readable schedule.

        Returns a list of route dicts, one per vehicle:
        {
            'vehicle':        vehicle number (1-based),
            'route':          list of customer IDs in visit order,
            'stops':          list of stop dicts (see below),
            'total_distance': float,
            'total_violation': float,
            'customers_served': int
        }

        Each stop dict:
        {
            'customer_id':   int,
            'x':             float,
            'y':             float,
            'demand':        float,
            'time_window':   (earliest, latest),
            'arrival_time':  float,
            'wait_time':     float,
            'service_start': float,
            'departure':     float,
            'violation':     float,
            'status':        str
        }
        """
        if self.solution is None:
            return []

        routes   = self.solution.decode_routes(self.depot)
        schedule = []

        for v_idx, route in enumerate(routes):
            stops        = []
            current_time = 0.0
            current_loc  = self.depot.get_location()
            total_dist   = 0.0
            total_viol   = 0.0

            for cid in route:
                c         = self.customers[cid]
                c_loc     = c.get_location()
                dist      = math.sqrt(
                    (current_loc[0] - c_loc[0]) ** 2 +
                    (current_loc[1] - c_loc[1]) ** 2
                )
                total_dist   += dist
                arrival_time  = current_time + dist
                earliest, latest = c.get_time_window()

                if arrival_time < earliest:
                    wait_time    = earliest - arrival_time
                    service_start = earliest
                    status       = "early (waited)"
                    violation    = 0.0
                elif arrival_time > latest:
                    wait_time    = 0.0
                    service_start = arrival_time
                    violation    = arrival_time - latest
                    total_viol  += violation
                    status       = "LATE"
                else:
                    wait_time    = 0.0
                    service_start = arrival_time
                    violation    = 0.0
                    status       = "on time"

                departure    = service_start + c.get_service_time()
                current_time = departure
                current_loc  = c_loc

                stops.append({
                    'customer_id'  : cid,
                    'x'            : c_loc[0],
                    'y'            : c_loc[1],
                    'demand'       : c.get_demand(),
                    'time_window'  : (earliest, latest),
                    'arrival_time' : round(arrival_time, 4),
                    'wait_time'    : round(wait_time, 4),
                    'service_start': round(service_start, 4),
                    'departure'    : round(departure, 4),
                    'violation'    : round(violation, 4),
                    'status'       : status
                })

            # add return leg to depot
            depot_loc  = self.depot.get_location()
            return_dist = math.sqrt(
                (current_loc[0] - depot_loc[0]) ** 2 +
                (current_loc[1] - depot_loc[1]) ** 2
            )
            total_dist += return_dist

            schedule.append({
                'vehicle'         : v_idx + 1,
                'route'           : route,
                'stops'           : stops,
                'total_distance'  : round(total_dist, 4),
                'total_violation' : round(total_viol, 4),
                'customers_served': len(route)
            })

        return schedule

    # ══════════════════════════════════════════════════════════════════════════
    # DISPLAY
    # ══════════════════════════════════════════════════════════════════════════

    def display_solution(self) -> None:
        """
        Prints the complete solution to the terminal:
            - Per-vehicle route table with arrival times and status
            - Overall metrics summary
        """
        if self.solution is None:
            print("[VRPTWSolver] No solution available. Call solve() first.")
            return

        print("\n" + "=" * 70)
        print("  SOLUTION SUMMARY")
        print("=" * 70)

        total_dist  = 0.0
        total_viol  = 0.0
        total_cust  = 0

        for vehicle in self.schedule:
            print(f"\n  Van {vehicle['vehicle']} "
                  f"| Customers: {vehicle['customers_served']} "
                  f"| Distance: {vehicle['total_distance']:.2f} "
                  f"| Violation: {vehicle['total_violation']:.2f}")
            print(f"  Route: Depot → {' → '.join(str(c) for c in vehicle['route'])} → Depot")
            print(f"  {'Cust':>5}  {'(x,y)':>12}  {'Window':>12}  "
                  f"{'Arrival':>8}  {'Wait':>6}  {'Status'}")
            print(f"  {'-'*65}")

            for stop in vehicle['stops']:
                e, l = stop['time_window']
                print(
                    f"  {stop['customer_id']:>5}  "
                    f"({stop['x']:>4.0f},{stop['y']:>4.0f})  "
                    f"[{e:>5.1f},{l:>5.1f}]  "
                    f"{stop['arrival_time']:>8.2f}  "
                    f"{stop['wait_time']:>6.2f}  "
                    f"{stop['status']}"
                )

            total_dist += vehicle['total_distance']
            total_viol += vehicle['total_violation']
            total_cust += vehicle['customers_served']

        print(f"\n{'=' * 70}")
        print(f"  OVERALL METRICS")
        print(f"{'=' * 70}")
        print(f"  Customers served:     {total_cust}")
        print(f"  Vehicles used:        {len(self.schedule)}")
        print(f"  Total distance:       {total_dist:.4f}")
        print(f"  Total violation:      {total_viol:.4f}")
        print(f"  Best fitness (F(x)):  {self.solution.get_fitness():.4f}")
        print(f"  Solve time:           {self.solve_time:.2f}s")
        print(f"  Generations run:      "
              f"{self.island_manager.current_generation + 1}")
        print(f"  Migration events:     "
              f"{len(self.island_manager.get_migration_log())}")
        print(f"{'=' * 70}\n")

    # ══════════════════════════════════════════════════════════════════════════
    # EXPORT
    # ══════════════════════════════════════════════════════════════════════════

    def export_solution(self, filepath: str) -> None:
        """
        Exports the solution to a CSV file.

        Columns:
            vehicle, customer_id, x, y, demand,
            time_window_earliest, time_window_latest,
            arrival_time, wait_time, service_start,
            departure, violation, status,
            route_distance, route_violation

        A summary row is appended at the end.

        Args:
            filepath: Output path e.g. "output/solution.csv"
        """
        if self.solution is None:
            print("[VRPTWSolver] No solution to export. Call solve() first.")
            return

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # header
            writer.writerow([
                'vehicle', 'customer_id', 'x', 'y', 'demand',
                'time_window_earliest', 'time_window_latest',
                'arrival_time', 'wait_time', 'service_start',
                'departure', 'violation', 'status',
                'route_total_distance', 'route_total_violation'
            ])

            total_dist = 0.0
            total_viol = 0.0

            for vehicle in self.schedule:
                for stop in vehicle['stops']:
                    e, l = stop['time_window']
                    writer.writerow([
                        vehicle['vehicle'],
                        stop['customer_id'],
                        stop['x'],
                        stop['y'],
                        stop['demand'],
                        e,
                        l,
                        stop['arrival_time'],
                        stop['wait_time'],
                        stop['service_start'],
                        stop['departure'],
                        stop['violation'],
                        stop['status'],
                        vehicle['total_distance'],
                        vehicle['total_violation']
                    ])
                total_dist += vehicle['total_distance']
                total_viol += vehicle['total_violation']

            # summary rows
            writer.writerow([])
            writer.writerow(['SUMMARY', '', '', '', '', '', '', '', '', '',
                             '', '', '', '', ''])
            writer.writerow(['vehicles_used', len(self.schedule)])
            writer.writerow(['total_distance', round(total_dist, 4)])
            writer.writerow(['total_violation', round(total_viol, 4)])
            writer.writerow(['best_fitness', round(self.solution.get_fitness(), 4)])
            writer.writerow(['solve_time_seconds', round(self.solve_time, 2)])
            writer.writerow(['generations',
                             self.island_manager.current_generation + 1])
            writer.writerow(['migration_events',
                             len(self.island_manager.get_migration_log())])

        print(f"[VRPTWSolver] Solution exported to: {filepath}")

    def export_fitness_history(self, filepath: str) -> None:
        """
        Exports the per-generation global fitness history to CSV.
        Used for plotting convergence charts in the parameter study.

        Args:
            filepath: Output path e.g. "output/fitness_history.csv"
        """
        if self.island_manager is None:
            print("[VRPTWSolver] No history available. Call solve() first.")
            return

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        history = self.island_manager.get_global_fitness_history()

        # also get per-island histories
        island_histories = [
            self.island_manager.get_island(i).get_fitness_history()
            for i in range(self.params.num_islands)
        ]

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            header = ['generation', 'global_best'] + \
                     [f'island_{i}_best' for i in range(self.params.num_islands)]
            writer.writerow(header)

            for gen in range(len(history)):
                row = [gen + 1, round(history[gen], 6)]
                for ih in island_histories:
                    if gen < len(ih):
                        row.append(round(ih[gen], 6))
                    else:
                        row.append('')
                writer.writerow(row)

        print(f"[VRPTWSolver] Fitness history exported to: {filepath}")

    # ══════════════════════════════════════════════════════════════════════════
    # ACCESSORS
    # ══════════════════════════════════════════════════════════════════════════

    def get_solution(self) -> Chromosome:
        return self.solution

    def get_customers(self) -> dict:
        return dict(self.customers)

    def get_depot(self) -> Depot:
        return self.depot

    def get_schedule(self) -> list:
        return list(self.schedule)

    def get_island_manager(self) -> IslandManager:
        return self.island_manager

    def __repr__(self) -> str:
        return (
            f"VRPTWSolver("
            f"loaded={self.is_loaded}, "
            f"customers={len(self.customers)}, "
            f"solved={self.solution is not None})"
        )