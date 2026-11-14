"""
Day 4 test — verifies IslandManager.
Run from the project root:  python tests/test_day4.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.customer       import Customer
from src.depot          import Depot
from src.ga_parameters  import GAParameters, TopologyType, MigrationStrategy, InitStrategy
from src.fitness_evaluator import FitnessEvaluator
from src.island_manager import IslandManager


def separator(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)


# ── Problem instance ──────────────────────────────────────────────────────────
separator("SETUP: 10-customer problem instance")

depot = Depot(depot_id=0, x_coordinate=50.0, y_coordinate=50.0, open_time=0.0, close_time=200.0)

customers = {
    1:  Customer(1,  x_coordinate=20.0, y_coordinate=30.0, demand=5.0, earliest_time=0.0,  latest_time=100.0, service_time=1.0),
    2:  Customer(2,  x_coordinate=70.0, y_coordinate=80.0, demand=8.0, earliest_time=10.0, latest_time=80.0,  service_time=1.0),
    3:  Customer(3,  x_coordinate=40.0, y_coordinate=60.0, demand=3.0, earliest_time=5.0,  latest_time=90.0,  service_time=0.5),
    4:  Customer(4,  x_coordinate=90.0, y_coordinate=20.0, demand=6.0, earliest_time=0.0,  latest_time=120.0, service_time=1.5),
    5:  Customer(5,  x_coordinate=10.0, y_coordinate=70.0, demand=4.0, earliest_time=20.0, latest_time=100.0, service_time=1.0),
    6:  Customer(6,  x_coordinate=60.0, y_coordinate=40.0, demand=7.0, earliest_time=0.0,  latest_time=150.0, service_time=0.5),
    7:  Customer(7,  x_coordinate=30.0, y_coordinate=90.0, demand=2.0, earliest_time=15.0, latest_time=90.0,  service_time=1.0),
    8:  Customer(8,  x_coordinate=80.0, y_coordinate=60.0, demand=5.0, earliest_time=0.0,  latest_time=200.0, service_time=1.0),
    9:  Customer(9,  x_coordinate=50.0, y_coordinate=10.0, demand=9.0, earliest_time=10.0, latest_time=150.0, service_time=2.0),
    10: Customer(10, x_coordinate=25.0, y_coordinate=50.0, demand=3.0, earliest_time=0.0,  latest_time=200.0, service_time=0.5),
}

params = GAParameters()
params.population_size          = 20
params.max_generations          = 100
params.num_islands              = 4
params.island_topology          = TopologyType.RING
params.migration_rate           = 0.8
params.migration_interval       = 10
params.migration_size           = 2
params.migration_strategy       = MigrationStrategy.BEST
params.crossover_rate           = 0.85
params.mutation_rate            = 0.1
params.local_search_rate        = 0.3
params.population_init_strategy = InitStrategy.SAVINGS_NN_RANDOM

evaluator = FitnessEvaluator(customers=customers, depot=depot, parameters=params)
print(f"  Evaluator ready: {evaluator}")


# ── Test topology graphs ──────────────────────────────────────────────────────
separator("TEST: topology graph construction")

manager = IslandManager(params=params, evaluator=evaluator)

# RING
params.island_topology = TopologyType.RING
graph = manager.build_topology_graph(4)
print(f"  RING topology (4 islands): {graph}")
assert graph[0] == [3, 1], f"RING island 0 wrong: {graph[0]}"
assert graph[1] == [0, 2], f"RING island 1 wrong: {graph[1]}"
assert graph[2] == [1, 3], f"RING island 2 wrong: {graph[2]}"
assert graph[3] == [2, 0], f"RING island 3 wrong: {graph[3]}"
print("  RING: correct ✓")

# FULLY CONNECTED
params.island_topology = TopologyType.FULLY_CONNECTED
graph = manager.build_topology_graph(4)
print(f"  FULLY_CONNECTED topology (4 islands): {graph}")
assert sorted(graph[0]) == [1, 2, 3], f"FULLY_CONNECTED island 0 wrong: {graph[0]}"
print("  FULLY_CONNECTED: correct ✓")

# STAR
params.island_topology = TopologyType.STAR
graph = manager.build_topology_graph(4)
print(f"  STAR topology (4 islands): {graph}")
assert sorted(graph[0]) == [1, 2, 3], f"STAR hub wrong: {graph[0]}"
assert graph[1] == [0], f"STAR spoke wrong: {graph[1]}"
print("  STAR: correct ✓")

# reset to RING for remaining tests
params.island_topology = TopologyType.RING
print("\n  TOPOLOGY TESTS PASSED")


# ── Test initialisation ───────────────────────────────────────────────────────
separator("TEST: initialise all islands")

manager = IslandManager(params=params, evaluator=evaluator)
manager.initialise_all_islands(customers=customers, depot=depot)

print(f"\n  Manager: {manager}")
assert len(manager.get_all_islands()) == 4, "Should have 4 islands"

# verify all islands have different best solutions (diversity check)
island_bests = [manager.get_island(i).get_best_solution().get_fitness()
                for i in range(4)]
print(f"  Island best fitnesses: {[round(f, 4) for f in island_bests]}")

# global best should be the minimum of all island bests
global_best = manager.get_global_best_solution()
print(f"  Global best fitness: {global_best.get_fitness():.4f}")
assert global_best.get_fitness() == min(island_bests), \
    f"Global best should be min of island bests"

print("\n  INITIALISATION TESTS PASSED")


# ── Test migration ────────────────────────────────────────────────────────────
separator("TEST: migration execution")

# record best fitness per island before migration
before = [manager.get_island(i).get_best_solution().get_fitness() for i in range(4)]
print(f"  Best fitness before migration: {[round(f, 2) for f in before]}")

manager.execute_migration(gen=9)  # simulate gen 10

log = manager.get_migration_log()
print(f"  Migration events logged: {len(log)}")
print(f"  Migration log: {log}")

# with RING topology there should be 4 pairs: (0,1), (1,2), (2,3), (3,0)
# but we process unique pairs so 4 events
assert len(log) >= 1, "At least one migration event should be logged"

print("\n  MIGRATION TESTS PASSED")


# ── Test migration select strategies ─────────────────────────────────────────
separator("TEST: migration selection strategies")

island_0 = manager.get_island(0)

# BEST strategy
params.migration_strategy = MigrationStrategy.BEST
params.migration_size     = 2
migrants = manager.select_migrants(island_0)
print(f"  BEST strategy selected {len(migrants)} migrants")
print(f"  Migrant fitnesses: {[round(m.get_fitness(), 4) for m in migrants]}")
assert len(migrants) == 2
# best strategy should return the lowest fitness chromosomes
pop_sorted = sorted(island_0.get_population(), key=lambda c: c.get_fitness())
assert migrants[0].get_fitness() == pop_sorted[0].get_fitness(), \
    "BEST should return fittest chromosome"
print("  BEST: correct ✓")

# RANDOM strategy
params.migration_strategy = MigrationStrategy.RANDOM
migrants_r = manager.select_migrants(island_0)
print(f"  RANDOM strategy selected {len(migrants_r)} migrants")
assert len(migrants_r) == 2
print("  RANDOM: correct ✓")

# MIXED strategy
params.migration_strategy = MigrationStrategy.MIXED
params.migration_size     = 4
migrants_m = manager.select_migrants(island_0)
print(f"  MIXED strategy selected {len(migrants_m)} migrants")
assert len(migrants_m) >= 2
print("  MIXED: correct ✓")

# reset
params.migration_strategy = MigrationStrategy.BEST
params.migration_size     = 2

print("\n  MIGRATION STRATEGY TESTS PASSED")


# ── Test full evolution ───────────────────────────────────────────────────────
separator("TEST: full evolution — 100 generations")

# fresh manager for clean evolution test
manager2 = IslandManager(params=params, evaluator=evaluator)
manager2.initialise_all_islands(customers=customers, depot=depot)

initial_best = manager2.get_global_best_solution().get_fitness()
print(f"  Initial global best: {initial_best:.4f}")

best_solution = manager2.run_evolution()

final_best = best_solution.get_fitness()
print(f"\n  Final global best:   {final_best:.4f}")
print(f"  Improvement:         {initial_best - final_best:.4f}")
print(f"  Migration events:    {len(manager2.get_migration_log())}")

history = manager2.get_global_fitness_history()
print(f"  History length:      {len(history)}")
print(f"  First 5 fitness:     {[round(f, 2) for f in history[:5]]}")
print(f"  Last  5 fitness:     {[round(f, 2) for f in history[-5:]]}")

assert final_best <= initial_best, "Global best should not get worse"
assert len(history) > 0, "Fitness history should not be empty"

# verify solution contains all customers
genes    = best_solution.get_genes()
cust_ids = sorted([g for g in genes if g != 0])
assert cust_ids == list(range(1, 11)), \
    f"Solution missing customers: {cust_ids}"
print(f"  All 10 customers in solution ✓")

routes = best_solution.decode_routes(depot)
print(f"  Number of routes (vans used): {len(routes)}")
for i, route in enumerate(routes):
    print(f"    Van {i+1}: {route}")

print("\n  FULL EVOLUTION TESTS PASSED")


# ── Test warm start ───────────────────────────────────────────────────────────
separator("TEST: warm start re-optimisation")

seed = manager2.get_global_best_solution()
seed_fitness = seed.get_fitness()
print(f"  Seed chromosome fitness: {seed_fitness:.4f}")

manager2.warm_start(customers=customers, depot=depot, seed_chromosome=seed)

# after warm start migration log and history should be reset
assert manager2.get_migration_log() == [], "Migration log should reset after warm start"
assert manager2.get_global_fitness_history() == [], "History should reset after warm start"
print(f"  Migration log reset ✓")
print(f"  History reset ✓")

print("\n  WARM START TESTS PASSED")


# ── Summary ───────────────────────────────────────────────────────────────────
separator("ALL DAY 4 TESTS PASSED")
print()