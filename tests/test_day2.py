"""
Day 2 test — verifies FitnessEvaluator.
Run from the project root:  python tests/test_day2.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.customer     import Customer
from src.depot        import Depot
from src.ga_parameters      import GAParameters
from src.chromosome   import Chromosome
from src.fitness_evaluator import FitnessEvaluator


def separator(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)


# ── Build a tiny problem instance ─────────────────────────────────────────────
#
#   Depot at (0, 0)
#   4 customers arranged in a simple grid:
#
#       C2(0,10)     C3(10,10)
#
#       Depot(0,0)   C1(10,0)    C4(20,0)
#
separator("SETUP: tiny 4-customer problem")

depot = Depot(depot_id=0, x_coordinate=0.0, y_coordinate=0.0, open_time=0.0, close_time=100.0)

# customers dict: {id: Customer}
customers = {
    1: Customer(1, x_coordinate=10.0, y_coordinate=0.0,  demand=5.0, earliest_time=0.0,  latest_time=50.0, service_time=1.0),
    2: Customer(2, x_coordinate=0.0,  y_coordinate=10.0, demand=5.0, earliest_time=0.0,  latest_time=50.0, service_time=1.0),
    3: Customer(3, x_coordinate=10.0, y_coordinate=10.0, demand=5.0, earliest_time=0.0,  latest_time=50.0, service_time=1.0),
    4: Customer(4, x_coordinate=20.0, y_coordinate=0.0,  demand=5.0, earliest_time=100.0, latest_time=120.0, service_time=1.0),  # tight window — will be late
}

params = GAParameters()
params.w1 = 1.0
params.w2 = 10.0
params.w3 = 5.0

evaluator = FitnessEvaluator(customers=customers, depot=depot, parameters=params)
print(evaluator)


# ── Test distance calculation ─────────────────────────────────────────────────
separator("TEST: distance calculation")

# depot (0,0) to customer 1 (10,0) should be exactly 10.0
d = evaluator.calc_distance((0, 0), (10, 0))
print(f"  Distance (0,0) to (10,0): {d:.4f}  (expected 10.0)")
assert abs(d - 10.0) < 1e-6, f"Distance wrong: {d}"

# depot to customer 3 (10,10) — diagonal = sqrt(200) ≈ 14.142
d2 = evaluator.calc_distance((0, 0), (10, 10))
print(f"  Distance (0,0) to (10,10): {d2:.4f}  (expected ~14.142)")
assert abs(d2 - 14.142) < 0.001, f"Distance wrong: {d2}"

print("  DISTANCE TESTS PASSED")


# ── Test total route distance ─────────────────────────────────────────────────
separator("TEST: total route distance")

# Route [1, 3]: depot(0,0) -> C1(10,0) -> C3(10,10) -> depot(0,0)
# = 10 + 10 + sqrt(200) = 10 + 10 + 14.142 = 34.142
route_a = [1, 3]
dist_a  = evaluator.calculate_route_distance(route_a)
expected_a = 10.0 + 10.0 + 14.142
print(f"  Route [1,3] total distance: {dist_a:.4f}  (expected ~{expected_a:.3f})")
assert abs(dist_a - expected_a) < 0.01, f"Route distance wrong: {dist_a}"

print("  TOTAL ROUTE DISTANCE TESTS PASSED")


# ── Test timeline simulation ───────────────────────────────────────────────────
separator("TEST: route timeline simulation")

# Route [1, 2]: depot(0,0) -> C1(10,0) -> C2(0,10) -> depot
# Both have wide time windows — no violations expected
route_b  = [1, 2]
timeline = evaluator.simulate_route_timeline(route_b)
print("  Timeline for route [1, 2]:")
for entry in timeline:
    print(f"    Customer {entry[0]}: arrival={entry[1]:.3f}, status={entry[2]}, deviation={entry[3]:.3f}")

# Customer 4 has earliest_time=100, latest_time=120
# Arrival at customer 4 from depot(0,0) = distance to (20,0) = 20.0 — well before window
# expect "early (waited)"
route_c  = [4]
timeline2 = evaluator.simulate_route_timeline(route_c)
print(f"\n  Timeline for route [4] (tight window [100,120]):")
for entry in timeline2:
    print(f"    Customer {entry[0]}: arrival={entry[1]:.3f}, status={entry[2]}, deviation={entry[3]:.3f}")
assert timeline2[0][2] == "early (waited)", f"Expected early wait, got: {timeline2[0][2]}"

print("\n  TIMELINE TESTS PASSED")


# ── Test time window violation ────────────────────────────────────────────────
separator("TEST: time window violation")

# Route [1, 2, 3] — all windows [0, 50] — no violations
route_no_viol = [1, 2, 3]
viol = evaluator.calc_time_window_violation(route_no_viol)
print(f"  Route [1,2,3] violation (wide windows): {viol:.4f}  (expected 0.0)")
assert viol == 0.0, f"Expected 0 violation, got {viol}"

# Manually create a customer with a window that will be violated
customers_test = {
    5: Customer(5, x_coordinate=50.0, y_coordinate=50.0, demand=1.0, earliest_time=0.0, latest_time=1.0, service_time=0.0)
}
evaluator_test = FitnessEvaluator(customers=customers_test, depot=depot, parameters=params)
# distance to (50,50) from (0,0) = sqrt(5000) ≈ 70.71 — arrives after latest=1.0
route_viol = [5]
viol2 = evaluator_test.calc_time_window_violation(route_viol)
print(f"  Route [5] violation (window [0,1], arrives ~70.7): {viol2:.4f}  (expected ~69.71)")
assert viol2 > 60.0, f"Expected large violation, got {viol2}"

print("\n  TIME WINDOW VIOLATION TESTS PASSED")


# ── Test full evaluate() ──────────────────────────────────────────────────────
separator("TEST: full chromosome evaluate()")

# Two vehicle routes: vehicle 1 serves [1,2], vehicle 2 serves [3]
# genes: [1, 2, 0, 3]
chrom = Chromosome(genes=[1, 2, 0, 3], island_id=0)
print(f"  Before evaluation: {chrom}")

fitness = evaluator.evaluate(chrom)
print(f"  After evaluation:  fitness={fitness:.4f}, evaluated={chrom.is_evaluated()}")
assert chrom.is_evaluated(), "Chromosome should be marked as evaluated"
assert chrom.get_fitness() == fitness, "Chromosome fitness should match returned value"
assert fitness > 0, "Fitness should be positive"

print(f"  Evaluation count: {evaluator.get_evaluation_count()}  (expected 1)")
assert evaluator.get_evaluation_count() == 1

# Already evaluated — evaluate_population should skip it
evaluator.evaluate_population([chrom])
print(f"  After evaluate_population on already-evaluated chrom: count={evaluator.get_evaluation_count()}  (expected still 1)")
assert evaluator.get_evaluation_count() == 1

print("\n  FULL EVALUATE TESTS PASSED")


# ── Test 2-opt ────────────────────────────────────────────────────────────────
separator("TEST: 2-opt local search")

# Route [1, 3, 2] vs [1, 2, 3] — test that 2-opt finds the shorter arrangement
route_before = [1, 3, 2]
dist_before  = evaluator.calculate_route_distance(route_before)
route_after  = evaluator.apply_two_opt(route_before)
dist_after   = evaluator.calculate_route_distance(route_after)

print(f"  Before 2-opt: route={route_before}, distance={dist_before:.4f}")
print(f"  After  2-opt: route={route_after},  distance={dist_after:.4f}")
assert dist_after <= dist_before + 1e-6, f"2-opt made route worse: {dist_after} > {dist_before}"

print("\n  2-OPT TESTS PASSED")


# ── Summary ───────────────────────────────────────────────────────────────────
separator("ALL DAY 2 TESTS PASSED")
print()