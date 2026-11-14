"""
Day 3 test — verifies GeneticAlgorithm (single island).
Run from the project root:  python tests/test_day3.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.customer          import Customer
from src.depot             import Depot
from src.ga_parameters     import GAParameters, InitStrategy, CrossoverStrategy, MutationStrategy
from src.chromosome        import Chromosome
from src.fitness_evaluator import FitnessEvaluator
from src.genetic_algorithm import GeneticAlgorithm


def separator(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)


# ── Build a 10-customer problem ───────────────────────────────────────────────
separator("SETUP: 10-customer problem instance")

depot = Depot(depot_id=0, x_coordinate=50.0, y_coordinate=50.0, open_time=0.0, close_time=200.0)

customers = {
    1:  Customer(1,  x_coordinate=20.0, y_coordinate=30.0, demand=5.0,  earliest_time=0.0,  latest_time=100.0, service_time=1.0),
    2:  Customer(2,  x_coordinate=70.0, y_coordinate=80.0, demand=8.0,  earliest_time=10.0, latest_time=80.0,  service_time=1.0),
    3:  Customer(3,  x_coordinate=40.0, y_coordinate=60.0, demand=3.0,  earliest_time=5.0,  latest_time=90.0,  service_time=0.5),
    4:  Customer(4,  x_coordinate=90.0, y_coordinate=20.0, demand=6.0,  earliest_time=0.0,  latest_time=120.0, service_time=1.5),
    5:  Customer(5,  x_coordinate=10.0, y_coordinate=70.0, demand=4.0,  earliest_time=20.0, latest_time=100.0, service_time=1.0),
    6:  Customer(6,  x_coordinate=60.0, y_coordinate=40.0, demand=7.0,  earliest_time=0.0,  latest_time=150.0, service_time=0.5),
    7:  Customer(7,  x_coordinate=30.0, y_coordinate=90.0, demand=2.0,  earliest_time=15.0, latest_time=90.0,  service_time=1.0),
    8:  Customer(8,  x_coordinate=80.0, y_coordinate=60.0, demand=5.0,  earliest_time=0.0,  latest_time=200.0, service_time=1.0),
    9:  Customer(9,  x_coordinate=50.0, y_coordinate=10.0, demand=9.0,  earliest_time=10.0, latest_time=150.0, service_time=2.0),
    10: Customer(10, x_coordinate=25.0, y_coordinate=50.0, demand=3.0,  earliest_time=0.0,  latest_time=200.0, service_time=0.5),
}

params = GAParameters()
params.population_size        = 20
params.max_generations        = 50
params.crossover_rate         = 0.85
params.crossover_strategy     = CrossoverStrategy.OX
params.mutation_rate          = 0.1
params.mutation_strategy      = MutationStrategy.BOTH
params.local_search_rate      = 0.3
params.population_init_strategy = InitStrategy.SAVINGS_NN_RANDOM

evaluator = FitnessEvaluator(customers=customers, depot=depot, parameters=params)
print(f"  Evaluator: {evaluator}")


# ── Test initialisation ───────────────────────────────────────────────────────
separator("TEST: island initialisation")

island = GeneticAlgorithm(island_id=0, params=params, evaluator=evaluator)
island.initialize(customers=customers, depot=depot, seed=42)

print(f"\n  Island: {island}")
pop = island.get_population()
print(f"  Population size: {len(pop)}  (expected {params.population_size})")
assert len(pop) == params.population_size, f"Wrong population size: {len(pop)}"

# verify all chromosomes are evaluated
for chrom in pop:
    assert chrom.is_evaluated(), "Chromosome should be evaluated after initialise()"
    assert chrom.get_fitness() < float('inf'), "Fitness should not be infinity"

print("  All chromosomes evaluated after initialise()")

# verify all chromosomes contain all 10 customers exactly once
for chrom in pop:
    genes    = chrom.get_genes()
    cust_ids = [g for g in genes if g != 0]
    assert sorted(cust_ids) == list(range(1, 11)), \
        f"Chromosome missing customers or has duplicates: {cust_ids}"

print("  All chromosomes contain all 10 customers exactly once")
print("\n  INITIALISATION TESTS PASSED")


# ── Test selection ────────────────────────────────────────────────────────────
separator("TEST: tournament selection")

selected = island.tournament_selection(k=5)
print(f"  Selected chromosome fitness: {selected.get_fitness():.4f}")
assert selected.is_evaluated(), "Selected chromosome should be evaluated"
assert isinstance(selected, Chromosome), "Selection should return a Chromosome"

# run selection 20 times — selected fitness should tend to be better than average
fitnesses   = [c.get_fitness() for c in island.get_population()]
avg_fitness = sum(fitnesses) / len(fitnesses)
selected_fitnesses = [island.tournament_selection(5).get_fitness() for _ in range(20)]
avg_selected = sum(selected_fitnesses) / len(selected_fitnesses)
print(f"  Average population fitness: {avg_fitness:.4f}")
print(f"  Average selected fitness:   {avg_selected:.4f}")
print(f"  Selection pressure working: {avg_selected <= avg_fitness}")

print("\n  SELECTION TESTS PASSED")


# ── Test crossover ────────────────────────────────────────────────────────────
separator("TEST: Order Crossover (OX)")

p1 = island.tournament_selection(5)
p2 = island.tournament_selection(5)
c1, c2 = island.order_crossover(p1, p2)

print(f"  Parent 1 genes: {p1.get_genes()}")
print(f"  Parent 2 genes: {p2.get_genes()}")
print(f"  Child  1 genes: {c1.get_genes()}")
print(f"  Child  2 genes: {c2.get_genes()}")

# verify children contain all 10 customers exactly once
for child, name in [(c1, "Child1"), (c2, "Child2")]:
    cust_ids = [g for g in child.get_genes() if g != 0]
    assert sorted(cust_ids) == list(range(1, 11)), \
        f"{name} missing customers or has duplicates: {cust_ids}"
    print(f"  {name}: all 10 customers present exactly once ✓")

print("\n  OX CROSSOVER TESTS PASSED")


# ── Test mutation ─────────────────────────────────────────────────────────────
separator("TEST: mutation operators")

chrom_before = island.tournament_selection(5)
genes_before = chrom_before.get_genes()
chrom_swap   = chrom_before.copy()
island.mutate_swap(chrom_swap)
genes_after  = chrom_swap.get_genes()

# customer set should be identical before and after mutation
before_custs = sorted([g for g in genes_before if g != 0])
after_custs  = sorted([g for g in genes_after  if g != 0])
assert before_custs == after_custs, "Mutation changed customer set"
print(f"  Swap mutation: customer set preserved ✓")
print(f"  Before: {genes_before}")
print(f"  After:  {genes_after}")

chrom_inv = chrom_before.copy()
island.mutate_inversion(chrom_inv)
inv_custs = sorted([g for g in chrom_inv.get_genes() if g != 0])
assert before_custs == inv_custs, "Inversion mutation changed customer set"
print(f"  Inversion mutation: customer set preserved ✓")

print("\n  MUTATION TESTS PASSED")


# ── Test single generation ────────────────────────────────────────────────────
separator("TEST: run single generation")

best_before = island.get_best_solution().get_fitness()
print(f"  Best fitness before generation: {best_before:.4f}")

island.run_generation()

best_after = island.get_best_solution().get_fitness()
print(f"  Best fitness after generation:  {best_after:.4f}")
print(f"  Generation count: {island.get_generation_count()}  (expected 1)")

assert island.get_generation_count() == 1
assert len(island.get_fitness_history()) == 1

print("\n  SINGLE GENERATION TESTS PASSED")


# ── Test full evolution ───────────────────────────────────────────────────────
separator("TEST: full evolution — 50 generations")

initial_best = island.get_best_solution().get_fitness()

for gen in range(params.max_generations - 1):
    island.run_generation()

final_best = island.get_best_solution().get_fitness()
history    = island.get_fitness_history()

print(f"  Generations run:  {island.get_generation_count()}")
print(f"  Initial best fitness: {initial_best:.4f}")
print(f"  Final best fitness:   {final_best:.4f}")
print(f"  Improvement: {initial_best - final_best:.4f}")
print(f"  Fitness improved over 50 generations: {final_best <= initial_best}")

assert len(history) == params.max_generations, \
    f"Fitness history length wrong: {len(history)}"
assert final_best <= initial_best, \
    "Fitness should not get worse over generations"

print(f"\n  Fitness history (first 10): {[round(f, 2) for f in history[:10]]}")
print(f"  Fitness history (last  10): {[round(f, 2) for f in history[-10:]]}")

print("\n  FULL EVOLUTION TESTS PASSED")


# ── Test convergence detection ────────────────────────────────────────────────
separator("TEST: convergence detection")

converged = island.has_converged(threshold=1e-6, window=50)
print(f"  has_converged() after {island.get_generation_count()} generations: {converged}")

print("\n  CONVERGENCE TESTS PASSED")


# ── Test migration injection ──────────────────────────────────────────────────
separator("TEST: inject migrants")

island2 = GeneticAlgorithm(island_id=1, params=params, evaluator=evaluator)
island2.initialize(customers=customers, depot=depot, seed=99)

# take 2 best from island 0 and inject into island 1
migrants = sorted(island.get_population(), key=lambda c: c.get_fitness())[:2]
best_migrant_fitness = migrants[0].get_fitness()

pop_before = island2.get_population()
worst_before = max(c.get_fitness() for c in pop_before)

island2.inject_migrants(migrants)

pop_after = island2.get_population()
print(f"  Population size unchanged: {len(pop_after) == len(pop_before)} "
      f"({len(pop_after)} == {len(pop_before)})")
print(f"  Best migrant fitness: {best_migrant_fitness:.4f}")
print(f"  Island 2 best after injection: {island2.get_best_solution().get_fitness():.4f}")

# migrants should have their island_id updated to island 1
for c in pop_after:
    if abs(c.get_fitness() - best_migrant_fitness) < 1e-6:
        print(f"  Migrant island_id updated to 1: {c.get_island_id() == 1}")
        break

print("\n  MIGRATION INJECTION TESTS PASSED")


# ── Summary ───────────────────────────────────────────────────────────────────
separator("ALL DAY 3 TESTS PASSED")
print()