"""
VRPTW Island GA — Main Entry Point
====================================
Run the full system from here.

Usage:
    python main.py                        # runs with default settings on C101_small
    python main.py data/C101_small.txt    # specify a different problem file

This file wires together all components in the correct order:
    1. Load the problem instance
    2. Configure the GA parameters
    3. Solve using the Island GA
    4. Display the solution
    5. Export results to output/
"""

import sys
import os

# make sure src is importable regardless of where you run from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (
    VRPTWSolver,
    GAParameters,
    CrossoverStrategy,
    MutationStrategy,
    TopologyType,
    MigrationStrategy,
    InitStrategy
)


def build_parameters():
    """
    Builds and returns the GA parameter configuration.
    Edit these values to change the system's behaviour
    and run your parameter study.
    """
    params = GAParameters()

    # ── Standard GA parameters ────────────────────────────────────────────
    params.population_size    = 50
    params.max_generations    = 200

    params.crossover_rate     = 0.85
    params.crossover_strategy = CrossoverStrategy.OX

    params.mutation_rate      = 0.1
    params.mutation_strategy  = MutationStrategy.BOTH

    params.local_search_rate  = 0.3

    params.population_init_strategy = InitStrategy.SAVINGS_NN_RANDOM

    # ── Island parameters ─────────────────────────────────────────────────
    params.num_islands        = 4
    params.island_topology    = TopologyType.RING

    params.migration_rate     = 0.8
    params.migration_interval = 10
    params.migration_size     = 2
    params.migration_strategy = MigrationStrategy.BEST

    # ── Fitness weights ───────────────────────────────────────────────────
    # w2 is high — we prioritise time window compliance over distance
    params.w1 = 1.0
    params.w2 = 10.0
    params.w3 = 5.0

    return params


def main():
    print("=" * 60)
    print("  VRPTW ISLAND GA — INTELLIGENT DELIVERY OPTIMISER")
    print("=" * 60)

    # ── determine problem file ─────────────────────────────────────────────
    if len(sys.argv) > 1:
        problem_file = sys.argv[1]
    else:
        problem_file = "data/C101_small.txt"

    # ── build solver ──────────────────────────────────────────────────────
    solver = VRPTWSolver()

    # ── step 1: load problem ──────────────────────────────────────────────
    solver.load_problem(problem_file)

    # ── step 2: configure parameters ─────────────────────────────────────
    params = build_parameters()
    solver.configure_parameters(params)

    # ── step 3: solve ─────────────────────────────────────────────────────
    solver.solve()

    # ── step 4: display solution ──────────────────────────────────────────
    solver.display_solution()

    # ── step 5: export results ────────────────────────────────────────────
    os.makedirs("output", exist_ok=True)

    solver.export_solution("output/solution.csv")
    solver.export_fitness_history("output/fitness_history.csv")

    print("\n  Results saved to output/")
    print("  solution.csv        — full route plan per vehicle")
    print("  fitness_history.csv — convergence data per generation")
    print("\n  Run visualisation next:")
    print("  python visualise.py")
    print()


if __name__ == "__main__":
    main()