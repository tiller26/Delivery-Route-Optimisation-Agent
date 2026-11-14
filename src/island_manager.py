import random
from src.genetic_algorithm import GeneticAlgorithm
from src.chromosome import Chromosome
from src.ga_parameters import TopologyType, MigrationStrategy


class IslandManager:
    """
    Orchestrates the Island-Based Genetic Algorithm.

    Creates and manages N independent GeneticAlgorithm island instances.
    Runs the evolutionary loop across all islands simultaneously.
    Executes periodic migration between connected islands.
    Tracks the global best solution across all islands.

    Called by VRPTWSolver — VRPTWSolver.solve() calls
    island_manager.run_evolution() which returns the global best Chromosome.
    """

    def __init__(self, params, evaluator):
        """
        Args:
            params:    GAParameters object
            evaluator: FitnessEvaluator shared across all islands
        """
        self.params            = params
        self.evaluator         = evaluator
        self.islands           = []          # list of GeneticAlgorithm instances
        self.topology_graph    = {}          # {island_id: [connected_island_ids]}
        self.global_best       = None        # best Chromosome across all islands
        self.global_history    = []          # best fitness per generation (global)
        self.migration_log     = []          # log of all migration events
        self.current_generation = 0

    # ══════════════════════════════════════════════════════════════════════════
    # INITIALISATION
    # ══════════════════════════════════════════════════════════════════════════

    def initialise_all_islands(self, customers: dict, depot):
        """
        Creates N island instances and seeds each one with a
        different random seed for maximum initial diversity.

        Also builds the island connectivity graph from the
        configured topology.

        Args:
            customers: dict {customer_id: Customer}
            depot:     Depot object
        """
        self.depot = depot
        self.islands = []
        n = self.params.num_islands

        print(f"\n[IslandManager] Initialising {n} islands "
              f"with topology: {self.params.island_topology.value}")

        for i in range(n):
            island = GeneticAlgorithm(
                island_id = i,
                params    = self.params,
                evaluator = self.evaluator
            )
            # each island gets a different seed — critical for diversity
            island.initialize(
                customers = customers,
                depot     = depot,
                seed      = (i + 1) * 137  # prime-based seeds for spread
            )
            self.islands.append(island)

        self.topology_graph = self.build_topology_graph(n)
        self.updateglobal_best()

        print(f"[IslandManager] All islands ready. "
              f"Global best fitness: {self.global_best.get_fitness():.4f}\n")

    # ══════════════════════════════════════════════════════════════════════════
    # TOPOLOGY
    # ══════════════════════════════════════════════════════════════════════════

    def build_topology_graph(self, n: int):
        """
        Builds the island connectivity graph based on the configured topology.

        Returns:
            dict {island_id: [list of connected island_ids]}

        RING:            each island connected to its two neighbours
        FULLY_CONNECTED: every island connected to every other
        STAR:            island 0 is the hub, all others connect only to hub
        RANDOM:          each pair connected with 50% probability (guaranteed connected)
        """
        topology = self.params.island_topology
        graph    = {i: [] for i in range(n)}

        if topology == TopologyType.RING:
            for i in range(n):
                left  = (i - 1) % n
                right = (i + 1) % n
                graph[i] = [left, right]

        elif topology == TopologyType.FULLY_CONNECTED:
            for i in range(n):
                graph[i] = [j for j in range(n) if j != i]

        elif topology == TopologyType.STAR:
            # island 0 is the hub
            for i in range(1, n):
                graph[0].append(i)
                graph[i].append(0)

        elif topology == TopologyType.RANDOM:
            # connect randomly but ensure every island has at least one connection
            rng = random.Random(self.params.num_islands * 7)
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.5:
                        graph[i].append(j)
                        graph[j].append(i)
            # guarantee connectivity — any island with no connections gets ring neighbours
            for i in range(n):
                if not graph[i]:
                    graph[i].append((i + 1) % n)
                    graph[(i + 1) % n].append(i)

        print(f"[IslandManager] Topology graph: {graph}")
        return graph

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN EVOLUTION LOOP
    # ══════════════════════════════════════════════════════════════════════════

    def run_evolution(self):
        """
        Runs the complete Island GA evolutionary process.

        Each generation:
            1. All islands run one generation independently
            2. Migration executes if due
            3. Global best is updated
            4. Convergence is checked

        Returns:
            The global best Chromosome found across all islands.
        """
        max_gen  = self.params.max_generations
        print(f"[IslandManager] Starting evolution — max {max_gen} generations\n")

        for gen in range(max_gen):
            self.current_generation = gen

            # ── Step 1: all islands evolve one generation ─────────────────────
            for island in self.islands:
                island.run_generation()

            # ── Step 2: migration (if due) ────────────────────────────────────
            if self.is_migration_due(gen):
                self.execute_migration(gen)

            # ── Step 3: update global best ────────────────────────────────────
            self.updateglobal_best()
            self.global_history.append(self.global_best.get_fitness())

            # ── Step 4: progress report every 50 generations ─────────────────
            if (gen + 1) % 50 == 0:
                island_bests = [
                    f"I{i}:{self.islands[i].get_best_solution().get_fitness():.2f}"
                    for i in range(len(self.islands))
                ]
                print(f"  Gen {gen+1:4d} | Global best: {self.global_best.get_fitness():.4f} "
                      f"| {' | '.join(island_bests)}")

            # ── Step 5: check global convergence ─────────────────────────────
            if self.check_global_convergence():
                print(f"\n[IslandManager] Converged at generation {gen + 1}. "
                      f"Global best: {self.global_best.get_fitness():.4f}")
                break

        print(f"\n[IslandManager] Evolution complete.")
        print(f"  Total generations:  {self.current_generation + 1}")
        print(f"  Migration events:   {len(self.migration_log)}")
        print(f"  Global best fitness:{self.global_best.get_fitness():.4f}")
        print(f"  Routes in solution: {self.global_best.get_route_count(self.depot)}")

        return self.global_best

    # ══════════════════════════════════════════════════════════════════════════
    # MIGRATION
    # ══════════════════════════════════════════════════════════════════════════

    def is_migration_due(self, gen: int):
        """
        Returns True if a migration event should be attempted this generation.
        Two conditions must both be true:
            1. generation is a multiple of migration_interval
            2. random check passes against migration_rate
        """
        if (gen + 1) % self.params.migration_interval != 0:
            return False
        return random.random() < self.params.migration_rate

    def execute_migration(self, gen: int):
        """
        Transfers individuals between all connected island pairs
        according to the configured topology and migration strategy.

        For each connected pair (A, B):
            - Select migrants from A → inject into B
            - Select migrants from B → inject into A
        """
        migrated_pairs = set()

        for island_id, neighbours in self.topology_graph.items():
            for neighbour_id in neighbours:

                # avoid processing the same pair twice
                pair = tuple(sorted([island_id, neighbour_id]))
                if pair in migrated_pairs:
                    continue
                migrated_pairs.add(pair)

                island_a = self.islands[island_id]
                island_b = self.islands[neighbour_id]

                # select migrants from A and inject into B
                migrants_a = self.select_migrants(island_a)
                island_b.inject_migrants(migrants_a)

                # select migrants from B and inject into A
                migrants_b = self.select_migrants(island_b)
                island_a.inject_migrants(migrants_b)

                # log the event
                self.migration_log.append({
                    'generation'   : gen + 1,
                    'from_island'  : island_id,
                    'to_island'    : neighbour_id,
                    'num_migrants' : len(migrants_a),
                    'best_migrant_fitness': migrants_a[0].get_fitness() if migrants_a else None
                })

    def select_migrants(self, island: GeneticAlgorithm):
        """
        Selects individuals from an island to migrate based on
        the configured migration strategy.

        BEST:   top migrationSize individuals by fitness
        RANDOM: random migrationSize individuals
        MIXED:  half best, half random
        """
        population   = island.get_population()
        n            = self.params.migration_size
        strategy     = self.params.migration_strategy

        if not population:
            return []

        sorted_pop = sorted(population, key=lambda c: c.get_fitness())

        if strategy == MigrationStrategy.BEST:
            selected = sorted_pop[:n]

        elif strategy == MigrationStrategy.RANDOM:
            selected = random.sample(population, min(n, len(population)))

        else:  # MIXED — half best, half random
            half     = max(1, n // 2)
            best     = sorted_pop[:half]
            rest     = sorted_pop[half:]
            random_  = random.sample(rest, min(half, len(rest))) if rest else []
            selected = best + random_

        # return copies so island populations stay independent
        return [c.copy() for c in selected]

    # ══════════════════════════════════════════════════════════════════════════
    # CONVERGENCE
    # ══════════════════════════════════════════════════════════════════════════

    def check_global_convergence(self, threshold: float = 1e-6, window: int = 50):
        """
        Returns True only when the global best fitness has not
        improved by more than threshold over the last window generations.

        Note: a single island may plateau while others improve.
        Termination only fires on global plateau — migration may
        restart a converged island before this point.
        """
        if len(self.global_history) < window:
            return False
        recent = self.global_history[-window:]
        return (max(recent) - min(recent)) < threshold

    # ══════════════════════════════════════════════════════════════════════════
    # GLOBAL BEST TRACKING
    # ══════════════════════════════════════════════════════════════════════════

    def updateglobal_best(self):
        """
        Scans all islands and updates the global best solution
        if any island has found a better one.
        """
        for island in self.islands:
            island_best = island.get_best_solution()
            if island_best is None:
                continue
            if (self.global_best is None or
                    island_best.get_fitness() < self.global_best.get_fitness()):
                self.global_best = island_best.copy()

    # ══════════════════════════════════════════════════════════════════════════
    # WARM START (used by re-optimisation)
    # ══════════════════════════════════════════════════════════════════════════

    def warm_start(self, customers: dict, depot, seed_chromosome: Chromosome):
        """
        Re-initialises all islands using a warm-start strategy.
        The current global best solution is injected into each
        island as one seed chromosome — this accelerates convergence
        when re-optimising after an environmental change.

        Called by VRPTWSolver during UC-14 (Trigger Re-optimisation).
        """
        print(f"\n[IslandManager] Warm-starting {len(self.islands)} islands "
              f"from seed fitness: {seed_chromosome.get_fitness():.4f}")

        for island in self.islands:
            island.initialise(customers=customers, depot=depot,
                              seed=(island.get_island_id() + 1) * 137)
            # inject the warm-start chromosome — replaces the worst individual
            island.inject_migrants([seed_chromosome.copy()])

        self.updateglobal_best()
        self.global_history = []
        self.migration_log  = []
        self.current_generation = 0

    # ══════════════════════════════════════════════════════════════════════════
    # ACCESSORS
    # ══════════════════════════════════════════════════════════════════════════

    def get_global_best_solution(self):
        return self.global_best

    def get_global_fitness_history(self):
        return list(self.global_history)

    def get_migration_log(self):
        return list(self.migration_log)

    def get_island(self, island_id: int):
        return self.islands[island_id]

    def get_all_islands(self):
        return list(self.islands)

    def __repr__(self):
        best_f = self.global_best.get_fitness() if self.global_best else "none"
        return (
            f"IslandManager("
            f"islands={len(self.islands)}, "
            f"topology={self.params.island_topology.value}, "
            f"generation={self.current_generation}, "
            f"global_best={best_f}, "
            f"migrations={len(self.migration_log)})"
        )