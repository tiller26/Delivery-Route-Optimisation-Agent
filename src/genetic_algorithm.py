import random
import math
from src.chromosome import Chromosome
from src.ga_parameters import (
    CrossoverStrategy, MutationStrategy,
    ElitismStrategy, InitStrategy
)


class GeneticAlgorithm:
    def __init__(self, island_id, params, evaluator):
        self.island_id = island_id
        self.params = params
        self.evaluator = evaluator

        # list of the chromosome objects
        self.population = []
        # best solution taht we find on the Island.
        self.best_solution = None
        # The best get_fitness() for a generation
        self.fitness_history = []
        self.generation = 0

        self.random_number_generator = random.Random()  

    #######################################################
    # Initialization Methods
    #######################################################

    def initialize(self, customers, depot, seed):
        # Initialize the random seed for reproducibility
        self.random_number_generator.seed(seed)
        random.seed(seed)

        customer_ids = list(customers.keys())
        pop_size = self.params.population_size
        strategy = self.params.population_init_strategy
        self.population = []

        if strategy == InitStrategy.SAVINGS_NN_RANDOM:
            n_random = pop_size // 4
            n_savings = pop_size // 4
            n_nn = pop_size - n_random - n_savings

            for _ in range(n_random):
                c = self.build_random_solution(customer_ids)
                self.population.append(c)

            for _ in range(n_savings):
                c = self.build_savings_solution(customers, depot)
                self.population.append(c)

            for _ in range(n_nn):
                c = self.build_nearest_neighbour_solution(customers, depot)
                self.population.append(c)

        elif strategy == InitStrategy.NN_RANDOM:
            n_nn = pop_size // 2
            n_random = pop_size - n_nn
            for _ in range(n_nn):
                self.population.append(self.build_nearest_neighbour_solution(customers, depot))
            for _ in range(n_random):
                self.population.append(self.build_random_solution(customer_ids))


        else: # Random only 
            for _ in range(pop_size):
                self.population.append(self.build_random_solution(customer_ids))

        
        # Assign island id to all chromosomes
        for chrom in self.population:
            chrom.set_island_id(self.island_id)

        # evaluate the initial population
        self.evaluator.evaluate_population(self.population)
        self.update_best()

        # Safe extraction check for strategy text name
        strategy_name = strategy.name if hasattr(strategy, 'name') else str(strategy)
        print(f" [Island {self.island_id}] Initialized population with strategy {strategy_name}. Best initial fitness: {self.best_solution.get_fitness():.4f}")

    #####################################################
    # Main Evolution Methods
    #####################################################

    def run_generation(self):
        pop_size = len(self.population)
        offspring = []

        # generate offspring until we have a full new population
        while len(offspring) < pop_size:
            parent1 = self.tournament_selection(k=5)
            parent2 = self.tournament_selection(k=5)

            # crossover
            if self.random_number_generator.random() < self.params.crossover_rate:
                if self.params.crossover_strategy == CrossoverStrategy.OX:
                    child1, child2 = self.order_crossover(parent1, parent2)
                else:
                    child1, child2 = self.pmx_crossover(parent1, parent2)
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()


            # mutation
            for child in [child1, child2]:
                if self.random_number_generator.random() < self.params.mutation_rate:
                    strategy = self.params.mutation_strategy
                    if strategy == MutationStrategy.SWAP:
                        self.mutate_swap(child)
                    elif strategy == MutationStrategy.INVERSION:
                        self.mutate_inversion(child)
                    else:
                        self.mutate_swap(child)
                        self.mutate_inversion(child)
                
                # 2-opt local search 
                if self.random_number_generator.random() < self.params.local_search_rate:
                    self.apply_two_opt_to_chromosome(child)

                child.set_island_id(self.island_id)
                offspring.append(child)

        # trim offspring to exact population size
        offspring = offspring[:pop_size]

        # evaluate all offspring
        self.evaluator.evaluate_population(offspring)

        # elitist replacement - merge elites from old population with offspring
        self.population = self.apply_elitism(offspring)

        self.generation += 1
        self.update_best()
        self.fitness_history.append(self.best_solution.get_fitness())

    ####################################################
    # Selection Methods
    ####################################################

    def tournament_selection(self, k=5):
        # Tournament selection samples k individuals and returns the best one

        candidates = self.random_number_generator.sample(self.population, min(k, len(self.population)))
        
        # Assume the first candidate is the best one to start
        best_candidate = candidates[0]
        best_fitness = best_candidate.get_fitness()

        # Compare against the rest of the candidates
        for candidate in candidates[1:]:
            current_fitness = candidate.get_fitness()
            if current_fitness < best_fitness:  # Assuming lower get_fitness() is better
                best_candidate = candidate
                best_fitness = current_fitness

        return best_candidate.copy()

    ####################################################
    # Crossover Methods
    ####################################################

    def order_crossover(self, parent1, parent2):
        # Order Crossover (OX) implementation 
        # Works on the customer-only part of the gene sequence to preserve valid route sequences.
        """
        Steps:
        1. Extract customer sequences from both parents (ignoring depot separators).
        2. Copy a random segment from parent1 to child1 and from parent2 to child2.
        3. Fill remaining positions in the order they appear in p2
        4. Reinsert depot markers proportionally
        """

        g1 = parent1.get_genes()
        g2 = parent2.get_genes()

        # separate customers from depot markers
        c1 = [gene for gene in g1 if gene != 0]
        c2 = [gene for gene in g2 if gene != 0]
        markers1 = g1.count(0)
        n = len(c1)

        if n == 0:
            return parent1.copy(), parent2.copy()

        # choose crossover segement
        start = self.random_number_generator.randint(0, n - 1)
        end = self.random_number_generator.randint(start, n - 1)

        # build child1: segment from p1 + remaining from p2
        child1_seq = self.ox_fill(c1, c2, start, end)
        child2_seq = self.ox_fill(c2, c1, start, end)

        # reinsert depot markers
        child1_genes = self.insert_depot_markers(child1_seq, markers1)
        child2_genes = self.insert_depot_markers(child2_seq, markers1)

        child1 = Chromosome(child1_genes, self.island_id)
        child2 = Chromosome(child2_genes, self.island_id)

        return child1, child2

    def ox_fill(self, base, filler, start, end):
        n = len(base)
        child = [None] * n

        # copy segment from base
        child[start:end+1] = base[start:end+1]
        segment_set = set(base[start:end+1])

        # fill remaining from filler in order
        filler_iter = [x for x in filler if x not in segment_set]
        fill_pos = [i for i in range(n) if child[i] is None]

        for pos, val in zip(fill_pos, filler_iter):
            child[pos] = val

        return child

    def pmx_crossover(self, parent1, parent2):
        # Partially Mapped Crossover (PMX) implementation
        # Similar to OX but with a mapping step to ensure valid offspring.
        g1 = parent1.get_genes()
        g2 = parent2.get_genes()

        c1 = [gene for gene in g1 if gene != 0]
        c2 = [gene for gene in g2 if gene != 0]
        markers1 = g1.count(0)
        n = len(c1)

        if n == 0:
            return parent1.copy(), parent2.copy()

        start = self.random_number_generator.randint(0, n - 1)
        end = self.random_number_generator.randint(start, n - 1)

        child1_seq = self.pmx_fill(c1, c2, start, end)
        child2_seq = self.pmx_fill(c2, c1, start, end)

        child1_genes = self.insert_depot_markers(child1_seq, markers1)
        child2_genes = self.insert_depot_markers(child2_seq, markers1)

        child1 = Chromosome(child1_genes, self.island_id)
        child2 = Chromosome(child2_genes, self.island_id)

        return child1, child2

    
    def pmx_fill(self, base, other, start, end):
        n = len(base)
        child = [None] * n

        # copy segment from base
        child[start:end+1] = base[start:end+1]

        # build mapping from other to base for the segment
        mapping = {}
        for i in range(start, end + 1):
            mapping[other[i]] = base[i]
            
        # fill remaining from other with mapping
        for i in range(n):
            if child[i] is None:
                val = other[i]
                while val in mapping:
                    val = mapping[val]
                child[i] = val

        return child

    def insert_depot_markers(self, customer_seq, num_markers):
        # This distributes customers across vehicles after crossover.

        if num_markers == 0 or len(customer_seq) == 0:
            return list(customer_seq)

        seq = list(customer_seq)
        result = []

        # split customer sequence into num_markers + 1 chunks
        n = len(seq)
        splits = sorted(self.random_number_generator.sample(range(1, n), min(num_markers, n - 1)))

        prev = 0
        for split in splits:
            result.extend(seq[prev:split])
            result.append(0)  # depot marker
            prev = split
        result.extend(seq[prev:])

        return result

    ###################################################
    # Mutation Operators
    ###################################################

    def mutate_swap(self, chromosome):
        # Swap mutation: exchange two randomly chosen customer positions.

        genes = chromosome.get_genes()
        customer_pos = [i for i, g in enumerate(genes) if g != 0]

        if len(customer_pos) < 2:
            return

        i,j = self.random_number_generator.sample(customer_pos, 2)
        genes[i], genes[j] = genes[j], genes[i]

        # write back
        for index, val in enumerate(genes):
            chromosome.set_gene(index, val)

    def mutate_inversion(self, chromosome):
        # Inversion mutation: reverse a randomly chosen sub-sequence.
        genes = chromosome.get_genes()
        customer_pos = [i for i, g in enumerate(genes) if g != 0]

        if len(customer_pos) < 2:
            return

        # choose two positions in the customer index list
        idx_i = self.random_number_generator.randint(0, len(customer_pos) - 2)
        idx_j = self.random_number_generator.randint(idx_i + 1, len(customer_pos) - 1)

        pos_i = customer_pos[idx_i]
        pos_j = customer_pos[idx_j]

        # reverse the sub-sequence between pos_i and pos_j
        genes[pos_i:pos_j + 1] = genes[pos_i:pos_j + 1][::-1]

        for idx, val in enumerate(genes):
            chromosome.set_gene(idx, val)


    ###################################################
    # 2-OPT local search 
    ###################################################

    def apply_two_opt_to_chromosome(self, chromosome):
        # Applies 2-opt local search to each vehicle sub-route in the chromosome. Updates the chromosome in place

        routes = chromosome.decode_routes(None)
        improved_routes = []

        for route in routes:
            improved = self.evaluator.apply_two_opt(route)
            improved_routes.append(improved)

        # rebuild gene sequence from improved routes
        new_genes = []
        for i, route in enumerate(improved_routes):
            new_genes.extend(route)
            if i < len(improved_routes) - 1 :
                new_genes.append(0) # depot marker between routes

        # update chromosome genes
        if len(new_genes) == chromosome.get_length():
            for idx, val in enumerate(new_genes):
                chromosome.genes[idx] = val
            chromosome.reset_evaluated()

    ####################################################
    # Elitism
    ####################################################

    def apply_elitism(self, offspring):
        # Preserves the top elite chromosomes from the current population by replacing the worst offsppring with them.

        strategy = self.params.elitism_strategy
        value = self.params.elitism_value
        pop_size = len(self.population)

        if strategy == ElitismStrategy.TOP_N:
            n_elites = int(value)
        else: # TOP_PERCENT
            n_elites = max(1, int(pop_size * value))

        # sort current population - best first
        sorted_pop = sorted(self.population, key=lambda c: c.get_fitness())
        elites = [c.copy() for c in sorted_pop[:n_elites]]

        # sort offspring - worst last
        sorted_offspring = sorted(offspring, key=lambda c: c.get_fitness())

        # replace worst offspring with elites
        new_population = sorted_offspring[:pop_size - n_elites] + elites

        return new_population

    ################################################################
    # Convergence
    ################################################################

    def has_converged(self, threshold: float = 1e-6, window: int = 50):
        # Returns True if this island has plateaued - no get_fitness() improvement greater than threshold over the last 'window' generations.
        if len(self.fitness_history) < window:
            return False

        recent  = self.fitness_history[-window:]
        return (max(recent) - min(recent)) < threshold


    ################################################################
    # Migration interface
    ################################################################

    def inject_migrants(self, migrants):
        # Receives migrants from another island(called by IslandManager). 
        # Replaces the weakest individuals in this island's population.

        if not migrants:
            return

        # sort population - worst last
        self.population.sort(key=lambda c: c.get_fitness())

        # replace worst with incoming migrants 
        n = min(len(migrants), len(self.population))
        for i in range(n):
            migrant = migrants[i].copy()
            migrant.set_island_id(self.island_id)
            self.population[-(i + 1)] = migrant

        self.update_best()

    #########################################################
    # Population Initialisation helpers
    #########################################################

    def build_random_solution(self, customer_ids):
        # Generates a random chromosome by shuffling customer IDs and inserting depot markers at random positions.
        ids = list(customer_ids)
        self.random_number_generator.shuffle(ids)

        n_vehicles = max(1, len(ids) // 5)  # rough estimate: 5 customers per vehicle

        n_markers = n_vehicles - 1

        genes = list(ids)
        for _ in range(n_markers):
            pos = self.random_number_generator.randint(1, len(genes) - 1)
            genes.insert(pos, 0)

        return Chromosome(genes, self.island_id)

    

    def build_savings_solution(self, customers, depot):
        # Implement the Clarke-Wright Savings Algorithm to create an initial solution
        depot_loc = depot.get_location()
        ids = list(customers.keys())

        savings = []
        for a in range(len(ids)):
            for b in range(a+1, len(ids)):
                i, j = ids[a], ids[b]
                ci, cj = customers[i], customers[j]
                s = (self.euclidean(depot_loc, ci.get_location()) +
                        self.euclidean(depot_loc, cj.get_location()) -
                        self.euclidean(ci.get_location(), cj.get_location()))
                savings.append((s, i, j))

        savings.sort(reverse=True)
        
        # Tels us which routes visists which customers. {Route x visits customer y}
        routes = {cid: [cid] for cid in ids}
        # This tells what route each customer is in. {Customer x in route y}
        route_of = {cid: cid for cid in ids}

        while savings:
            limit = min(3, len(savings))
            index = self.random_number_generator.randint(0, limit - 1)
            s, i, j = savings.pop(index)

            ri_key = route_of.get(i)
            rj_key = route_of.get(j)

            if ri_key is None or rj_key is None:
                continue 
            # Checks for the same route
            if ri_key == rj_key:
                continue

            rj = routes[rj_key]
            ri = routes[ri_key]

            if ri[-1] == i and rj[0] == j:
                merged = ri + rj
                new_key = ri_key
                routes[new_key] = merged
                del routes[rj_key]
                for cid in rj:
                    route_of[cid] = ri_key

            elif ri[0] == i and rj[-1] == j:
                merged = rj + ri
                new_key = rj_key
                routes[new_key] = merged
                del routes[ri_key]
                for cid in ri:
                    route_of[cid] = rj_key

        route_list = list(routes.values())
        genes = []
        for idx, route in enumerate(route_list):
            genes.extend(route)
            if idx < len(route_list) - 1:
                # This is to separate the routes.
                genes.append(0)


        return Chromosome(genes, self.island_id)

    def build_nearest_neighbour_solution(self, customers, depot):
        unvisited = set(customers.keys())
        routes = []
        depot_loc = depot.get_location()

        k = self.random_number_generator.choice(list(unvisited))
        
        while unvisited:
            route = []
            current_loc = depot_loc
            current_time = 0.0


            while unvisited:
                candidates = []
                for cid in unvisited:
                    c = customers[cid]
                    dist = self.euclidean(current_loc, c.get_location())
                    arrival_time = current_time + dist
                    _, latest_time = c.get_time_window()
                    if arrival_time <= latest_time:
                        candidates.append((dist, cid))

                if not candidates:
                    break

                candidates.sort(key = lambda x: x[0])
                top_k = candidates[:k]

                chosen_dist, chosen_id = self.random_number_generator.choice(top_k)
                
                c = customers[chosen_id]
                earliest, _ = c.get_time_window()
                arrival_time = current_time + chosen_dist
                current_time = max(arrival_time, earliest) + c.get_service_time()
                current_loc = c.get_location()
                route.append(chosen_id)
                unvisited.remove(chosen_id)

            if route:
                routes.append(route)

        genes = []
        for i, route in enumerate(routes):
            genes.extend(route)
            if i < len(routes) - 1:
                genes.append(0)

        return Chromosome(genes, self.island_id)
                    
    ########################################################
    # Accessors
    ########################################################

    def get_best_solution(self):
        return self.best_solution

    def get_fitness_history(self):
        return list(self.fitness_history)

    def get_population(self):
        return list(self.population)

    def get_generation_count(self):
        return self.generation

    def get_island_id(self):
        return self.island_id

    
    #########################################################
    # Private Helpers
    #########################################################

    def update_best(self):
        # Updates the island's best solution if a better one is found

        if not self.population:
            return 

        current_best = min(self.population, key=lambda c: c.get_fitness())
        if(self.best_solution is None) or (current_best.get_fitness() < self.best_solution.get_fitness()):
            self.best_solution = current_best.copy()

    def euclidean(self, loc1, loc2):
        # Euclidean distance between two (x, y) tiples.
        return math.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)

    def __repr__(self):
        best_f = self.best_solution.get_fitness() if self.best_solution else "none"

        return (f"GeneticAlgorithm("
                f"island={self.island_id}, "
                f"pop={len(self.population)}, "
                f"gen={self.generation}, "
                f"best={best_f}"
                )