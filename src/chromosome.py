from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .depot import Depot


# The class that represents a single candidate solution.
class Chromosome:
    # Variables
    fitness_score = None
    evaluated = False
    generation_created = None

    def __init__(self, genes, island_id):
        self.genes = genes
        self.island_id = island_id

    # Gene access
    def get_genes(self):
        return self.genes

    def set_gene(self, index, value):
        self.genes[index] = value
        self.evaluated = False   # Need to check this.

    def get_length(self):
        return len(self.genes)

    def get_fitness(self):
        return self.fitness_score

    def set_fitness(self, score):
        self.fitness_score = score
        self.evaluated = True

    def is_evaluated(self):
        return self.evaluated

    def reset_evaluated(self):
        self.fitness_score = None
        self.evaluated = False

    def get_island_id(self):
        return self.island_id

    def set_island_id(self, island_id):
        self.island_id = island_id

    def decode_routes(self, depot: Depot):
        # This method decodes the chromosome's genes into a list of routes.
        # Each route is a list of customer IDs, starting and ending with the depot.
        routes = []
        current_route = []  # Start EMPTY
        for gene in self.genes:
            if gene == 0:  # 0 is a separator for routes
                if current_route:
                    routes.append(current_route)
                    current_route = []  # Start a new route
            else:
                current_route.append(gene)  # Add customer ID to the current route
        if current_route:  # If there's an unfinished route, close it
            routes.append(current_route)
        return routes

    def get_route_count(self, depot: Depot):
        # Pass the required depot object directly into the decoder method
        return len(self.decode_routes(depot))

    def copy(self):
        new_chrom = Chromosome(self.genes.copy(), self.island_id)
        
        # Transfer the evaluation data so it doesn't get amnesia
        new_chrom.fitness_score = self.fitness_score
        new_chrom.evaluated = self.evaluated
        new_chrom.generation_created = self.generation_created 
        
        return new_chrom

    def to_String(self):
        return f"Chromosome(genes={self.genes}, fitness_score={self.fitness_score}, evaluated={self.evaluated}, generation_created={self.generation_created}, island_id={self.island_id})"