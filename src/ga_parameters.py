from enum import Enum

#############################
# Enumurations
#############################

class CrossoverStrategy(Enum):
    OX = "OX"   # Order Crossover
    PMX = "PMX" # Partially Mapped Crossover
    BOTH = "BOTH"

class MutationStrategy(Enum):
    SWAP = "SWAP"
    INVERSION = "INVERSION"
    BOTH = "BOTH"

class ElitismStrategy(Enum):
    TOP_N = "TOP_N"               # preserve top N individuals 
    TOP_PERCENT = "TOP_PERCENT"   # preserve top N% of population

class TopologyType(Enum):
    RING = "RING"
    FULLY_CONNECTED = "FULLY_CONNECTED"
    STAR = "STAR"
    RANDOM = "RANDOM"

class MigrationStrategy(Enum):
    BEST = "BEST"
    RANDOM = "RANDOM"
    MIXED = "MIXED"

class InitStrategy(Enum):
    RANDOM_ONLY = "RANDOM_ONLY"
    NN_RANDOM = "NN_RANDOM"
    SAVINGS_NN_RANDOM = "SAVINGS_NN_RANDOM"


# This is a dedicated class to hold my parameters.
class GAParameters:

    def __init__(self, config_map=None):
        if config_map is None:
            config_map = {}

        # Standard GA Parameters 
        self.population_size = config_map.get("population_size", 100)
        self.max_generations = config_map.get("max_generations", 1000)
        self.crossover_rate = config_map.get("crossover_rate", 0.8)
        self.crossover_strategy = config_map.get("crossover_strategy", "PMX")
        self.mutation_rate = config_map.get("mutation_rate", 0.01)
        self.mutation_strategy = config_map.get("mutation_strategy", "SWAP")
        self.elitism_strategy = config_map.get("elitism_strategy", "TOP_N")
        self.elitism_value = config_map.get("elitism_value", 0.1)
        
        # Island architecture parameters
        self.num_islands = config_map.get("num_islands", 5)
        self.island_topology = config_map.get("island_topology", "RING")
        self.migration_rate = config_map.get("migration_rate", 0.1)
        self.migration_interval = config_map.get("migration_interval", 10)
        self.migration_size = config_map.get("migration_size", 5)
        self.migration_strategy = config_map.get("migration_strategy", "BEST")
        
        # Additional operator parameters
        self.local_search_rate = config_map.get("local_search_rate", 0.1)
        self.population_strategy = config_map.get("population_strategy", "RANDOM_ONLY")
        
        # Fitness function weights
        self.w1 = config_map.get("w1", 1.0)
        self.w2 = config_map.get("w2", 1.0)
        self.w3 = config_map.get("w3", 1.0)

    def validate(self):
        # Helper function to extract string values from Enums or use the string directly
        def get_val(attr):
            return attr.name if hasattr(attr, 'name') else (attr.value if hasattr(attr, 'value') else str(attr))

        if self.population_size <= 0:
            raise ValueError("Population size must be greater than 0.")
            
        if self.max_generations <= 0:
            raise ValueError("Max generations must be greater than 0.")
            
        if not (0 <= self.crossover_rate <= 1):
            raise ValueError("Crossover rate must be between 0 and 1.")
            
        if get_val(self.crossover_strategy) not in ["PMX", "OX", "BOTH"]:
            raise ValueError("Crossover strategy must be 'PMX', 'OX', or 'BOTH'.")
            
        if not (0 <= self.mutation_rate <= 1):
            raise ValueError("Mutation rate must be between 0 and 1.")
            
        elitism_strat = get_val(getattr(self, 'elitism_strategy', 'TOP_N'))
        if elitism_strat not in ["TOP_N", "PERCENTAGE", "TOP_PERCENT"]:
            raise ValueError("Elitism strategy must be 'TOP_N', 'PERCENTAGE', or 'TOP_PERCENT'.")
            
        if getattr(self, 'elitism_value', 0) < 0:
            raise ValueError("Elitism value must be non-negative.")
            
        if self.num_islands < 1:
            raise ValueError("Number of islands must be at least 1.")
            
        if get_val(self.island_topology) not in ["RING", "STAR", "MESH", "FULLY_CONNECTED", "RANDOM"]:
            raise ValueError("Island topology must be 'RING', 'STAR', 'MESH', 'FULLY_CONNECTED', or 'RANDOM'.")
            
        if not (0 <= self.migration_rate <= 1):
            raise ValueError("Migration rate must be between 0 and 1.")
            
        if self.migration_interval < 1:
            raise ValueError("Migration interval must be at least 1 generation.")
            
        if self.migration_size < 1:
            raise ValueError("Migration size must be at least 1 individual.")
            
        if get_val(self.migration_strategy) not in ["BEST", "RANDOM", "MIXED"]:
            raise ValueError("Migration strategy must be 'BEST', 'RANDOM', or 'MIXED'.")
            
        if not (0 <= self.local_search_rate <= 1):
            raise ValueError("Local search rate must be between 0 and 1.")
            
        pop_strat_attr = getattr(self, 'population_strategy', getattr(self, 'population_init_strategy', None))
        pop_strat = get_val(pop_strat_attr)
        if pop_strat not in ["RANDOM_ONLY", "HEURISTIC_ONLY", "MIXED", "SAVINGS_NN_RANDOM", "NN_RANDOM"]:
            raise ValueError("Population strategy must be 'RANDOM_ONLY', 'HEURISTIC_ONLY', 'MIXED', 'NN_RANDOM', or 'SAVINGS_NN_RANDOM'.")
        
        
    def get_Defaults():
        return GAParameters({
            "population_size": 100,
            "max_generations": 1000,
            "crossover_rate": 0.8,
            "crossover_strategy": "PMX",
            "mutation_rate": 0.01,
            "mutation_strategy": "SWAP",
            "elitism_strategy": "TOP_N",
            "elitism_value": 0.1,
            "num_islands": 5,
            "island_topology": "RING",
            "migration_rate": 0.1,
            "migration_interval": 10,
            "migration_size": 5,
            "migration_strategy": "BEST",
            "local_search_rate": 0.1,
            "population_strategy": "RANDOM_ONLY",
            "w1": 1.0,
            "w2": 1.0,
            "w3": 1.0
        })

    def to_String(self):
        return f"GAParameters(population_size={self.population_size}, max_generations={self.max_generations}, crossover_rate={self.crossover_rate}, crossover_strategy='{self.crossover_strategy}', mutation_rate={self.mutation_rate}, mutation_strategy='{self.mutation_strategy}', elitism_strategy='{self.elitism_strategy}', elitism_value={self.elitism_value}, num_islands={self.num_islands}, island_topology='{self.island_topology}', migration_rate={self.migration_rate}, migration_interval={self.migration_interval}, migration_size={self.migration_size}, migration_strategy='{self.migration_strategy}', local_search_rate={self.local_search_rate}, population_strategy='{self.population_strategy}', w1={self.w1}, w2={self.w2}, w3={self.w3})"

