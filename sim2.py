"""
Bacterial evolution simulator with birth-death (Gillespie) dynamics.

This module implements a more biologically realistic simulation where:
- Multiple genes can evolve simultaneously
- Each gene can have multiple copies (gene duplication)
- Population dynamics follow continuous birth-death process
- As long as one gene copy is functional, the organism survives
- Fitness is multiplicative across genes

Designed for easy integration with Streamlit and visualization.
"""

import networkx as nx
import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import heapq

# Import word graph builder from sim1
from sim1 import build_word_graph, word_fitness


@dataclass(frozen=True)
class Genome:
    """
    Immutable genome representation with multiple genes.

    Each gene can have multiple copies (gene duplication).
    Frozen dataclass makes it hashable for use in dictionaries.

    Attributes:
        genes: Tuple of tuples, where each inner tuple is (gene_id, word_sequence)
                Example: (('gene1', 'WORD'), ('gene1', 'WORD'), ('gene2', 'COLD'))
                represents 2 copies of gene1 and 1 copy of gene2
    """
    genes: Tuple[Tuple[str, str], ...]

    @staticmethod
    def from_config(gene_config: Dict[str, Tuple[str, int]]) -> 'Genome':
        """
        Create genome from configuration.

        Args:
            gene_config: {gene_id: (start_word, n_copies)}

        Example:
            gene_config = {
                'gene1': ('WORD', 2),  # 2 copies of gene1, both starting at WORD
                'gene2': ('COLD', 1)   # 1 copy of gene2, starting at COLD
            }
        """
        genes = []
        for gene_id, (start_word, n_copies) in sorted(gene_config.items()):
            for _ in range(n_copies):
                genes.append((gene_id, start_word))
        return Genome(genes=tuple(genes))

    def get_genes_dict(self) -> Dict[str, List[str]]:
        """Get genes organized by gene_id."""
        result = defaultdict(list)
        for gene_id, word in self.genes:
            result[gene_id].append(word)
        return dict(result)

    def mutate_copy(self, copy_idx: int, new_word: str) -> 'Genome':
        """Create new genome with one copy mutated."""
        new_genes = list(self.genes)
        gene_id = new_genes[copy_idx][0]
        new_genes[copy_idx] = (gene_id, new_word)
        return Genome(genes=tuple(new_genes))

    def __str__(self):
        genes_dict = self.get_genes_dict()
        return ', '.join([f"{gid}: {words}" for gid, words in genes_dict.items()])


def gene_fitness(word: str, target: str, word_graph: nx.Graph,
                s: float = 0.5, baseline: float = 1.0) -> float:
    """
    Fitness contribution of a single gene copy.

    Args:
        word: Current sequence
        target: Target sequence
        word_graph: Word graph for computing distances
        s: Selection coefficient (benefit of being at target) - default 0.5 for strong selection
        baseline: Baseline fitness for non-target sequences

    Returns:
        Fitness value (1.0 = neutral, 1+s = at target, <1.0 = deleterious)
    """
    if word == target:
        return baseline + s  # Beneficial - 50% advantage by default

    # Compute distance to target
    if word not in word_graph or target not in word_graph:
        return 0.01  # Very unfit if not in graph

    try:
        distance = nx.shortest_path_length(word_graph, word, target)
    except nx.NetworkXNoPath:
        distance = float('inf')

    if distance == float('inf'):
        return 0.01  # Very unfit if no path

    # Steeper fitness gradient toward target
    # Closer words have higher fitness, providing stronger directional selection
    max_distance = 8  # Typical max distance in 4-letter word space

    # Exponential decay provides steeper gradient near target
    # Distance 1 from target: ~0.88, Distance 4: ~0.60, Distance 8: ~0.30
    fitness = baseline * (0.7 ** (distance / 2.0))

    # Floor at 0.3 for valid words (more costly than before)
    return max(fitness, 0.3)


def genome_fitness(genome: Genome, targets: Dict[str, str],
                   word_graph: nx.Graph, s: float = 0.5,
                   model: str = 'multiplicative') -> float:
    """
    Calculate overall fitness of a genome.

    For each gene, the best copy determines fitness (gene redundancy).
    Fitness is then combined across genes.

    Args:
        genome: Genome to evaluate
        targets: {gene_id: target_word}
        word_graph: Word graph for distance computation
        s: Selection coefficient
        model: 'multiplicative' or 'additive' (for future)

    Returns:
        Overall fitness value
    """
    genes_dict = genome.get_genes_dict()

    if model == 'multiplicative':
        fitness = 1.0
        for gene_id, target in targets.items():
            if gene_id not in genes_dict:
                return 0.0  # Missing essential gene

            # Best copy wins - as long as one copy is good, gene functions
            copy_fitnesses = [
                gene_fitness(copy, target, word_graph, s)
                for copy in genes_dict[gene_id]
            ]
            best_copy_fitness = max(copy_fitnesses)

            # Multiplicative across genes
            fitness *= best_copy_fitness

        return fitness

    elif model == 'additive':
        # For future implementation
        raise NotImplementedError("Additive model not yet implemented")

    else:
        raise ValueError(f"Unknown fitness model: {model}")


class BacterialPopulation:
    """
    Population of bacteria with different genotypes.

    Tracks genotype frequencies and provides methods for
    population-level statistics.
    """

    def __init__(self, carrying_capacity: int):
        self.genotypes: Dict[Genome, int] = {}
        self.carrying_capacity = carrying_capacity
        self.total_size = 0

    def add_individual(self, genome: Genome, count: int = 1):
        """Add individuals with given genome."""
        self.genotypes[genome] = self.genotypes.get(genome, 0) + count
        self.total_size += count

    def remove_individual(self, genome: Genome):
        """Remove one individual with given genome."""
        if genome in self.genotypes and self.genotypes[genome] > 0:
            self.genotypes[genome] -= 1
            self.total_size -= 1
            if self.genotypes[genome] == 0:
                del self.genotypes[genome]

    def get_random_individual(self) -> Optional[Genome]:
        """Sample random individual proportional to frequency."""
        if self.total_size == 0:
            return None

        # Weighted random choice
        genomes = list(self.genotypes.keys())
        weights = [self.genotypes[g] for g in genomes]
        return random.choices(genomes, weights=weights, k=1)[0]

    def diversity(self) -> int:
        """Number of distinct genotypes."""
        return len(self.genotypes)

    def dominant_genotype(self) -> Tuple[Optional[Genome], int]:
        """Most common genotype and its count."""
        if not self.genotypes:
            return None, 0
        dominant = max(self.genotypes.items(), key=lambda x: x[1])
        return dominant[0], dominant[1]

    def get_frequency_dict(self) -> Dict[Genome, float]:
        """Get genotype frequencies (proportions)."""
        if self.total_size == 0:
            return {}
        return {g: count/self.total_size for g, count in self.genotypes.items()}


class GillespieSimulation:
    """
    Gillespie algorithm for bacterial evolution with birth-death dynamics.

    This implements the continuous-time birth-death process where:
    - Each genotype has a birth rate (proportional to fitness)
    - Death rate increases with population density (carrying capacity)
    - Mutations occur at birth with probability mu

    Designed for easy integration with Streamlit visualization.
    """

    def __init__(self,
                 gene_config: Dict[str, Tuple[str, int]],
                 targets: Dict[str, str],
                 word_graph: nx.Graph,
                 N_carrying_capacity: int = 1000,
                 mu: float = 0.01,
                 s: float = 0.5,
                 birth_rate_base: float = 1.5,
                 death_rate_base: float = 0.5,
                 seed: Optional[int] = None):
        """
        Initialize simulation.

        Args:
            gene_config: {gene_id: (start_word, n_copies)}
            targets: {gene_id: target_word}
            word_graph: NetworkX graph of valid words
            N_carrying_capacity: Population carrying capacity
            mu: Mutation rate per gene copy per birth event
                (each copy mutates independently, so more copies = more mutations)
            s: Selection coefficient (fitness benefit at target)
            birth_rate_base: Base birth rate
            death_rate_base: Base death rate (increases with density)
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.gene_config = gene_config
        self.targets = targets
        self.word_graph = word_graph
        self.N_carrying_capacity = N_carrying_capacity
        self.mu = mu
        self.s = s
        self.birth_rate_base = birth_rate_base
        self.death_rate_base = death_rate_base

        # Initialize population with founder genome
        self.population = BacterialPopulation(N_carrying_capacity)
        founder = Genome.from_config(gene_config)
        self.population.add_individual(founder, count=10)  # Start with 10 individuals

        # Simulation state
        self.time = 0.0
        self.generation = 0

        # Metrics tracking for explored space and mutational expansion
        self.all_genotypes_seen = set()  # All unique genotypes ever encountered
        self.all_words_seen = set()  # All unique words (phenotypes) ever encountered
        self.max_diversity = 0  # Maximum genotypic diversity reached
        self.max_edit_distance = 0  # Maximum edit distance from starting words

        # Initialize with founder
        self.all_genotypes_seen.add(founder)
        for _, word in founder.genes:
            self.all_words_seen.add(word)

        # Store starting words for distance calculations
        self.starting_words = set()
        for gene_id, (start_word, n_copies) in gene_config.items():
            self.starting_words.add(start_word)

        # History for visualization
        self.history = []
        self.record_state()

    def get_birth_rate(self, genome: Genome) -> float:
        """Birth rate for a genotype (proportional to fitness)."""
        fitness = genome_fitness(genome, self.targets, self.word_graph, self.s)
        return self.birth_rate_base * fitness

    def get_death_rate(self) -> float:
        """
        Death rate (increases with population density).

        Implements logistic growth through density-dependent mortality.
        Stronger density dependence creates more selection pressure.
        """
        density = self.population.total_size / self.N_carrying_capacity
        # Quadratic density dependence: stronger regulation near carrying capacity
        return self.death_rate_base * (1 + 2 * density ** 1.5)

    def compute_rates(self) -> Tuple[float, float]:
        """
        Compute total birth and death rates.

        Returns:
            (total_birth_rate, total_death_rate)
        """
        total_birth = sum(
            self.get_birth_rate(genome) * count
            for genome, count in self.population.genotypes.items()
        )
        total_death = self.get_death_rate() * self.population.total_size
        return total_birth, total_death

    def birth_event(self):
        """Execute a birth event (with possible mutation).

        Each gene copy has independent mutation probability mu.
        This is biologically realistic: more gene copies = more mutation supply.
        """
        # Sample parent proportional to birth rate
        genomes = list(self.population.genotypes.keys())
        birth_rates = [
            self.get_birth_rate(g) * self.population.genotypes[g]
            for g in genomes
        ]

        if sum(birth_rates) == 0:
            return  # No viable births

        parent = random.choices(genomes, weights=birth_rates, k=1)[0]

        # Each gene copy mutates independently with probability mu
        offspring_genes = []
        for copy_idx, (gene_id, current_word) in enumerate(parent.genes):
            if random.random() < self.mu:
                # This copy mutates
                if current_word in self.word_graph:
                    neighbors = list(self.word_graph.neighbors(current_word))
                    if neighbors:
                        new_word = random.choice(neighbors)
                        offspring_genes.append((gene_id, new_word))
                    else:
                        # No valid neighbors, copy unchanged
                        offspring_genes.append((gene_id, current_word))
                else:
                    # Word not in graph, copy unchanged
                    offspring_genes.append((gene_id, current_word))
            else:
                # No mutation, perfect copy
                offspring_genes.append((gene_id, current_word))

        offspring = Genome(genes=tuple(offspring_genes))

        # Track new genotypes and words
        if offspring not in self.all_genotypes_seen:
            self.all_genotypes_seen.add(offspring)

        for _, word in offspring.genes:
            if word not in self.all_words_seen:
                self.all_words_seen.add(word)
                # Update max edit distance when we see a new word
                self._update_max_edit_distance(word)

        self.population.add_individual(offspring)

    def death_event(self):
        """Execute a death event (random individual dies)."""
        dead = self.population.get_random_individual()
        if dead:
            self.population.remove_individual(dead)

    def _update_max_edit_distance(self, word: str):
        """Update maximum edit distance from any starting word."""
        for start_word in self.starting_words:
            try:
                dist = nx.shortest_path_length(self.word_graph, start_word, word)
                if dist > self.max_edit_distance:
                    self.max_edit_distance = dist
            except nx.NetworkXNoPath:
                pass  # Can't reach this word from start

    def step(self) -> Dict:
        """
        Execute one Gillespie step.

        Returns:
            State dictionary with current metrics (for Streamlit updates)
        """
        if self.population.total_size == 0:
            return self.get_state()

        # Compute rates
        birth_rate, death_rate = self.compute_rates()
        total_rate = birth_rate + death_rate

        if total_rate == 0:
            return self.get_state()  # No events possible

        # Sample time to next event (exponential distribution)
        dt = np.random.exponential(1.0 / total_rate)
        self.time += dt

        # Sample which event occurs
        if random.random() < birth_rate / total_rate:
            self.birth_event()
        else:
            self.death_event()

        self.generation += 1

        # Record state periodically
        if self.generation % 100 == 0:
            self.record_state()

        return self.get_state()

    def record_state(self):
        """
        Record current state for history.

        Stores full population snapshot for visualization and time-scrubbing.
        """
        # Update max diversity if current diversity is higher
        current_diversity = self.population.diversity()
        if current_diversity > self.max_diversity:
            self.max_diversity = current_diversity

        state = self.get_state()

        # Also store genotype distribution for visualization
        state['genotypes'] = dict(self.population.genotypes)  # Copy current genotypes

        self.history.append(state)

    def get_state(self) -> Dict:
        """Get current state as dictionary (Streamlit-friendly)."""
        dominant_genome, dominant_count = self.population.dominant_genotype()

        # Check if any gene has reached target
        genes_at_target = {}
        if dominant_genome:
            genes_dict = dominant_genome.get_genes_dict()
            for gene_id, target in self.targets.items():
                if gene_id in genes_dict:
                    genes_at_target[gene_id] = target in genes_dict[gene_id]
                else:
                    genes_at_target[gene_id] = False

        return {
            'time': self.time,
            'generation': self.generation,
            'population_size': self.population.total_size,
            'diversity': self.population.diversity(),
            'dominant_genome': str(dominant_genome) if dominant_genome else None,
            'dominant_frequency': dominant_count / self.population.total_size if self.population.total_size > 0 else 0,
            'genes_at_target': genes_at_target,
            'all_targets_reached': all(genes_at_target.values()) if genes_at_target else False,
            # Explored space metrics
            'explored_genotypes': len(self.all_genotypes_seen),
            'explored_words': len(self.all_words_seen),
            'max_diversity': self.max_diversity,
            'max_edit_distance': self.max_edit_distance
        }

    def run(self,
            max_generations: int = 10000,
            max_time: float = 1000.0,
            stop_at_target: bool = True,
            callback: Optional[Callable] = None) -> Dict:
        """
        Run simulation until stopping condition.

        Args:
            max_generations: Maximum number of events
            max_time: Maximum simulation time
            stop_at_target: Stop when all genes reach target
            callback: Optional callback function called each step (for live updates)

        Returns:
            Final state dictionary
        """
        while self.generation < max_generations and self.time < max_time:
            if self.population.total_size == 0:
                break

            state = self.step()

            if callback:
                callback(state)

            if stop_at_target and state['all_targets_reached']:
                self.record_state()
                break

        return self.get_state()

    def get_history_df(self) -> pd.DataFrame:
        """
        Get simulation history as DataFrame (for plotting).

        Returns DataFrame with columns:
        - time, generation, population_size, diversity, dominant_frequency
        - explored_genotypes, explored_words, max_diversity, max_edit_distance
        - Plus one column per gene: gene_X_at_target
        """
        if not self.history:
            return pd.DataFrame()

        # Flatten history
        rows = []
        for state in self.history:
            row = {
                'time': state['time'],
                'generation': state['generation'],
                'population_size': state['population_size'],
                'diversity': state['diversity'],
                'genotypes': state['genotypes'],
                'dominant_frequency': state['dominant_frequency'],
                'explored_genotypes': state.get('explored_genotypes', 0),
                'explored_words': state.get('explored_words', 0),
                'max_diversity': state.get('max_diversity', 0),
                'max_edit_distance': state.get('max_edit_distance', 0)
            }
            # Add gene-specific columns
            for gene_id, at_target in state['genes_at_target'].items():
                row[f'{gene_id}_at_target'] = at_target
            rows.append(row)

        return pd.DataFrame(rows)

    def get_genotype_frequencies_df(self) -> pd.DataFrame:
        """
        Get current genotype frequencies as DataFrame.

        Returns DataFrame with columns:
        - genotype (string representation)
        - frequency
        - count
        - fitness
        """
        freq_dict = self.population.get_frequency_dict()

        rows = []
        for genome, freq in freq_dict.items():
            rows.append({
                'genotype': str(genome),
                'frequency': freq,
                'count': self.population.genotypes[genome],
                'fitness': genome_fitness(genome, self.targets, self.word_graph, self.s)
            })

        df = pd.DataFrame(rows)
        return df.sort_values('frequency', ascending=False)

    def get_gene_diversity_df(self) -> pd.DataFrame:
        """
        Get per-gene sequence diversity as DataFrame.

        Returns DataFrame showing all observed sequences for each gene.
        """
        # Collect all sequences per gene across all genotypes
        gene_sequences = defaultdict(lambda: defaultdict(int))

        for genome, count in self.population.genotypes.items():
            genes_dict = genome.get_genes_dict()
            for gene_id, sequences in genes_dict.items():
                for seq in sequences:
                    gene_sequences[gene_id][seq] += count

        # Convert to DataFrame
        rows = []
        for gene_id in sorted(gene_sequences.keys()):
            total = sum(gene_sequences[gene_id].values())
            for seq, count in sorted(gene_sequences[gene_id].items(),
                                    key=lambda x: x[1], reverse=True):
                rows.append({
                    'gene_id': gene_id,
                    'sequence': seq,
                    'count': count,
                    'frequency': count / total,
                    'is_target': seq == self.targets.get(gene_id, '')
                })

        return pd.DataFrame(rows)

    def get_word_frequencies(self, gene_id: Optional[str] = None) -> Dict[str, int]:
        """
        Get word (sequence) frequencies in current population.

        Counts how many copies of each word exist across all genomes.
        Useful for visualization: node size = word frequency.

        Args:
            gene_id: If specified, only count words from this gene.
                    If None, count all words from all genes.

        Returns:
            Dictionary mapping word -> count
        """
        word_counts = defaultdict(int)

        for genome, count in self.population.genotypes.items():
            for gid, word in genome.genes:
                if gene_id is None or gid == gene_id:
                    word_counts[word] += count

        return dict(word_counts)

    def get_genome_cooccurrence(self, gene_id: Optional[str] = None) -> List[Tuple[str, str, int]]:
        """
        Get pairs of words that coexist in the same genome.

        For multi-copy genomes, returns which words appear together.
        Useful for visualization: draw edges between co-occurring words.

        Args:
            gene_id: If specified, only consider words from this gene.

        Returns:
            List of (word1, word2, count) tuples where count is number of genomes
            containing both words.
        """
        cooccurrence = defaultdict(int)

        for genome, count in self.population.genotypes.items():
            # Get words from this genome
            if gene_id is None:
                words = [word for _, word in genome.genes]
            else:
                words = [word for gid, word in genome.genes if gid == gene_id]

            # Count pairs (use set to avoid duplicates within genome)
            unique_words = list(set(words))
            for i in range(len(unique_words)):
                for j in range(i+1, len(unique_words)):
                    w1, w2 = sorted([unique_words[i], unique_words[j]])
                    cooccurrence[(w1, w2)] += count

        # Convert to list of tuples
        return [(w1, w2, cnt) for (w1, w2), cnt in cooccurrence.items()]

    def get_population_snapshot(self, gene_id: Optional[str] = None) -> Dict:
        """
        Get complete population state for visualization.

        Returns:
            Dictionary with:
            - word_frequencies: {word: count}
            - cooccurrences: [(word1, word2, count), ...]
            - targets: {gene_id: target_word}
            - state: current simulation state
        """
        return {
            'word_frequencies': self.get_word_frequencies(gene_id),
            'cooccurrences': self.get_genome_cooccurrence(gene_id),
            'targets': self.targets,
            'state': self.get_state()
        }

    def get_snapshot_at_generation(self, generation_idx: int, gene_id: Optional[str] = None) -> Dict:
        """
        Get population snapshot at a specific generation from history.

        Useful for time-scrubbing in visualization.

        Args:
            generation_idx: Index in history (0 to len(history)-1)
            gene_id: If specified, only get data for this gene

        Returns:
            Snapshot dictionary similar to get_population_snapshot()
        """
        if not self.history or generation_idx >= len(self.history):
            return None

        state = self.history[generation_idx]
        genotypes = state.get('genotypes', {})

        # Reconstruct word frequencies from stored genotypes
        word_counts = defaultdict(int)
        cooccurrence = defaultdict(int)

        for genome, count in genotypes.items():
            # Extract words from genome
            if gene_id is None:
                words = [word for _, word in genome.genes]
            else:
                words = [word for gid, word in genome.genes if gid == gene_id]

            # Count words
            for word in words:
                word_counts[word] += count

            # Count cooccurrences
            unique_words = list(set(words))
            for i in range(len(unique_words)):
                for j in range(i+1, len(unique_words)):
                    w1, w2 = sorted([unique_words[i], unique_words[j]])
                    cooccurrence[(w1, w2)] += count

        return {
            'word_frequencies': dict(word_counts),
            'cooccurrences': [(w1, w2, cnt) for (w1, w2), cnt in cooccurrence.items()],
            'targets': self.targets,
            'state': state
        }


def run_replicate_experiment(gene_config: Dict[str, Tuple[str, int]],
                            targets: Dict[str, str],
                            word_graph: nx.Graph,
                            n_replicates: int = 10,
                            sim_init_kwargs: Optional[Dict] = None,
                            sim_run_kwargs: Optional[Dict] = None) -> List[GillespieSimulation]:
    """
    Run multiple replicate simulations (like multiple flask experiments).

    Args:
        gene_config: Gene configuration
        targets: Target sequences
        word_graph: Word graph
        n_replicates: Number of replicate populations
        sim_init_kwargs: Arguments passed to GillespieSimulation.__init__()
        sim_run_kwargs: Arguments passed to sim.run()

    Returns:
        List of completed simulations
    """
    if sim_init_kwargs is None:
        sim_init_kwargs = {}
    if sim_run_kwargs is None:
        sim_run_kwargs = {}

    simulations = []

    for i in range(n_replicates):
        print(f"Running replicate {i+1}/{n_replicates}...")

        sim = GillespieSimulation(
            gene_config=gene_config,
            targets=targets,
            word_graph=word_graph,
            seed=i,  # Different seed for each replicate
            **sim_init_kwargs
        )

        final_state = sim.run(**sim_run_kwargs)
        simulations.append(sim)

        if final_state['all_targets_reached']:
            print(f"  ✓ Success at generation {final_state['generation']}")
        else:
            print(f"  ✗ Failed (pop size: {final_state['population_size']})")

    return simulations


def compare_gene_copy_numbers(start_word: str,
                              target_word: str,
                              word_graph: nx.Graph,
                              copy_numbers: List[int] = [1, 2, 3],
                              n_replicates: int = 10,
                              sim_init_kwargs: Optional[Dict] = None,
                              sim_run_kwargs: Optional[Dict] = None) -> pd.DataFrame:
    """
    Compare evolutionary outcomes for different numbers of gene copies.

    This addresses the key question: does gene duplication help evolution?

    Args:
        start_word: Starting sequence
        target_word: Target sequence
        word_graph: Word graph
        copy_numbers: List of copy numbers to test
        n_replicates: Replicates per condition
        sim_init_kwargs: Arguments for GillespieSimulation.__init__()
        sim_run_kwargs: Arguments for sim.run()

    Returns:
        DataFrame with summary statistics per condition
    """
    results = []

    for n_copies in copy_numbers:
        print(f"\n{'='*60}")
        print(f"Testing {n_copies} gene cop{'y' if n_copies == 1 else 'ies'}")
        print(f"{'='*60}")

        gene_config = {'gene1': (start_word, n_copies)}
        targets = {'gene1': target_word}

        sims = run_replicate_experiment(
            gene_config=gene_config,
            targets=targets,
            word_graph=word_graph,
            n_replicates=n_replicates,
            sim_init_kwargs=sim_init_kwargs,
            sim_run_kwargs=sim_run_kwargs
        )

        # Collect statistics
        success_count = sum(1 for s in sims if s.get_state()['all_targets_reached'])
        generations = [s.generation for s in sims]
        times = [s.time for s in sims]

        results.append({
            'n_copies': n_copies,
            'success_rate': success_count / n_replicates,
            'avg_generations': np.mean(generations),
            'std_generations': np.std(generations),
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_generations': np.min(generations),
            'max_generations': np.max(generations)
        })

    return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    print("Building word graph...")
    word_graph, valid_words, _ = build_word_graph(word_length=4, max_delta=1)
    print(f"Graph has {word_graph.number_of_nodes()} nodes\n")

    # Example 1: Single gene with duplication
    print("="*60)
    print("EXAMPLE 1: Single gene evolution (WORD → GENE)")
    print("="*60)

    gene_config = {
        'gene1': ('WORD', 2)  # 2 copies
    }
    targets = {
        'gene1': 'GENE'
    }

    sim = GillespieSimulation(
        gene_config=gene_config,
        targets=targets,
        word_graph=word_graph,
        N_carrying_capacity=1000,
        mu=0.05,
        s=0.2,
        seed=42
    )

    print("\nRunning simulation...")
    final = sim.run(max_generations=50000)

    print(f"\nFinal state:")
    print(f"  Generation: {final['generation']}")
    print(f"  Time: {final['time']:.2f}")
    print(f"  Population: {final['population_size']}")
    print(f"  Diversity: {final['diversity']} genotypes")
    print(f"  Target reached: {final['all_targets_reached']}")
    print(f"  Dominant: {final['dominant_genome']}")

    # Show history
    history_df = sim.get_history_df()
    print(f"\nHistory shape: {history_df.shape}")
    print(history_df.head())

    # Example 2: Compare copy numbers
    print("\n" + "="*60)
    print("EXAMPLE 2: Comparing 1 vs 2 vs 3 gene copies")
    print("="*60)

    comparison_df = compare_gene_copy_numbers(
        start_word='WORD',
        target_word='GENE',
        word_graph=word_graph,
        copy_numbers=[1, 2, 3],
        n_replicates=5,
        sim_init_kwargs={
            'N_carrying_capacity': 1000,
            'mu': 0.05,
            's': 0.2
        },
        sim_run_kwargs={
            'max_generations': 50000
        }
    )

    print("\nComparison results:")
    print(comparison_df.to_string(index=False))
