# sim2.py - Birth-Death Evolution Simulator

## Overview

`sim2.py` implements a biologically realistic bacterial evolution simulator using the **Gillespie birth-death process**. This is Phase 1 of the enhanced simulation framework.

## Key Features

### 1. **Realistic Population Dynamics**
- Continuous-time birth-death process (Gillespie algorithm)
- Births proportional to fitness
- Deaths increase with population density (carrying capacity)
- Mutations occur during reproduction

### 2. **Multi-Gene Genomes**
- Each genome can have multiple genes
- Each gene can have multiple copies (gene duplication)
- **"Best copy wins"**: As long as ONE copy is functional, the organism survives

### 3. **Fitness Model**
- **Multiplicative fitness** across genes (literature standard)
- Per-gene fitness based on distance to target
- Selection coefficient `s` controls advantage of reaching target
- Valid mutations are not lethal (fitness ≥ 0.5)

### 4. **Streamlit-Ready Design**
- Clean DataFrame outputs for all metrics
- Callback support for live visualization
- Time series tracking of:
  - Population size
  - Genotypic diversity
  - Dominant genotype frequency
  - Per-gene progress toward target

## Core Classes

### `Genome`
```python
# Immutable, hashable genome representation
genome = Genome.from_config({
    'gene1': ('WORD', 2),  # 2 copies of gene1
    'gene2': ('COLD', 1)   # 1 copy of gene2
})
```

### `BacterialPopulation`
```python
# Tracks genotype frequencies
population = BacterialPopulation(carrying_capacity=1000)
population.add_individual(genome)
diversity = population.diversity()
```

### `GillespieSimulation`
```python
# Main simulation engine
sim = GillespieSimulation(
    gene_config={'gene1': ('WORD', 2)},
    targets={'gene1': 'GENE'},
    word_graph=word_graph,
    N_carrying_capacity=1000,
    mu=0.01,  # mutation rate
    s=0.1     # selection coefficient
)

# Run simulation
final = sim.run(max_generations=10000, stop_at_target=True)

# Get data for visualization
history_df = sim.get_history_df()
genotypes_df = sim.get_genotype_frequencies_df()
gene_diversity_df = sim.get_gene_diversity_df()
```

## Key Functions

### Fitness Functions

```python
# Per-gene fitness (distance-based)
fitness = gene_fitness(word, target, word_graph, s=0.1)

# Overall genome fitness (multiplicative)
fitness = genome_fitness(genome, targets, word_graph, s=0.1)
```

### Experimental Functions

```python
# Run replicate experiments (multiple flasks)
sims = run_replicate_experiment(
    gene_config, targets, word_graph,
    n_replicates=10
)

# Compare different gene copy numbers
comparison_df = compare_gene_copy_numbers(
    start_word='WORD',
    target_word='GENE',
    word_graph=word_graph,
    copy_numbers=[1, 2, 3],
    n_replicates=10
)
```

## Output DataFrames

### 1. History DataFrame
Time series of population metrics:
- `time`, `generation`
- `population_size`, `diversity`
- `dominant_frequency`
- `gene1_at_target`, `gene2_at_target`, ...

### 2. Genotype Frequencies DataFrame
Current genotype distribution:
- `genotype` (string representation)
- `frequency`, `count`
- `fitness`

### 3. Gene Diversity DataFrame
Per-gene sequence diversity:
- `gene_id`, `sequence`
- `count`, `frequency`
- `is_target` (boolean)

## Example Usage

### Single Gene Evolution
```python
from sim2 import *

word_graph, _, _ = build_word_graph(word_length=4)

sim = GillespieSimulation(
    gene_config={'gene1': ('WORD', 2)},
    targets={'gene1': 'GENE'},
    word_graph=word_graph,
    N_carrying_capacity=500,
    mu=0.1,
    s=0.3
)

final = sim.run(max_generations=20000)
history_df = sim.get_history_df()

# Plot with matplotlib/streamlit
import matplotlib.pyplot as plt
plt.plot(history_df['generation'], history_df['population_size'])
plt.xlabel('Generation')
plt.ylabel('Population Size')
```

### Multi-Gene Evolution
```python
gene_config = {
    'gene1': ('WORD', 2),  # duplicated
    'gene2': ('COLD', 1)   # essential
}
targets = {
    'gene1': 'GENE',
    'gene2': 'WARM'
}

sim = GillespieSimulation(gene_config, targets, word_graph, mu=0.1, s=0.2)
final = sim.run(max_generations=30000)

# Check which gene reached target first
gene_div = sim.get_gene_diversity_df()
print(gene_div[gene_div['is_target']])
```

### Replicate Experiments
```python
# Run 10 independent populations
sims = run_replicate_experiment(
    gene_config={'gene1': ('WORD', 2)},
    targets={'gene1': 'GENE'},
    word_graph=word_graph,
    n_replicates=10
)

# Analyze variability
success_rate = sum(s.get_state()['all_targets_reached'] for s in sims) / 10
```

### Compare Gene Copy Numbers
```python
# Does gene duplication help?
comparison = compare_gene_copy_numbers(
    start_word='WORD',
    target_word='GENE',
    word_graph=word_graph,
    copy_numbers=[1, 2, 3, 4],
    n_replicates=20
)

# Results show success_rate, avg_generations, etc.
print(comparison)
```

## Streamlit Integration

The simulation is designed for easy Streamlit integration:

```python
import streamlit as st
from sim2 import *

# Setup
word_graph, _, _ = build_word_graph(word_length=4)
sim = GillespieSimulation(...)

# Live updates
progress_bar = st.progress(0)
chart = st.line_chart()

def update_callback(state):
    progress_bar.progress(state['generation'] / max_gen)
    # Update chart with new data point

sim.run(callback=update_callback)

# Final visualization
history_df = sim.get_history_df()
st.line_chart(history_df.set_index('generation')['population_size'])
st.dataframe(sim.get_genotype_frequencies_df())
```

## Performance

- Single simulation (10K generations): ~1-5 seconds
- Replicate experiment (10 flasks): ~10-50 seconds
- Comparison study (3 conditions, 10 reps): ~1-2 minutes

Even faster in C++/Rust if needed for large-scale studies.

## Next Steps (Future Phases)

1. **Phase 2**: Multiple genes in complex genomes
2. **Phase 3**: Gene space network analysis & visualization
3. **Phase 4**: Interactive 3D gene space explorer
4. **Phase 5**: Stickbreaking fitness model (diminishing returns)

## Key Biological Insights

This model captures:
- ✓ Gene duplication allows risky exploration
- ✓ Population bottlenecks affect diversity
- ✓ Clonal interference in large populations
- ✓ Stochastic extinction in small populations
- ✓ Realistic mutation-selection balance
