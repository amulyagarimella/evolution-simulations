# Comparison: sim1 vs sim2 with 1 Gene Copy

## Summary

**sim1** and **sim2 with n_copies=1** are **conceptually similar** but **algorithmically different**.

## Key Similarities

1. **Single gene copy**: Both track evolution when there's only one functional copy of a gene
2. **Death on invalid mutation**: In both cases, an invalid mutation can end the evolutionary trajectory
3. **Selection and drift**: Both incorporate fitness-based selection and stochastic effects

## Key Differences

### sim1: Trajectory-Based Evolution
- **Algorithm**: Explicit Moran model with trajectory tracking
- **Population**: Tracks a single lineage through genotype space
- **Dynamics**:
  - One individual at a time
  - Each step: attempt mutation, accept/reject based on fitness
  - If mutation is invalid, trajectory ends (or can restart)
- **Parameters**:
  - `N_e`: Effective population size (affects drift strength)
  - `n_copies`: Number of gene copies evolving in parallel
- **Use case**: Understanding evolutionary paths and trajectories

### sim2: Population-Based Birth-Death Process
- **Algorithm**: Gillespie algorithm (continuous-time stochastic simulation)
- **Population**: Explicit population of individuals with genotypes
- **Dynamics**:
  - Birth-death events in continuous time
  - Birth rate ∝ fitness
  - Death rate ∝ population density (carrying capacity)
  - Population size fluctuates stochastically
- **Parameters**:
  - `N_carrying_capacity`: Maximum sustainable population
  - `mu`: Mutation rate per birth
  - `s`: Selection coefficient
  - `n_copies`: Gene copies per individual
- **Use case**: Understanding population dynamics, clonal interference, diversity

## When is sim2 with n_copies=1 equivalent to sim1?

### Approximate equivalence conditions:

1. **Very small population**: Set `N_carrying_capacity ≈ 1-10` in sim2
2. **High selection**: Both use strong selection relative to drift
3. **Same mutation rate**: Adjust `mu` in sim2 to match effective mutation rate

### Example comparison:

```python
# sim1 setup
N_e = 1000
n_copies = 1
edits_per_step = 1

# Approximately equivalent sim2 setup
N_carrying_capacity = 5  # Small population
n_copies = 1
mu = 0.1  # Adjust to match expected mutation frequency
s = 0.5   # Strong selection
```

## Practical Differences

Even with similar parameters, you'll observe:

1. **Stochasticity**: sim2 has more stochastic variation due to birth-death dynamics
2. **Population fluctuations**: sim2 population can go extinct randomly
3. **Time scales**: sim2 uses continuous time (generations), sim1 uses discrete steps
4. **Computational cost**: sim2 is slower for small populations due to event-based simulation

## Recommendation

- **Use sim1** for: Quick trajectory exploration, teaching, visualizing single paths
- **Use sim2** for: Realistic population dynamics, studying diversity, clonal interference, gene duplications

## Verification

To verify similarity, run both simulations with:
- Same start and target words
- Comparable selection strength
- Multiple replicates

Compare:
- Success rate (reaching target)
- Average time/generations to target
- Distribution of trajectories through genotype space

Expected result: **Similar trends but different fine-scale dynamics**
