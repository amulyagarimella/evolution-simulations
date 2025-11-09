# Word Evolution Simulator

Interactive visualization and statistical analysis of evolution through word mutations, demonstrating how gene duplications enable evolutionary innovation.

## Features

- **Interactive Web App**: Visualize evolution step-by-step with density-based coloring showing attractor basins
- **Gene Duplication Simulation**: Compare essential gene evolution (1 copy) vs. evolution with duplications (2+ copies)
- **Statistical Analysis CLI**: Run batch simulations and collect performance statistics

## Quick Start

### Web App

```bash
pip install -r requirements.txt
streamlit run app.py
```

### CLI Statistics Tool

```bash
# Run 10 simulations with default parameters
python run_stats.py run --num_runs=10

# Compare essential gene (1 copy) vs duplications (3 copies)
python run_stats.py run --num_runs=50 --n_copies=1 --output=essential
python run_stats.py run --num_runs=50 --n_copies=3 --output=duplications

# Auto-compare 1-5 gene copies (automatically generates plots!)
python run_stats.py compare --num_runs=50 --max_copies=5

# Generate plots from existing results
python run_stats.py plot --pattern="compare_*.json"

# List saved results
python run_stats.py list_results
```

## Tips

- Start with small `num_runs` (10-20) to test parameters
- Use `compare` to see the dramatic effect of gene duplications
- Higher `n_e` = stronger selection, lower = more genetic drift
- Results are saved automatically with timestamps in `results/` directory
