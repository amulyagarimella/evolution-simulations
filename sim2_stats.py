#!/usr/bin/env python3
"""
CLI tool for running Gillespie simulation statistics and parameter sweeps.

Runs the bacterial evolution simulation (sim2) multiple times and collects statistics:
- Time to reach target (continuous time)
- Number of generations (birth/death events)
- Success rate
- Population dynamics (size, diversity)
- Explored genotype/phenotype space

Designed for comparing gene copy numbers and parameter sweeps.
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import trio
import fire
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
from sim2 import GillespieSimulation, build_word_graph

# Results directory configuration
RESULTS_DIR = Path("results/sim2/")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Claude-inspired color palette (warm, inviting tones)
CLAUDE_COLORS = {
    'primary': '#D97757',      # Warm coral/orange
    'secondary': '#C75146',    # Deep coral
    'accent1': '#E8A87C',      # Light peach
    'accent2': '#B85C50',      # Rust
    'dark': '#5D4E37',         # Coffee brown
    'gradient': ['#E8A87C', '#D97757', '#C75146', '#B85C50', '#9B4F47']  # Light to dark
}


# ============================================================================
# CONFIGURATION AND STYLE SETUP
# ============================================================================

def setup_publication_style():
    """Configure matplotlib and seaborn for publication-quality figures."""
    sns.set_context("talk", font_scale=1.1)
    sns.set_style("whitegrid", {
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'axes.edgecolor': '.15',
        'axes.linewidth': 1.2,
    })

    # Set CMU Serif font for publication quality
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['CMU Serif', 'Computer Modern', 'Times']
    mpl.rcParams['pdf.fonttype'] = 42  # TrueType fonts (editable in Illustrator)
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVG (not paths)
    mpl.rcParams['axes.unicode_minus'] = False  # Use proper minus sign

    # Higher quality settings
    mpl.rcParams['figure.dpi'] = 150
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = 0.1


def get_color_palette(n_colors: int) -> List[str]:
    """Get appropriate color palette for n_colors conditions."""
    if n_colors <= len(CLAUDE_COLORS['gradient']):
        return CLAUDE_COLORS['gradient'][:n_colors]
    return sns.color_palette("YlOrRd", n_colors=n_colors)


# ============================================================================
# DATA STRUCTURES AND STATISTICS
# ============================================================================

class SimulationStats:
    """Container for Gillespie simulation run statistics."""

    def __init__(self):
        self.runs = []

    def add_run(self, run_data: Dict):
        """Add a single run's statistics."""
        self.runs.append(run_data)

    def get_summary(self) -> Dict:
        """Get summary statistics across all runs."""
        if not self.runs:
            return {}

        successful_runs = [r for r in self.runs if r['success']]

        summary = self._compute_all_runs_stats()
        if successful_runs:
            summary.update(self._compute_success_only_stats(successful_runs))

        return summary

    def _compute_all_runs_stats(self) -> Dict:
        """Compute statistics across all runs."""
        return {
            'total_runs': len(self.runs),
            'successful_runs': sum(r['success'] for r in self.runs),
            'success_rate': sum(r['success'] for r in self.runs) / len(self.runs) * 100,
            'avg_generations': np.mean([r['generations'] for r in self.runs]),
            'std_generations': np.std([r['generations'] for r in self.runs]),
            'avg_sim_time': np.mean([r['sim_time'] for r in self.runs]),
            'std_sim_time': np.std([r['sim_time'] for r in self.runs]),
            'avg_wall_time': np.mean([r['wall_time'] for r in self.runs]),
            'avg_final_pop': np.mean([r['final_population'] for r in self.runs]),
            'std_final_pop': np.std([r['final_population'] for r in self.runs]),
            'avg_explored_words': np.mean([r['explored_words'] for r in self.runs]),
            'std_explored_words': np.std([r['explored_words'] for r in self.runs]),
            'min_explored_words': np.min([r['explored_words'] for r in self.runs]),
            'max_explored_words': np.max([r['explored_words'] for r in self.runs]),
            'avg_explored_genotypes': np.mean([r['explored_genotypes'] for r in self.runs]),
            'std_explored_genotypes': np.std([r['explored_genotypes'] for r in self.runs]),
            'avg_max_edit_distance': np.mean([r['max_edit_distance'] for r in self.runs]),
            'std_max_edit_distance': np.std([r['max_edit_distance'] for r in self.runs]),
        }

    def _compute_success_only_stats(self, successful_runs: List[Dict]) -> Dict:
        """Compute statistics for successful runs only."""
        return {
            'avg_generations_success': np.mean([r['generations'] for r in successful_runs]),
            'std_generations_success': np.std([r['generations'] for r in successful_runs]),
            'avg_sim_time_success': np.mean([r['sim_time'] for r in successful_runs]),
            'std_sim_time_success': np.std([r['sim_time'] for r in successful_runs]),
            'min_generations_success': np.min([r['generations'] for r in successful_runs]),
            'max_generations_success': np.max([r['generations'] for r in successful_runs]),
            'avg_explored_words_success': np.mean([r['explored_words'] for r in successful_runs]),
            'std_explored_words_success': np.std([r['explored_words'] for r in successful_runs]),
            'avg_explored_genotypes_success': np.mean([r['explored_genotypes'] for r in successful_runs]),
            'std_explored_genotypes_success': np.std([r['explored_genotypes'] for r in successful_runs]),
        }

    def to_json(self, filename: str):
        """Save detailed results to JSON file."""
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(item) for item in obj]
            return obj

        data = {
            'runs': convert_numpy(self.runs),
            'summary': convert_numpy(self.get_summary())
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def to_csv(self, filename: str):
        """Save results to CSV file for easy analysis."""
        rows = []
        for run in self.runs:
            row = {
                'run_number': run['run_number'],
                'success': run['success'],
                'generations': run['generations'],
                'sim_time': run['sim_time'],
                'wall_time': run['wall_time'],
                'final_population': run['final_population'],
                'max_diversity': run['max_diversity'],
                'explored_genotypes': run['explored_genotypes'],
                'explored_words': run['explored_words'],
                'max_edit_distance': run['max_edit_distance'],
            }
            row.update({f'param_{k}': v for k, v in run['params'].items()})
            rows.append(row)

        pd.DataFrame(rows).to_csv(filename, index=False)

    def to_summary_file(self, filename: str):
        """Save summary statistics to a text file."""
        summary = self.get_summary()

        with open(filename, 'w') as f:
            f.write("="*60 + "\n")
            f.write("GILLESPIE SIMULATION STATISTICS SUMMARY\n")
            f.write("="*60 + "\n")
            self._write_basic_stats(f, summary)
            self._write_exploration_stats(f, summary)
            if summary['successful_runs'] > 0:
                self._write_success_stats(f, summary)
            f.write("="*60 + "\n")

    def _write_basic_stats(self, f, summary: Dict):
        """Write basic statistics to file."""
        f.write(f"Total Runs:              {summary['total_runs']}\n")
        f.write(f"Successful Runs:         {summary['successful_runs']}\n")
        f.write(f"Success Rate:            {summary['success_rate']:.1f}%\n")
        f.write(f"\nAverage Generations:     {summary['avg_generations']:.1f} ± {summary['std_generations']:.1f}\n")
        f.write(f"Average Simulation Time: {summary['avg_sim_time']:.2f} ± {summary['std_sim_time']:.2f}\n")
        f.write(f"Average Wall Time:       {summary['avg_wall_time']:.2f}s\n")
        f.write(f"Average Final Pop:       {summary['avg_final_pop']:.1f} ± {summary['std_final_pop']:.1f}\n")

    def _write_exploration_stats(self, f, summary: Dict):
        """Write exploration statistics to file."""
        f.write(f"\n--- Explored Space Statistics ---\n")
        f.write(f"Words Explored:          {summary['avg_explored_words']:.1f} ± {summary['std_explored_words']:.1f}\n")
        f.write(f"  Range: {summary['min_explored_words']:.0f} - {summary['max_explored_words']:.0f}\n")
        f.write(f"Genotypes Explored:      {summary['avg_explored_genotypes']:.1f} ± {summary['std_explored_genotypes']:.1f}\n")
        f.write(f"Max Edit Distance:       {summary['avg_max_edit_distance']:.1f} ± {summary['std_max_edit_distance']:.1f}\n")

    def _write_success_stats(self, f, summary: Dict):
        """Write success-only statistics to file."""
        f.write(f"\n--- Successful Runs Only ---\n")
        f.write(f"Generations to Success:  {summary['avg_generations_success']:.1f} ± {summary['std_generations_success']:.1f}\n")
        f.write(f"  Range: {summary['min_generations_success']:.0f} - {summary['max_generations_success']:.0f}\n")
        f.write(f"Sim Time to Success:     {summary['avg_sim_time_success']:.2f} ± {summary['std_sim_time_success']:.2f}\n")
        f.write(f"Words Explored:          {summary['avg_explored_words_success']:.1f} ± {summary['std_explored_words_success']:.1f}\n")
        f.write(f"Genotypes Explored:      {summary['avg_explored_genotypes_success']:.1f} ± {summary['std_explored_genotypes_success']:.1f}\n")

    def save_all(self, base_path: str):
        """Save results in all formats (JSON, CSV, summary)."""
        self.to_json(str(base_path) + ".json")
        self.to_csv(str(base_path) + ".csv")
        self.to_summary_file(str(base_path) + ".txt")

    def print_summary(self):
        """Print summary statistics to console."""
        summary = self.get_summary()
        print("\n" + "="*60)
        print("GILLESPIE SIMULATION STATISTICS SUMMARY")
        print("="*60)
        self._print_basic_stats(summary)
        self._print_exploration_stats(summary)
        if summary['successful_runs'] > 0:
            self._print_success_stats(summary)
        print("="*60 + "\n")

    def _print_basic_stats(self, summary: Dict):
        """Print basic statistics."""
        print(f"Total Runs:              {summary['total_runs']}")
        print(f"Successful Runs:         {summary['successful_runs']}")
        print(f"Success Rate:            {summary['success_rate']:.1f}%")
        print(f"\nAverage Generations:     {summary['avg_generations']:.1f} ± {summary['std_generations']:.1f}")
        print(f"Average Simulation Time: {summary['avg_sim_time']:.2f} ± {summary['std_sim_time']:.2f}")
        print(f"Average Wall Time:       {summary['avg_wall_time']:.2f}s")
        print(f"Average Final Pop:       {summary['avg_final_pop']:.1f} ± {summary['std_final_pop']:.1f}")

    def _print_exploration_stats(self, summary: Dict):
        """Print exploration statistics."""
        print(f"\n--- Explored Space Statistics ---")
        print(f"Words Explored:          {summary['avg_explored_words']:.1f} ± {summary['std_explored_words']:.1f}")
        print(f"  Range: {summary['min_explored_words']:.0f} - {summary['max_explored_words']:.0f}")
        print(f"Genotypes Explored:      {summary['avg_explored_genotypes']:.1f} ± {summary['std_explored_genotypes']:.1f}")
        print(f"Max Edit Distance:       {summary['avg_max_edit_distance']:.1f} ± {summary['std_max_edit_distance']:.1f}")

    def _print_success_stats(self, summary: Dict):
        """Print success-only statistics."""
        print(f"\n--- Successful Runs Only ---")
        print(f"Generations to Success:  {summary['avg_generations_success']:.1f} ± {summary['std_generations_success']:.1f}")
        print(f"  Range: {summary['min_generations_success']:.0f} - {summary['max_generations_success']:.0f}")
        print(f"Sim Time to Success:     {summary['avg_sim_time_success']:.2f} ± {summary['std_sim_time_success']:.2f}")
        print(f"Words Explored:          {summary['avg_explored_words_success']:.1f} ± {summary['std_explored_words_success']:.1f}")
        print(f"Genotypes Explored:      {summary['avg_explored_genotypes_success']:.1f} ± {summary['std_explored_genotypes_success']:.1f}")


# ============================================================================
# SIMULATION EXECUTION
# ============================================================================

def run_single_simulation(word_graph, gene_config: Dict, targets: Dict,
                         N_carrying_capacity: int = 1000,
                         mu: float = 0.01,
                         s: float = 0.5,
                         birth_rate_base: float = 1.5,
                         death_rate_base: float = 0.5,
                         max_generations: int = 50000,
                         max_time: float = 1000.0,
                         seed: Optional[int] = None) -> Dict:
    """Run a single Gillespie simulation and return statistics."""
    sim = GillespieSimulation(
        gene_config=gene_config,
        targets=targets,
        word_graph=word_graph,
        N_carrying_capacity=N_carrying_capacity,
        mu=mu,
        s=s,
        birth_rate_base=birth_rate_base,
        death_rate_base=death_rate_base,
        seed=seed
    )

    start_time = time.time()
    final_state = sim.run(
        max_generations=max_generations,
        max_time=max_time,
        stop_at_target=True
    )
    wall_time = time.time() - start_time

    return {
        'success': final_state['all_targets_reached'],
        'generations': final_state['generation'],
        'sim_time': final_state['time'],
        'wall_time': wall_time,
        'final_population': final_state['population_size'],
        'final_diversity': final_state['diversity'],
        'max_diversity': sim.max_diversity,
        'explored_genotypes': len(sim.all_genotypes_seen),
        'explored_words': len(sim.all_words_seen),
        'max_edit_distance': sim.max_edit_distance,
        'dominant_genome': final_state['dominant_genome'],
    }


def run_batch_simulations(num_runs: int,
                         start_word: str,
                         target_word: str,
                         n_copies: int = 1,
                         N_carrying_capacity: int = 1000,
                         mu: float = 0.01,
                         s: float = 0.5,
                         birth_rate_base: float = 1.5,
                         death_rate_base: float = 0.5,
                         max_generations: int = 50000,
                         max_time: float = 1000.0,
                         verbose: bool = True) -> SimulationStats:
    """Run multiple simulation trials and collect statistics."""
    # Build word graph once (expensive operation)
    if verbose:
        print(f"Building word graph for {len(start_word)}-letter words...")

    word_graph, valid_words, _ = build_word_graph(
        word_length=len(start_word), max_delta=1
    )

    if verbose:
        print(f"Graph built: {word_graph.number_of_nodes()} nodes, "
              f"{word_graph.number_of_edges()} edges\n")

    # Validate words
    if start_word not in word_graph:
        raise ValueError(f"Start word '{start_word}' not in dictionary!")
    if target_word not in word_graph:
        raise ValueError(f"Target word '{target_word}' not in dictionary!")

    gene_config = {'gene1': (start_word, n_copies)}
    targets = {'gene1': target_word}

    stats = SimulationStats()

    for run_num in range(1, num_runs + 1):
        if verbose:
            print(f"Run {run_num}/{num_runs}...", end=" ", flush=True)

        result = run_single_simulation(
            word_graph=word_graph,
            gene_config=gene_config,
            targets=targets,
            N_carrying_capacity=N_carrying_capacity,
            mu=mu,
            s=s,
            birth_rate_base=birth_rate_base,
            death_rate_base=death_rate_base,
            max_generations=max_generations,
            max_time=max_time,
            seed=run_num
        )

        run_data = {
            'run_number': run_num,
            **result,
            'params': {
                'start_word': start_word,
                'target_word': target_word,
                'n_copies': n_copies,
                'N_carrying_capacity': N_carrying_capacity,
                'mu': mu,
                's': s,
                'birth_rate_base': birth_rate_base,
                'death_rate_base': death_rate_base,
            }
        }

        stats.add_run(run_data)

        if verbose:
            status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
            print(f"{status} - gen={result['generations']}, "
                  f"time={result['sim_time']:.1f}, "
                  f"pop={result['final_population']}, "
                  f"wall_time={result['wall_time']:.2f}s")

    return stats


def run_single_parameter_combination(args: Tuple) -> Tuple[Dict, SimulationStats]:
    """
    Run simulations for a single parameter combination.
    Helper function for parallel execution.
    """
    (combo_num, total_combinations, num_runs, start_word, target_word,
     n_copies, mu, s, N, output_prefix, timestamp) = args

    print(f"[Worker] Combination {combo_num}/{total_combinations}: "
          f"n_copies={n_copies}, mu={mu}, s={s}, N={N}")

    stats = run_batch_simulations(
        num_runs=num_runs,
        start_word=start_word,
        target_word=target_word,
        n_copies=n_copies,
        N_carrying_capacity=N,
        mu=mu,
        s=s,
        max_generations=50000,
        max_time=1000.0,
        verbose=False
    )

    summary = stats.get_summary()
    result_dict = {
        'n_copies': n_copies,
        'mu': mu,
        's': s,
        'N_carrying_capacity': N,
        **summary
    }

    # Save individual results
    output_path = (RESULTS_DIR /
                  f"{output_prefix}_{start_word}_to_{target_word}_"
                  f"n{n_copies}_mu{mu}_s{s}_N{N}_{timestamp}")
    stats.save_all(str(output_path))

    print(f"[Worker] Completed combination {combo_num}/{total_combinations}")

    return result_dict, stats


# ============================================================================
# PLOTTING UTILITIES
# ============================================================================

def save_plot(fig, output_path: Path, plot_name: str):
    """Save plot to file and close figure."""
    plt.tight_layout()
    plt.savefig(output_path, format='svg')
    plt.close()
    return output_path


def create_line_plot_with_error(ax, x, y_mean, y_std, color, marker='o', label=None):
    """Create line plot with shaded error region."""
    ax.plot(x, y_mean, marker=marker, linewidth=3, markersize=10,
            color=color, markeredgecolor='white', markeredgewidth=2, label=label)
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.25, color=color)


def format_parameter_name(param: str) -> str:
    """Format parameter name for display."""
    return param.replace('_', ' ').title()


# ============================================================================
# COMPARISON PLOTS
# ============================================================================

def plot_comparison_results(result_files: List[Path],
                           comparison_var: str = 'n_copies',
                           output_dir: Path = RESULTS_DIR):
    """Generate publication-quality comparison plots from multiple result files."""
    setup_publication_style()

    # Load all results
    all_data = []
    for filepath in result_files:
        with open(filepath, 'r') as f:
            data = json.load(f)
            if data['runs']:
                comp_value = data['runs'][0]['params'][comparison_var]
                for run in data['runs']:
                    all_data.append({
                        comparison_var: comp_value,
                        'success': run['success'],
                        'generations': run['generations'],
                        'sim_time': run['sim_time'],
                        'final_population': run['final_population'],
                        'explored_genotypes': run['explored_genotypes'],
                        'explored_words': run['explored_words'],
                        'max_diversity': run['max_diversity'],
                    })

    df = pd.DataFrame(all_data)
    df_success = df[df['success'] == True]

    x = sorted(df[comparison_var].unique())
    colors = get_color_palette(len(x))

    plots_saved = {}
    plots_saved.update(_plot_success_rate(df, comparison_var, x, colors, output_dir))
    plots_saved.update(_plot_evolutionary_time(df_success, comparison_var, x, output_dir))
    plots_saved.update(_plot_generations(df_success, comparison_var, x, output_dir))
    plots_saved.update(_plot_genotypes(df, comparison_var, x, output_dir))
    plots_saved.update(_plot_distributions(df_success, comparison_var, x, colors, output_dir))

    return plots_saved


def _plot_success_rate(df, comparison_var, x, colors, output_dir):
    """Plot success rate vs comparison variable."""
    fig, ax = plt.subplots(figsize=(10, 7))

    grouped = df.groupby(comparison_var)
    success_rate = grouped['success'].mean() * 100
    ax.bar(x, success_rate.values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)

    ax.set_xlabel(format_parameter_name(comparison_var))
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(f'Evolution Success Rate vs {format_parameter_name(comparison_var)}',
                 fontweight='bold', pad=20)
    ax.set_ylim([0, 105])
    sns.despine()

    plot_path = output_dir / f'plot_success_rate_vs_{comparison_var}.svg'
    return {'success_rate': save_plot(fig, plot_path, 'success_rate')}


def _plot_evolutionary_time(df_success, comparison_var, x, output_dir):
    """Plot evolutionary time vs comparison variable."""
    if len(df_success) == 0:
        return {}

    fig, ax = plt.subplots(figsize=(10, 7))
    grouped = df_success.groupby(comparison_var)
    avg_time = grouped['sim_time'].mean()
    std_time = grouped['sim_time'].std()

    create_line_plot_with_error(ax, x, avg_time.values, std_time.values,
                                CLAUDE_COLORS['primary'])

    ax.set_xlabel(format_parameter_name(comparison_var))
    ax.set_ylabel('Evolutionary Time to Target')
    ax.set_title(f'Time to Reach Target vs {format_parameter_name(comparison_var)}',
                 fontweight='bold', pad=20)
    sns.despine()

    plot_path = output_dir / f'plot_time_vs_{comparison_var}.svg'
    return {'evolutionary_time': save_plot(fig, plot_path, 'time')}


def _plot_generations(df_success, comparison_var, x, output_dir):
    """Plot generations vs comparison variable."""
    if len(df_success) == 0:
        return {}

    fig, ax = plt.subplots(figsize=(10, 7))
    grouped = df_success.groupby(comparison_var)
    avg_gens = grouped['generations'].mean()
    std_gens = grouped['generations'].std()

    create_line_plot_with_error(ax, x, avg_gens.values, std_gens.values,
                                CLAUDE_COLORS['accent2'], marker='s')

    ax.set_xlabel(format_parameter_name(comparison_var))
    ax.set_ylabel('Generations (Events) to Target')
    ax.set_title(f'Number of Events vs {format_parameter_name(comparison_var)}',
                 fontweight='bold', pad=20)
    sns.despine()

    plot_path = output_dir / f'plot_generations_vs_{comparison_var}.svg'
    return {'generations': save_plot(fig, plot_path, 'generations')}


def _plot_genotypes(df, comparison_var, x, output_dir):
    """Plot explored genotypes vs comparison variable."""
    fig, ax = plt.subplots(figsize=(10, 7))
    grouped = df.groupby(comparison_var)
    avg_genotypes = grouped['explored_genotypes'].mean()
    std_genotypes = grouped['explored_genotypes'].std()

    create_line_plot_with_error(ax, x, avg_genotypes.values, std_genotypes.values,
                                CLAUDE_COLORS['secondary'], marker='d')

    ax.set_xlabel(format_parameter_name(comparison_var))
    ax.set_ylabel('Unique Genotypes Explored')
    ax.set_title(f'Genotypic Diversity vs {format_parameter_name(comparison_var)}',
                 fontweight='bold', pad=20)
    sns.despine()

    plot_path = output_dir / f'plot_genotypes_vs_{comparison_var}.svg'
    return {'genotypes': save_plot(fig, plot_path, 'genotypes')}


def _plot_distributions(df_success, comparison_var, x, colors, output_dir):
    """Plot distributions of time and generations."""
    if len(df_success) == 0:
        return {}

    plots = {}

    # Time distribution
    fig, ax = plt.subplots(figsize=(12, 7))
    for val, color in zip(x, colors):
        data_subset = df_success[df_success[comparison_var] == val]['sim_time']
        if len(data_subset) > 0:
            label = f'{val}'
            if comparison_var == 'n_copies':
                label += f' {"copy" if val == 1 else "copies"}'
            sns.kdeplot(data=data_subset, ax=ax, linewidth=3, alpha=0.7,
                       label=label, color=color, fill=True)

    ax.set_xlabel('Evolutionary Time to Target')
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution of Time to Target by {format_parameter_name(comparison_var)}',
                 fontweight='bold', pad=20)
    ax.legend(title=format_parameter_name(comparison_var), frameon=True, fancybox=True, shadow=True)
    sns.despine()

    plot_path = output_dir / f'plot_time_distribution_{comparison_var}.svg'
    plots['time_dist'] = save_plot(fig, plot_path, 'time_dist')

    # Generations distribution
    fig, ax = plt.subplots(figsize=(12, 7))
    for val, color in zip(x, colors):
        data_subset = df_success[df_success[comparison_var] == val]['generations']
        if len(data_subset) > 0:
            label = f'{val}'
            if comparison_var == 'n_copies':
                label += f' {"copy" if val == 1 else "copies"}'
            sns.kdeplot(data=data_subset, ax=ax, linewidth=3, alpha=0.7,
                       label=label, color=color, fill=True)

    ax.set_xlabel('Generations (Events) to Target')
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution of Events to Target by {format_parameter_name(comparison_var)}',
                 fontweight='bold', pad=20)
    ax.legend(title=format_parameter_name(comparison_var), frameon=True, fancybox=True, shadow=True)
    sns.despine()

    plot_path = output_dir / f'plot_generations_distribution_{comparison_var}.svg'
    plots['generations_dist'] = save_plot(fig, plot_path, 'gens_dist')

    return plots


# ============================================================================
# GRID SEARCH PLOTS
# ============================================================================

def plot_grid_search_results(df: pd.DataFrame,
                             output_dir: Path = RESULTS_DIR,
                             start: str = "WORD",
                             target: str = "GENE",
                             timestamp: str = ""):
    """Generate publication-quality plots for grid search results."""
    setup_publication_style()
    plots_saved = {}

    # Determine varying parameters
    param_cols = ['n_copies', 'mu', 's', 'N_carrying_capacity']
    varying_params = [col for col in param_cols if df[col].nunique() > 1]

    if len(varying_params) < 2:
        print("Warning: Need at least 2 varying parameters for grid plots")
        return plots_saved

    # Create heatmaps for each metric
    x_param, y_param = varying_params[0], varying_params[1]
    other_params = varying_params[2:]

    # Get fixed parameter subset if needed
    df_plot = _get_representative_subset(df, other_params) if len(varying_params) > 2 else df.copy()

    # Create heatmaps
    plots_saved.update(_create_heatmaps(df_plot, x_param, y_param, start, target, timestamp, output_dir))

    # Create summary plots
    plots_saved.update(_create_summary_plots(df, varying_params, start, target, timestamp, output_dir))

    # Generate copy number comparison plots
    plots_saved.update(plot_grid_search_by_copies(df, output_dir, start, target, timestamp))

    return plots_saved


def _get_representative_subset(df: pd.DataFrame, other_params: List[str]) -> pd.DataFrame:
    """Get representative subset by fixing additional parameters."""
    fixed_params = {p: df[p].iloc[0] for p in other_params}
    df_subset = df.copy()
    for p, val in fixed_params.items():
        df_subset = df_subset[df_subset[p] == val]
    return df_subset


def _create_heatmaps(df_plot, x_param, y_param, start, target, timestamp, output_dir):
    """Create heatmap plots for different metrics."""
    plots_saved = {}

    metrics = {
        'success_rate': ('Success Rate (%)', 'YlOrRd', 0, 100),
        'avg_sim_time_success': ('Avg Time to Success', 'viridis', None, None),
        'avg_generations_success': ('Avg Generations to Success', 'viridis', None, None),
        'avg_final_pop': ('Avg Final Population', 'viridis', None, None),
    }

    for metric, (label, cmap, vmin, vmax) in metrics.items():
        if metric not in df_plot.columns:
            continue

        try:
            pivot = df_plot.pivot_table(values=metric, index=y_param, columns=x_param, aggfunc='mean')

            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap=cmap, ax=ax,
                       cbar_kws={'label': label}, vmin=vmin, vmax=vmax,
                       linewidths=0.5, linecolor='white')

            ax.set_xlabel(format_parameter_name(x_param), fontsize=14)
            ax.set_ylabel(format_parameter_name(y_param), fontsize=14)
            ax.set_title(f'{label} by {format_parameter_name(x_param)} and {format_parameter_name(y_param)}',
                        fontweight='bold', pad=20, fontsize=16)

            plot_path = output_dir / f'grid_heatmap_{metric}_{start}_to_{target}_{timestamp}.svg'
            plots_saved[f'heatmap_{metric}'] = save_plot(fig, plot_path, f'heatmap_{metric}')
        except Exception as e:
            print(f"Warning: Could not create heatmap for {metric}: {e}")

    return plots_saved


def _create_summary_plots(df, varying_params, start, target, timestamp, output_dir):
    """Create summary comparison plots."""
    if 'success_rate' not in df.columns:
        return {}

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Top 10 by success rate
    _plot_top_combinations(axes[0], df, 'success_rate', 'Top 10 Parameter Combinations by Success Rate',
                          CLAUDE_COLORS['primary'])

    # Top 10 fastest
    if 'avg_sim_time_success' in df.columns:
        df_success = df[df['success_rate'] > 0].copy()
        if len(df_success) > 0:
            _plot_top_combinations(axes[1], df_success, 'avg_sim_time_success',
                                  'Top 10 Fastest Parameter Combinations',
                                  CLAUDE_COLORS['accent1'], ascending=True)

    # Parameter effect on success rate
    if len(varying_params) > 0:
        param = varying_params[0]
        sns.boxplot(data=df, x=param, y='success_rate', ax=axes[2], color=CLAUDE_COLORS['secondary'])
        axes[2].set_ylabel('Success Rate (%)')
        axes[2].set_xlabel(format_parameter_name(param))
        axes[2].set_title(f'Success Rate Distribution by {format_parameter_name(param)}', fontweight='bold')
        sns.despine(ax=axes[2])

    # Parameter effect on time
    if 'avg_sim_time_success' in df.columns and len(varying_params) > 0:
        df_success = df[df['success_rate'] > 0].copy()
        if len(df_success) > 0:
            param = varying_params[0]
            sns.boxplot(data=df_success, x=param, y='avg_sim_time_success', ax=axes[3],
                       color=CLAUDE_COLORS['accent2'])
            axes[3].set_ylabel('Avg Time to Success')
            axes[3].set_xlabel(format_parameter_name(param))
            axes[3].set_title(f'Time Distribution by {format_parameter_name(param)}', fontweight='bold')
            sns.despine(ax=axes[3])

    plot_path = output_dir / f'grid_summary_{start}_to_{target}_{timestamp}.svg'
    return {'summary': save_plot(fig, plot_path, 'summary')}


def _plot_top_combinations(ax, df, metric, title, color, ascending=False):
    """Plot top parameter combinations for a given metric."""
    top_n = 10
    top_combos = df.nsmallest(top_n, metric) if ascending else df.nlargest(top_n, metric)

    ax.barh(range(len(top_combos)), top_combos[metric].values, color=color, alpha=0.8)
    ax.set_yticks(range(len(top_combos)))
    ax.set_yticklabels([
        f"n={r['n_copies']}, μ={r['mu']:.3f}, s={r['s']:.2f}, N={r['N_carrying_capacity']}"
        for _, r in top_combos.iterrows()
    ], fontsize=9)
    ax.set_xlabel(format_parameter_name(metric))
    ax.set_title(title, fontweight='bold')
    ax.invert_yaxis()
    sns.despine(ax=ax)


# ============================================================================
# GENE COPY NUMBER COMPARISON PLOTS
# ============================================================================

def plot_grid_search_by_copies(df: pd.DataFrame,
                               output_dir: Path = RESULTS_DIR,
                               start: str = "WORD",
                               target: str = "GENE",
                               timestamp: str = ""):
    """Generate comparison plots for grid search results by gene copy number."""
    setup_publication_style()
    plots_saved = {}

    if df['n_copies'].nunique() <= 1:
        return plots_saved

    other_params = ['mu', 's', 'N_carrying_capacity']
    varying_other = [p for p in other_params if df[p].nunique() > 1]

    if not varying_other:
        return plots_saved

    n_copies_vals = sorted(df['n_copies'].unique())
    n_colors = len(n_copies_vals)
    copy_colors = get_color_palette(n_colors)
    color_map = {n: copy_colors[i] for i, n in enumerate(n_copies_vals)}

    # Create plots based on number of varying parameters
    if len(varying_other) == 1:
        plots_saved.update(_plot_single_param_comparison(df, varying_other[0], n_copies_vals,
                                                         color_map, start, target, timestamp, output_dir))
    elif len(varying_other) >= 2:
        plots_saved.update(_plot_multi_param_comparison(df, varying_other, n_copies_vals,
                                                        color_map, start, target, timestamp, output_dir))

    return plots_saved


def _plot_single_param_comparison(df, param, n_copies_vals, color_map, start, target, timestamp, output_dir):
    """Create comparison plots for single varying parameter."""
    param_vals = sorted(df[param].unique())
    n_subplots = len(param_vals)

    n_cols = min(2, n_subplots)
    n_rows = (n_subplots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 7 * n_rows))
    if n_subplots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, param_val in enumerate(param_vals):
        ax = axes[idx]
        df_subset = df[df[param] == param_val]
        grouped = df_subset.groupby('n_copies')

        success_rate = grouped['success_rate'].mean()
        std_rate = grouped['success_rate'].std()

        x_pos = range(len(n_copies_vals))
        ax.bar(x_pos, [success_rate.get(n, 0) for n in n_copies_vals],
               yerr=[std_rate.get(n, 0) for n in n_copies_vals],
               color=[color_map[n] for n in n_copies_vals],
               alpha=0.8, edgecolor='white', linewidth=2, capsize=5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(n) for n in n_copies_vals])
        ax.set_xlabel('Number of Gene Copies')
        ax.set_ylabel('Success Rate (%)')
        ax.set_ylim([0, 105])
        ax.set_title(f'{format_parameter_name(param)} = {param_val}', fontweight='bold')
        sns.despine(ax=ax)

    # Hide unused subplots
    for idx in range(n_subplots, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'Success Rate vs Gene Copy Number\n({start} → {target})',
                fontsize=16, fontweight='bold', y=1.02)

    plot_path = output_dir / f'grid_success_rate_by_copies_{start}_to_{target}_{timestamp}.svg'
    return {'success_rate_by_copies': save_plot(fig, plot_path, 'success_by_copies')}


def _plot_multi_param_comparison(df, varying_other, n_copies_vals, color_map, start, target, timestamp, output_dir):
    """Create comparison plots for multiple varying parameters."""
    param1, param2 = varying_other[0], varying_other[1]
    param1_vals = sorted(df[param1].unique())
    param2_vals = sorted(df[param2].unique())

    plots = {}

    # Success rate plot
    plots.update(_create_grid_subplot(df, param1_vals, param2_vals, n_copies_vals, color_map,
                                     param1, param2, 'success_rate', 'Success Rate (%)',
                                     start, target, timestamp, output_dir, plot_type='bar'))

    # Time plot
    if 'avg_sim_time_success' in df.columns:
        df_success = df[df['success_rate'] > 0].copy()
        if len(df_success) > 0:
            plots.update(_create_grid_subplot(df_success, param1_vals, param2_vals, n_copies_vals, color_map,
                                             param1, param2, 'avg_sim_time_success', 'Time to Success',
                                             start, target, timestamp, output_dir, plot_type='line'))

    # Generations plot
    if 'avg_generations_success' in df.columns:
        df_success = df[df['success_rate'] > 0].copy()
        if len(df_success) > 0:
            plots.update(_create_grid_subplot(df_success, param1_vals, param2_vals, n_copies_vals, color_map,
                                             param1, param2, 'avg_generations_success', 'Generations to Success',
                                             start, target, timestamp, output_dir, plot_type='line'))

    return plots


def _create_grid_subplot(df, param1_vals, param2_vals, n_copies_vals, color_map,
                         param1, param2, metric, metric_label, start, target, timestamp, output_dir, plot_type='bar'):
    """Create grid of subplots for parameter combinations."""
    n_rows = len(param2_vals)
    n_cols = len(param1_vals)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for i, param2_val in enumerate(param2_vals):
        for j, param1_val in enumerate(param1_vals):
            ax = axes[i][j] if n_rows > 1 else axes[j]
            df_subset = df[(df[param1] == param1_val) & (df[param2] == param2_val)]

            if len(df_subset) > 0:
                grouped = df_subset.groupby('n_copies')

                if plot_type == 'bar':
                    _plot_bar_comparison(ax, grouped, n_copies_vals, color_map, metric)
                else:
                    _plot_line_comparison(ax, grouped, n_copies_vals, color_map, metric)

            # Set labels
            if i == n_rows - 1:
                ax.set_xlabel('Gene Copies')
            if j == 0:
                ax.set_ylabel(metric_label)

            title = f'{format_parameter_name(param1)}={param1_val}\n{format_parameter_name(param2)}={param2_val}'
            ax.set_title(title, fontsize=10, fontweight='bold')

    fig.suptitle(f'{metric_label} vs Gene Copy Number\n({start} → {target})',
                fontsize=16, fontweight='bold', y=1.0)

    metric_name = metric.replace('avg_', '').replace('_success', '')
    plot_path = output_dir / f'grid_{metric_name}_by_copies_{start}_to_{target}_{timestamp}.svg'
    return {f'{metric_name}_by_copies': save_plot(fig, plot_path, f'{metric_name}_by_copies')}


def _plot_bar_comparison(ax, grouped, n_copies_vals, color_map, metric):
    """Plot bar chart comparison."""
    metric_mean = grouped[metric].mean()
    metric_std = grouped[metric].std()

    x_pos = range(len(n_copies_vals))
    ax.bar(x_pos, [metric_mean.get(n, 0) for n in n_copies_vals],
           yerr=[metric_std.get(n, 0) for n in n_copies_vals],
           color=[color_map[n] for n in n_copies_vals],
           alpha=0.8, edgecolor='white', linewidth=2, capsize=5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(n) for n in n_copies_vals])
    ax.set_ylim([0, 105])
    sns.despine(ax=ax)


def _plot_line_comparison(ax, grouped, n_copies_vals, color_map, metric):
    """Plot line chart comparison."""
    metric_mean = grouped[metric].mean()
    metric_std = grouped[metric].std()

    available_copies = [n for n in n_copies_vals if n in metric_mean.index]

    if available_copies:
        x_plot = [n_copies_vals.index(n) for n in available_copies]
        y_vals = [metric_mean[n] for n in available_copies]
        y_err = [metric_std.get(n, 0) for n in available_copies]

        color = CLAUDE_COLORS['primary'] if 'time' in metric else CLAUDE_COLORS['accent2']
        marker = 'o' if 'time' in metric else 's'

        ax.plot(x_plot, y_vals, marker=marker, linewidth=3, markersize=8,
               color=color, markeredgecolor='white', markeredgewidth=2)
        ax.fill_between(x_plot, [y - e for y, e in zip(y_vals, y_err)],
                       [y + e for y, e in zip(y_vals, y_err)],
                       alpha=0.25, color=color)

    ax.set_xticks(range(len(n_copies_vals)))
    ax.set_xticklabels([str(n) for n in n_copies_vals])
    sns.despine(ax=ax)


# ============================================================================
# EXPLORED WORDS DISTRIBUTION PLOTS
# ============================================================================

def plot_explored_words_distributions(output_dir: Path = RESULTS_DIR,
                                      results_pattern: str = "grid_WORD_to_GENE_*.json"):
    """Create publication-quality KDE plots showing explored words distributions."""
    setup_publication_style()
    plots_saved = {}

    # Load all results
    json_files = sorted(output_dir.glob(results_pattern))
    if not json_files:
        print(f"No result files found matching pattern: {results_pattern}")
        return plots_saved

    df = _load_explored_words_data(json_files)
    if df.empty:
        print("No data found in result files")
        return plots_saved

    mutation_rates = sorted(df['mu'].unique())
    n_copies_vals = sorted(df['n_copies'].unique())
    color_map = {n: c for n, c in zip(n_copies_vals, get_color_palette(len(n_copies_vals)))}

    # Create different plot types
    plots_saved['kde_by_mu'] = _create_kde_plots(df, mutation_rates, n_copies_vals, color_map, output_dir)
    plots_saved['violin_by_mu'] = _create_violin_plots(df, mutation_rates, n_copies_vals, color_map, output_dir)
    plots_saved['boxplot_by_mu'] = _create_boxplot_with_points(df, mutation_rates, n_copies_vals, color_map, output_dir)

    _print_explored_words_summary(df, mutation_rates, n_copies_vals)

    return plots_saved


def _load_explored_words_data(json_files: List[Path]) -> pd.DataFrame:
    """Load explored words data from JSON files."""
    all_data = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        if 'runs' in data and len(data['runs']) > 0:
            params = data['runs'][0]['params']
            for run in data['runs']:
                all_data.append({
                    'explored_words': run['explored_words'],
                    'success': run['success'],
                    'n_copies': params['n_copies'],
                    'N': params['N_carrying_capacity'],
                    'mu': params['mu'],
                    's': params['s'],
                    'generations': run['generations']
                })

    return pd.DataFrame(all_data)


def _create_kde_plots(df, mutation_rates, n_copies_vals, color_map, output_dir):
    """Create KDE distribution plots."""
    fig, axes = plt.subplots(1, len(mutation_rates), figsize=(8*len(mutation_rates), 6), sharey=True)
    if len(mutation_rates) == 1:
        axes = [axes]

    for idx, mu in enumerate(mutation_rates):
        ax = axes[idx]
        mu_subset = df[df['mu'] == mu]

        for n in n_copies_vals:
            subset = mu_subset[mu_subset['n_copies'] == n]
            if len(subset) > 0:
                mean_val = subset['explored_words'].mean()
                sns.kdeplot(data=subset['explored_words'], ax=ax, linewidth=3,
                           label=f'n={n} (μ={mean_val:.1f})',
                           color=color_map[n], fill=True, alpha=0.5)

        ax.set_xlabel('Explored Words', fontsize=14, fontweight='bold')
        ax.set_ylabel('Density' if idx == 0 else '', fontsize=14, fontweight='bold')
        ax.set_title(f'Mutation Rate μ = {mu}', fontsize=16, fontweight='bold')
        ax.legend(title='Gene Copies', fontsize=11, title_fontsize=12,
                 frameon=True, fancybox=True, shadow=True, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        sns.despine(ax=ax)

    fig.suptitle('Distribution of Explored Words by Mutation Rate and Gene Copy Number',
                 fontsize=18, fontweight='bold', y=1.02)

    plot_path = output_dir / 'explored_words_kde_by_mu_and_copies.svg'
    save_plot(fig, plot_path, 'kde')
    print(f"Saved: {plot_path.name}")
    return plot_path


def _create_violin_plots(df, mutation_rates, n_copies_vals, color_map, output_dir):
    """Create violin distribution plots."""
    fig, axes = plt.subplots(1, len(mutation_rates), figsize=(8*len(mutation_rates), 6), sharey=True)
    if len(mutation_rates) == 1:
        axes = [axes]

    for idx, mu in enumerate(mutation_rates):
        mu_subset = df[df['mu'] == mu]

        parts = axes[idx].violinplot(
            [mu_subset[mu_subset['n_copies'] == n]['explored_words'].values for n in n_copies_vals],
            positions=range(len(n_copies_vals)),
            widths=0.7,
            showmeans=True,
            showmedians=True
        )

        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(color_map[n_copies_vals[i]])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)

        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
            if partname in parts:
                parts[partname].set_edgecolor('black')
                parts[partname].set_linewidth(1.5)

        axes[idx].set_xticks(range(len(n_copies_vals)))
        axes[idx].set_xticklabels([str(n) for n in n_copies_vals])
        axes[idx].set_xlabel('Number of Gene Copies (n)', fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('Explored Words' if idx == 0 else '', fontsize=14, fontweight='bold')
        axes[idx].set_title(f'Mutation Rate μ = {mu}', fontsize=16, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')
        sns.despine(ax=axes[idx])

    fig.suptitle('Explored Words Distribution: Violins by Gene Copies and Mutation Rate',
                 fontsize=18, fontweight='bold', y=1.02)

    plot_path = output_dir / 'explored_words_violin_by_mu_and_copies.svg'
    save_plot(fig, plot_path, 'violin')
    print(f"Saved: {plot_path.name}")
    return plot_path


def _create_boxplot_with_points(df, mutation_rates, n_copies_vals, color_map, output_dir):
    """Create box plots with individual points overlaid."""
    fig, axes = plt.subplots(1, len(mutation_rates), figsize=(8*len(mutation_rates), 6), sharey=True)
    if len(mutation_rates) == 1:
        axes = [axes]

    for idx, mu in enumerate(mutation_rates):
        mu_subset = df[df['mu'] == mu]

        box_data = [mu_subset[mu_subset['n_copies'] == n]['explored_words'].values for n in n_copies_vals]

        bp = axes[idx].boxplot(box_data,
                               positions=range(len(n_copies_vals)),
                               widths=0.5,
                               patch_artist=True,
                               showfliers=False,
                               medianprops=dict(color='black', linewidth=2),
                               boxprops=dict(linewidth=1.5),
                               whiskerprops=dict(linewidth=1.5),
                               capprops=dict(linewidth=1.5))

        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(color_map[n_copies_vals[i]])
            box.set_alpha(0.7)

        # Add strip plot
        for i, n in enumerate(n_copies_vals):
            y_data = mu_subset[mu_subset['n_copies'] == n]['explored_words'].values
            x_data = np.random.normal(i, 0.04, size=len(y_data))
            axes[idx].scatter(x_data, y_data, alpha=0.3, s=20,
                            color=color_map[n], edgecolors='black', linewidth=0.5)

        axes[idx].set_xticks(range(len(n_copies_vals)))
        axes[idx].set_xticklabels([str(n) for n in n_copies_vals])
        axes[idx].set_xlabel('Number of Gene Copies (n)', fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('Explored Words' if idx == 0 else '', fontsize=14, fontweight='bold')
        axes[idx].set_title(f'Mutation Rate μ = {mu}', fontsize=16, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')
        sns.despine(ax=axes[idx])

    fig.suptitle('Explored Words Distribution: Box Plots with Individual Points',
                 fontsize=18, fontweight='bold', y=1.02)

    plot_path = output_dir / 'explored_words_boxplot_by_mu_and_copies.svg'
    save_plot(fig, plot_path, 'boxplot')
    print(f"Saved: {plot_path.name}")
    return plot_path


def _print_explored_words_summary(df, mutation_rates, n_copies_vals):
    """Print summary statistics for explored words."""
    print("\n" + "="*70)
    print("EXPLORED WORDS SUMMARY STATISTICS")
    print("="*70)
    print(f"\nOverall: Mean={df['explored_words'].mean():.1f}, Median={df['explored_words'].median():.1f}")
    print("\nBy Mutation Rate and Gene Copies:")
    for mu in mutation_rates:
        print(f"\n  μ = {mu}:")
        mu_subset = df[df['mu'] == mu]
        for n in n_copies_vals:
            subset = mu_subset[mu_subset['n_copies'] == n]
            if len(subset) > 0:
                print(f"    n={n}: Mean={subset['explored_words'].mean():.1f}, "
                      f"Median={subset['explored_words'].median():.1f}, "
                      f"Std={subset['explored_words'].std():.1f}")
    print("="*70 + "\n")


# ============================================================================
# CLI INTERFACE
# ============================================================================

class SimulationCLI:
    """CLI for running Gillespie simulation statistics and parameter sweeps."""

    def run(self,
            num_runs: int = 10,
            start: str = "WORD",
            target: str = "GENE",
            n_copies: int = 1,
            mu: float = 0.01,
            s: float = 0.5,
            N: int = 1000,
            birth_rate: float = 1.5,
            death_rate: float = 0.5,
            max_gen: int = 50000,
            max_time: float = 1000.0,
            output: Optional[str] = None,
            quiet: bool = False):
        """
        Run Gillespie simulation statistics collection.

        Examples:
            python sim2_stats.py run --num_runs=10
            python sim2_stats.py run --num_runs=20 --n_copies=3 --mu=0.02
        """
        start, target = start.upper(), target.upper()

        if len(start) != len(target):
            print("ERROR: Start and target words must be the same length!")
            return

        if not quiet:
            print(f"\nRunning {num_runs} Gillespie simulations:")
            print(f"  {start} → {target}")
            print(f"  n_copies={n_copies}, mu={mu}, s={s}, N={N}\n")

        stats = run_batch_simulations(
            num_runs=num_runs,
            start_word=start,
            target_word=target,
            n_copies=n_copies,
            N_carrying_capacity=N,
            mu=mu,
            s=s,
            birth_rate_base=birth_rate,
            death_rate_base=death_rate,
            max_generations=max_gen,
            max_time=max_time,
            verbose=not quiet
        )

        stats.print_summary()

        if output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = f"{start}_to_{target}_n{n_copies}_mu{mu}_s{s}_{timestamp}"

        output_path = RESULTS_DIR / output
        stats.save_all(str(output_path))

        print(f"Results saved to:")
        print(f"  {output_path}.json (detailed data)")
        print(f"  {output_path}.csv (tabular data)")
        print(f"  {output_path}.txt (summary)\n")

    def sweep_grid(self,
                   num_runs: int = 10,
                   start: str = "WORD",
                   target: str = "GENE",
                   n_copies_values: str = "1,2,3,4,5",
                   mu_values: str = "0.01,0.02",
                   s_values: str = "0.3,0.5",
                   N_values: str = "500,1000",
                   output_prefix: str = "grid"):
        """
        Run a full parameter grid search.

        Examples:
            python sim2_stats.py sweep_grid --num_runs=20
            python sim2_stats.py sweep_grid --num_runs=10 --n_copies_values="1,2" --mu_values="0.01,0.02,0.05"
        """
        start, target = start.upper(), target.upper()

        # Parse parameter lists
        n_copies_list = [int(x) for x in (n_copies_values.split(',') if isinstance(n_copies_values, str) else n_copies_values)]
        mu_list = [float(x) for x in (mu_values.split(',') if isinstance(mu_values, str) else mu_values)]
        s_list = [float(x) for x in (s_values.split(',') if isinstance(s_values, str) else s_values)]
        N_list = [int(x) for x in (N_values.split(',') if isinstance(N_values, str) else N_values)]

        total_combinations = len(n_copies_list) * len(mu_list) * len(s_list) * len(N_list)

        print(f"\n{'='*70}")
        print(f"PARAMETER GRID SEARCH")
        print(f"{'='*70}")
        print(f"Evolution: {start} → {target}")
        print(f"Parameters to vary:")
        print(f"  n_copies: {n_copies_list}")
        print(f"  mu: {mu_list}")
        print(f"  s: {s_list}")
        print(f"  N: {N_list}")
        print(f"Total combinations: {total_combinations}")
        print(f"Runs per combination: {num_runs}")
        print(f"Total simulations: {total_combinations * num_runs}")
        print(f"{'='*70}\n")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare parameter combinations
        combinations = []
        combo_num = 0
        for n_copies in n_copies_list:
            for mu in mu_list:
                for s in s_list:
                    for N in N_list:
                        combo_num += 1
                        combinations.append((
                            combo_num, total_combinations, num_runs, start, target,
                            n_copies, mu, s, N, output_prefix, timestamp
                        ))

        # Run in parallel using trio
        print(f"Running {total_combinations} parameter combinations in parallel...\n")
        max_workers = int(os.environ.get('SIM2_MAX_WORKERS', os.cpu_count() or 4))
        print(f"Using {max_workers} parallel workers with trio\n")

        all_results = trio.run(self._run_all_combinations, combinations, max_workers, total_combinations)
        all_results.sort(key=lambda x: (x['n_copies'], x['mu'], x['s'], x['N_carrying_capacity']))

        # Print and save results
        print(f"\n{'='*70}")
        print("GRID SEARCH SUMMARY")
        print(f"{'='*70}\n")

        df = pd.DataFrame(all_results)
        print(df[['n_copies', 'mu', 's', 'N_carrying_capacity',
                  'success_rate', 'avg_generations_success']].to_string(index=False))

        comparison_path = RESULTS_DIR / f"grid_search_{start}_to_{target}_{timestamp}.csv"
        df.to_csv(comparison_path, index=False)
        print(f"\nGrid search results saved to: {comparison_path}")

        # Generate plots
        print("\nGenerating grid search plots...")
        plots = plot_grid_search_results(df, output_dir=RESULTS_DIR,
                                        start=start, target=target, timestamp=timestamp)

        print("\n✓ Plots saved:")
        for plot_type, path in plots.items():
            print(f"  {plot_type}: {path.name}")
        print()

    async def _run_all_combinations(self, combinations, max_workers, total_combinations):
        """Run all parameter combinations in parallel using trio."""
        all_results = []
        results_lock = trio.Lock()
        completed = 0
        concurrency_limit = trio.Semaphore(max_workers)

        async def run_single_combo(combo):
            nonlocal completed
            combo_num = combo[0]
            try:
                async with concurrency_limit:
                    result_dict, stats = await trio.to_thread.run_sync(
                        run_single_parameter_combination, combo
                    )

                    async with results_lock:
                        all_results.append(result_dict)
                        completed += 1
                        print(f"[Main] Progress: {completed}/{total_combinations} combinations completed")
            except Exception as exc:
                async with results_lock:
                    completed += 1
                    print(f"[Main] Combination {combo_num} generated an exception: {exc}")

        async with trio.open_nursery() as nursery:
            for combo in combinations:
                nursery.start_soon(run_single_combo, combo)

        return all_results

    def plot_grid(self,
                  csv_file: Optional[str] = None,
                  start: Optional[str] = None,
                  target: Optional[str] = None,
                  timestamp: Optional[str] = None):
        """
        Plot existing grid search results from a CSV file.

        Examples:
            python sim2_stats.py plot_grid
            python sim2_stats.py plot_grid --csv_file=results/sim2/grid_search_WORD_to_GENE_20240101_120000.csv
        """
        # Find and load CSV file
        csv_path = self._find_csv_file(csv_file)
        if csv_path is None:
            return

        # Extract metadata
        start, target, timestamp = self._extract_metadata(csv_path, start, target, timestamp)

        # Load CSV
        print(f"\nLoading grid search results from: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} parameter combinations")
        except Exception as e:
            print(f"ERROR: Could not load CSV file: {e}")
            return

        # Validate columns
        required_cols = ['n_copies', 'mu', 's', 'N_carrying_capacity']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"ERROR: CSV file missing required columns: {missing}")
            return

        # Generate plots
        print(f"\nGenerating plots for {start} → {target}...")
        plots = plot_grid_search_results(df, output_dir=RESULTS_DIR,
                                        start=start, target=target, timestamp=timestamp)

        print("\n✓ Grid plots saved:")
        for plot_type, path in plots.items():
            print(f"  {plot_type}: {path.name}")

        # Generate explored words plots
        print("\nGenerating explored words distribution plots...")
        json_pattern = f"grid_{start}_to_{target}_*.json"
        explored_plots = plot_explored_words_distributions(
            output_dir=RESULTS_DIR,
            results_pattern=json_pattern
        )

        if explored_plots:
            print("\n✓ Explored words plots saved:")
            for plot_type, path in explored_plots.items():
                print(f"  {plot_type}: {path.name}")

        print()

    def _find_csv_file(self, csv_file: Optional[str]) -> Optional[Path]:
        """Find CSV file to use."""
        if csv_file is None:
            grid_files = sorted(RESULTS_DIR.glob("grid_search_*.csv"))
            if not grid_files:
                print("ERROR: No grid search CSV files found in results/sim2/")
                print("Please specify a CSV file with --csv_file")
                return None
            csv_file = str(grid_files[-1])
            print(f"Using most recent grid search file: {csv_file}")

        csv_path = Path(csv_file)
        if not csv_path.exists():
            print(f"ERROR: CSV file not found: {csv_path}")
            return None

        return csv_path

    def _extract_metadata(self, csv_path: Path, start: Optional[str],
                         target: Optional[str], timestamp: Optional[str]) -> Tuple[str, str, str]:
        """Extract metadata from filename."""
        filename = csv_path.stem

        if start is None or target is None or timestamp is None:
            parts = filename.replace("grid_search_", "").split("_")
            if "to" in parts:
                to_idx = parts.index("to")
                if start is None:
                    start = "_".join(parts[:to_idx]).upper()
                if target is None and len(parts) >= to_idx + 3:
                    target = "_".join(parts[to_idx+1:-2]).upper()
                    if timestamp is None:
                        timestamp = "_".join(parts[-2:])
                else:
                    target = parts[to_idx+1].upper() if to_idx+1 < len(parts) else "GENE"
                    if timestamp is None and to_idx+2 < len(parts):
                        timestamp = "_".join(parts[to_idx+2:])
            else:
                start = start or "WORD"
                target = target or "GENE"
                match = re.search(r'(\d{8}_\d{6})', filename)
                timestamp = match.group(1) if match else datetime.now().strftime("%Y%m%d_%H%M%S")

        return start, target, timestamp

    def plot_explored_words(self, results_pattern: str = "grid_WORD_to_GENE_*.json"):
        """
        Generate publication-quality plots of explored words distributions.

        Examples:
            python sim2_stats.py plot_explored_words
            python sim2_stats.py plot_explored_words --results_pattern="grid_*.json"
        """
        print("\nGenerating explored words distribution plots...")
        print(f"Searching for files matching: {results_pattern}")

        plots = plot_explored_words_distributions(
            output_dir=RESULTS_DIR,
            results_pattern=results_pattern
        )

        if plots:
            print("\n✓ Plots generated successfully:")
            for plot_type, path in plots.items():
                print(f"  {plot_type}: {path.name}")
            print()
        else:
            print("\nNo plots generated. Check that result files exist.")

    def list_results(self):
        """List all saved results in the results directory."""
        results = sorted(RESULTS_DIR.glob("*.json"))

        if not results:
            print("No results found in results/sim2/ directory")
            return

        print(f"\nFound {len(results)} result files:\n")
        for r in results:
            print(f"  {r.name}")
        print()


if __name__ == '__main__':
    fire.Fire(SimulationCLI)
