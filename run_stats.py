#!/usr/bin/env python3
"""
CLI tool for running evolution simulation statistics.

Runs the word evolution simulation multiple times and collects statistics:
- Number of attempts needed to reach target
- Number of steps per attempt
- Success rate
- Time statistics

Designed to be importable for use in the web app later.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import fire
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
from sim1 import build_word_graph, simulate_word_evolution

# Claude-inspired color palette (warm, inviting tones)
CLAUDE_COLORS = {
    'primary': '#D97757',      # Warm coral/orange
    'secondary': '#C75146',    # Deep coral
    'accent1': '#E8A87C',      # Light peach
    'accent2': '#B85C50',      # Rust
    'dark': '#5D4E37',         # Coffee brown
    'gradient': ['#E8A87C', '#D97757', '#C75146', '#B85C50', '#9B4F47']  # Light to dark
}

def setup_publication_style():
    """Configure matplotlib and seaborn for publication-quality figures."""
    # Seaborn talk preset for larger fonts/elements
    sns.set_context("talk", font_scale=1.1)
    sns.set_style("whitegrid", {
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'axes.edgecolor': '.15',
        'axes.linewidth': 1.2,
    })

    # Set CMU Serif font for publication quality
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['CMU Serif', 'Computer Modern', 'DejaVu Serif', 'Times']
    mpl.rcParams['pdf.fonttype'] = 42  # TrueType fonts (editable in Illustrator)
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVG (not paths)
    mpl.rcParams['axes.unicode_minus'] = False  # Use proper minus sign

    # Higher quality settings
    mpl.rcParams['figure.dpi'] = 150
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = 0.1


# Results directory configuration
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


class SimulationStats:
    """Container for simulation run statistics."""

    def __init__(self):
        self.runs = []  # List of run results

    def add_run(self, run_data: Dict):
        """Add a single run's statistics."""
        self.runs.append(run_data)

    def get_summary(self) -> Dict:
        """Get summary statistics across all runs."""
        if not self.runs:
            return {}

        total_attempts = [r['attempts'] for r in self.runs]
        total_steps = [r['total_steps'] for r in self.runs]
        success_rates = [r['success'] for r in self.runs]
        run_times = [r['duration'] for r in self.runs]

        return {
            'total_runs': len(self.runs),
            'successful_runs': sum(success_rates),
            'success_rate': sum(success_rates) / len(self.runs) * 100,
            'avg_attempts_to_success': sum(total_attempts) / len(self.runs),
            'avg_total_steps': sum(total_steps) / len(self.runs),
            'avg_run_time_seconds': sum(run_times) / len(self.runs),
            'min_attempts': min(total_attempts),
            'max_attempts': max(total_attempts),
            'min_steps': min(total_steps),
            'max_steps': max(total_steps),
        }

    def to_json(self, filename: str):
        """Save detailed results to JSON file."""
        data = {
            'runs': self.runs,
            'summary': self.get_summary()
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def to_csv(self, filename: str):
        """Save results to CSV file for easy analysis."""
        # Flatten runs data for CSV
        rows = []
        for run in self.runs:
            row = {
                'run_number': run['run_number'],
                'success': run['success'],
                'attempts': run['attempts'],
                'total_steps': run['total_steps'],
                'duration': run['duration'],
                'avg_steps_per_attempt': run['total_steps'] / run['attempts'],
            }
            # Add parameters
            row.update({f'param_{k}': v for k, v in run['params'].items()})
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)

    def to_summary_file(self, filename: str):
        """Save summary statistics to a text file."""
        summary = self.get_summary()

        with open(filename, 'w') as f:
            f.write("="*60 + "\n")
            f.write("SIMULATION STATISTICS SUMMARY\n")
            f.write("="*60 + "\n")
            f.write(f"Total Runs:              {summary['total_runs']}\n")
            f.write(f"Successful Runs:         {summary['successful_runs']}\n")
            f.write(f"Success Rate:            {summary['success_rate']:.1f}%\n")
            f.write(f"\nAverage Attempts/Run:    {summary['avg_attempts_to_success']:.2f}\n")
            f.write(f"Average Total Steps:     {summary['avg_total_steps']:.1f}\n")
            f.write(f"Average Run Time:        {summary['avg_run_time_seconds']:.2f}s\n")
            f.write(f"\nMin/Max Attempts:        {summary['min_attempts']} / {summary['max_attempts']}\n")
            f.write(f"Min/Max Steps:           {summary['min_steps']} / {summary['max_steps']}\n")
            f.write("="*60 + "\n")

    def save_all(self, base_path: str):
        """Save results in all formats (JSON, CSV, summary)."""
        base = Path(base_path)
        self.to_json(str(base.with_suffix('.json')))
        self.to_csv(str(base.with_suffix('.csv')))
        self.to_summary_file(str(base.with_suffix('.txt')))

    def print_summary(self):
        """Print summary statistics to console."""
        summary = self.get_summary()

        print("\n" + "="*60)
        print("SIMULATION STATISTICS SUMMARY")
        print("="*60)
        print(f"Total Runs:              {summary['total_runs']}")
        print(f"Successful Runs:         {summary['successful_runs']}")
        print(f"Success Rate:            {summary['success_rate']:.1f}%")
        print(f"\nAverage Attempts/Run:    {summary['avg_attempts_to_success']:.2f}")
        print(f"Average Total Steps:     {summary['avg_total_steps']:.1f}")
        print(f"Average Run Time:        {summary['avg_run_time_seconds']:.2f}s")
        print(f"\nMin/Max Attempts:        {summary['min_attempts']} / {summary['max_attempts']}")
        print(f"Min/Max Steps:           {summary['min_steps']} / {summary['max_steps']}")
        print("="*60 + "\n")


def run_until_success(word_graph, N_e: int, start_word: str, target_word: str,
                     n_steps: int = 100, edits_per_step: int = 1,
                     n_copies: int = 1, max_attempts: int = 1000) -> Tuple[bool, Dict]:
    """
    Run simulation until success or max_attempts reached.

    Returns:
        (success, stats_dict) where stats_dict contains:
            - attempts: number of attempts made
            - total_steps: total steps across all attempts
            - steps_per_attempt: list of steps for each attempt
            - success: whether target was reached
    """
    attempts = 0
    total_steps = 0
    steps_per_attempt = []

    while attempts < max_attempts:
        attempts += 1

        # Run simulation
        trajectory = simulate_word_evolution(
            word_graph, N_e, start_word, target_word,
            n_steps=n_steps,
            edits_per_step=edits_per_step,
            n_copies=n_copies
        )

        steps_this_attempt = len(trajectory)
        total_steps += steps_this_attempt
        steps_per_attempt.append(steps_this_attempt)

        # Check if we reached the target
        reached_target = False
        for step_data, visit_density in trajectory:
            for copy_id, current, attempted, accepted, status in step_data:
                if accepted and attempted == target_word:
                    reached_target = True
                    break
            if reached_target:
                break

        if reached_target:
            return True, {
                'attempts': attempts,
                'total_steps': total_steps,
                'steps_per_attempt': steps_per_attempt,
                'success': True
            }

    # Failed to reach target
    return False, {
        'attempts': attempts,
        'total_steps': total_steps,
        'steps_per_attempt': steps_per_attempt,
        'success': False
    }


def run_batch_simulations(num_runs: int, N_e: int, start_word: str, target_word: str,
                         n_steps: int = 100, edits_per_step: int = 1,
                         n_copies: int = 1, max_attempts: int = 1000,
                         verbose: bool = True) -> SimulationStats:
    """
    Run multiple simulation trials and collect statistics.

    Args:
        num_runs: Number of successful runs to complete
        N_e: Effective population size (selection strength)
        start_word: Starting word
        target_word: Target word
        n_steps: Maximum steps per attempt
        edits_per_step: Number of edits per mutation
        n_copies: Number of gene copies
        max_attempts: Maximum attempts per run before giving up
        verbose: Whether to print progress

    Returns:
        SimulationStats object with all results
    """
    # Build word graph once (expensive operation)
    if verbose:
        print(f"Building word graph for {len(start_word)}-letter words...")

    word_graph, valid_words, invalid_words = build_word_graph(
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

    stats = SimulationStats()

    for run_num in range(1, num_runs + 1):
        if verbose:
            print(f"Run {run_num}/{num_runs}...", end=" ", flush=True)

        start_time = time.time()

        success, run_stats = run_until_success(
            word_graph, N_e, start_word, target_word,
            n_steps=n_steps,
            edits_per_step=edits_per_step,
            n_copies=n_copies,
            max_attempts=max_attempts
        )

        duration = time.time() - start_time

        run_data = {
            'run_number': run_num,
            'success': success,
            'attempts': run_stats['attempts'],
            'total_steps': run_stats['total_steps'],
            'steps_per_attempt': run_stats['steps_per_attempt'],
            'duration': duration,
            'params': {
                'N_e': N_e,
                'start_word': start_word,
                'target_word': target_word,
                'n_steps': n_steps,
                'edits_per_step': edits_per_step,
                'n_copies': n_copies,
            }
        }

        stats.add_run(run_data)

        if verbose:
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"{status} - {run_stats['attempts']} attempts, "
                  f"{run_stats['total_steps']} total steps, {duration:.2f}s")

    return stats


def plot_comparison_results(result_files: List[Path], output_dir: Path = RESULTS_DIR):
    """
    Generate 4 publication-quality comparison plots from multiple result files.

    Args:
        result_files: List of paths to JSON result files
        output_dir: Directory to save plots

    Returns:
        Dictionary mapping plot names to file paths
    """
    # Setup publication style
    setup_publication_style()

    # Load all results
    all_data = []
    for filepath in result_files:
        with open(filepath, 'r') as f:
            data = json.load(f)
            n_copies = data['runs'][0]['params']['n_copies']

            # Extract metrics for each run
            for run in data['runs']:
                if run['success']:
                    all_data.append({
                        'n_copies': n_copies,
                        'attempts': run['attempts'],
                        'total_steps': run['total_steps'],
                        'steps_per_success': run['total_steps'] / run['attempts']
                    })

    df = pd.DataFrame(all_data)

    # Group by n_copies
    grouped = df.groupby('n_copies')

    plots_saved = {}

    # Create color palette based on number of conditions
    x = sorted(df['n_copies'].unique())
    n_conditions = len(x)
    if n_conditions <= len(CLAUDE_COLORS['gradient']):
        colors = CLAUDE_COLORS['gradient'][:n_conditions]
    else:
        # Generate more colors if needed
        colors = sns.color_palette("YlOrRd", n_colors=n_conditions)

    # PLOT 1: Line plot - Average attempts/run vs # copies
    fig, ax = plt.subplots(figsize=(10, 7))

    avg_attempts = grouped['attempts'].mean()
    std_attempts = grouped['attempts'].std()

    ax.plot(x, avg_attempts.values, marker='o', linewidth=3, markersize=10,
            color=CLAUDE_COLORS['primary'], markeredgecolor='white', markeredgewidth=2)
    ax.fill_between(x,
                     avg_attempts.values - std_attempts.values,
                     avg_attempts.values + std_attempts.values,
                     alpha=0.25, color=CLAUDE_COLORS['primary'])

    ax.set_xlabel('Number of Gene Copies')
    ax.set_ylabel('Average Attempts per Run')
    ax.set_title('Evolutionary Efficiency: Gene Duplications Reduce Restart Attempts',
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    sns.despine()

    plot1_path = output_dir / 'plot_avg_attempts_vs_copies.svg'
    plt.tight_layout()
    plt.savefig(plot1_path, format='svg')
    plt.close()
    plots_saved['avg_attempts'] = plot1_path

    # PLOT 2: Line plot - Average steps/success vs # copies
    fig, ax = plt.subplots(figsize=(10, 7))

    avg_steps = grouped['steps_per_success'].mean()
    std_steps = grouped['steps_per_success'].std()

    ax.plot(x, avg_steps.values, marker='s', linewidth=3, markersize=10,
            color=CLAUDE_COLORS['secondary'], markeredgecolor='white', markeredgewidth=2)
    ax.fill_between(x,
                     avg_steps.values - std_steps.values,
                     avg_steps.values + std_steps.values,
                     alpha=0.25, color=CLAUDE_COLORS['secondary'])

    ax.set_xlabel('Number of Gene Copies')
    ax.set_ylabel('Average Steps per Successful Attempt')
    ax.set_title('Evolutionary Path Length vs Gene Copy Number',
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    sns.despine()

    plot2_path = output_dir / 'plot_avg_steps_vs_copies.svg'
    plt.tight_layout()
    plt.savefig(plot2_path, format='svg')
    plt.close()
    plots_saved['avg_steps'] = plot2_path

    # PLOT 3: KDE - Distribution of attempts/run
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, (n_copy, color) in enumerate(zip(x, colors)):
        data_subset = df[df['n_copies'] == n_copy]['attempts']
        sns.kdeplot(data=data_subset, ax=ax, linewidth=3, alpha=0.7,
                   label=f'{n_copy} {"copy" if n_copy == 1 else "copies"}',
                   color=color, fill=True)

    ax.set_xlabel('Attempts per Run')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Restart Attempts by Gene Copy Number',
                 fontweight='bold', pad=20)
    ax.legend(title='Gene Copies', frameon=True, fancybox=True, shadow=True)
    sns.despine()

    plot3_path = output_dir / 'plot_attempts_distribution.svg'
    plt.tight_layout()
    plt.savefig(plot3_path, format='svg')
    plt.close()
    plots_saved['attempts_dist'] = plot3_path

    # PLOT 4: KDE - Distribution of steps/success
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, (n_copy, color) in enumerate(zip(x, colors)):
        data_subset = df[df['n_copies'] == n_copy]['steps_per_success']
        sns.kdeplot(data=data_subset, ax=ax, linewidth=3, alpha=0.7,
                   label=f'{n_copy} {"copy" if n_copy == 1 else "copies"}',
                   color=color, fill=True)

    ax.set_xlabel('Steps per Successful Attempt')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Evolutionary Path Lengths by Gene Copy Number',
                 fontweight='bold', pad=20)
    ax.legend(title='Gene Copies', frameon=True, fancybox=True, shadow=True)
    sns.despine()

    plot4_path = output_dir / 'plot_steps_distribution.svg'
    plt.tight_layout()
    plt.savefig(plot4_path, format='svg')
    plt.close()
    plots_saved['steps_dist'] = plot4_path

    return plots_saved


class SimulationCLI:
    """CLI for running word evolution simulation statistics."""

    def run(self,
            num_runs: int = 10,
            start: str = "WORD",
            target: str = "GENE",
            n_e: int = 1000,
            n_steps: int = 100,
            edits_per_step: int = 1,
            n_copies: int = 1,
            max_attempts: int = 1000,
            output: Optional[str] = None,
            quiet: bool = False):
        """
        Run simulation statistics collection.

        Args:
            num_runs: Number of simulation runs to perform
            start: Starting word
            target: Target word
            n_e: Effective population size (selection strength)
            n_steps: Maximum steps per attempt
            edits_per_step: Number of edits per mutation
            n_copies: Number of gene copies (1=essential gene, >1=duplications)
            max_attempts: Maximum attempts per run before giving up
            output: Output file basename (will create .json, .csv, .txt). If None, auto-generated.
            quiet: Suppress progress output

        Examples:
            python run_stats.py run --num_runs=10
            python run_stats.py run --num_runs=20 --n_copies=3
            python run_stats.py run --num_runs=50 --n_copies=1 --output=essential_gene
        """
        start = start.upper()
        target = target.upper()

        # Validate word lengths
        if len(start) != len(target):
            print("ERROR: Start and target words must be the same length!")
            return

        if not quiet:
            print(f"\nRunning {num_runs} simulations:")
            print(f"  {start} → {target}")
            print(f"  N_e={n_e}, n_copies={n_copies}, edits_per_step={edits_per_step}\n")

        # Run simulations
        stats = run_batch_simulations(
            num_runs=num_runs,
            N_e=n_e,
            start_word=start,
            target_word=target,
            n_steps=n_steps,
            edits_per_step=edits_per_step,
            n_copies=n_copies,
            max_attempts=max_attempts,
            verbose=not quiet
        )

        # Print summary
        stats.print_summary()

        # Save results
        if output is None:
            # Auto-generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = f"{start}_to_{target}_copies{n_copies}_{timestamp}"

        output_path = RESULTS_DIR / output
        stats.save_all(str(output_path))

        print(f"Results saved to:")
        print(f"  {output_path}.json (detailed data)")
        print(f"  {output_path}.csv (tabular data)")
        print(f"  {output_path}.txt (summary)\n")

        return stats

    def compare(self,
                num_runs: int = 20,
                start: str = "WORD",
                target: str = "GENE",
                n_e: int = 1000,
                n_steps: int = 100,
                edits_per_step: int = 1,
                max_copies: int = 5):
        """
        Compare simulation performance across different numbers of gene copies.

        This is useful for demonstrating the advantage of gene duplications!

        Args:
            num_runs: Number of runs per configuration
            start: Starting word
            target: Target word
            n_e: Effective population size
            n_steps: Maximum steps per attempt
            edits_per_step: Number of edits per mutation
            max_copies: Maximum number of copies to test (will test 1 to max_copies)

        Examples:
            python run_stats.py compare --num_runs=50
            python run_stats.py compare --num_runs=100 --max_copies=3
        """
        start = start.upper()
        target = target.upper()

        print(f"\n{'='*70}")
        print(f"COMPARING GENE COPIES: 1 to {max_copies}")
        print(f"{'='*70}")
        print(f"Evolution: {start} → {target}")
        print(f"Runs per configuration: {num_runs}")
        print(f"{'='*70}\n")

        all_results = []

        for n_copies in range(1, max_copies + 1):
            print(f"\n{'─'*70}")
            print(f"Testing with {n_copies} gene copy/copies...")
            print(f"{'─'*70}")

            stats = run_batch_simulations(
                num_runs=num_runs,
                N_e=n_e,
                start_word=start,
                target_word=target,
                n_steps=n_steps,
                edits_per_step=edits_per_step,
                n_copies=n_copies,
                max_attempts=1000,
                verbose=True
            )

            summary = stats.get_summary()
            all_results.append({
                'n_copies': n_copies,
                **summary
            })

            # Save individual results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = RESULTS_DIR / f"compare_{start}_to_{target}_copies{n_copies}_{timestamp}"
            stats.save_all(str(output_path))

        # Create comparison table
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}\n")

        df = pd.DataFrame(all_results)
        print(df.to_string(index=False))

        # Save comparison results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_path = RESULTS_DIR / f"comparison_{start}_to_{target}_{timestamp}.csv"
        df.to_csv(comparison_path, index=False)

        print(f"\nComparison saved to: {comparison_path}")

        # Generate plots automatically
        print("\nGenerating comparison plots...")
        result_files = [RESULTS_DIR / f"compare_{start}_to_{target}_copies{i}_{timestamp}.json"
                       for i in range(1, max_copies + 1)]

        plots = plot_comparison_results(result_files, RESULTS_DIR)

        print("\n✓ Plots saved:")
        for plot_type, path in plots.items():
            print(f"  {plot_type}: {path.name}")
        print()

        return df

    def list_results(self):
        """List all saved results in the results directory."""
        results = sorted(RESULTS_DIR.glob("*.json"))

        if not results:
            print("No results found in results/ directory")
            return

        print(f"\nFound {len(results)} result files:\n")
        for r in results:
            print(f"  {r.name}")
        print()

    def plot(self, *result_files: str, pattern: str = None):
        """
        Generate comparison plots from result files.

        Creates 4 plots:
        1. Line plot: Average attempts/run vs # gene copies
        2. Line plot: Average steps/success vs # gene copies
        3. Histogram: Distribution of attempts/run (color per # copies)
        4. Histogram: Distribution of steps/success (color per # copies)

        Args:
            result_files: Paths to JSON result files (relative to results/ or absolute)
            pattern: Glob pattern to match files (e.g., "compare_*.json")

        Examples:
            # Plot specific files
            python run_stats.py plot results/file1.json results/file2.json

            # Plot all comparison files
            python run_stats.py plot --pattern="compare_*.json"

            # Plot most recent comparison
            python run_stats.py plot --pattern="comparison_*.csv"
        """
        # Determine which files to plot
        files_to_plot = []

        if pattern:
            # Use glob pattern
            files_to_plot = sorted(RESULTS_DIR.glob(pattern))
            if not files_to_plot:
                print(f"No files found matching pattern: {pattern}")
                return
        elif result_files:
            # Use specified files
            for f in result_files:
                path = Path(f)
                if not path.is_absolute():
                    path = RESULTS_DIR / path
                if not path.exists():
                    # Try adding .json extension
                    path = path.with_suffix('.json')
                if path.exists():
                    files_to_plot.append(path)
                else:
                    print(f"Warning: File not found: {f}")
        else:
            # Auto-detect: find files with different n_copies
            all_results = sorted(RESULTS_DIR.glob("*.json"))

            # Group by start/target, find sets with different n_copies
            by_params = {}
            for filepath in all_results:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    if data['runs']:
                        params = data['runs'][0]['params']
                        key = (params['start_word'], params['target_word'],
                               params['N_e'], params['edits_per_step'])
                        if key not in by_params:
                            by_params[key] = []
                        by_params[key].append((filepath, params['n_copies']))

            # Find the most recent set with multiple n_copies
            for key, file_list in sorted(by_params.items(), reverse=True):
                if len(file_list) > 1:
                    files_to_plot = [f[0] for f in sorted(file_list, key=lambda x: x[1])]
                    print(f"Auto-detected comparison: {key[0]} → {key[1]}")
                    break

            if not files_to_plot:
                print("No comparison sets found. Please specify files or use --pattern")
                print("\nAvailable files:")
                self.list_results()
                return

        # Filter to only .json files
        json_files = [f for f in files_to_plot if f.suffix == '.json']

        if not json_files:
            print("No JSON result files found!")
            return

        print(f"\nGenerating plots from {len(json_files)} result files...")
        for f in json_files:
            print(f"  - {f.name}")

        # Generate plots
        plots = plot_comparison_results(json_files, RESULTS_DIR)

        print("\n✓ Plots saved:")
        for plot_type, path in plots.items():
            print(f"  {plot_type}: {path.name}")
        print()

        return plots


if __name__ == '__main__':
    fire.Fire(SimulationCLI)
