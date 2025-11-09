"""
Demo script for sim2.py showing Streamlit-ready visualizations.

Run this to see examples of:
1. Single gene evolution with duplication
2. Multi-gene evolution
3. Replicate experiments
4. Comparison across gene copy numbers
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sim2 import *

# Setup plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def demo_single_gene():
    """Demo 1: Single gene evolution with duplication."""
    print("="*70)
    print("DEMO 1: Single gene evolution (WORD → GENE) with 2 copies")
    print("="*70)

    word_graph, _, _ = build_word_graph(word_length=4, max_delta=1)

    gene_config = {'gene1': ('WORD', 2)}
    targets = {'gene1': 'GENE'}

    sim = GillespieSimulation(
        gene_config=gene_config,
        targets=targets,
        word_graph=word_graph,
        N_carrying_capacity=500,
        mu=0.1,
        s=0.3,
        seed=42
    )

    print("\nRunning simulation...")
    final = sim.run(max_generations=20000, stop_at_target=True)

    print(f"\nResults:")
    print(f"  Generation: {final['generation']}")
    print(f"  Time: {final['time']:.2f}")
    print(f"  Target reached: {final['all_targets_reached']}")
    print(f"  Dominant genotype: {final['dominant_genome']}")

    # Get Streamlit-ready DataFrames
    history_df = sim.get_history_df()
    genotypes_df = sim.get_genotype_frequencies_df()
    gene_div_df = sim.get_gene_diversity_df()

    print(f"\n📊 Streamlit-ready outputs:")
    print(f"  history_df: {history_df.shape} - time series data")
    print(f"  genotypes_df: {genotypes_df.shape} - current genotype frequencies")
    print(f"  gene_div_df: {gene_div_df.shape} - per-gene sequence diversity")

    # Example plot: Population size over time
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Population size
    axes[0, 0].plot(history_df['generation'], history_df['population_size'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Population Size')
    axes[0, 0].set_title('Population Dynamics')
    axes[0, 0].axhline(500, color='gray', linestyle='--', alpha=0.5, label='Carrying capacity')
    axes[0, 0].legend()

    # Plot 2: Diversity
    axes[0, 1].plot(history_df['generation'], history_df['diversity'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Number of Genotypes')
    axes[0, 1].set_title('Genotypic Diversity')

    # Plot 3: Dominant frequency
    axes[1, 0].plot(history_df['generation'], history_df['dominant_frequency'], 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Dominant Frequency')
    axes[1, 0].set_title('Clonal Sweeps')

    # Plot 4: Gene at target
    if 'gene1_at_target' in history_df.columns:
        axes[1, 1].fill_between(history_df['generation'], 0, history_df['gene1_at_target'].astype(int),
                                alpha=0.3, label='Gene1 at target')
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].set_ylabel('Target Reached')
    axes[1, 1].set_title('Adaptive Progress')
    axes[1, 1].set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig('results/sim2_demo_single_gene.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to results/sim2_demo_single_gene.png")
    plt.close()

    return sim


def demo_multi_gene():
    """Demo 2: Multiple genes evolving simultaneously."""
    print("\n" + "="*70)
    print("DEMO 2: Multi-gene evolution")
    print("="*70)

    word_graph, _, _ = build_word_graph(word_length=4, max_delta=1)

    # Two genes, each with different copy numbers
    gene_config = {
        'gene1': ('WORD', 2),  # 2 copies
        'gene2': ('COLD', 1)   # 1 copy (essential)
    }
    targets = {
        'gene1': 'GENE',
        'gene2': 'WARM'
    }

    sim = GillespieSimulation(
        gene_config=gene_config,
        targets=targets,
        word_graph=word_graph,
        N_carrying_capacity=500,
        mu=0.1,
        s=0.2,
        seed=123
    )

    print("\nRunning multi-gene simulation...")
    final = sim.run(max_generations=30000, stop_at_target=True)

    print(f"\nResults:")
    print(f"  Generation: {final['generation']}")
    print(f"  All targets reached: {final['all_targets_reached']}")
    print(f"  Gene1 at target: {final['genes_at_target']['gene1']}")
    print(f"  Gene2 at target: {final['genes_at_target']['gene2']}")

    # Show gene-level diversity
    gene_div_df = sim.get_gene_diversity_df()
    print(f"\n📊 Gene sequence diversity:")
    print(gene_div_df[gene_div_df['frequency'] > 0.05].to_string(index=False))

    return sim


def demo_replicates():
    """Demo 3: Replicate experiments (multiple flasks)."""
    print("\n" + "="*70)
    print("DEMO 3: Replicate experiments")
    print("="*70)

    word_graph, _, _ = build_word_graph(word_length=4, max_delta=1)

    gene_config = {'gene1': ('WORD', 2)}
    targets = {'gene1': 'GENE'}

    print("\nRunning 5 replicate populations...")
    sims = run_replicate_experiment(
        gene_config=gene_config,
        targets=targets,
        word_graph=word_graph,
        n_replicates=5,
        sim_init_kwargs={
            'N_carrying_capacity': 500,
            'mu': 0.1,
            's': 0.3
        },
        sim_run_kwargs={
            'max_generations': 20000
        }
    )

    # Compare outcomes across replicates
    print(f"\n📊 Replicate outcomes:")
    for i, sim in enumerate(sims, 1):
        state = sim.get_state()
        status = "✓" if state['all_targets_reached'] else "✗"
        print(f"  Flask {i}: {status} Gen={state['generation']:5d}, Pop={state['population_size']:4d}")

    return sims


def demo_copy_number_comparison():
    """Demo 4: Compare different numbers of gene copies."""
    print("\n" + "="*70)
    print("DEMO 4: Comparing gene copy numbers (1 vs 2 vs 3)")
    print("="*70)

    word_graph, _, _ = build_word_graph(word_length=4, max_delta=1)

    comparison_df = compare_gene_copy_numbers(
        start_word='WORD',
        target_word='GENE',
        word_graph=word_graph,
        copy_numbers=[1, 2, 3],
        n_replicates=10,
        sim_init_kwargs={
            'N_carrying_capacity': 500,
            'mu': 0.1,
            's': 0.3
        },
        sim_run_kwargs={
            'max_generations': 30000
        }
    )

    print("\n📊 Comparison results:")
    print(comparison_df.to_string(index=False))

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Success rate
    axes[0].bar(comparison_df['n_copies'], comparison_df['success_rate'], color='steelblue')
    axes[0].set_xlabel('Number of Gene Copies')
    axes[0].set_ylabel('Success Rate')
    axes[0].set_title('Effect of Gene Duplication on Success')
    axes[0].set_ylim(0, 1.1)
    axes[0].set_xticks(comparison_df['n_copies'])

    # Average generations to success
    axes[1].bar(comparison_df['n_copies'], comparison_df['avg_generations'],
               yerr=comparison_df['std_generations'], color='coral', capsize=5)
    axes[1].set_xlabel('Number of Gene Copies')
    axes[1].set_ylabel('Avg Generations')
    axes[1].set_title('Speed of Adaptation')
    axes[1].set_xticks(comparison_df['n_copies'])

    plt.tight_layout()
    plt.savefig('results/sim2_demo_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to results/sim2_demo_comparison.png")
    plt.close()

    return comparison_df


if __name__ == "__main__":
    # Run all demos
    print("\n🧬 SIM2 DEMONSTRATIONS - Gillespie Birth-Death Simulations\n")

    # Demo 1: Single gene
    sim1 = demo_single_gene()

    # Demo 2: Multiple genes
    sim2 = demo_multi_gene()

    # Demo 3: Replicates
    sims3 = demo_replicates()

    # Demo 4: Comparison
    comparison = demo_copy_number_comparison()

    print("\n" + "="*70)
    print("✓ All demos complete!")
    print("="*70)
    print("\nKey features for Streamlit integration:")
    print("  • Clean DataFrame outputs (.get_history_df(), etc.)")
    print("  • Callback support for live updates")
    print("  • Replicate experiments for statistical power")
    print("  • Ready-made comparison functions")
    print("\nNext steps:")
    print("  • Integrate into existing Streamlit app")
    print("  • Add real-time visualization with st.line_chart()")
    print("  • Create gene space exploration visualizations")
