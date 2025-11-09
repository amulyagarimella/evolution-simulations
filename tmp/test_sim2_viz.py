"""
Quick test of sim2 visualization functions.
"""

import matplotlib.pyplot as plt
from sim1 import build_word_graph
from sim2 import GillespieSimulation
from sim2_viz import compute_graph_layout, create_population_figure

print("🧬 Testing sim2 visualization...")

# 1. Build word graph
print("\n1. Building word graph...")
word_graph, _, _ = build_word_graph(word_length=4)
print(f"   ✓ Graph has {word_graph.number_of_nodes()} nodes")

# 2. Compute layout
print("\n2. Computing layout...")
pos = compute_graph_layout(word_graph)
print(f"   ✓ Layout computed for {len(pos)} nodes")

# 3. Run quick simulation
print("\n3. Running short simulation (COLD → WARM)...")
gene_config = {'gene1': ('COLD', 2)}
targets = {'gene1': 'WARM'}

sim = GillespieSimulation(
    gene_config=gene_config,
    targets=targets,
    word_graph=word_graph,
    N_carrying_capacity=300,
    mu=0.15,
    s=0.3,
    seed=123
)

final = sim.run(max_generations=5000, stop_at_target=True)
print(f"   ✓ Simulation complete: Gen {final['generation']}, Pop {final['population_size']}")
print(f"   Target reached: {final['all_targets_reached']}")

# 4. Test visualization at different time points
print("\n4. Creating visualizations...")

# Initial state
snapshot_start = sim.get_snapshot_at_generation(0)
fig1 = create_population_figure(word_graph, snapshot_start, pos, title="Initial Population")
plt.savefig('results/test_viz_start.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: results/test_viz_start.png")
plt.close(fig1)

# Middle state
mid_idx = len(sim.history) // 2
snapshot_mid = sim.get_snapshot_at_generation(mid_idx)
fig2 = create_population_figure(word_graph, snapshot_mid, pos, title=f"Mid-simulation")
plt.savefig('results/test_viz_mid.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: results/test_viz_mid.png")
plt.close(fig2)

# Final state
snapshot_end = sim.get_snapshot_at_generation(len(sim.history) - 1)
fig3 = create_population_figure(word_graph, snapshot_end, pos, title="Final Population")
plt.savefig('results/test_viz_end.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: results/test_viz_end.png")
plt.close(fig3)

# 5. Test helper methods
print("\n5. Testing helper methods...")
word_freq = sim.get_word_frequencies()
print(f"   ✓ Word frequencies: {len(word_freq)} unique words")
print(f"     Top 5: {list(word_freq.items())[:5]}")

cooccur = sim.get_genome_cooccurrence()
print(f"   ✓ Cooccurrences: {len(cooccur)} pairs")
if cooccur:
    print(f"     Sample: {cooccur[0]}")

snapshot = sim.get_population_snapshot()
print(f"   ✓ Population snapshot: {len(snapshot['word_frequencies'])} words active")

print("\n✅ All tests passed! Visualization system working.")
print("\nTo see the interactive app, run:")
print("   streamlit run sim2_app.py")
