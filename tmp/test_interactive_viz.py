"""
Test the new interactive Plotly visualization.
"""

from sim1 import build_word_graph
from sim2 import GillespieSimulation
from sim2_viz import compute_graph_layout, create_interactive_population_figure

print("🧬 Testing interactive visualization...")

# 1. Build word graph
print("\n1. Building word graph...")
word_graph, _, _ = build_word_graph(word_length=4)
print(f"   ✓ Graph has {word_graph.number_of_nodes()} nodes")

# 2. Compute layout
print("\n2. Computing layout...")
pos = compute_graph_layout(word_graph)
print(f"   ✓ Layout computed")

# 3. Run quick simulation
print("\n3. Running simulation (COLD → WARM)...")
gene_config = {'gene1': ('COLD', 2)}
targets = {'gene1': 'WARM'}

sim = GillespieSimulation(
    gene_config=gene_config,
    targets=targets,
    word_graph=word_graph,
    N_carrying_capacity=300,
    mu=0.15,
    s=0.3,
    seed=42
)

final = sim.run(max_generations=3000, stop_at_target=True)
print(f"   ✓ Simulation complete: Gen {final['generation']}, Pop {final['population_size']}")

# 4. Create interactive visualization
print("\n4. Creating interactive Plotly figure...")
snapshot = sim.get_population_snapshot()
fig = create_interactive_population_figure(word_graph, snapshot, pos)

# Save as HTML
html_file = 'results/test_interactive_viz.html'
fig.write_html(html_file)
print(f"   ✓ Saved interactive plot to {html_file}")
print(f"   Open in browser to test hover tooltips!")

# Test at different time points
print("\n5. Testing snapshots at different generations...")
for i, gen_name in [(0, 'start'), (len(sim.history)//2, 'mid'), (len(sim.history)-1, 'end')]:
    snapshot = sim.get_snapshot_at_generation(i)
    fig = create_interactive_population_figure(word_graph, snapshot, pos)
    html_file = f'results/test_interactive_viz_{gen_name}.html'
    fig.write_html(html_file)
    print(f"   ✓ Saved {html_file}")

print("\n✅ Interactive visualization test complete!")
print("\nFeatures to test in browser:")
print("  • Hover over nodes to see word labels and fitness")
print("  • Target word shown as green star (🎯)")
print("  • All nodes colored by fitness (distance to target)")
print("  • Node size shows frequency in population")
print("  • Orange dashed lines show gene copies in same genome")
print("  • Zoom and pan controls in top right")
