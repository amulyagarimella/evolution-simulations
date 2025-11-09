# Sim2 Visualization Guide

## ✅ Complete Implementation

All components are built and tested! The sim2 visualization system "rhymes" with sim1's approach while showing population dynamics.

## 🎨 What's Been Built

### 1. **sim2.py** - Enhanced with Visualization Methods
- `get_word_frequencies(gene_id)` - Get word counts in population
- `get_genome_cooccurrence(gene_id)` - Get word pairs that coexist in genomes
- `get_population_snapshot(gene_id)` - Complete state for visualization
- `get_snapshot_at_generation(idx, gene_id)` - Time-scrubbing support
- Enhanced `record_state()` - Stores full genotype distribution in history

### 2. **sim2_viz.py** - Visualization Functions
- `compute_graph_layout()` - Reuses sim1's NetworkX layout
- `draw_population_network()` - Main visualization function
  - Heat map style: node size = frequency
  - Shows all copies for multi-copy genomes
  - Orange dashed lines = cooccurring words
  - Green = target, blue intensity = abundance
- `create_population_figure()` - Generate complete figure
- `create_side_by_side_comparison()` - Compare multiple states
- Multiple color schemes available

### 3. **sim2_app.py** - Streamlit Interactive App
- **Sidebar controls**:
  - Gene configuration (start, target, # copies)
  - Population parameters (capacity, mutation rate, selection)
  - Simulation settings
- **Main visualization**:
  - Network graph with population heat map
  - Time controls (play, pause, next, prev, slider)
  - Generation scrubbing
- **Statistics panels**:
  - Real-time metrics (generation, pop size, diversity, target status)
  - Word frequency table
  - Time series plots (population, diversity, clonal sweeps, progress)

### 4. **test_sim2_viz.py** - Test Suite
Validates all components work correctly.

## 🚀 How to Run

### Interactive Streamlit App
```bash
streamlit run sim2_app.py
```

Then:
1. Configure parameters in sidebar
2. Click "🚀 Run Simulation"
3. Watch the simulation run
4. Use slider to scrub through time
5. Explore statistics and plots

### Command Line Test
```bash
python test_sim2_viz.py
```

Generates three visualization snapshots:
- `results/test_viz_start.png` - Initial population
- `results/test_viz_mid.png` - Mid-simulation
- `results/test_viz_end.png` - Final state

## 📊 Visualization Features

### "Rhyming" with sim1
✅ Same word graph network layout
✅ NetworkX spring/Kamada-Kawai positioning
✅ Color-coded nodes (unexplored, active, target)
✅ Animation over time
✅ Clear visual progression

### Population-Specific Enhancements
✨ **Node size** = frequency (heat map effect)
✨ **Multiple active nodes** = population cloud
✨ **Orange dashed edges** = cooccurring words (multi-copy genomes)
✨ **Blue intensity gradient** = relative abundance
✨ **Time scrubbing** with slider

## 🎯 Example Use Cases

### 1. Basic Evolution
```python
from sim1 import build_word_graph
from sim2 import GillespieSimulation
from sim2_viz import compute_graph_layout, create_population_figure

# Setup
word_graph, _, _ = build_word_graph(word_length=4)
pos = compute_graph_layout(word_graph)

# Simulate
sim = GillespieSimulation(
    gene_config={'gene1': ('COLD', 2)},
    targets={'gene1': 'WARM'},
    word_graph=word_graph,
    mu=0.1, s=0.2
)
sim.run(max_generations=10000)

# Visualize
snapshot = sim.get_population_snapshot()
fig = create_population_figure(word_graph, snapshot, pos)
plt.show()
```

### 2. Time-lapse
```python
# Get snapshots at different time points
start = sim.get_snapshot_at_generation(0)
mid = sim.get_snapshot_at_generation(len(sim.history)//2)
end = sim.get_snapshot_at_generation(len(sim.history)-1)

# Create comparison
from sim2_viz import create_side_by_side_comparison
fig = create_side_by_side_comparison(
    word_graph,
    [start, mid, end],
    pos,
    titles=['Start', 'Middle', 'End']
)
```

### 3. Multi-Copy Visualization
When genomes have multiple copies (e.g., ['COLD', 'BOLD']), the visualization:
- Shows both COLD and BOLD as active nodes
- Draws orange dashed edge between them
- Node sizes reflect total copies across population
- Demonstrates gene redundancy visually

## 📈 What the Visualization Shows

### Network Layout
- **Background (light gray)**: Unexplored words in dictionary
- **Active nodes (blue)**: Words present in current population
  - Size ∝ number of individuals with this word
  - Intensity ∝ relative frequency
- **Target nodes (green)**: Goal words
- **Cooccurrence edges (orange dashed)**: Words in same genome

### Temporal Evolution
- Watch "clouds" of population spread through gene space
- See clonal sweeps (one node grows large)
- Observe genetic diversity (many nodes vs few)
- Track progress toward target

### Multi-Copy Dynamics
- See both copies of duplicated genes
- Visualize exploration: one copy safe, one exploring
- Orange connections show which words coexist

## 🔍 Interpreting the Visualization

**Dense cloud**: High genetic diversity, many variants coexisting
**Single large node**: Clonal sweep, one genotype dominates
**Spreading cloud**: Population exploring gene space
**Converging cloud**: Selection driving population toward target
**Orange web**: Complex multi-copy genome structure
**Path to green**: Successful evolutionary trajectory

## 🎨 Customization

The visualization is highly customizable:

```python
# Different color schemes
from sim2_viz import set_color_scheme
colors = set_color_scheme('heatmap')  # or 'minimal'

# Custom node sizing
draw_population_network(
    ...,
    node_size_scale=2000,  # Larger nodes
    show_cooccurrence=False,  # Hide orange edges
    show_labels=True  # Show word labels
)
```

## 📁 Files Generated

From test run:
- ✅ `results/test_viz_start.png` - Initial state
- ✅ `results/test_viz_mid.png` - Mid-simulation
- ✅ `results/test_viz_end.png` - Final state

## 🐛 Troubleshooting

**Streamlit not installed:**
```bash
pip install streamlit
```

**Visualization too slow:**
- Reduce carrying capacity (fewer genotypes)
- Increase `frame_interval` in animation
- Use 'kamada_kawai' layout for large graphs

**Can't see all words:**
- Set `show_labels=True` (but cluttered for >20 words)
- Use expander "Word Frequencies" table in app
- Zoom in on specific regions

## 🎓 Next Steps

Now that visualization is working:

1. **Experiment with parameters** in Streamlit app
   - Compare 1 vs 2 vs 3 gene copies
   - Try different mutation rates
   - Observe different evolutionary trajectories

2. **Create publication figures**
   - Use `create_population_figure()` for static images
   - Export high-res PNG/SVG for papers
   - Side-by-side comparisons

3. **Gene space analysis (Phase 3)**
   - Network connectivity metrics
   - Accessibility analysis
   - 3D exploration viewer

4. **Rust optimization (if needed)**
   - Port hot paths to Rust
   - Parallelize replicate experiments
   - Real-time animation with larger populations

## 💡 Key Insights Visible

The visualization makes these concepts graspable:

- **Gene duplication enables exploration**: See population spread when copies exist
- **Clonal interference**: Multiple beneficial mutations competing
- **Genetic drift**: Random fluctuations in small populations
- **Population bottlenecks**: Sudden diversity collapses
- **Adaptive landscapes**: Paths through gene space toward fitness peaks

Enjoy exploring evolutionary dynamics! 🧬
