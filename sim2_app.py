"""
Streamlit app for visualizing sim2 population evolution.

Interactive visualization of birth-death bacterial evolution on word graphs.
Includes all visualization functions for a clean, unified codebase.
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
from sklearn.manifold import MDS

# Set matplotlib style for better aesthetics
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('default')
        # Manually set some style parameters
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3

from sim1 import build_word_graph
from sim2 import GillespieSimulation, genome_fitness


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def compute_graph_layout(word_graph: nx.Graph, layout_type: str = 'auto',
                         seed: int = 42) -> Dict[str, Tuple[float, float]]:
    """
    Compute 2D layout for word graph based on edit distance.

    Uses edit distance (shortest path length in word graph) to position nodes
    such that words with similar sequences are close together in 2D space.

    Args:
        word_graph: NetworkX graph of words
        layout_type: 'edit_distance', 'spring', 'kamada_kawai', or 'auto'
        seed: Random seed for reproducible layouts

    Returns:
        Dictionary mapping word -> (x, y) position
    """
    print("Computing graph layout...")

    if layout_type == 'auto':
        # Use same logic as app.py: kamada_kawai for large graphs, spring for smaller
        n_nodes = len(word_graph.nodes())
        if n_nodes > 1000:
            layout_type = 'kamada_kawai'
        else:
            layout_type = 'spring'

    if layout_type == 'edit_distance':
        print(f"Using edit distance-based layout (MDS) for {len(word_graph.nodes())} nodes")
        pos = _compute_edit_distance_layout(word_graph, seed=seed)
    elif layout_type == 'kamada_kawai':
        print(f"Using Kamada-Kawai layout for {len(word_graph.nodes())} nodes")
        pos = nx.kamada_kawai_layout(word_graph)
    else:  # spring
        print(f"Using spring layout for {len(word_graph.nodes())} nodes")
        # Use EXACT same parameters as app.py
        pos = nx.spring_layout(word_graph, k=0.5, iterations=50, seed=seed)

    return pos


def _compute_edit_distance_layout(word_graph: nx.Graph, seed: int = 42) -> Dict[str, Tuple[float, float]]:
    """
    Compute 2D layout using edit distance (shortest path length) via MDS.

    Filters out unreachable words (isolated nodes or disconnected components).

    Args:
        word_graph: NetworkX graph of words
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping word -> (x, y) position (only for reachable words)
    """
    # Find connected components and filter to largest component
    connected_components = list(nx.connected_components(word_graph))

    if not connected_components:
        return {}

    # Use the largest connected component
    largest_component = max(connected_components, key=len)
    n_removed = word_graph.number_of_nodes() - len(largest_component)

    if n_removed > 0:
        print(f"Removing {n_removed} unreachable word(s) from {len(connected_components)} disconnected component(s)")

    # Create subgraph with only reachable words
    reachable_graph = word_graph.subgraph(largest_component).copy()
    words = list(reachable_graph.nodes())
    n_words = len(words)

    if n_words == 0:
        return {}

    if n_words == 1:
        return {words[0]: (0.0, 0.0)}

    print(f"Computing edit distance matrix for {n_words} reachable words...")

    # Compute distance matrix using shortest path lengths (edit distance)
    distance_matrix = np.zeros((n_words, n_words))
    all_pairs_dist = dict(nx.all_pairs_shortest_path_length(reachable_graph))

    for i, word1 in enumerate(words):
        dists_from_word1 = all_pairs_dist.get(word1, {})
        for j, word2 in enumerate(words):
            if i == j:
                distance_matrix[i, j] = 0.0
            else:
                dist = dists_from_word1.get(word2, 10.0)
                distance_matrix[i, j] = dist

    print("Embedding into 2D space using MDS...")

    # Use MDS to embed in 2D space
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=seed,
              max_iter=300, n_init=10, normalized_stress='auto')

    embedding = mds.fit_transform(distance_matrix)

    # Normalize to reasonable scale - use MUCH larger scale to spread nodes out and prevent overlap
    embedding = embedding - embedding.mean(axis=0)
    scale = np.abs(embedding).max()
    if scale > 0:
        # Scale based on number of nodes to ensure adequate spacing
        # More nodes need MUCH more space to prevent overlap
        # Use very aggressive scaling - aim for at least 2-3 pixels between smallest nodes
        # Smallest nodes are ~5 pixels, so we want spacing of at least 7-8 pixels minimum
        # For n nodes in area A, average spacing is roughly sqrt(A/n)
        # We want spacing > 8, so A > 64*n, which means radius > 8*sqrt(n/pi) ~ 4.5*sqrt(n)
        base_scale = 15.0  # Much larger base
        # Scale factor increases aggressively with node count
        # For 100 nodes: factor ~2, for 1000 nodes: factor ~6.3
        node_factor = np.sqrt(n_words / 10.0) if n_words > 10 else 1.0
        final_scale = base_scale * max(1.0, node_factor)
        embedding = embedding / scale * final_scale

    # Create position dictionary
    pos = {word: (float(embedding[i, 0]), float(embedding[i, 1]))
           for i, word in enumerate(words)}

    print(f"Layout computed with stress={mds.stress_:.4f}")

    return pos


def create_interactive_population_figure(word_graph: nx.Graph,
                                         snapshot: Dict,
                                         pos: Dict[str, Tuple[float, float]],
                                         title: str = "",
                                         node_size_scale: float = 20.0,
                                         sim: Optional[GillespieSimulation] = None,
                                         current_generation_idx: Optional[int] = None) -> go.Figure:
    """
    Create interactive Plotly visualization with hover tooltips and fitness coloring.

    Args:
        word_graph: NetworkX word graph
        snapshot: Population snapshot from get_population_snapshot()
        pos: Node positions from compute_graph_layout()
        title: Plot title
        node_size_scale: Scaling factor for node sizes
        sim: Optional GillespieSimulation object
        current_generation_idx: Optional current generation index

    Returns:
        Plotly Figure object
    """
    word_frequencies = snapshot['word_frequencies']
    targets = snapshot['targets']
    state = snapshot['state']

    # Get all words
    all_words = set(word_graph.nodes())
    active_words = set(word_frequencies.keys())
    target_words = set(targets.values())

    # Find explored but inactive words
    explored_inactive_words = set()
    if sim is not None and current_generation_idx is not None:
        for i in range(current_generation_idx):
            if i < len(sim.history):
                prev_snapshot = sim.get_snapshot_at_generation(i)
                if prev_snapshot:
                    prev_freqs = prev_snapshot.get('word_frequencies', {})
                    if prev_freqs:
                        explored_inactive_words.update(set(prev_freqs.keys()))

        explored_inactive_words = explored_inactive_words - active_words - target_words

    # Compute fitness for all words
    target_word = list(targets.values())[0] if targets else None
    word_fitnesses = {}
    if target_word:
        for word in all_words:
            if word == target_word:
                word_fitnesses[word] = 1.0
            else:
                try:
                    dist = nx.shortest_path_length(word_graph, word, target_word)
                    max_dist = 10
                    word_fitnesses[word] = max(0.0, 1.0 - (dist / max_dist))
                except nx.NetworkXNoPath:
                    word_fitnesses[word] = 0.0
    else:
        word_fitnesses = {word: 0.5 for word in all_words}

    # Create node traces
    # Background nodes (unexplored)
    bg_words = all_words - active_words - target_words - explored_inactive_words
    bg_x, bg_y, bg_text, bg_fitness = [], [], [], []

    for word in bg_words:
        if word in pos:
            x, y = pos[word]
            bg_x.append(x)
            bg_y.append(y)
            bg_text.append(f"{word}<br>Fitness: {word_fitnesses.get(word, 0):.2f}<br>(not visited)")
            bg_fitness.append(word_fitnesses.get(word, 0))

    bg_trace = go.Scatter(
        x=bg_x, y=bg_y,
        mode='markers',
        hoverinfo='text',
        text=bg_text,
        marker=dict(
            size=3,  # Much smaller to reduce overlap
            color=bg_fitness,
            colorscale='Greys',
            opacity=0.3,  # More visible since smaller
            line=dict(width=0)
        ),
        name='Unexplored words',
        showlegend=True
    )

    # Explored but inactive nodes
    explored_x, explored_y, explored_text, explored_sizes, explored_fitness = [], [], [], [], []

    if explored_inactive_words:
        max_historical_freq = 1
        if sim is not None and current_generation_idx is not None:
            for i in range(current_generation_idx):
                if i < len(sim.history):
                    prev_snapshot = sim.get_snapshot_at_generation(i)
                    if prev_snapshot:
                        prev_freqs = prev_snapshot.get('word_frequencies', {})
                        if prev_freqs:
                            max_historical_freq = max(max_historical_freq, max(prev_freqs.values()))

        for word in explored_inactive_words:
            if word in pos:
                x, y = pos[word]
                fitness = word_fitnesses.get(word, 0)

                # Find last frequency
                last_freq = 0
                if sim is not None and current_generation_idx is not None:
                    for i in range(min(current_generation_idx, len(sim.history)) - 1, -1, -1):
                        prev_snapshot = sim.get_snapshot_at_generation(i)
                        if prev_snapshot:
                            prev_freqs = prev_snapshot.get('word_frequencies', {})
                            if prev_freqs and word in prev_freqs:
                                last_freq = prev_freqs[word]
                                break

                explored_x.append(x)
                explored_y.append(y)
                explored_text.append(f"<b>{word}</b><br>Visits: {last_freq}<br>Fitness: {fitness:.2f}<br>(explored but inactive)")
                # Much smaller sizes to prevent overlap: 4 + (density/max) * 6 (range: 4-10)
                size = (last_freq / max_historical_freq) * 6 + 4 if max_historical_freq > 0 else 4
                explored_sizes.append(max(size, 4))
                explored_fitness.append(fitness)

    # Custom fitness colorscale: Red (low) -> Yellow (medium) -> Green (high)
    fitness_colorscale = [[0, '#d73027'], [0.5, '#fee08b'], [1, '#1a9850']]
    
    explored_trace = go.Scatter(
        x=explored_x, y=explored_y,
        mode='markers',
        hoverinfo='text',
        text=explored_text,
        marker=dict(
            size=explored_sizes,
            color=explored_fitness,
            colorscale=fitness_colorscale,
            cmin=0,
            cmax=1,
            opacity=0.8,  # More visible since smaller
            line=dict(width=0.5, color='white'),
            colorbar=dict(
                title="Fitness",
                thickness=15,
                len=0.5,
                y=0.25
            )
        ),
        name='Explored but inactive',
        showlegend=True
    )

    # Build edges from mutation history FIRST to identify source nodes
    edge_x, edge_y = [], []
    edge_list = []  # Store (x0, y0, x1, y1) tuples for annotations
    source_nodes = set()  # Track nodes that are sources of edges
    
    # Collect edges from simulation history (mutations between generations)
    if sim is not None and current_generation_idx is not None and current_generation_idx > 0:
        # Get previous generation's active words
        prev_snapshot = sim.get_snapshot_at_generation(current_generation_idx - 1)
        if prev_snapshot:
            prev_freqs = prev_snapshot.get('word_frequencies', {})
            prev_words = set(prev_freqs.keys())
            current_words = set(word_frequencies.keys())
            
            # Find words that appeared in current generation (mutations from previous)
            new_words = current_words - prev_words
            
            # For each new word, try to find a likely parent (closest word in previous generation)
            for new_word in new_words:
                if new_word in pos:
                    # Find closest word in previous generation (by edit distance in graph)
                    best_parent = None
                    min_dist = float('inf')
                    for prev_word in prev_words:
                        if prev_word in pos:
                            try:
                                dist = nx.shortest_path_length(word_graph, prev_word, new_word)
                                if dist < min_dist:
                                    min_dist = dist
                                    best_parent = prev_word
                            except nx.NetworkXNoPath:
                                continue
                    
                    if best_parent and best_parent in pos and min_dist <= 2:  # Only show edges for close mutations
                        x0, y0 = pos[best_parent]
                        x1, y1 = pos[new_word]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                        edge_list.append((x0, y0, x1, y1))
                        source_nodes.add(best_parent)  # Track source nodes
            
            # Also show edges for direct mutations (edit distance 1) between words in consecutive generations
            # This captures mutations even when exploring already-explored sites
            for current_word in current_words:
                if current_word in pos:
                    # Check if this word increased in frequency (likely from mutations)
                    freq_increase = word_frequencies.get(current_word, 0) - prev_freqs.get(current_word, 0)
                    
                    # Find words in previous generation that are direct neighbors (edit distance 1)
                    for prev_word in prev_words:
                        if prev_word in pos and prev_word != current_word:
                            try:
                                dist = nx.shortest_path_length(word_graph, prev_word, current_word)
                                if dist == 1:  # Direct mutation (single letter change)
                                    # Show edge if:
                                    # 1. Current word is new, OR
                                    # 2. Current word increased significantly, OR
                                    # 3. Previous word decreased significantly (population moved away)
                                    prev_freq = prev_freqs.get(prev_word, 0)
                                    current_freq = word_frequencies.get(current_word, 0)
                                    prev_decrease = prev_freq - word_frequencies.get(prev_word, prev_freq)
                                    
                                    if (current_word in new_words or 
                                        freq_increase > 5 or 
                                        (prev_decrease > 5 and current_freq > 0)):
                                        x0, y0 = pos[prev_word]
                                        x1, y1 = pos[current_word]
                                        # Avoid duplicate edges
                                        if (x0, y0, x1, y1) not in edge_list and (x1, y1, x0, y0) not in edge_list:
                                            edge_x.extend([x0, x1, None])
                                            edge_y.extend([y0, y1, None])
                                            edge_list.append((x0, y0, x1, y1))
                                            source_nodes.add(prev_word)  # Track source nodes
                            except (nx.NetworkXNoPath, KeyError):
                                continue

    # Active nodes (in population)
    # Separate source nodes (with arrows) from non-source nodes
    active_x, active_y, active_text, active_sizes, active_fitness = [], [], [], [], []
    source_x, source_y, source_text, source_sizes, source_fitness = [], [], [], [], []
    max_freq = max(word_frequencies.values()) if word_frequencies else 1

    for word in active_words:
        if word in pos and word not in target_words:
            x, y = pos[word]
            freq = word_frequencies[word]
            fitness = word_fitnesses.get(word, 0)
            text = f"<b>{word}</b><br>Count: {freq}<br>Fitness: {fitness:.2f}<br>Freq: {freq/sum(word_frequencies.values())*100:.1f}%"
            # Much smaller sizes to prevent overlap: 6 + (freq/max) * 8 (range: 6-14)
            size = (freq / max_freq) * 8 + 6

            if word in source_nodes:
                # Source nodes (where arrows start) - more transparent
                source_x.append(x)
                source_y.append(y)
                source_text.append(text)
                source_sizes.append(size)
                source_fitness.append(fitness)
            else:
                # Non-source nodes - normal opacity
                active_x.append(x)
                active_y.append(y)
                active_text.append(text)
                active_sizes.append(size)
                active_fitness.append(fitness)

    active_trace = go.Scatter(
        x=active_x, y=active_y,
        mode='markers',
        hoverinfo='text',
        text=active_text,
        marker=dict(
            size=active_sizes,
            color=active_fitness,
            colorscale=fitness_colorscale,
            cmin=0,
            cmax=1,
            opacity=1.0,  # Fully opaque for visibility
            line=dict(width=1.5, color='white')
        ),
        name='Active copies',
        showlegend=True
    )
    
    # Source nodes trace (more transparent)
    source_trace = None
    if source_x:
        source_trace = go.Scatter(
            x=source_x, y=source_y,
            mode='markers',
            hoverinfo='text',
            text=source_text,
            marker=dict(
                size=source_sizes,
                color=source_fitness,
                colorscale=fitness_colorscale,
                cmin=0,
                cmax=1,
                opacity=0.5,  # More visible
                line=dict(width=1.5, color='white')
            ),
            name='Active copies',
            showlegend=False  # Don't duplicate in legend
        )

    # Target nodes
    target_x, target_y, target_text, target_sizes = [], [], [], []

    for word in target_words:
        if word in pos:
            x, y = pos[word]
            freq = word_frequencies.get(word, 0)

            target_x.append(x)
            target_y.append(y)
            if freq > 0:
                target_text.append(f"<b>🎯 {word} (TARGET)</b><br>Visits: {freq}<br>✅ TARGET REACHED!")
                target_sizes.append((freq / max_freq) * 8 + 8)  # Range: 8-16
            else:
                target_text.append(f"<b>🎯 {word} (TARGET)</b><br>Not yet reached")
                target_sizes.append(8)

    target_trace = go.Scatter(
        x=target_x, y=target_y,
        mode='markers',
        hoverinfo='text',
        text=target_text,
        marker=dict(
            size=target_sizes if target_sizes else [8],
            color='#22c55e',  # Solid green
            symbol='star',
            opacity=1.0,
            line=dict(width=0)  # No outline/stroke
        ),
        name='Target word',
        showlegend=True
    )

    # Create edge trace (will be added last to appear on top)
    edge_trace = None
    if edge_x:
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=2.5, color='rgba(60, 60, 60, 0.25)'),  # Uniform 0.25 opacity
            hoverinfo='skip',
            showlegend=False
        )

    # Create figure with edges on top layer
    traces = [bg_trace, explored_trace, active_trace]
    if source_trace:
        traces.append(source_trace)
    traces.append(target_trace)
    if edge_trace:
        traces.append(edge_trace)
    fig = go.Figure(data=traces)
    
    # Build annotations for edge arrows (annotations are always on top)
    # In Plotly, arrows point FROM (ax, ay) TO (x, y)
    annotations = []
    if edge_list:
        for x0, y0, x1, y1 in edge_list:
            # Calculate arrow position at the END of the edge (95% to account for arrowhead size)
            arrow_x = x0 + 0.99 * (x1 - x0)
            arrow_y = y0 + 0.99 * (y1 - y0)
            annotations.append(dict(
                x=arrow_x,  # Arrow points TO this position (at end of edge)
                y=arrow_y,
                ax=x0,      # Arrow starts FROM source
                ay=y0,
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,  # Smaller arrowhead
                arrowwidth=1,  # Thinner arrow shaft
                arrowcolor='rgba(32, 32, 32, 0.25)',  # Uniform 0.25 opacity
            ))

    # Calculate axis ranges
    all_x = bg_x + explored_x + active_x + target_x
    all_y = bg_y + explored_y + active_y + target_y

    if all_x and all_y:
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Add padding (15% on each side) to ensure nodes aren't at edges
        x_padding = x_range * 0.15 if x_range > 0 else 2
        y_padding = y_range * 0.15 if y_range > 0 else 2
        
        # Ensure minimum padding for very small ranges
        x_padding = max(x_padding, 1.0)
        y_padding = max(y_padding, 1.0)
        
        x_axis_range = [x_min - x_padding, x_max + x_padding]
        y_axis_range = [y_min - y_padding, y_max + y_padding]
        
        # Ensure aspect ratio is roughly square for better visualization
        x_span = x_axis_range[1] - x_axis_range[0]
        y_span = y_axis_range[1] - y_axis_range[0]
        if x_span > y_span * 1.5:
            # X is too wide, expand Y
            center_y = (y_axis_range[0] + y_axis_range[1]) / 2
            y_axis_range = [center_y - x_span / 2, center_y + x_span / 2]
        elif y_span > x_span * 1.5:
            # Y is too tall, expand X
            center_x = (x_axis_range[0] + x_axis_range[1]) / 2
            x_axis_range = [center_x - y_span / 2, center_x + y_span / 2]
    else:
        # Default range if no nodes
        x_axis_range = [-10, 10]
        y_axis_range = [-10, 10]

    # Update layout
    gen = state['generation']
    pop_size = state['population_size']
    diversity = state['diversity']
    target_reached = state['all_targets_reached']

    if title:
        full_title = title
    else:
        full_title = f"Generation {gen} | Active Genotypes: {diversity} | Node size = frequency, Color = fitness to target"

    layout_dict = dict(
        title=dict(
            text=full_title,
            x=0.5,
            xanchor='center',
            font=dict(size=14)
        ),
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=60),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=x_axis_range,
            autorange=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=y_axis_range,
            autorange=False
        ),
        plot_bgcolor='white',
        height=600
    )
    
    if annotations:
        layout_dict['annotations'] = annotations
    
    fig.update_layout(**layout_dict)

    return fig


# ============================================================================
# STREAMLIT APP
# ============================================================================

# Page configuration
st.set_page_config(
    page_title="Bacterial Evolution Simulator",
    page_icon="🧬",
    layout="wide"
)

# Title
st.title("🧬 Bacterial Population Evolution")
st.markdown("""
Visualize evolution with **birth-death dynamics** on a word graph.
Watch populations spread through genetic space as they evolve toward a target.
""")

# Initialize session state
if 'simulation' not in st.session_state:
    st.session_state.simulation = None
if 'word_graph' not in st.session_state:
    st.session_state.word_graph = None
if 'pos' not in st.session_state:
    st.session_state.pos = None
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0

# Sidebar controls
st.sidebar.header("⚙️ Simulation Parameters")

# Gene configuration
st.sidebar.subheader("Gene Configuration")
start_word = st.sidebar.text_input("Start word", value="GENE", max_chars=4).upper()
target_word = st.sidebar.text_input("Target word", value="WORD", max_chars=4).upper()
n_copies = st.sidebar.slider("Number of gene copies", min_value=1, max_value=5, value=2,
                             help="1 = essential gene, >1 = gene duplication")

st.sidebar.caption("💡 Try: COLD→BOLD (easy), WORD→WORM (easy), COLD→WARM (hard)")

# Population parameters
st.sidebar.subheader("Population Parameters")
N_capacity = st.sidebar.slider("Carrying capacity", min_value=100, max_value=2000,
                               value=300, step=50)
mu = st.sidebar.slider("Mutation rate", min_value=0.01, max_value=0.3,
                      value=0.15, step=0.01, format="%.2f")
s = st.sidebar.slider("Selection coefficient", min_value=0.05, max_value=1.0,
                     value=0.5, step=0.05, format="%.2f")

# Simulation parameters
st.sidebar.subheader("Simulation Settings")
max_generations = st.sidebar.slider("Max generations", min_value=1000, max_value=50000,
                                   value=10000, step=1000)
stop_at_target = st.sidebar.checkbox("Stop when target reached", value=True)

# Run button
if st.sidebar.button("🚀 Run Simulation", type="primary"):
    with st.spinner("Building word graph..."):
        word_graph, valid_words, _ = build_word_graph(word_length=len(start_word))
        st.session_state.word_graph = word_graph
        # Use same layout logic as app.py
        pos = compute_graph_layout(word_graph, layout_type='auto')
        st.session_state.pos = pos

    # Validate words
    if start_word not in word_graph:
        st.error(f"❌ Start word '{start_word}' not found in dictionary!")
        st.stop()
    if target_word not in word_graph:
        st.error(f"❌ Target word '{target_word}' not found in dictionary!")
        st.stop()

    with st.spinner(f"Running simulation: {start_word} → {target_word}..."):
        gene_config = {'gene1': (start_word, n_copies)}
        targets = {'gene1': target_word}

        sim = GillespieSimulation(
            gene_config=gene_config,
            targets=targets,
            word_graph=word_graph,
            N_carrying_capacity=N_capacity,
            mu=mu,
            s=s,
            seed=42
        )

        # Run simulation
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(state):
            progress = state['generation'] / max_generations
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Generation {state['generation']:,} | Pop: {state['population_size']}")

        final_state = sim.run(
            max_generations=max_generations,
            stop_at_target=stop_at_target,
            callback=update_progress
        )

        progress_bar.empty()
        status_text.empty()

        st.session_state.simulation = sim
        st.session_state.current_frame = len(sim.history) - 1 if sim.history else 0

        # Show result
        if final_state['all_targets_reached']:
            st.success(f"✅ Target reached at generation {final_state['generation']:,}!")
        elif final_state['population_size'] == 0:
            st.error("❌ Population went extinct!")
        else:
            st.warning(f"⏱️ Simulation ended at generation {final_state['generation']:,} without reaching target.")

# Display visualization
if st.session_state.simulation is not None and st.session_state.word_graph is not None:
    sim = st.session_state.simulation
    word_graph = st.session_state.word_graph
    pos = st.session_state.pos

    st.markdown("---")
    st.header("📊 Visualization")

    # Generation slider
    if len(sim.history) > 1:
        max_frame = len(sim.history) - 1
        if st.session_state.current_frame > max_frame:
            st.session_state.current_frame = max_frame

        frame_idx = st.slider(
            "Generation (drag to scrub through time)",
            min_value=0,
            max_value=max_frame,
            value=st.session_state.current_frame,
            key='generation_slider'
        )
        st.session_state.current_frame = frame_idx
    else:
        st.session_state.current_frame = 0

    # Get snapshot
    snapshot = sim.get_snapshot_at_generation(st.session_state.current_frame)

    if snapshot:
        # Two-column layout like app.py
        col1, col2 = st.columns([1, 2])
        
        state = snapshot['state']
        word_frequencies = snapshot['word_frequencies']
        targets = snapshot['targets']
        target_word = list(targets.values())[0] if targets else None
        
        with col1:
            st.markdown(f"**Generation:** {state['generation']}")
            st.markdown(f"**Population Size:** {state['population_size']}")
            st.markdown(f"**Active Genotypes:** {state['diversity']}")
            
            # Show active genotypes (top 5 plus target)
            st.markdown("---")
            st.markdown("**Active Genotypes:**")
            sorted_words = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)
            
            # Show target word first if present
            if target_word and target_word in word_frequencies:
                target_freq = word_frequencies[target_word]
                st.markdown(f"🎯 **{target_word}**: {target_freq} ✅ TARGET")
            
            # Show top 5 non-target words
            non_target_words = [(word, freq) for word, freq in sorted_words if word != target_word]
            top_5 = non_target_words[:5]
            
            for word, freq in top_5:
                freq_pct = (freq / state['population_size']) * 100
                st.markdown(f"**{word}**: {freq} ({freq_pct:.1f}%)")
            
            # Show population changes from previous generation
            st.markdown("---")
            st.markdown("**Population Changes:**")
            if st.session_state.current_frame > 0:
                prev_snapshot = sim.get_snapshot_at_generation(st.session_state.current_frame - 1)
                if prev_snapshot:
                    prev_freqs = prev_snapshot.get('word_frequencies', {})
                    prev_pop = prev_snapshot.get('state', {}).get('population_size', 0)
                    
                    pop_change = state['population_size'] - prev_pop
                    if pop_change > 0:
                        st.markdown(f"Population: +{pop_change} ⬆️")
                    elif pop_change < 0:
                        st.markdown(f"Population: {pop_change} ⬇️")
                    else:
                        st.markdown(f"Population: No change")
                    
                    # Show new genotypes
                    new_words = set(word_frequencies.keys()) - set(prev_freqs.keys())
                    if new_words:
                        st.markdown(f"**New genotypes:** {', '.join(sorted(new_words))}")
                    
                    # Show extinct genotypes
                    extinct_words = set(prev_freqs.keys()) - set(word_frequencies.keys())
                    if extinct_words:
                        st.markdown(f"**Extinct genotypes:** {', '.join(sorted(extinct_words))}")
                    
                    # Show frequency changes
                    changed_words = []
                    for word in set(word_frequencies.keys()) & set(prev_freqs.keys()):
                        change = word_frequencies[word] - prev_freqs[word]
                        if abs(change) > 0:
                            changed_words.append((word, change))
                    
                    if changed_words:
                        st.markdown("**Frequency changes:**")
                        for word, change in sorted(changed_words, key=lambda x: abs(x[1]), reverse=True)[:5]:
                            if change > 0:
                                st.markdown(f"  {word}: +{change}")
                            else:
                                st.markdown(f"  {word}: {change}")
            else:
                st.markdown("Initial generation")
        
        with col2:
            # Main interactive visualization
            fig = create_interactive_population_figure(
                word_graph,
                snapshot,
                pos,
                node_size_scale=10,  # Reduced to prevent overlap (not used directly, but kept for compatibility)
                sim=sim,
                current_generation_idx=st.session_state.current_frame
            )
            st.plotly_chart(fig, use_container_width=True)

        # Statistics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)

        state = snapshot['state']
        with col1:
            st.metric("Generation", f"{state['generation']:,}")
        with col2:
            st.metric("Population Size", state['population_size'])
        with col3:
            st.metric("Genotypic Diversity", state['diversity'])
        with col4:
            target_status = "✅ Reached" if state['all_targets_reached'] else "⏳ Evolving"
            st.metric("Target Status", target_status)

        # Explored space metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Explored Genotypes", state.get('explored_genotypes', 0))
        with col2:
            st.metric("Explored Words", state.get('explored_words', 0))
        with col3:
            st.metric("Max Diversity Reached", state.get('max_diversity', 0))
        with col4:
            st.metric("Max Distance from Start", state.get('max_edit_distance', 0))

        # Word frequencies
        with st.expander("📈 Word Frequencies in Population"):
            word_freq = snapshot['word_frequencies']
            if word_freq:
                freq_df_data = {
                    'Word': list(word_freq.keys()),
                    'Count': list(word_freq.values()),
                    'Frequency': [f"{(v/sum(word_freq.values()))*100:.1f}%"
                                 for v in word_freq.values()]
                }
                freq_df = pd.DataFrame(freq_df_data)
                freq_df = freq_df.sort_values('Count', ascending=False)
                st.dataframe(freq_df, use_container_width=True, hide_index=True)

        # Time series plots
        with st.expander("📉 Population Dynamics Over Time"):
            history_df = sim.get_history_df()

            fig, axes = plt.subplots(2, 2, figsize=(15, 9))
            fig.patch.set_facecolor('#ffffff')
            
            # Modern color palette
            colors = {
                'primary': '#3b82f6',
                'secondary': '#10b981',
                'accent': '#f59e0b',
                'danger': '#ef4444',
                'current': '#ef4444',
                'capacity': '#6b7280'
            }

            # Population size
            axes[0, 0].plot(history_df['generation'], history_df['population_size'],
                          linewidth=3, color=colors['primary'], label='Population Size', zorder=3)
            axes[0, 0].axhline(N_capacity, color=colors['capacity'], linestyle='--', 
                             linewidth=2, alpha=0.7, label='Carrying Capacity', zorder=2)
            axes[0, 0].axvline(state['generation'], color=colors['current'], linestyle=':', 
                             linewidth=2.5, alpha=0.8, label='Current Generation', zorder=4)
            axes[0, 0].set_xlabel('Generation', fontsize=12, fontweight='medium', color='#374151')
            axes[0, 0].set_ylabel('Population Size', fontsize=12, fontweight='medium', color='#374151')
            axes[0, 0].set_title('Population Dynamics', fontsize=14, fontweight='bold', color='#111827', pad=12)
            axes[0, 0].legend(frameon=True, fancybox=True, shadow=True, fontsize=10, 
                            framealpha=0.95, loc='best')
            axes[0, 0].grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
            axes[0, 0].set_facecolor('#fafafa')
            axes[0, 0].spines['top'].set_visible(False)
            axes[0, 0].spines['right'].set_visible(False)
            axes[0, 0].spines['left'].set_color('#e5e7eb')
            axes[0, 0].spines['bottom'].set_color('#e5e7eb')

            # Diversity
            axes[0, 1].plot(history_df['generation'], history_df['diversity'],
                          linewidth=3, color=colors['secondary'], zorder=3)
            axes[0, 1].axvline(state['generation'], color=colors['current'], linestyle=':', 
                             linewidth=2.5, alpha=0.8, zorder=4)
            axes[0, 1].set_xlabel('Generation', fontsize=12, fontweight='medium', color='#374151')
            axes[0, 1].set_ylabel('Number of Genotypes', fontsize=12, fontweight='medium', color='#374151')
            axes[0, 1].set_title('Genotypic Diversity', fontsize=14, fontweight='bold', color='#111827', pad=12)
            axes[0, 1].grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
            axes[0, 1].set_facecolor('#fafafa')
            axes[0, 1].spines['top'].set_visible(False)
            axes[0, 1].spines['right'].set_visible(False)
            axes[0, 1].spines['left'].set_color('#e5e7eb')
            axes[0, 1].spines['bottom'].set_color('#e5e7eb')

            # Dominant frequency
            axes[1, 0].plot(history_df['generation'], history_df['dominant_frequency'],
                          linewidth=3, color=colors['danger'], zorder=3)
            axes[1, 0].axvline(state['generation'], color=colors['current'], linestyle=':', 
                             linewidth=2.5, alpha=0.8, zorder=4)
            axes[1, 0].set_xlabel('Generation', fontsize=12, fontweight='medium', color='#374151')
            axes[1, 0].set_ylabel('Dominant Frequency', fontsize=12, fontweight='medium', color='#374151')
            axes[1, 0].set_title('Clonal Sweeps', fontsize=14, fontweight='bold', color='#111827', pad=12)
            axes[1, 0].set_ylim(0, 1.05)
            axes[1, 0].grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
            axes[1, 0].set_facecolor('#fafafa')
            axes[1, 0].spines['top'].set_visible(False)
            axes[1, 0].spines['right'].set_visible(False)
            axes[1, 0].spines['left'].set_color('#e5e7eb')
            axes[1, 0].spines['bottom'].set_color('#e5e7eb')

            # Target reached
            if 'gene1_at_target' in history_df.columns:
                axes[1, 1].fill_between(
                    history_df['generation'],
                    0,
                    history_df['gene1_at_target'].astype(int),
                    alpha=0.5,
                    label='Target reached',
                    color=colors['secondary'],
                    zorder=2
                )
                axes[1, 1].axvline(state['generation'], color=colors['current'], linestyle=':', 
                                 linewidth=2.5, alpha=0.8, zorder=4)
                axes[1, 1].set_xlabel('Generation', fontsize=12, fontweight='medium', color='#374151')
                axes[1, 1].set_ylabel('Target Reached', fontsize=12, fontweight='medium', color='#374151')
                axes[1, 1].set_title('Adaptive Progress', fontsize=14, fontweight='bold', color='#111827', pad=12)
                axes[1, 1].set_ylim(-0.1, 1.1)
                axes[1, 1].grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
                axes[1, 1].set_facecolor('#fafafa')
                axes[1, 1].spines['top'].set_visible(False)
                axes[1, 1].spines['right'].set_visible(False)
                axes[1, 1].spines['left'].set_color('#e5e7eb')
                axes[1, 1].spines['bottom'].set_color('#e5e7eb')
                axes[1, 1].legend(frameon=True, fancybox=True, shadow=True, fontsize=10, 
                                framealpha=0.95, loc='best')

            plt.tight_layout(pad=2.5)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # Explored space metrics plots
        with st.expander("🗺️ Explored Space Expansion"):
            history_df = sim.get_history_df()

            fig, axes = plt.subplots(2, 2, figsize=(15, 9))
            fig.patch.set_facecolor('#ffffff')
            
            # Modern color palette
            colors = {
                'purple': '#a855f7',
                'orange': '#f59e0b',
                'pink': '#ec4899',
                'cyan': '#06b6d4',
                'green': '#10b981',
                'current': '#ef4444'
            }

            # Explored words
            axes[0, 0].plot(history_df['generation'], history_df['explored_words'],
                          linewidth=3, color=colors['purple'], zorder=3)
            axes[0, 0].axvline(state['generation'], color=colors['current'], linestyle=':', 
                             linewidth=2.5, alpha=0.8, label='Current Generation', zorder=4)
            axes[0, 0].set_xlabel('Generation', fontsize=12, fontweight='medium', color='#374151')
            axes[0, 0].set_ylabel('Unique Words Explored', fontsize=12, fontweight='medium', color='#374151')
            axes[0, 0].set_title('Phenotypic Space Exploration', fontsize=14, fontweight='bold', color='#111827', pad=12)
            axes[0, 0].legend(frameon=True, fancybox=True, shadow=True, fontsize=10, 
                            framealpha=0.95, loc='best')
            axes[0, 0].grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
            axes[0, 0].set_facecolor('#fafafa')
            axes[0, 0].spines['top'].set_visible(False)
            axes[0, 0].spines['right'].set_visible(False)
            axes[0, 0].spines['left'].set_color('#e5e7eb')
            axes[0, 0].spines['bottom'].set_color('#e5e7eb')

            # Explored genotypes
            axes[0, 1].plot(history_df['generation'], history_df['explored_genotypes'],
                          linewidth=3, color=colors['orange'], zorder=3)
            axes[0, 1].axvline(state['generation'], color=colors['current'], linestyle=':', 
                             linewidth=2.5, alpha=0.8, zorder=4)
            axes[0, 1].set_xlabel('Generation', fontsize=12, fontweight='medium', color='#374151')
            axes[0, 1].set_ylabel('Unique Genotypes Explored', fontsize=12, fontweight='medium', color='#374151')
            axes[0, 1].set_title('Genotypic Space Exploration', fontsize=14, fontweight='bold', color='#111827', pad=12)
            axes[0, 1].grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
            axes[0, 1].set_facecolor('#fafafa')
            axes[0, 1].spines['top'].set_visible(False)
            axes[0, 1].spines['right'].set_visible(False)
            axes[0, 1].spines['left'].set_color('#e5e7eb')
            axes[0, 1].spines['bottom'].set_color('#e5e7eb')

            # Max edit distance
            axes[1, 0].plot(history_df['generation'], history_df['max_edit_distance'],
                          linewidth=3, color=colors['pink'], zorder=3)
            axes[1, 0].axvline(state['generation'], color=colors['current'], linestyle=':', 
                             linewidth=2.5, alpha=0.8, zorder=4)
            axes[1, 0].set_xlabel('Generation', fontsize=12, fontweight='medium', color='#374151')
            axes[1, 0].set_ylabel('Max Edit Distance from Start', fontsize=12, fontweight='medium', color='#374151')
            axes[1, 0].set_title('Mutational Distance Expansion', fontsize=14, fontweight='bold', color='#111827', pad=12)
            axes[1, 0].grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
            axes[1, 0].set_facecolor('#fafafa')
            axes[1, 0].spines['top'].set_visible(False)
            axes[1, 0].spines['right'].set_visible(False)
            axes[1, 0].spines['left'].set_color('#e5e7eb')
            axes[1, 0].spines['bottom'].set_color('#e5e7eb')

            # Max diversity reached
            axes[1, 1].plot(history_df['generation'], history_df['max_diversity'],
                          linewidth=3, color=colors['cyan'], label='Peak Diversity', zorder=3)
            axes[1, 1].plot(history_df['generation'], history_df['diversity'],
                          linewidth=2, color=colors['green'], alpha=0.7, label='Current Diversity', zorder=3)
            axes[1, 1].axvline(state['generation'], color=colors['current'], linestyle=':', 
                             linewidth=2.5, alpha=0.8, label='Current Generation', zorder=4)
            axes[1, 1].set_xlabel('Generation', fontsize=12, fontweight='medium', color='#374151')
            axes[1, 1].set_ylabel('Genotypic Diversity', fontsize=12, fontweight='medium', color='#374151')
            axes[1, 1].set_title('Peak Diversity Tracking', fontsize=14, fontweight='bold', color='#111827', pad=12)
            axes[1, 1].legend(frameon=True, fancybox=True, shadow=True, fontsize=10, 
                            framealpha=0.95, loc='best')
            axes[1, 1].grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
            axes[1, 1].set_facecolor('#fafafa')
            axes[1, 1].spines['top'].set_visible(False)
            axes[1, 1].spines['right'].set_visible(False)
            axes[1, 1].spines['left'].set_color('#e5e7eb')
            axes[1, 1].spines['bottom'].set_color('#e5e7eb')

            plt.tight_layout(pad=2.5)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

else:
    # Welcome message
    st.info("👈 Configure parameters in the sidebar and click **Run Simulation** to begin!")

    st.markdown("""
    ### About This Simulator

    This tool simulates bacterial evolution using **birth-death dynamics** (Gillespie algorithm):

    - **Population-based**: Multiple genotypes coexist and compete
    - **Gene duplication**: Simulate 1-5 copies of a gene
    - **Realistic dynamics**: Births proportional to fitness, deaths increase with density
    - **Visual exploration**: Watch populations spread through genetic space

    #### How to Use
    1. Set start and target words (4 letters each)
    2. Choose number of gene copies (1 = essential, >1 = duplication)
    3. Adjust population and mutation parameters
    4. Run simulation and explore results
    5. Drag the slider to scrub through time and see evolution progress
    6. **Hover over nodes** to see word labels, frequencies, and fitness values

    #### Visualization Guide
    - **Node size** = frequency in population (bigger = more common)
    - **Node color** = fitness (darker blue = closer to target)
    - **Blue nodes** = currently active genotypes in population
    - **Orange nodes** = explored but extinct (were in population before)
    - **Green star (🎯)** = target word (goal of evolution)
    - **Gray background** = unexplored words (never been in population)
    - **Hover over any node** to see: word, count, frequency, and fitness
    - **Fitness color bar** shows fitness scale (0 = far from target, 1 = at target)

    #### Interactive Features
    - **Slider**: Drag to scrub through evolutionary history (defaults to final state)
    - **Hover tooltips**: Mouse over nodes to see details
    - **Zoom & Pan**: Use Plotly controls (top right) to zoom and pan the graph
    - **Legend**: Click legend items to toggle visibility of different node types
    """)

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit • Powered by NetworkX and Plotly • Enhanced interactive visualization*")
