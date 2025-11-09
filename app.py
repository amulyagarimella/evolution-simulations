import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import plotly.graph_objects as go
from sim1 import build_word_graph, simulate_word_evolution, word_fitness

# Page configuration
st.set_page_config(
    page_title="Word Evolution Simulator",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Word Evolution Simulator")
st.markdown("""
This simulation demonstrates evolution through word mutations. Starting from one word,
the system attempts to evolve toward a target word through random mutations (single letter changes),
with selection based on fitness (similarity to target).
""")

# Sidebar controls
st.sidebar.header("Simulation Parameters")

start_word = st.sidebar.text_input("Start Word", "WORD").upper()
target_word = st.sidebar.text_input("Target Word", "GENE").upper()

N_e = st.sidebar.slider(
    "Effective Population Size (N_e)",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100,
    help="Higher values = stronger selection, lower values = more genetic drift"
)

edits_per_step = st.sidebar.slider(
    "Edits Per Step",
    min_value=1,
    max_value=3,
    value=1,
    help="Number of single-letter changes to make in each mutation"
)

n_copies = st.sidebar.slider(
    "Number of Gene Copies",
    min_value=1,
    max_value=5,
    value=1,
    help="1 = essential gene (any invalid mutation kills organism), >1 = duplications allow exploration"
)

n_steps = st.sidebar.slider(
    "Maximum Steps",
    min_value=50,
    max_value=500,
    value=100,
    step=10,
    help="Maximum number of mutation attempts before giving up"
)

max_attempts = st.sidebar.slider(
    "Max Attempts",
    min_value=1,
    max_value=20,
    value=10,
    help="How many times to retry if simulation doesn't reach target"
)

# Validate inputs
if len(start_word) != 4 or len(target_word) != 4:
    st.error("Both words must be exactly 4 letters long!")
    st.stop()

# Initialize session state
if 'word_graph' not in st.session_state:
    st.session_state.word_graph = None
    st.session_state.valid_words = None
    st.session_state.invalid_words = None

if 'trajectory' not in st.session_state:
    st.session_state.trajectory = None
    st.session_state.pos = None

# Build word graph (only once)
if st.session_state.word_graph is None:
    with st.spinner("Building word graph... This may take a minute..."):
        st.session_state.word_graph, st.session_state.valid_words, st.session_state.invalid_words = build_word_graph(
            word_length=4, max_delta=1
        )
    st.success(f"Word graph built! {st.session_state.word_graph.number_of_nodes()} nodes, {st.session_state.word_graph.number_of_edges()} edges")

# Run simulation button
if st.sidebar.button("🚀 Run Simulation", type="primary"):
    word_graph = st.session_state.word_graph

    # Check if start and target words are in the graph
    if start_word not in word_graph:
        st.error(f"'{start_word}' is not a valid word in the dictionary!")
        st.stop()
    if target_word not in word_graph:
        st.error(f"'{target_word}' is not a valid word in the dictionary!")
        st.stop()

    # Run simulation with retries
    attempt = 0
    trajectory = None
    success = False

    progress_bar = st.progress(0)
    status_text = st.empty()

    while attempt < max_attempts and not success:
        attempt += 1
        status_text.text(f"Attempt {attempt}/{max_attempts}...")
        progress_bar.progress(attempt / max_attempts)

        trajectory = simulate_word_evolution(
            word_graph, N_e, start_word, target_word,
            n_steps=n_steps,
            edits_per_step=edits_per_step,
            n_copies=n_copies
        )

        # Check if we reached the target
        # With new format: trajectory is list of (step_data, visit_density)
        # where step_data is list of (copy_id, current, attempted, accepted, status)
        reached_target = False
        for step_data, visit_density in trajectory:
            for copy_id, current, attempted, accepted, status in step_data:
                if accepted and attempted == target_word:
                    reached_target = True
                    break
            if reached_target:
                break

        if reached_target:
            success = True
            break

    progress_bar.empty()
    status_text.empty()

    if success:
        st.success(f"✅ Success! Reached '{target_word}' in attempt {attempt} with {len(trajectory)} steps")
        st.session_state.trajectory = trajectory
        st.session_state.attempt_num = attempt

        # Compute layout for visualization
        with st.spinner("Computing graph layout..."):
            if len(word_graph.nodes()) > 1000:
                st.session_state.pos = nx.kamada_kawai_layout(word_graph)
            else:
                st.session_state.pos = nx.spring_layout(word_graph, k=0.5, iterations=50, seed=42)
    else:
        # Get last active words
        last_step_data, _ = trajectory[-1] if trajectory else ([], {})
        last_words = set()
        for copy_id, current, attempted, accepted, status in last_step_data:
            if accepted:
                last_words.add(attempted)
            else:
                last_words.add(current)
        last_words_str = ", ".join(sorted(last_words)) if last_words else "none"
        st.error(f"❌ Failed to reach '{target_word}' after {max_attempts} attempts. Last words: {last_words_str}")
        st.info("Try increasing 'Max Attempts' or 'Maximum Steps', or try adding more gene copies")

# Display trajectory if available
if st.session_state.trajectory is not None:
    trajectory = st.session_state.trajectory
    pos = st.session_state.pos
    word_graph = st.session_state.word_graph
    valid_words = st.session_state.valid_words

    st.markdown("---")
    st.header("📊 Simulation Results")

    # Build the evolutionary paths for each copy
    # Format: {copy_id: [list of words in order]}
    copy_paths = {}
    copy_status = {}  # Track which copies are alive/dead

    # Initialize starting positions
    first_step_data, _ = trajectory[0]
    for copy_id, current, attempted, accepted, status in first_step_data:
        copy_paths[copy_id] = [current]
        copy_status[copy_id] = 'alive'

    # Build paths through trajectory
    for step_data, visit_density in trajectory:
        for copy_id, current, attempted, accepted, status in step_data:
            if status == 'invalid':
                copy_status[copy_id] = 'dead'
            elif accepted:
                if copy_id in copy_paths:
                    copy_paths[copy_id].append(attempted)
                    if attempted == target_word:
                        copy_status[copy_id] = 'success'

    # Get final visit density
    _, final_density = trajectory[-1]

    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Steps", len(trajectory))
    with col2:
        alive_count = sum(1 for s in copy_status.values() if s in ['alive', 'success'])
        st.metric("Active Copies", f"{alive_count}/{len(copy_paths)}")
    with col3:
        total_mutations = sum(len(step_data) for step_data, _ in trajectory)
        print(trajectory[0])
        accepted_count = sum(1 for step_data, _ in trajectory
                           for _, _, _, accepted, _ in step_data if accepted)
        st.metric("Accepted Mutations", f"{accepted_count}/{total_mutations}")
    with col4:
        success_count = sum(1 for s in copy_status.values() if s == 'success')
        st.metric("Reached Target", f"{success_count} copies")

    # Show evolutionary paths
    st.subheader("Evolutionary Paths")
    for copy_id, path in copy_paths.items():
        status_emoji = {"alive": "🟢", "dead": "💀", "success": "✅"}
        status = copy_status.get(copy_id, 'alive')
        st.write(f"**Copy {copy_id}** {status_emoji[status]}: {' → '.join(path[:20])}" +
                (" ..." if len(path) > 20 else ""))

    # Interactive trajectory viewer
    st.markdown("---")
    st.subheader("🔍 Explore Trajectory Step-by-Step")

    frame = st.slider(
        "Step",
        min_value=0,
        max_value=len(trajectory) - 1,
        value=len(trajectory) - 1,  # Default to END result
        help="Slide to explore each step of the evolution"
    )

    # Get current step info
    step_data, step_density = trajectory[frame]

    # Build current positions of all copies up to this frame
    current_copy_positions = {}  # {copy_id: current_word}
    copy_paths_so_far = {}  # {copy_id: [words]}

    # Initialize
    first_step_data, _ = trajectory[0]
    for copy_id, current, attempted, accepted, status in first_step_data:
        current_copy_positions[copy_id] = current
        copy_paths_so_far[copy_id] = [current]

    # Build up to current frame
    for i in range(frame + 1):
        frame_step_data, _ = trajectory[i]
        for copy_id, current, attempted, accepted, status in frame_step_data:
            if status == 'invalid':
                # This copy died
                if copy_id in current_copy_positions:
                    del current_copy_positions[copy_id]
            elif accepted:
                current_copy_positions[copy_id] = attempted
                if copy_id in copy_paths_so_far:
                    copy_paths_so_far[copy_id].append(attempted)

    # Track rejected nodes
    rejected_nodes = set()
    for i in range(frame + 1):
        frame_step_data, _ = trajectory[i]
        for copy_id, current, attempted, accepted, status in frame_step_data:
            if not accepted:
                rejected_nodes.add(attempted)

    # Display step information
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"**Step:** {frame}")
        st.markdown(f"**Active Copies:** {len(current_copy_positions)}")

        # Show each copy's current state
        for copy_id, current_word in current_copy_positions.items():
            st.markdown(f"**Copy {copy_id}:** `{current_word}`")

        # Show mutations attempted in this step
        st.markdown("---")
        st.markdown("**Mutations this step:**")
        for copy_id, current, attempted, accepted, status in step_data:
            if status == 'invalid':
                st.markdown(f"Copy {copy_id}: `{current}` → `{attempted}` ❌ DIED")
            elif accepted:
                st.markdown(f"Copy {copy_id}: `{current}` → `{attempted}` ✅")
            else:
                st.markdown(f"Copy {copy_id}: `{current}` → `{attempted}` ⚠️ Rejected")

    with col2:
        # Create interactive Plotly visualization with hover tooltips (like sim2_viz)
        # Compute fitness for all words (distance to target)
        word_fitnesses = {}
        for word in valid_words:
            if word == target_word:
                word_fitnesses[word] = 1.0
            else:
                try:
                    dist = nx.shortest_path_length(word_graph, word, target_word)
                    max_dist = 10
                    word_fitnesses[word] = max(0.0, 1.0 - (dist / max_dist))
                except nx.NetworkXNoPath:
                    word_fitnesses[word] = 0.0

        # Categorize nodes
        all_visited_words = set(step_density.keys())
        unexplored = [w for w in valid_words if w not in all_visited_words and w != target_word]

        # Background nodes (unexplored)
        bg_x, bg_y, bg_text, bg_fitness = [], [], [], []
        for word in unexplored:
            if word in pos:
                x, y = pos[word]
                bg_x.append(x)
                bg_y.append(y)
                bg_text.append(f"{word}<br>Fitness: {word_fitnesses.get(word, 0):.2f}<br>(not visited)")
                bg_fitness.append(word_fitnesses.get(word, 0))

        bg_trace = go.Scatter(
            x=bg_x,
            y=bg_y,
            mode='markers',
            hoverinfo='text',
            text=bg_text,
            marker=dict(
                size=7,
                color=bg_fitness,
                colorscale='Greys',
                opacity=0.2,
                line=dict(width=0)
            ),
            name='Unexplored words',
            showlegend=True
        )

        # Explored but inactive nodes (visited in past but not currently active)
        explored_inactive_x, explored_inactive_y, explored_inactive_text, explored_inactive_sizes, explored_inactive_fitness = [], [], [], [], []
        if step_density:
            explored_words = list(step_density.keys())
            explored_words = [w for w in explored_words 
                           if w != target_word and w not in current_copy_positions.values()]
            
            if explored_words:
                densities = [step_density[w] for w in explored_words]
                max_density = max(densities) if densities else 1
                
                for word in explored_words:
                    if word in pos:
                        x, y = pos[word]
                        density = step_density[word]
                        fitness = word_fitnesses.get(word, 0.0)
                        
                        explored_inactive_x.append(x)
                        explored_inactive_y.append(y)
                        explored_inactive_text.append(f"<b>{word}</b><br>Visits: {density}<br>Fitness: {fitness:.2f}<br>(explored but inactive)")
                        explored_inactive_sizes.append(10 + (density / max_density) * 15)  # Range: 10-25 pixels
                        explored_inactive_fitness.append(fitness)

        # Custom fitness colorscale for explored nodes (slightly muted)
        explored_colorscale = [[0, '#d73027'], [0.5, '#fee08b'], [1, '#1a9850']]
        
        explored_inactive_trace = go.Scatter(
            x=explored_inactive_x,
            y=explored_inactive_y,
            mode='markers',
            hoverinfo='text',
            text=explored_inactive_text,
            marker=dict(
                size=explored_inactive_sizes,
                color=explored_inactive_fitness,
                colorscale=explored_colorscale,
                cmin=0,
                cmax=1,
                opacity=0.6,
                line=dict(width=1.5, color='white'),
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

        # Build edges from mutation paths FIRST to identify source nodes
        edge_x, edge_y = [], []
        edge_list = []  # Store (x0, y0, x1, y1) tuples for annotations
        source_nodes = set()  # Track nodes that are sources of edges
        
        # Collect all edges from copy paths
        for copy_id, path in copy_paths_so_far.items():
            for i in range(len(path) - 1):
                from_word = path[i]
                to_word = path[i + 1]
                if from_word in pos and to_word in pos: 
                    x0, y0 = pos[from_word]
                    x1, y1 = pos[to_word]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_list.append((x0, y0, x1, y1))
                    source_nodes.add(from_word)  # Track source nodes

        # Current active copy positions
        # Separate source nodes (with arrows) from non-source nodes
        current_x, current_y, current_text, current_fitness = [], [], [], []
        source_x, source_y, source_text, source_fitness = [], [], [], []
        
        if current_copy_positions:
            for copy_id, word in current_copy_positions.items():
                if word in pos and word != target_word:  # Exclude target word to avoid duplicate
                    x, y = pos[word]
                    fitness = word_fitnesses.get(word, 0.0)
                    text = f"<b>{word}</b> (Copy {copy_id})<br>Fitness: {fitness:.2f}<br>Current position"
                    
                    if word in source_nodes:
                        # Source nodes (where arrows start) - more transparent
                        source_x.append(x)
                        source_y.append(y)
                        source_text.append(text)
                        source_fitness.append(fitness)
                    else:
                        # Non-source nodes - normal opacity
                        current_x.append(x)
                        current_y.append(y)
                        current_text.append(text)
                        current_fitness.append(fitness)

        # Custom fitness colorscale: Red (low) -> Yellow (medium) -> Green (high)
        fitness_colorscale = [[0, '#d73027'], [0.5, '#fee08b'], [1, '#1a9850']]
        
        # Non-source nodes trace
        current_trace = go.Scatter(
            x=current_x,
            y=current_y,
            mode='markers',
            hoverinfo='text',
            text=current_text,
            marker=dict(
                size=25,  # Larger to stand out as current positions
                color=current_fitness,
                colorscale=fitness_colorscale,
                cmin=0,
                cmax=1,
                opacity=0.9,
                line=dict(width=3, color='white')
            ),
            name='Active copies',
            showlegend=True
        )
        
        # Source nodes trace (more transparent)
        source_trace = None
        if source_x:
            source_trace = go.Scatter(
                x=source_x,
                y=source_y,
                mode='markers',
                hoverinfo='text',
                text=source_text,
                marker=dict(
                    size=25,
                    color=source_fitness,
                    colorscale=fitness_colorscale,
                    cmin=0,
                    cmax=1,
                    opacity=0.5,  # More transparent for source nodes
                    line=dict(width=3, color='white')
                ),
                name='Active copies',
                showlegend=False  # Don't duplicate in legend
            )

        # Target word as green star (scales with density like other nodes)
        target_x, target_y, target_text, target_size = [], [], [], []
        if target_word in pos:
            x, y = pos[target_word]
            target_density = step_density.get(target_word, 0)
            
            # Calculate max_density for scaling (same as used for visited nodes)
            all_densities = list(step_density.values()) if step_density else []
            max_density = max(all_densities) if all_densities else 1
            
            # Scale target size with density using same formula as visited nodes
            if target_density > 0:
                target_size_val = 10 + (target_density / max_density) * 15  # Range: 10-25 pixels
            else:
                target_size_val = 10  # Minimum size when not visited
            
            target_x.append(x)
            target_y.append(y)
            if target_density > 0:
                target_text.append(f"<b>🎯 {target_word} (TARGET)</b><br>Visits: {target_density}<br>✅ TARGET REACHED!")
            else:
                target_text.append(f"<b>🎯 {target_word} (TARGET)</b><br>Not yet reached")
            target_size.append(target_size_val)

        target_trace = go.Scatter(
            x=target_x,
            y=target_y,
            mode='markers',
            hoverinfo='text',
            text=target_text,
            marker=dict(
                size=target_size if target_size else [10],
                color='#22c55e',  # Solid green
                symbol='star',
                opacity=1.0,
                line=dict(width=0)  # No outline/stroke
            ),
            name='Target word',
            showlegend=True
        )

        # Create edge trace (will be added last to appear on top)
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=2.5, color='rgba(60, 60, 60, 0.25)'),  # Uniform 0.25 opacity
            hoverinfo='skip',
            showlegend=False
        )

        # Create figure with edges on top layer
        traces = [bg_trace, explored_inactive_trace, current_trace]
        if source_trace:
            traces.append(source_trace)
        traces.extend([target_trace, edge_trace])
        fig = go.Figure(data=traces)
        
        # Add arrows for edges using annotations (annotations are always on top)
        # In Plotly, arrows point FROM (ax, ay) TO (x, y)
        annotations = []
        for x0, y0, x1, y1 in edge_list:
            # Calculate arrow position at the END of the edge (99% to account for arrowhead size)
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
        
        if annotations:
            fig.update_layout(annotations=annotations)

        # Calculate axis ranges from all node positions to prevent resizing when traces are toggled
        all_x = bg_x + explored_inactive_x + current_x + target_x
        all_y = bg_y + explored_inactive_y + current_y + target_y
        
        if all_x and all_y:
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            # Add padding (10% on each side)
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_padding = x_range * 0.1 if x_range > 0 else 1
            y_padding = y_range * 0.1 if y_range > 0 else 1
            x_axis_range = [x_min - x_padding, x_max + x_padding]
            y_axis_range = [y_min - y_padding, y_max + y_padding]
        else:
            # Default range if no nodes
            x_axis_range = [-10, 10]
            y_axis_range = [-10, 10]

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Step {frame} | Active Copies: {len(current_copy_positions)} | Node size = visit frequency, Color = fitness to target",
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
                autorange=False  # Keep fixed range when traces are toggled
            ),
            yaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                range=y_axis_range,
                autorange=False  # Keep fixed range when traces are toggled
            ),
            plot_bgcolor='white',
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

# Instructions at the bottom
st.markdown("---")
st.markdown("""
### 📖 How to Use
1. Set your parameters in the sidebar
2. Click "Run Simulation" to start
3. Use the slider to explore each step of the evolution (defaults to showing the final result)
4. Experiment with different parameters to see how they affect evolution!

### 🎯 Parameter Guide
- **N_e (Population Size)**: Higher = stronger selection, lower = more drift
- **Edits Per Step**: How many mutations happen at once (1 = single letter, 2 = two letters, etc.)
- **Number of Gene Copies**:
  - **1 copy = Essential gene**: Any invalid mutation KILLS the organism (simulation restarts)
  - **2+ copies = With duplications**: Copies evolve independently; one copy can die while others continue exploring
- **Maximum Steps**: How long to run before giving up
- **Max Attempts**: How many times to retry if target isn't reached

### 🧬 Understanding the Visualization
- **🎯 Green star**: Target word (goal of evolution)
- **Node size**: Larger nodes have been visited more often (proportional to visit frequency)
- **Node color**: Shows fitness to target
  - **Blue nodes**: Currently active copy positions
  - **Orange nodes**: Explored but inactive (visited in past but not currently active)
- **Gray background**: Unexplored words (never visited)
- **Hover over nodes**: See word labels, visit counts, and fitness values

### 🔬 Biological Model
This simulates evolution of an essential gene:
- **Without duplications (1 copy)**: Organism must maintain function - any lethal mutation kills it
- **With duplications (2+ copies)**: Some copies can explore "invalid" spaces (lose function) while others maintain viability
- This demonstrates why **gene duplications are a major source of evolutionary innovation**!
""")
