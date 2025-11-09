import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
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
        accepted_count = sum(1 for step_data, _ in trajectory
                           for _, _, accepted, _ in step_data if accepted)
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
        # Create visualization with DENSITY-BASED COLORING
        fig, ax = plt.subplots(figsize=(10, 8))

        # Categorize nodes by visit density
        all_visited_words = set(step_density.keys())
        unexplored = [w for w in valid_words if w not in all_visited_words]

        # Draw unexplored nodes (light gray)
        if unexplored:
            nx.draw_networkx_nodes(word_graph, pos, nodelist=unexplored,
                                   node_color='lightgray',
                                   node_size=50, ax=ax, alpha=0.3)

        # Draw VISITED nodes with DENSITY-BASED COLORING
        # Use colormap to show attractor basins
        if step_density:
            visited_words = list(step_density.keys())
            densities = [step_density[w] for w in visited_words]
            max_density = max(densities) if densities else 1

            # Create colormap - blue (low density) to red (high density)
            import matplotlib.cm as cm
            cmap = cm.get_cmap('YlOrRd')  # Yellow-Orange-Red for heat map

            # Normalize densities
            norm_densities = [d / max_density for d in densities]

            # Draw each visited node with density-based color
            for word, density, norm_density in zip(visited_words, densities, norm_densities):
                # Skip current active words - we'll draw them separately
                if word not in current_copy_positions.values():
                    nx.draw_networkx_nodes(word_graph, pos, nodelist=[word],
                                         node_color=[cmap(norm_density)],
                                         node_size=100 + density * 20,  # Size also shows density
                                         ax=ax, alpha=0.7)

        # Draw edges for all evolutionary paths
        all_path_edges = set()
        for copy_id, path in copy_paths_so_far.items():
            if len(path) > 1:
                path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                all_path_edges.update(path_edges)

        if all_path_edges:
            nx.draw_networkx_edges(word_graph, pos, edgelist=list(all_path_edges),
                                   edge_color='green', width=1.5, ax=ax, alpha=0.4)

        # Highlight CURRENT positions of active copies
        if current_copy_positions:
            nx.draw_networkx_nodes(word_graph, pos,
                                   nodelist=list(current_copy_positions.values()),
                                   node_color='red', node_size=250, ax=ax,
                                   edgecolors='darkred', linewidths=2)

        # Draw edges to attempted mutations in this step
        for copy_id, current, attempted, accepted, status in step_data:
            if current in word_graph and attempted in word_graph:
                edge_color = 'orange' if status == 'invalid' else ('green' if accepted else 'yellow')
                nx.draw_networkx_edges(word_graph, pos, edgelist=[(current, attempted)],
                                       edge_color=edge_color, width=2, style='dashed', ax=ax)

                # Highlight attempted mutations
                if status == 'invalid':
                    nx.draw_networkx_nodes(word_graph, pos, nodelist=[attempted],
                                           node_color='orange', node_size=150,
                                           alpha=0.7, ax=ax, edgecolors='red', linewidths=2)

        ax.set_title(f"Step {frame} | Active Copies: {len(current_copy_positions)}\nColor intensity = visit density (attractor basins)")
        ax.axis('off')

        st.pyplot(fig)
        plt.close()

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
- **Color intensity (yellow→orange→red)**: Shows visit density - darker/redder areas are "attractor basins" where evolution tends to get stuck
- **Node size**: Larger nodes have been visited more often
- **Red nodes with dark border**: Current active copy positions
- **Green edges**: Evolutionary paths taken
- **Orange edges with ❌**: Invalid mutations that killed a copy
- **Yellow dashed edges**: Rejected mutations (fitness-based)

### 🔬 Biological Model
This simulates evolution of an essential gene:
- **Without duplications (1 copy)**: Organism must maintain function - any lethal mutation kills it
- **With duplications (2+ copies)**: Some copies can explore "invalid" spaces (lose function) while others maintain viability
- This demonstrates why **gene duplications are a major source of evolutionary innovation**!
""")
