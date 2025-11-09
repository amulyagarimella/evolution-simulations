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
            edits_per_step=edits_per_step
        )

        # Check if we reached the target
        current = start_word
        for step_current, step_attempted, step_accepted, step_status in trajectory:
            if step_accepted:
                current = step_attempted

        if current == target_word:
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
        st.error(f"❌ Failed to reach '{target_word}' after {max_attempts} attempts. Last word: {current}")
        st.info("Try increasing 'Max Attempts' or 'Maximum Steps', or adjust N_e")

# Display trajectory if available
if st.session_state.trajectory is not None:
    trajectory = st.session_state.trajectory
    pos = st.session_state.pos
    word_graph = st.session_state.word_graph
    valid_words = st.session_state.valid_words

    st.markdown("---")
    st.header("📊 Simulation Results")

    # Build the evolutionary path
    path = [trajectory[0][0]]  # Start word
    for i in range(len(trajectory)):
        prev_current, prev_attempted, prev_accepted, prev_status = trajectory[i]
        if prev_accepted:
            path.append(prev_attempted)

    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Steps", len(trajectory))
    with col2:
        st.metric("Path Length", len(path))
    with col3:
        accepted_count = sum(1 for _, _, accepted, _ in trajectory if accepted)
        st.metric("Accepted Mutations", accepted_count)
    with col4:
        rejection_rate = (1 - accepted_count / len(trajectory)) * 100
        st.metric("Rejection Rate", f"{rejection_rate:.1f}%")

    # Show evolutionary path
    st.subheader("Evolutionary Path")
    st.write(" → ".join(path))

    # Interactive trajectory viewer
    st.markdown("---")
    st.subheader("🔍 Explore Trajectory Step-by-Step")

    frame = st.slider(
        "Step",
        min_value=0,
        max_value=len(trajectory) - 1,
        value=0,
        help="Slide to explore each step of the evolution"
    )

    # Get current step info
    current, attempted, accepted, status = trajectory[frame]

    # Build path up to current frame
    current_path = [trajectory[0][0]]
    for i in range(frame + 1):
        _, step_attempted, step_accepted, _ = trajectory[i]
        if step_accepted:
            current_path.append(step_attempted)

    current_word = current_path[-1]

    # Track rejected nodes
    rejected_nodes = set()
    for i in range(frame + 1):
        _, step_attempted, step_accepted, _ = trajectory[i]
        if not step_accepted:
            rejected_nodes.add(step_attempted)

    # Display step information
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"**Step:** {frame}")
        st.markdown(f"**Current Word:** `{current_word}`")
        st.markdown(f"**Attempted:** `{attempted}`")

        if status == 'invalid':
            st.markdown(f"**Result:** ❌ Invalid word")
        elif accepted:
            st.markdown(f"**Result:** ✅ Accepted")
        else:
            st.markdown(f"**Result:** ⚠️ Rejected (fitness)")

        current_fitness = word_fitness(current_word, target_word)
        attempted_fitness = word_fitness(attempted, target_word)
        st.markdown(f"**Current Fitness:** {current_fitness:.3f}")
        st.markdown(f"**Attempted Fitness:** {attempted_fitness:.3f}")

    with col2:
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))

        # Draw unexplored nodes
        unexplored = [w for w in valid_words if w not in current_path and w not in rejected_nodes]
        if unexplored:
            nx.draw_networkx_nodes(word_graph, pos, nodelist=unexplored,
                                   node_color='lightblue',
                                   node_size=100, ax=ax, alpha=0.6)

        # Draw rejected nodes
        if rejected_nodes:
            nx.draw_networkx_nodes(word_graph, pos, nodelist=list(rejected_nodes),
                                   node_color='gray', alpha=0.4,
                                   node_size=80, ax=ax)

        # Draw evolutionary path
        if len(current_path) > 1:
            path_edges = [(current_path[i], current_path[i+1]) for i in range(len(current_path)-1)]
            nx.draw_networkx_edges(word_graph, pos, edgelist=path_edges,
                                   edge_color='green', width=2, ax=ax)
            path_without_current = [w for w in current_path if w != current_word]
            if path_without_current:
                nx.draw_networkx_nodes(word_graph, pos, nodelist=path_without_current,
                                       node_color='lightgreen', node_size=120, ax=ax)

        # Highlight current word
        nx.draw_networkx_nodes(word_graph, pos, nodelist=[current_word],
                               node_color='red', node_size=200, ax=ax)

        # Draw edge to attempted mutation
        if current in word_graph and attempted in word_graph:
            nx.draw_networkx_edges(word_graph, pos, edgelist=[(current, attempted)],
                                   edge_color='orange', width=2, style='dashed', ax=ax)

        # Highlight attempted mutation
        if status == 'invalid':
            nx.draw_networkx_nodes(word_graph, pos, nodelist=[attempted],
                                   node_color='orange', node_size=150,
                                   alpha=0.5, ax=ax)
        elif accepted:
            nx.draw_networkx_nodes(word_graph, pos, nodelist=[attempted],
                                   node_color='green', node_size=150, ax=ax)
        else:
            nx.draw_networkx_nodes(word_graph, pos, nodelist=[attempted],
                                   node_color='yellow', node_size=150, ax=ax)

        ax.set_title(f"Step {frame}: {current_word} → {attempted}")
        ax.axis('off')

        st.pyplot(fig)
        plt.close()

# Instructions at the bottom
st.markdown("---")
st.markdown("""
### 📖 How to Use
1. Set your parameters in the sidebar
2. Click "Run Simulation" to start
3. Use the slider to explore each step of the evolution
4. Experiment with different parameters to see how they affect evolution!

### 🎯 Parameter Guide
- **N_e (Population Size)**: Higher = stronger selection, lower = more drift
- **Edits Per Step**: How many mutations happen at once (1 = single letter, 2 = two letters, etc.)
- **Maximum Steps**: How long to run before giving up
- **Max Attempts**: How many times to retry if target isn't reached
""")
