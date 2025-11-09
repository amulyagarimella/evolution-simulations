import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import random
import enchant  # for valid English words
import os

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Build word graph
def build_word_graph(word_length=4, max_delta=1, use_common_words=True):
    """
    Returns graph where nodes are words, edges connect words within max_delta changes.
    OPTIMIZED: Only checks valid words and generates neighbors directly.
    Uses common words dictionary to keep graph manageable and meaningful.
    """
    d = enchant.Dict("en_US")

    print("Loading valid words...")

    import string

    # Load common 4-letter words from file to constrain the dictionary
    common_words = set()
    if use_common_words:
        common_words_file = os.path.join(os.path.dirname(__file__), "common_words.txt")
        if os.path.exists(common_words_file):
            with open(common_words_file, 'r') as f:
                common_words = {line.strip().upper() for line in f if line.strip()}
            print(f"Loaded {len(common_words)} common words from file")
        else:
            print(f"Warning: {common_words_file} not found, using unrestricted dictionary")
            use_common_words = False

    if use_common_words:
        # Only explore words in the common words set
        seed_words = list(common_words)
    else:
        seed_words = ["WORD"]

    valid_words = set()
    to_check = set(seed_words)
    checked = set()

    print("Building word graph by exploring neighbors...")
    word_graph = nx.Graph()

    # Expand by checking neighbors of valid words and build edges simultaneously
    while to_check:
        word = to_check.pop()
        if word in checked:
            continue
        checked.add(word)

        word_upper = word.upper()

        # If using common words filter, skip if not in the list
        if use_common_words and word_upper not in common_words:
            continue

        if d.check(word_upper):
            valid_words.add(word_upper)
            word_graph.add_node(word_upper, valid=True)

            # Generate neighbors (words that differ by 1 letter)
            for i in range(word_length):
                for c in string.ascii_uppercase:
                    if c != word_upper[i]:
                        neighbor = word_upper[:i] + c + word_upper[i+1:]

                        # If neighbor already found valid, add edge
                        if neighbor in valid_words:
                            word_graph.add_edge(word_upper, neighbor)
                        # Otherwise queue for checking
                        elif neighbor not in checked:
                            # Only queue if in common words (when filtering)
                            if not use_common_words or neighbor in common_words:
                                to_check.add(neighbor)

    print(f"Graph built with {word_graph.number_of_nodes()} nodes and {word_graph.number_of_edges()} edges")
    return word_graph, list(valid_words), []

# Fitness function for word game
def word_fitness(word, target="GENE"):
    """
    Fitness = similarity to target.
    Could also add: common English words have higher base fitness.
    """
    # Distance from target
    distance_from_target = sum(c1 != c2 for c1, c2 in zip(word, target))
    
    # Fitness decreases with distance
    return 1.0 / (1.0 + distance_from_target)

# Evolution simulation
def simulate_word_evolution(word_graph, N_e, start_word, target_word, n_steps=100, show_invalid=True):
    """
    Returns list of (current_word, attempted_word, accepted) tuples for animation.
    """
    trajectory = []
    current = start_word
    
    for step in range(n_steps):
        # Try a random mutation
        neighbors = list(word_graph.neighbors(current))
        if not neighbors:
            break
            
        attempted = random.choice(neighbors)
        
        # Check if valid
        is_valid = word_graph.nodes[attempted]['valid']
        
        if not is_valid:
            # Invalid - selection removes it
            trajectory.append((current, attempted, False, 'invalid'))
            continue
        
        # Valid - check fitness
        current_fitness = word_fitness(current, target_word)
        attempted_fitness = word_fitness(attempted, target_word)
        
        # Selection: accept if better, sometimes accept if neutral/slightly worse
        if attempted_fitness >= current_fitness:
            accept = True
        else:
            # Drift: occasionally accept slightly deleterious
            s = (attempted_fitness - current_fitness) / current_fitness
            p_accept = np.exp(2 * N_e * s)  # Simplified from fixation probability
            accept = (random.random() < p_accept)
        
        trajectory.append((current, attempted, accept, 'valid'))

        if accept:
            current = attempted
            if current == target_word:
                # Reached target - stop simulation
                break
    
    return trajectory

# Visualization
def animate_evolution(word_graph, trajectory, valid_words, invalid_words, attempt_num=1):
    """
    Animate the evolutionary trajectory.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Layout graph - with constrained dictionary, spring layout is fast enough
    print("Computing graph layout...")
    if len(word_graph.nodes()) > 1000:
        # For very large graphs, use Kamada-Kawai (faster than spring, better than random)
        pos = nx.kamada_kawai_layout(word_graph)
        print("Using Kamada-Kawai layout")
    else:
        # Spring layout clusters similar words nicely
        pos = nx.spring_layout(word_graph, k=0.5, iterations=50, seed=42)
        print(f"Using spring layout for {len(word_graph.nodes())} nodes")

    def update(frame):
        ax.clear()

        if frame >= len(trajectory):
            return

        current, attempted, accepted, status = trajectory[frame]

        # Build the evolutionary path up to this point (including current frame)
        path = [trajectory[0][0]]  # Start with the initial word
        for i in range(frame + 1):
            prev_current, prev_attempted, prev_accepted, prev_status = trajectory[i]
            if prev_accepted:
                path.append(prev_attempted)

        # The actual current word is the last word in the path
        current_word = path[-1]

        # Track all nodes that were explored but rejected
        rejected_nodes = set()
        for i in range(frame + 1):
            step_current, step_attempted, step_accepted, step_status = trajectory[i]
            if not step_accepted:
                rejected_nodes.add(step_attempted)

        # Draw all unexplored valid nodes
        unexplored = [w for w in valid_words if w not in path and w not in rejected_nodes]
        if unexplored:
            nx.draw_networkx_nodes(word_graph, pos, nodelist=unexplored,
                                   node_color='lightblue',
                                   node_size=100, ax=ax, alpha=0.6)

        # Draw rejected nodes in gray
        if rejected_nodes:
            nx.draw_networkx_nodes(word_graph, pos, nodelist=list(rejected_nodes),
                                   node_color='gray', alpha=0.4,
                                   node_size=80, ax=ax)

        # Invalid words - gray (if any)
        if invalid_words:
            nx.draw_networkx_nodes(word_graph, pos, nodelist=invalid_words,
                                   node_color='lightgray', alpha=0.3,
                                   node_size=50, ax=ax)

        # Don't draw all edges - only show edges for the evolutionary path
        # This keeps the visualization clean and focused

        # Draw the evolutionary path in green with edges
        if len(path) > 1:
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            nx.draw_networkx_edges(word_graph, pos, edgelist=path_edges,
                                   edge_color='green', width=2, ax=ax)
            # Draw path nodes in green (except current which will be red)
            path_without_current = [w for w in path if w != current_word]
            if path_without_current:
                nx.draw_networkx_nodes(word_graph, pos, nodelist=path_without_current,
                                       node_color='lightgreen', node_size=120, ax=ax)

        # Highlight current word in red
        nx.draw_networkx_nodes(word_graph, pos, nodelist=[current_word],
                               node_color='red', node_size=200, ax=ax)

        # Draw edge from current to attempted
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

        ax.set_title(f"Attempt {attempt_num} | Step {frame}: Current word = {current_word} | Path length: {len(path)}")
        ax.axis('off')

    anim = FuncAnimation(fig, update, frames=len(trajectory), interval=200, repeat=False)
    plt.show()
    return anim

if __name__ == "__main__":
    N_e = 1000
    start_word = "WORD"
    target_word = "GENE"

    word_graph, valid, invalid = build_word_graph(word_length=4, max_delta=1)

    # Repeat simulation until we reach the target word
    attempt = 0
    while True:
        attempt += 1
        print(f"\nAttempt {attempt}: Running simulation from {start_word} to {target_word}...")
        trajectory = simulate_word_evolution(word_graph, N_e, start_word, target_word, n_steps=100)

        # Check if we reached the target
        # Build the final path to see the last word
        current = start_word
        for step_current, step_attempted, step_accepted, step_status in trajectory:
            if step_accepted:
                current = step_attempted

        if current == target_word:
            print(f"Success! Reached {target_word} in attempt {attempt}")
            break
        else:
            print(f"Failed to reach {target_word}. Last word: {current}")

    print(f"\nTotal attempts needed: {attempt}")
    print(f"Showing successful trajectory with {len(trajectory)} steps...")
    animate_evolution(word_graph, trajectory, valid, invalid, attempt_num=attempt)