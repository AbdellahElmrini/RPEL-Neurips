import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm, trange


def generate_connected_graph_old(n, e, seed=None):
    """
    Function to generate a random connected graph with n nodes and e edges.
    """
    if e < n - 1 or e > n * (n - 1) // 2:
        raise ValueError("Invalid number of edges for a simple connected graph.")

    # Generate a random spanning tree
    #G = nx.generators.trees.random_tree(n, seed=seed)
    G = nx.random_labeled_tree(n, seed=seed)
    G = nx.Graph(G)  # Ensure it's an undirected graph

    # Add extra edges randomly until the edge budget is met
    existing_edges = set(G.edges())
    possible_edges = set(
        (i, j) for i in range(n) for j in range(i + 1, n)
    ) - existing_edges

    extra_edges_needed = e - (n - 1)
    random.seed(seed)
    extra_edges = random.sample(possible_edges, extra_edges_needed)
    G.add_edges_from(extra_edges)

    return G


def generate_connected_graph(n, e, seed=None):
    """
    Generate a random connected simple undirected graph with n nodes and e edges.
    """
    if e < n - 1 or e > n * (n - 1) // 2:
        raise ValueError("Invalid number of edges for a simple connected graph.")

    G = nx.random_labeled_tree(n, seed=seed)

    existing_edges = {tuple(sorted(edge)) for edge in G.edges()}

    possible_edges = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if (i, j) not in existing_edges
    ]

    extra_edges_needed = e - (n - 1)

    rng = random.Random(seed)
    extra_edges = rng.sample(possible_edges, extra_edges_needed)

    G.add_edges_from(extra_edges)

    return G


def relabel_graph_for_byzantine(G, nb_honests, nb_workers, strategy='betweenness', seed=None):
    """
    Relabels G so that adversarially chosen nodes occupy the last
    (nb_workers - nb_honests) indices, which the rest of the code
    treats as Byzantine.

    Strategies:
        'random'          : uniform random selection (current baseline)
        'betweenness'     : select highest betweenness centrality nodes
        'max_local_degree': greedy maximization of max honest Byzantine-neighbor count
        'target_min_degree': greedy maximization of worst-off honest Byzantine fraction
    """
    b = nb_workers - nb_honests
    nodes = list(range(nb_workers))

    if strategy == 'random':
        rng = random.Random(seed)
        byz_set = set(rng.sample(nodes, b))

    elif strategy == 'betweenness':
        centrality = nx.betweenness_centrality(G)
        byz_set = set(sorted(nodes, key=lambda n: centrality[n], reverse=True)[:b])

    elif strategy == 'max_local_degree':
        # Greedy: at each step add the node that maximizes
        # max_{i honest} |N(i) ∩ Byzantine|
        byz_set = set()
        for _ in range(b):
            best_node, best_score = None, -1
            for candidate in nodes:
                if candidate in byz_set:
                    continue
                temp_byz = byz_set | {candidate}
                score = max(
                    sum(1 for nbr in G.neighbors(i) if nbr in temp_byz)
                    for i in nodes if i not in temp_byz
                )
                if score > best_score:
                    best_score, best_node = score, candidate
            byz_set.add(best_node)
    elif strategy == 'target_min_degree':
        # Greedy: at each step select the candidate node that maximizes
        # the Byzantine fraction |N(i) ∩ Byzantine| / deg(i)
        # of the worst-off honest node, i.e. the one with lowest degree.
        # Ties in the target honest node are broken by picking the lowest
        # degree honest node, reflecting that low-degree nodes are most
        # vulnerable since a single Byzantine neighbor is a larger fraction
        # of their neighborhood.
        # Once an honest node has strictly more than half its neighbors
        # Byzantine, it is already compromised and excluded from the
        # scoring — the attack focuses on nodes not yet saturated.
        byz_set = set()
        # Precompute neighbor sets once for efficiency
        neighbors = {n: set(G.neighbors(n)) for n in nodes}
        degrees   = {n: len(neighbors[n]) for n in nodes}

        for _ in range(b):
            best_node, best_score, best_target_deg = None, -1.0, float('inf')

            for candidate in nodes:
                if candidate in byz_set:
                    continue
                temp_byz = byz_set | {candidate}
                honest   = [n for n in nodes if n not in temp_byz]

                # Byzantine fraction for each honest node under temp_byz
                byz_fractions = {
                    i: len(neighbors[i] & temp_byz) / degrees[i]
                    for i in honest
                }

                # Restrict to honest nodes not yet compromised, i.e.
                # those with Byzantine fraction <= 1/2
                unsaturated = [i for i in honest if byz_fractions[i] <= 0.5]

                if not unsaturated:
                    # All honest nodes are already compromised under temp_byz;
                    # fall back to picking any candidate to complete the set
                    best_node = candidate
                    break

                # Focus on the unsaturated honest node with the highest
                # Byzantine fraction among those with the lowest degree
                target = min(
                    unsaturated,
                    key=lambda i: ( -byz_fractions[i], degrees[i])
                )
                score  = byz_fractions[target]

                # Prefer higher Byzantine fraction on the target;
                # break ties by preferring lower degree targets
                if (score > best_score or
                        (score == best_score and degrees[target] < best_target_deg)):
                    best_score      = score
                    best_node       = candidate
                    best_target_deg = degrees[target]

            byz_set.add(best_node)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Build relabeling: honest nodes -> 0..nb_honests-1
    #                   byzantine nodes -> nb_honests..nb_workers-1
    honest_list = [n for n in nodes if n not in byz_set]
    byz_list    = list(byz_set)

    mapping = {old: new for new, old in enumerate(honest_list)}
    mapping.update({old: nb_honests + new for new, old in enumerate(byz_list)})

    return nx.relabel_nodes(G, mapping)

def graph_byz_robust(G,byz):
    """
    Function to verify if a graph is Byzantine robust
    in the sense, no honest node has a Byzantine majority
    """
    corrupted_nodes = []
    is_robust = True
    for node in G.nodes():
        if node in byz:
            continue
        neighbors = list(G.neighbors(node))
        byz_neighbors = len([n for n in neighbors if n in byz])
        honest_neighbors = len(neighbors) - byz_neighbors + 1
        if byz_neighbors >= honest_neighbors:
            corrupted_nodes.append(node)
            is_robust = False
    return is_robust, corrupted_nodes

if __name__ == "__main__":
    """combinations = list(itertools.combinations(range(20), 3))

    n_exp = 1000
    n_success = 0
    for i in trange(n_exp):
        G = generate_connected_graph(20, 60, seed = i)
        for byz in combinations:
            is_robust, corrupted_nodes = graph_byz_robust(G, byz)
            if not is_robust:
                #print(f"Byzantine nodes: {byz}, Corrupted nodes: {corrupted_nodes}")
                n_success += 1
                break
    """
    combinations = list(itertools.combinations(range(30), 6))
    print(len(combinations))

    n_exp = 100
    n_success = 0
    for i in trange(n_exp):
        G = generate_connected_graph(30, 225, seed = i)
        for byz in combinations:
            is_robust, corrupted_nodes = graph_byz_robust(G, byz)
            if not is_robust:
                #print(f"Byzantine nodes: {byz}, Corrupted nodes: {corrupted_nodes}")
                n_success += 1
                break

    print(f"Number of successful experiments: {n_success} out of {n_exp}")
    
    """G = generate_connected_graph(30, 225, seed = 0)
    byz = np.random.choice(range(20), size=3, replace=False)
    node_colors = ['red' if node in byz else 'blue' for node in G.nodes()]
    nx.draw_networkx(G, node_color=node_colors)

    

    plt.savefig("graph_for_robustness.png")

    byz = np.random.choice(range(20), size = 3, replace=False)
    print(byz)

    print(graph_byz_robust(G, byz))"""