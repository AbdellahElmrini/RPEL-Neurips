import numpy as np
import networkx as nx

def LaplacianGossipMatrix(G):
    max_degree = max([G.degree(node) for node in G.nodes()])

    sorted_nodes = sorted(G.nodes())
    W = np.eye(G.number_of_nodes()) - 1/max_degree * nx.laplacian_matrix(G, nodelist=sorted_nodes ).toarray()
    return W