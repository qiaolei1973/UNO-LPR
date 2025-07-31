import pandas as pd
import numpy as np
import networkx as nx
import json
from tqdm import tqdm
from gensim.models import Word2Vec
import random
import matplotlib.pyplot as plt
import os


def process_assist09_data():
    file_path = './raw_data/assist09/skill_builder_data.csv'
    data = pd.read_csv(file_path, sep=',', encoding='latin1')
    
    data = data.dropna(subset=['user_id', 'assistment_id', 'skill_id', 'correct'])
    
    data['sequence_id'] = data['assistment_id']
    data['log_id'] = data['order_id']
    data['action'] = data['correct'].apply(lambda x: 1 if x > 0 else 0)
    
    user_counts = data['user_id'].value_counts()
    valid_users = user_counts[user_counts >= 5].index
    data = data[data['user_id'].isin(valid_users)]
    
    data['session_id'] = data['assistment_id']
    data['timestamp'] = data['log_id']
    data['item_id'] = data['skill_id'].astype(int)
    columns_to_keep = ['user_id', 'session_id', 'item_id', 'timestamp', 'action']
    data = data[columns_to_keep]
    data['new_item_id'] = pd.factorize(data['item_id'])[0] + 1
    data['new_user_id'] = pd.factorize(data['user_id'])[0] + 1
    item_id_mapping = dict(zip(data['item_id'], data['new_item_id']))
    item_id_mapping = {k: int(v) if isinstance(v, (np.int64, np.int32)) else v 
                      for k, v in item_id_mapping.items()}
    
    with open('./processed_data/assist09/item_id_mapping.json', 'w') as f:
        json.dump(item_id_mapping, f, indent=2)
    
    data['item_id'] = data['new_item_id']
    data['user_id'] = data['new_user_id']
    data = data[columns_to_keep]
    data = data.sort_values(by=['user_id', 'session_id', 'timestamp'])
    
    data.to_csv('./processed_data/assist09/assist09_processed.csv', index=False)
    

    user_lengths = data.groupby('user_id').size()

    long_users = user_lengths[user_lengths < 5]
    print(long_users.sort_values(ascending=False))


    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    user_lengths.hist(bins=50, edgecolor='black')
    plt.title('All Users Interaction Length Distribution')
    plt.xlabel('Number of Interactions')
    plt.ylabel('User Count')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    user_lengths[user_lengths < 200].hist(bins=50, color='orange', edgecolor='black')
    plt.title('Users (Length < 200)')
    plt.xlabel('Number of Interactions')
    plt.ylabel('User Count')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    user_lengths[user_lengths >= 200].hist(bins=50, color='green', edgecolor='black')
    plt.title('Users (Length >= 200)')
    plt.xlabel('Number of Interactions')
    plt.ylabel('User Count')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = 'user_interaction_length_distribution.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    return data, item_id_mapping

def build_transition_graph(data, item_id_mapping):
    all_items = set(item_id_mapping.values())
    concept_num = len(all_items)
    graph = np.zeros((concept_num + 1, concept_num + 1))  
    
    grouped = data.groupby('user_id')
    for user_id, group in tqdm(grouped, desc='Building transition graph'):
        group = group.sort_values('timestamp')
        items = group['item_id'].values
        
        
        for j in range(len(items) - 1):
            pre = items[j]
            next_item = items[j + 1]
            graph[pre, next_item] += 1
    
    
    def normalize_graph(graph):
        graph = np.array(graph, dtype=np.float32)
        np.fill_diagonal(graph, 0)  
        
        
        rowsum = graph.sum(axis=1)
        colsum = graph.sum(axis=0)
        isolated_nodes = (rowsum == 0) & (colsum == 0)
        
        
        rowsum[isolated_nodes] = 1  
        r_inv = np.power(rowsum, -1)
        r_inv[isolated_nodes] = 0    
        norm_graph = np.diag(r_inv).dot(graph)
        
        return norm_graph
    
    return normalize_graph(graph)

def save_transition_graph(graph, item_id_mapping):
    G = nx.DiGraph()
    
    
    for orig_id, new_id in item_id_mapping.items():
        G.add_node(new_id, original_id=orig_id)
    
    
    n_items = len(item_id_mapping)
    for i in range(1, n_items + 1):
        for j in range(1, n_items + 1):
            if graph[i, j] > 0:
                G.add_edge(i, j, weight=graph[i, j])
    
    
    while not nx.is_directed_acyclic_graph(G):
        cycle = nx.find_cycle(G)
        G.remove_edge(*cycle[-1])
    
    
    save_path = './processed_data/assist09/prerequest_graph.gexf'
    nx.write_gexf(G, save_path)
    print(f"Saved transition graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    return G

def generate_embeddings(knowledge_graph, output_path):
    
    WALK_LENGTH = 100
    NUM_WALKS = 100
    EMBEDDING_SIZE = 200
    WINDOW_SIZE = 5
    
    
    node2vec = Node2VecWrapper(
        graph=knowledge_graph,
        is_directed=True,
        p=0.25,
        q=0.5
    )
    walks = node2vec.simulate_walks(
        num_walks=NUM_WALKS,
        walk_length=WALK_LENGTH
    )
    
    
    model = Word2Vec(
        sentences=walks,
        vector_size=EMBEDDING_SIZE,
        window=WINDOW_SIZE,
        min_count=0,
        sg=1,
        hs=0,
        workers=8,
        epochs=10
    )
    
    
    embedding_matrix = np.stack([
        model.wv[node] 
        for node in sorted(knowledge_graph.nodes())
    ])
    
    
    np.save(output_path, embedding_matrix)
    print(f"Embeddings saved to {output_path}")

class Node2VecWrapper:
    def __init__(self, graph, is_directed, p, q):
        self.graph = graph
        self.is_directed = is_directed
        self.p = p
        self.q = q
        for u, v in graph.edges():
            graph[u][v]['weight'] = graph[u][v].get('weight', 1.0)
        self._preprocess_transition_probs()
    
    def _preprocess_transition_probs(self):
        self.alias_nodes = {}
        for node in self.graph.nodes():
            probs = [self.graph[node][nbr]['weight'] for nbr in sorted(self.graph.neighbors(node))]
            norm_probs = [p/sum(probs) for p in probs]
            self.alias_nodes[node] = alias_setup(norm_probs)
        
        self.alias_edges = {}
        if self.is_directed:
            for edge in self.graph.edges():
                self.alias_edges[edge] = self._get_alias_edge(*edge)
        else:
            for edge in self.graph.edges():
                self.alias_edges[edge] = self._get_alias_edge(*edge)
                self.alias_edges[(edge[1], edge[0])] = self._get_alias_edge(edge[1], edge[0])
    
    def _get_alias_edge(self, src, dst):
        unnormalized_probs = []
        for dst_nbr in sorted(self.graph.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(self.graph[dst][dst_nbr]['weight']/self.p)
            elif self.graph.has_edge(dst_nbr, src):
                unnormalized_probs.append(self.graph[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(self.graph[dst][dst_nbr]['weight']/self.q)
        norm_probs = [p/sum(unnormalized_probs) for p in unnormalized_probs]
        return alias_setup(norm_probs)
    
    def simulate_walks(self, num_walks, walk_length):
        walks = []
        nodes = list(self.graph.nodes())
        for _ in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self._node2vec_walk(walk_length, node))
        return walks
    
    def _node2vec_walk(self, walk_length, start_node):
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(self.graph.neighbors(cur))
            if not cur_nbrs:
                break
            if len(walk) == 1:
                walk.append(cur_nbrs[alias_draw(*self.alias_nodes[cur])])
            else:
                prev = walk[-2]
                edge = (prev, cur)
                next_node = cur_nbrs[alias_draw(*self.alias_edges[edge])]
                walk.append(next_node)
        return walk

def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int32)
    smaller, larger = [], []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)
    while smaller and larger:
        small = smaller.pop()
        large = larger.pop()
        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J, q

def alias_draw(J, q):
    K = len(J)
    kk = int(np.random.rand() * K)
    return kk if np.random.rand() < q[kk] else J[kk]

def print_statistics(data):
    print("\n===== Dataset Statistics =====")
    print(f"Total records: {len(data)}")
    print(f"Unique users: {data['user_id'].nunique()}")
    print(f"Unique items: {data['item_id'].nunique()}")
    print(f"Average actions per user: {data.groupby('user_id')['action'].count().mean():.2f}")
    print(f"Average unique items per user: {data.groupby('user_id')['item_id'].nunique().mean():.2f}")
    print(f"Global action rate: {data['action'].mean():.4f}")
    print("Sample data:")
    print(data.head(3))

def process_assist09():
    
    data, item_id_mapping = process_assist09_data()
    print_statistics(data)
    
    
    transition_graph = build_transition_graph(data, item_id_mapping)
    G = save_transition_graph(transition_graph, item_id_mapping)
    
    
    output_path = "./processed_data/assist09/GraphEmbeddings.npy"
    generate_embeddings(G, output_path)
    
    
    embeddings = np.load(output_path)
    print(f"\nEmbedding matrix shape: {embeddings.shape}")

if __name__ == "__main__":
    process_assist09()