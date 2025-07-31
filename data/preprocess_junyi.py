import pandas as pd
import networkx as nx
import json
import numpy as np

def _read():
    file_path = './raw_data/junyi/junyi_ProblemLog_for_PSLC.txt'
    columns_to_keep = [0, 1, -5, 8, 10]
    data = pd.read_csv(file_path, sep='\t' )
    columns_to_keep = [data.columns[idx] for idx in columns_to_keep]
    data = data[columns_to_keep]
    data.columns = ['user_id', 'session_id', 'item_id', 'timestamp', 'action']
    data = data.dropna(subset=['item_id', 'action', 'user_id', 'session_id', 'timestamp'])
 
    user_counts = data['user_id'].value_counts()
    item_counts = data['item_id'].value_counts()
    valid_users = user_counts[user_counts >= 5].index
    valid_items = item_counts[item_counts >= 5].index
    data = data[data['user_id'].isin(valid_users) & data['item_id'].isin(valid_items)]
    data = data.reset_index(drop=True)

    data['action'] = data['action'].map({'INCORRECT': 0, 'CORRECT': 1, 'HINT': 0})


    data['new_item_id'] = pd.factorize(data['item_id'])[0]+1
    item_id_mapping = dict(zip(data['item_id'], data['new_item_id']))

    item_id_mapping = {k: int(v) if isinstance(v, (np.int64, np.int32)) else v 
                  for k, v in item_id_mapping.items()}

    with open('./processed_data/junyi/item_id_mapping.json', 'w') as f:
        json.dump(item_id_mapping, f, indent=2)

    data['item_id'] = pd.factorize(data['item_id'])[0] + 1
    data['user_id'] = pd.factorize(data['user_id'])[0] + 1
    
    data = data[['user_id', 'session_id', 'item_id', 'timestamp', 'action']]

    data['session_id'] = data.groupby('user_id')['session_id'].transform(lambda x: pd.factorize(x)[0])

    data = data.sort_values(by=['user_id', 'session_id', 'timestamp'])

    data.to_csv('./processed_data/junyi/junyi_processed.csv', index=False)

    return data, item_id_mapping


def build_knowledge_graph(data_path, item_id_mapping):
    df = pd.read_csv(data_path)
    
    df = df[['name', 'prerequisites']].dropna(subset=['name', 'prerequisites'])
    df['prerequisites'] = df['prerequisites'].str.split(',')
    print(df.head(10))
    df['mapped_id'] = df['name'].map(item_id_mapping)
    df['mapped_prereqs'] = df['prerequisites'].apply(
        lambda x: [item_id_mapping[p] for p in x if p in item_id_mapping]
    )
    
    df = df.dropna(subset=['mapped_id']).copy()
    print(df.head(10))
    G = nx.DiGraph()
    for k,node in item_id_mapping.items():
        G.add_node(int(node), original_name=k, is_isolated=True)

    for _, row in df.iterrows():
        G.add_node(int(row['mapped_id']), original_name=row['name'])
        for prereq in row['mapped_prereqs']:
            G.add_edge(int(prereq), int(row['mapped_id']))
    print(f'(include {len(G.nodes())}nodes, {len(G.edges())} edges)')
    return make_dag(G)

def make_dag(G):
    while not nx.is_directed_acyclic_graph(G):
        cycle = nx.find_cycle(G)
        G.remove_edge(*cycle[-1])
    print(f'(include {len(G.nodes())}nodes, {len(G.edges())} edges)')

    return G

def _graph(item_id_mapping:None):

    if not item_id_mapping:
        with open('./processed_data/junyi/item_id_mapping.json') as f:
            item_id_mapping = json.load(f)
    
    G = build_knowledge_graph('./raw_data/junyi/junyi_Exercise_table.csv', item_id_mapping)

    save_G_path = './processed_data/junyi/prerequest_graph.gexf'
    nx.write_gexf(G, save_G_path)

def process_junyi():
    data, item_id_mapping = _read()
    _graph(item_id_mapping)

    avg_problems_per_student = data.groupby('user_id')['item_id'].count().mean()
    avg_unique_problems_per_student = data.groupby('user_id')['item_id'].nunique().mean()
    print(f"Average problems per student: {avg_problems_per_student:.2f}")
    print(f"Average unique problems per student: {avg_unique_problems_per_student:.2f}")
    print(f"Total unique users: {data['user_id'].nunique()}")
    print(f"Total unique items: {data['item_id'].nunique()}")
    print(f"total records: {len(data)}")
    print(data.head(2))


import networkx as nx
from gensim.models import Word2Vec
import numpy as np
import random

def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int32)
    
    smaller = []
    larger = []
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
            probs = [
                self.graph[node][nbr]['weight']
                for nbr in sorted(self.graph.neighbors(node))
            ]
            norm_const = sum(probs)
            norm_probs = [p/norm_const for p in probs]
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
        
        norm_const = sum(unnormalized_probs)
        normalized_probs = [p/norm_const for p in unnormalized_probs]
        return alias_setup(normalized_probs)
    
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
    
def generate_junyi_embeddings(knowledge_graph, output_path):
    WALK_LENGTH = 100    
    NUM_WALKS = 100      
    EMBEDDING_SIZE = 300 
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
        model.wv[str(node)] 
        for node in sorted(knowledge_graph.nodes())
    ])
    
    
    np.save(output_path, embedding_matrix)
    print(f"Embeddings saved to {output_path}")


import networkx as nx

process_junyi()
knowledge_graph = nx.read_gexf('./processed_data/junyi/prerequest_graph.gexf')
output_path="./processed_data/junyi/GraphEmbeddings.npy"


generate_junyi_embeddings(
    knowledge_graph=knowledge_graph,
    output_path=output_path)


embeddings = np.load(output_path)
print(f"Embedding matrix shape: {embeddings.shape}")
print(embeddings)
    