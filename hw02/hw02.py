
import argparse, copy, random 
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def load_cora(cora_dir):
    
    content = pd.read_csv(f"{cora_dir}/cora.content", sep="\t", header=None)
    cites   = pd.read_csv(f"{cora_dir}/cora.cites",   sep="\t", header=None)

    paper_ids = content[0].values
    id_map = {pid: i for i, pid in enumerate(paper_ids)}

    features = torch.tensor(content.iloc[:, 1:-1].values, dtype=torch.float32)

    edges = cites.map(id_map.get).values
    G = nx.Graph()
    G.add_nodes_from(range(len(paper_ids)))
    G.add_edges_from(edges)
    return G, features

def build_adj(G):
    N = G.number_of_nodes()
    
    if G.number_of_edges() > 0:
        src, dst = zip(*G.edges())
        src = list(src)
        dst = list(dst)
    else:
        src, dst = [], []
    
    row_indices = src + dst + list(range(N))  
    col_indices = dst + src + list(range(N))  

    adj_indices = torch.tensor([row_indices, col_indices], dtype=torch.long)
    adj_values = torch.ones(len(row_indices), dtype=torch.float32)
    adj_matrix = torch.sparse_coo_tensor(adj_indices, adj_values, (N, N))
    
    adj_dense = adj_matrix.to_dense()
    
    deg = adj_dense.sum(1)  
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    
    D_indices = torch.arange(N).unsqueeze(0).repeat(2, 1)
    D = torch.sparse_coo_tensor(D_indices, deg_inv_sqrt, (N, N))
    
    D_dense = D.to_dense()
    normalized = torch.mm(torch.mm(D_dense, adj_dense), D_dense)
    
    normalized_sparse = normalized.to_sparse_coo()
    
    return normalized_sparse

class GraphConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=False)

    def forward(self, x, adj):
        h = adj.matmul(x) 
        torch.sparse.mm(adj, x)
        return torch.relu(self.linear(h))

class GCNLink(nn.Module):
    def __init__(self, in_dim, hidden_dim=16, dropout=0.1):
        super().__init__()
        self.g1 = GraphConv(in_dim, hidden_dim)
        self.g2 = GraphConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(2 * hidden_dim, 1)
        self.drop = nn.Dropout(dropout)

    def node_encode(self, x, adj):
        h1 = self.drop(self.g1(x, adj))
        return self.g2(h1, adj)

    def forward(self, x, adj, edge_index):
        h = self.node_encode(x, adj)
        src, dst = edge_index
        z = torch.cat([h[src], h[dst]], dim=1)
        return torch.sigmoid(self.lin(z)).squeeze()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int,   default=30)
    parser.add_argument("--lr",       type=float, default=1e-2)
    parser.add_argument("--hidden",   type=int,   default=16)
    parser.add_argument("--dropout",  type=float, default=0.1)
    parser.add_argument("--patience", type=int,   default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    set_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    cora_dir="data/cora"
    G, X = load_cora(cora_dir)
    adj = build_adj(G)

    X = X.to(device)
    adj = adj.to(device) if hasattr(adj, 'to') else adj

   
    edges = np.array(G.edges())
    train_edges, test_edges = train_test_split(edges, test_size=0.2, random_state=SEED)
    train_edges, val_edges  = train_test_split(train_edges,  test_size=0.125, random_state=SEED)

    model = GCNLink(in_dim=X.size(1), hidden_dim=args.hidden, dropout=args.dropout)
    model = model.to(device)
    


if __name__ == "__main__":
    main()